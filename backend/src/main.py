from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from qdrant_client import models

from src.api.v1.router import api_router
from src.core.background import shutdown_background_tasks
from src.core.config import get_settings, project_root
from src.core.rate_limit import limiter
from src.database import close_database, init_database
from src.dependencies import close_query_service
from src.rag.vector_store import close_cached_qdrant_client, get_qdrant_client_for_settings
from src.schemas.admin import HealthResponse


def _configure_logging() -> None:
    log_config_path = project_root() / "config" / "logging_config.yaml"
    if log_config_path.exists():
        with open(log_config_path, encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")


def _ensure_qdrant_collection(settings) -> None:
    client = get_qdrant_client_for_settings(settings)
    collection_name = settings.qdrant_collection_name
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=settings.embedding_dense_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "bge_m3_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        logging.getLogger(__name__).info("Created Qdrant collection: %s", collection_name)
    _ensure_qdrant_payload_indexes(client, collection_name)


def _ensure_qdrant_payload_indexes(client, collection_name: str) -> None:
    logger = logging.getLogger(__name__)
    for field_name in ["owner_id", "collection_id", "material_id", "language", "modality"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            logger.debug(
                "Qdrant payload index already exists or could not be created",
                extra={"collection": collection_name, "field": field_name, "error": str(exc)},
            )
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="content_text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.MULTILINGUAL,
                lowercase=True,
            ),
            wait=True,
        )
    except Exception as exc:
        logger.debug(
            "Qdrant text payload index already exists or could not be created",
            extra={"collection": collection_name, "field": "content_text", "error": str(exc)},
        )


async def _recover_stuck_materials() -> None:
    """Reset materials left in intermediate states by a previous crash."""
    from src.models.material import Material
    from src.models.common import PipelineStatus, utc_now

    stuck_statuses = [
        PipelineStatus.PARSING.value,
        PipelineStatus.PARSED.value,
        PipelineStatus.CHUNKING.value,
        PipelineStatus.EMBEDDING.value,
        PipelineStatus.INDEXING.value,
    ]
    logger = logging.getLogger(__name__)
    stuck = await Material.find({"status": {"$in": stuck_statuses}}).to_list()
    if not stuck:
        return
    logger.warning(
        "Recovering %d materials stuck in intermediate pipeline states",
        len(stuck),
        extra={"material_ids": [str(m.id) for m in stuck]},
    )
    for material in stuck:
        material.status = PipelineStatus.FAILED.value
        material.failed_stage = material.status
        material.error_message = "Process crashed mid-pipeline; retry to re-process."
        material.updated_at = utc_now()
        await material.save()
    logger.info("Marked %d stuck materials as failed — use retry endpoint to re-process.", len(stuck))


async def _ensure_seed_user(settings) -> None:
    """Create the default user from SEED_USER_EMAIL / SEED_USER_PASSWORD if missing.

    Runs on every startup so the account survives a full data reset.
    No-op when credentials are not configured or the user already exists.
    """
    email = settings.seed_user_email
    password = settings.seed_user_password
    if not email or not password:
        return
    from src.models.user import User
    from src.services.auth_service import AuthService, AuthError
    logger = logging.getLogger(__name__)
    existing = await User.find_one({"email": email.lower().strip()})
    if existing:
        return
    try:
        auth = AuthService(settings)
        await auth.register(email=email, password=password, display_name="Admin")
        logger.info("Seed user created", extra={"email": email})
    except AuthError as exc:
        logger.warning("Seed user already exists or could not be created", extra={"error": str(exc)})


async def _reenqueue_uploaded_materials(settings) -> None:
    """Re-enqueue materials stuck in 'uploaded' state (asyncio.create_task lost on restart)."""
    from src.models.material import Material
    from src.models.common import PipelineStatus
    from src.services.material_service import MaterialService

    logger = logging.getLogger(__name__)
    uploaded = await Material.find({"status": PipelineStatus.UPLOADED.value}).to_list()
    if not uploaded:
        return
    logger.info(
        "Re-enqueueing %d uploaded materials whose pipeline task was lost on restart",
        len(uploaded),
        extra={"material_ids": [str(m.id) for m in uploaded]},
    )
    service = MaterialService(settings=settings)
    ok = 0
    for material in uploaded:
        try:
            await service.retry_material(material_id=str(material.id), owner_id=material.owner_id)
            ok += 1
        except Exception as exc:
            logger.warning(
                "Failed to re-enqueue uploaded material",
                extra={"material_id": str(material.id), "error": str(exc)},
            )
    logger.info("Re-enqueued %d/%d uploaded materials", ok, len(uploaded))


async def _warmup_models(settings, logger: logging.Logger) -> None:
    """Pre-load BGE-M3 embedder and CrossEncoder reranker into RAM at startup."""
    import asyncio

    async def _load_embedder():
        try:
            from src.rag.embedder import get_cached_bge_m3_model
            await asyncio.to_thread(get_cached_bge_m3_model, settings)
            logger.info("BGE-M3 embedder warmed up")
        except Exception as exc:
            logger.warning("BGE-M3 embedder warmup failed (will warm on first query): %s", exc)

    async def _load_reranker():
        try:
            from src.rag.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(settings)
            await reranker._aload_model()
            logger.info("CrossEncoder reranker warmed up")
        except Exception as exc:
            logger.warning("CrossEncoder reranker warmup failed (will warm on first query): %s", exc)

    # Sequentially, not gathered: initializing two torch models concurrently
    # races on device placement and raises "Cannot copy out of meta tensor".
    await _load_embedder()
    await _load_reranker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    settings = get_settings()
    logger = logging.getLogger(__name__)
    if not settings.api_auth_enabled:
        logger.warning("API auth is disabled; do not expose this service directly to the internet")
    elif not settings.api_key:
        logger.critical("API auth is enabled but AGENTBOOK_API_KEY is not configured; scoped API requests will fail closed")
    await init_database(settings)
    _ensure_qdrant_collection(settings)
    if not settings.testing:
        await _ensure_seed_user(settings)
        await _recover_stuck_materials()
        await _reenqueue_uploaded_materials(settings)
    # Pre-warm heavy models as a background task so startup doesn't block.
    # create_task before yield = task runs concurrently while server serves traffic.
    if not settings.testing:
        import asyncio
        asyncio.create_task(_warmup_models(settings, logger))
    yield
    await shutdown_background_tasks()
    await close_query_service()
    close_cached_qdrant_client()
    await close_database()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service=settings.app_name)

    return app


app = create_app()
