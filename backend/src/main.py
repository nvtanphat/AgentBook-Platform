from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from qdrant_client import models

from src.api.v1.router import api_router
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
    yield
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
