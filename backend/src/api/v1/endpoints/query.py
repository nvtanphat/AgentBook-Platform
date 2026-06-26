import asyncio
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from src.core.rate_limit import limiter
from src.dependencies import get_query_service, get_study_guide_service, get_summary_service, verify_owner_access
from src.schemas.common import APIResponse
from src.schemas.query import (
    CompareRequest,
    CompareResponse,
    QueryByGraphRequest,
    QueryByImageRequest,
    QueryRequest,
    QueryResponse,
    StudyGuideRequest,
    StudyGuideResponse,
    SummaryRequest,
    SummaryResponse,
)
from src.services.query_service import QueryService
from src.services.study_guide_service import StudyGuideService
from src.services.summary_service import SummaryService

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)

# Cap concurrent heavy query executions. On a CPU box, the embedding/rerank steps
# are serialized by a process-wide semaphore; letting many abandoned/retried
# requests run at once thrashes the CPU and cascades every query into multi-minute
# latency. Bounding concurrency keeps each query close to its standalone time.
_QUERY_CONCURRENCY = asyncio.Semaphore(2)


async def _run_cancellable(request: Request, make_coro):
    """Run a query coroutine under the concurrency cap, cancelling it if the
    client disconnects so abandoned work stops instead of piling up on the CPU.
    """
    async with _QUERY_CONCURRENCY:
        task = asyncio.ensure_future(make_coro())
        watcher = asyncio.ensure_future(_until_disconnect(request))
        try:
            done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            # Client went away first — stop the pipeline.
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            logger.info("Query cancelled — client disconnected before completion")
            raise asyncio.CancelledError()
        finally:
            watcher.cancel()


async def _until_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(1.0)


@router.post("/ask", response_model=APIResponse[QueryResponse])
@limiter.limit("15/minute")
async def ask(
    request: Request,
    body: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[QueryResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await _run_cancellable(request, lambda: query_service.ask(body))
    except asyncio.CancelledError:
        # Client disconnected before completion — the connection is already gone,
        # so the status code is moot; just stop without running the pipeline further.
        raise HTTPException(status_code=499, detail="Client closed request.") from None
    except ValueError as exc:
        logger.warning("Invalid query request", extra={"owner_id": body.owner_id, "collection_id": body.collection_id, "error": str(exc)})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid query request.") from exc
    except Exception as exc:
        logger.exception("Query pipeline failed", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query pipeline failed. Please retry later.",
        ) from exc
    return APIResponse(success=True, message="Query answered successfully", data=result, error=None)


@router.post("/ask-stream")
@limiter.limit("15/minute")
async def ask_stream(
    request: Request,
    body: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> StreamingResponse:
    verify_owner_access(request, body.owner_id)

    async def generate():
        # CPU-bound steps (BGE-M3 embedding, local LLM generation) can run for
        # minutes without emitting an SSE event. A silent gap that long makes
        # ngrok / proxies / the browser treat the stream as dead and drop it
        # ("connection error while receiving data"). Emit a keepalive comment
        # every HEARTBEAT_SECONDS of silence — SSE comment lines (": …") are
        # ignored by EventSource but keep the TCP connection alive.
        heartbeat_seconds = 15.0
        # Hold a concurrency slot for the stream's lifetime; released on completion
        # or client disconnect (Starlette throws GeneratorExit → finally runs).
        async with _QUERY_CONCURRENCY:
            agen = query_service.ask_stream(body).__aiter__()
            pending: asyncio.Future | None = None
            try:
                while True:
                    if pending is None:
                        # One task pulls the next event; on a heartbeat timeout we keep
                        # awaiting THIS SAME task (never cancel it) so the underlying
                        # async generator is not corrupted mid-step.
                        pending = asyncio.ensure_future(agen.__anext__())
                    done, _ = await asyncio.wait({pending}, timeout=heartbeat_seconds)
                    if pending not in done:
                        yield ": keepalive\n\n"
                        continue
                    task, pending = pending, None
                    try:
                        line = task.result()
                    except StopAsyncIteration:
                        break
                    yield line
            except ValueError:
                logger.warning("Invalid streaming query request", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
                yield f"event: error\ndata: {json.dumps({'message': 'Invalid query request.'})}\n\n"
            except Exception:
                logger.exception("Streaming query pipeline failed", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
                yield f"event: error\ndata: {json.dumps({'message': 'Query pipeline failed. Please retry later.'})}\n\n"
            finally:
                # Client disconnect / completion: cancel any in-flight pull and close
                # the upstream generator so the pipeline doesn't keep running detached.
                if pending is not None:
                    pending.cancel()
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask-graph", response_model=APIResponse[QueryResponse])
@limiter.limit("15/minute")
async def ask_graph(
    request: Request,
    body: QueryByGraphRequest,
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[QueryResponse]:
    """GraphRAG anchored query — user selected node(s)/edge(s) on the
    knowledge graph; answer is grounded in chunks reachable from those
    entities + their K-hop neighbours."""
    verify_owner_access(request, body.owner_id)
    if not body.entity_ids and not body.relation_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="entity_ids or relation_ids is required for graph-anchored query",
        )
    try:
        result = await query_service.ask_with_graph_anchor(body)
    except ValueError as exc:
        logger.warning("Invalid graph-query request", extra={"owner_id": body.owner_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid query request.") from exc
    except Exception as exc:
        logger.exception("Graph-query pipeline failed", extra={"owner_id": body.owner_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Graph-query pipeline failed. Please retry later.",
        ) from exc
    return APIResponse(success=True, message="Graph query answered successfully", data=result, error=None)


_IMAGE_QUERY_MAX_BYTES = 8 * 1024 * 1024  # 8 MB cap for upload-as-query
_IMAGE_QUERY_ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"}


@router.post("/ask-image", response_model=APIResponse[QueryResponse])
@limiter.limit("8/minute")
async def ask_image(
    request: Request,
    image: UploadFile = File(...),
    owner_id: str = Form(...),
    collection_id: str | None = Form(default=None),
    material_ids: str | None = Form(default=None),
    conversation_id: str = Form(default="default"),
    query_text: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
    answer_language: str | None = Form(default=None),
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[QueryResponse]:
    """Image-as-Query: user uploads an image (with optional caption text).
    Backend SigLIP-embeds the upload, finds visually similar figures in the
    user's collection, then synthesises a grounded answer."""
    verify_owner_access(request, owner_id)

    if image.content_type and image.content_type.lower() not in _IMAGE_QUERY_ALLOWED_CT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image content-type: {image.content_type}",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")
    if len(image_bytes) > _IMAGE_QUERY_MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image exceeds 8 MB limit")

    parsed_material_ids: list[str] = []
    if material_ids:
        try:
            decoded = json.loads(material_ids)
            if isinstance(decoded, list):
                parsed_material_ids = [str(m) for m in decoded if isinstance(m, (str, int))]
        except json.JSONDecodeError:
            parsed_material_ids = [m.strip() for m in material_ids.split(",") if m.strip()]

    try:
        body = QueryByImageRequest(
            owner_id=owner_id,
            collection_id=collection_id,
            material_ids=parsed_material_ids,
            conversation_id=conversation_id,
            query_text=query_text,
            top_k=top_k,
            answer_language=answer_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request fields: {exc}")

    try:
        result = await query_service.ask_with_image(
            request=body,
            image_bytes=image_bytes,
            image_filename=image.filename or "upload",
        )
    except ValueError as exc:
        logger.warning("Invalid image-query request", extra={"owner_id": owner_id, "collection_id": collection_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid query request.") from exc
    except Exception as exc:
        logger.exception("Image-query pipeline failed", extra={"owner_id": owner_id, "collection_id": collection_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image-query pipeline failed. Please retry later.",
        ) from exc
    return APIResponse(success=True, message="Image query answered successfully", data=result, error=None)


@router.post("/compare", response_model=APIResponse[CompareResponse])
@limiter.limit("10/minute")
async def compare(
    request: Request,
    body: CompareRequest,
    query_service: QueryService = Depends(get_query_service),
) -> APIResponse[CompareResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await query_service.compare(body)
    except ValueError as exc:
        logger.warning("Invalid compare request", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid compare request.") from exc
    except Exception as exc:
        logger.exception("Compare pipeline failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Compare pipeline failed. Please retry later.") from exc
    return APIResponse(success=True, message="Comparison generated successfully", data=result, error=None)


@router.post("/summarize", response_model=APIResponse[SummaryResponse])
async def summarize(
    request: Request,
    body: SummaryRequest,
    summary_service: SummaryService = Depends(get_summary_service),
) -> APIResponse[SummaryResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await summary_service.summarize(body)
    except ValueError as exc:
        logger.warning("Invalid summary request", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid summary request.") from exc
    except Exception as exc:
        logger.exception("Summary pipeline failed", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Summary pipeline failed. Please retry later.") from exc
    return APIResponse(success=True, message="Summary generated successfully", data=result, error=None)


@router.post("/study-guide", response_model=APIResponse[StudyGuideResponse])
async def study_guide(
    request: Request,
    body: StudyGuideRequest,
    study_guide_service: StudyGuideService = Depends(get_study_guide_service),
) -> APIResponse[StudyGuideResponse]:
    verify_owner_access(request, body.owner_id)
    try:
        result = await study_guide_service.build(body)
    except ValueError as exc:
        logger.warning("Invalid study guide request", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid study guide request.") from exc
    except Exception as exc:
        logger.exception("Study guide pipeline failed", extra={"owner_id": body.owner_id, "collection_id": body.collection_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Study guide pipeline failed. Please retry later.") from exc
    return APIResponse(success=True, message="Study guide generated successfully", data=result, error=None)
