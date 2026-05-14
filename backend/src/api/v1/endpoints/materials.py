from __future__ import annotations

import json
import logging
from pathlib import Path

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import ValidationError
from qdrant_client import models as qdrant_models

from src.core.config import Settings
from src.core.security import UploadValidationError, ensure_child_path, stream_upload_to_temp
from src.dependencies import get_app_settings, get_material_service, verify_owner_access
from src.models.chunk import Chunk
from src.models.material import Material, get_material_pages
from src.models.pipeline_job import PipelineJob
from src.rag.vector_store import get_qdrant_client_for_settings
from src.schemas.common import APIResponse
from src.schemas.material import (
    DebugBBox,
    DebugBlock,
    DebugChunk,
    DebugPage,
    MaterialBatchUploadItem,
    MaterialBatchUploadResponse,
    MaterialDebugResponse,
    MaterialResponse,
    MaterialStatusResponse,
    MaterialUploadMetadata,
    MaterialUploadResponse,
)
from src.services.material_service import MaterialService

router = APIRouter(prefix="/materials", tags=["materials"])
logger = logging.getLogger(__name__)

STATUS_PROGRESS = {
    "uploaded": 10,
    "parsing": 30,
    "parsed": 55,
    "indexing": 80,
    "indexed": 100,
    "failed": 100,
}


@router.get("", response_model=APIResponse[list[MaterialResponse]])
async def list_materials(
    request: Request,
    owner_id: str = Query(..., min_length=1),
    collection_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> APIResponse[list[MaterialResponse]]:
    verify_owner_access(request, owner_id)
    match_query: dict = {"owner_id": owner_id}
    if collection_id:
        try:
            col_oid = PydanticObjectId(collection_id)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="collection_id must be a valid ObjectId") from exc
        match_query["collection_id"] = col_oid
    docs = await Material.aggregate([
        {"$match": match_query},
        {"$sort": {"updated_at": -1}},
        {"$limit": limit},
        {"$project": {"pages": 0}},
    ]).to_list()
    results = [
        MaterialResponse(
            material_id=str(doc["_id"]),
            collection_id=str(doc["collection_id"]),
            owner_id=doc["owner_id"],
            filename=doc["filename"],
            original_name=doc["original_name"],
            file_type=doc["file_type"],
            status=doc["status"],
            subject=doc.get("subject"),
            topic=doc.get("topic"),
            page_count=doc.get("page_count"),
            version=doc["version"],
        )
        for doc in docs
    ]
    return APIResponse(success=True, message="Materials loaded successfully", data=results, error=None)


@router.post(
    "/upload",
    response_model=APIResponse[MaterialUploadResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_material(
    request: Request,
    metadata: str = Form(...),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
    material_service: MaterialService = Depends(get_material_service),
) -> APIResponse[MaterialUploadResponse]:
    try:
        parsed_metadata = MaterialUploadMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be valid JSON") from exc
    verify_owner_access(request, parsed_metadata.owner_id, settings)

    streamed = None
    try:
        streamed = await stream_upload_to_temp(
            file,
            settings.max_upload_size_bytes,
            settings.data_dir / "cache" / "uploads",
        )
        result = await material_service.upload_material_from_temp(
            metadata=parsed_metadata,
            original_filename=file.filename or "upload.bin",
            content_type=file.content_type,
            temp_path=streamed.temp_path,
            file_size_bytes=streamed.size_bytes,
            checksum_sha256=streamed.checksum_sha256,
            head=streamed.head,
        )
    except UploadValidationError as exc:
        logger.exception("Upload validation failed")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload request.") from exc
    except LookupError as exc:
        logger.exception("Upload target not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload target not found.") from exc
    except ValueError as exc:
        logger.exception("Invalid upload request")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload request.") from exc
    finally:
        if streamed is not None:
            streamed.temp_path.unlink(missing_ok=True)

    return APIResponse(success=True, message="Material uploaded successfully", data=result, error=None)


@router.post(
    "/batch_upload",
    response_model=APIResponse[MaterialBatchUploadResponse],
    status_code=status.HTTP_207_MULTI_STATUS,
)
async def batch_upload_materials(
    request: Request,
    metadata: str = Form(...),
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_app_settings),
    material_service: MaterialService = Depends(get_material_service),
) -> APIResponse[MaterialBatchUploadResponse]:
    try:
        raw_metadata = json.loads(metadata)
        if isinstance(raw_metadata, dict):
            metadata_items = [MaterialUploadMetadata.model_validate(raw_metadata) for _ in files]
        elif isinstance(raw_metadata, list):
            if len(raw_metadata) != len(files):
                raise ValueError("metadata list length must match files length")
            metadata_items = [MaterialUploadMetadata.model_validate(item) for item in raw_metadata]
        else:
            raise ValueError("metadata must be an object or list")
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        logger.exception("Invalid batch metadata")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata must be valid JSON matching the upload schema") from exc

    for item in metadata_items:
        verify_owner_access(request, item.owner_id, settings)

    results: list[MaterialBatchUploadItem] = []
    for upload_file, item_metadata in zip(files, metadata_items, strict=True):
        streamed = None
        filename = upload_file.filename or "upload.bin"
        try:
            streamed = await stream_upload_to_temp(
                upload_file,
                settings.max_upload_size_bytes,
                settings.data_dir / "cache" / "uploads",
            )
            data = await material_service.upload_material_from_temp(
                metadata=item_metadata,
                original_filename=filename,
                content_type=upload_file.content_type,
                temp_path=streamed.temp_path,
                file_size_bytes=streamed.size_bytes,
                checksum_sha256=streamed.checksum_sha256,
                head=streamed.head,
            )
            results.append(MaterialBatchUploadItem(filename=filename, success=True, data=data))
        except (UploadValidationError, LookupError, ValueError) as exc:
            logger.exception("Batch material upload item failed", extra={"filename": filename})
            results.append(MaterialBatchUploadItem(filename=filename, success=False, error="Material upload failed. Please retry later."))
        finally:
            if streamed is not None:
                streamed.temp_path.unlink(missing_ok=True)

    success_count = sum(1 for item in results if item.success)
    return APIResponse(
        success=success_count > 0,
        message=f"{success_count}/{len(results)} file(s) uploaded successfully",
        data=MaterialBatchUploadResponse(results=results),
        error=None if success_count > 0 else "All uploads failed",
    )


@router.get("/{material_id}/status", response_model=APIResponse[MaterialStatusResponse])
async def material_status(
    request: Request,
    material_id: str,
    owner_id: str = Query(..., min_length=1),
) -> APIResponse[MaterialStatusResponse]:
    verify_owner_access(request, owner_id)
    try:
        mid = PydanticObjectId(material_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_id must be a valid ObjectId") from exc

    material = await Material.get(mid)
    if material is None or material.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found for this owner")

    job = await PipelineJob.find_one(PipelineJob.material_id == mid)
    stage = job.stage if job else material.status
    normalized_stage = stage.lower()
    normalized_status = material.status.lower()
    progress = STATUS_PROGRESS.get(normalized_stage, STATUS_PROGRESS.get(normalized_status, 0))

    return APIResponse(
        success=True,
        message="Material status loaded successfully",
        data=MaterialStatusResponse(
            material_id=str(material.id),
            collection_id=str(material.collection_id),
            status=material.status,
            stage=stage,
            progress_pct=progress,
            failed_stage=material.failed_stage or (job.failed_stage if job else None),
            error_message=material.error_message or (job.last_error if job else None),
        ),
        error=None,
    )


@router.get("/{material_id}/debug", response_model=APIResponse[MaterialDebugResponse])
async def material_debug(
    request: Request,
    material_id: str,
    owner_id: str = Query(..., min_length=1),
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[MaterialDebugResponse]:
    verify_owner_access(request, owner_id, settings)
    try:
        mid = PydanticObjectId(material_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_id must be a valid ObjectId") from exc

    material = await Material.get(mid)
    if material is None or material.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found for this owner")

    material_pages = await get_material_pages(material)
    pages = [
        DebugPage(
            page_number=p.page_number,
            width=p.width,
            height=p.height,
            ocr_confidence=p.ocr_confidence,
            blocks=[
                DebugBlock(
                    block_id=b.block_id,
                    block_index=b.block_index,
                    block_type=b.block_type,
                    content=b.content,
                    language=b.language,
                    bbox=DebugBBox(**b.bbox.model_dump()) if b.bbox else None,
                    ocr_confidence=b.ocr_confidence,
                    reading_order=b.reading_order,
                )
                for b in p.blocks
            ],
        )
        for p in material_pages
    ]

    chunk_docs = await Chunk.find(
        Chunk.owner_id == owner_id,
        Chunk.material_id == mid,
    ).to_list()
    chunks = [
        DebugChunk(
            chunk_id=str(c.id),
            content=c.content,
            language=c.language,
            modality=c.modality,
            token_count=c.token_count,
            source_block_ids=c.source_block_ids,
            source_pages=c.source_pages,
            chunk_strategy=c.chunk_strategy,
            embedding_model=c.embedding_model,
        )
        for c in chunk_docs
    ]

    vector_count = 0
    try:
        qdrant = get_qdrant_client_for_settings(settings)
        result = qdrant.count(
            collection_name=settings.qdrant_collection_name,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="material_id",
                        match=qdrant_models.MatchValue(value=str(mid)),
                    )
                ]
            ),
            exact=True,
        )
        vector_count = result.count
    except Exception:
        vector_count = 0

    is_image = material.file_type.lower() in {"png", "jpg", "jpeg"}
    raw_image_url = (
        f"{settings.api_v1_prefix}/materials/{material_id}/raw?owner_id={owner_id}" if is_image else None
    )

    return APIResponse(
        success=True,
        message="Debug data loaded successfully",
        data=MaterialDebugResponse(
            material_id=str(material.id),
            collection_id=str(material.collection_id),
            owner_id=material.owner_id,
            original_name=material.original_name,
            file_type=material.file_type,
            status=material.status,
            modality=material.modality,
            language=material.language,
            page_count=material.page_count or len(pages),
            pages=pages,
            chunks=chunks,
            qdrant_vector_count=vector_count,
            raw_image_url=raw_image_url,
        ),
        error=None,
    )


@router.get("/{material_id}/raw")
async def material_raw_file(
    request: Request,
    material_id: str,
    owner_id: str = Query(..., min_length=1),
    settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    verify_owner_access(request, owner_id, settings)
    try:
        mid = PydanticObjectId(material_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_id must be a valid ObjectId") from exc

    material = await Material.get(mid)
    if material is None or material.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found for this owner")

    storage_path = Path(material.storage_path)
    if storage_path.is_absolute():
        target = storage_path
    else:
        target = settings.data_dir / storage_path
        if not target.exists():
            target = settings.data_dir.parent / storage_path
    target = target.resolve()

    root = settings.data_dir.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Path traversal denied")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw file not found")

    return FileResponse(target, filename=material.original_name)


@router.post("/{material_id}/retry", response_model=APIResponse[dict])
async def retry_material(
    request: Request,
    material_id: str,
    owner_id: str = Query(..., min_length=1),
    material_service: MaterialService = Depends(get_material_service),
) -> APIResponse[dict]:
    verify_owner_access(request, owner_id)
    try:
        result = await material_service.retry_material(material_id=material_id, owner_id=owner_id)
    except ValueError as exc:
        logger.exception("Invalid retry request", extra={"material_id": material_id, "owner_id": owner_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid retry request.") from exc
    except LookupError as exc:
        logger.exception("Retry target not found", extra={"material_id": material_id, "owner_id": owner_id})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found.") from exc
    return APIResponse(success=True, message="Pipeline retry queued", data=result, error=None)


@router.delete("/{material_id}", response_model=APIResponse[dict])
async def delete_material(
    request: Request,
    material_id: str,
    owner_id: str = Query(..., min_length=1),
    material_service: MaterialService = Depends(get_material_service),
) -> APIResponse[dict]:
    verify_owner_access(request, owner_id)
    try:
        counts = await material_service.delete_material(material_id=material_id, owner_id=owner_id)
    except ValueError as exc:
        logger.exception("Invalid material deletion request", extra={"material_id": material_id, "owner_id": owner_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid material deletion request.") from exc
    except LookupError as exc:
        logger.exception("Material deletion target not found", extra={"material_id": material_id, "owner_id": owner_id})
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found.") from exc
    return APIResponse(success=True, message="Material deleted successfully", data=counts, error=None)
