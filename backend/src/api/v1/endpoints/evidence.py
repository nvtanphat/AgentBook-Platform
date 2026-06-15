from __future__ import annotations

from pathlib import Path

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from src.core.config import Settings
from src.core.security import UploadValidationError, ensure_child_path
from src.dependencies import get_app_settings, verify_owner_access
from src.models.material import Material, get_material_page, get_material_pages
from src.schemas.common import APIResponse
from src.schemas.evidence import BoundingBoxSchema, EvidenceBlockSchema, EvidencePageResponse

router = APIRouter(prefix="/evidence", tags=["evidence"])

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


@router.get("/figure/{doc_id}")
async def get_figure_image(
    request: Request,
    doc_id: str,
    block_id: str = Query(..., min_length=1),
    owner_id: str = Query(..., min_length=1),
    settings: Settings = Depends(get_app_settings),
) -> FileResponse:
    """Serve the cropped/embedded figure image for a specific block."""
    verify_owner_access(request, owner_id)
    try:
        material = await Material.get(PydanticObjectId(doc_id))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="doc_id must be a valid ObjectId") from exc
    if material is None or material.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    # Search all pages for the block
    pages = await get_material_pages(material)
    target_block = None
    for mat_page in pages:
        for block in mat_page.blocks:
            if block.block_id == block_id:
                target_block = block
                break
        if target_block is not None:
            break

    if target_block is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Block not found")
    if target_block.block_type != "figure":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Block is not a figure")

    img_path_str = (target_block.extra or {}).get("figure_image_path")
    if not img_path_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figure image not available")

    img_path = Path(img_path_str)
    try:
        ensure_child_path(settings.data_dir, img_path)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied") from exc

    if not img_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figure image file not found on disk")

    suffix = img_path.suffix.lower().lstrip(".")
    media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
    return FileResponse(path=str(img_path), media_type=media_type, filename=img_path.name)


@router.get("/{doc_id}/{page}", response_model=APIResponse[EvidencePageResponse])
async def get_evidence_page(
    request: Request,
    doc_id: str,
    page: int,
    owner_id: str = Query(..., min_length=1),
    collection_id: str | None = Query(default=None),
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[EvidencePageResponse]:
    verify_owner_access(request, owner_id)
    try:
        material = await Material.get(PydanticObjectId(doc_id))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="doc_id must be a valid ObjectId") from exc
    if material is None or material.owner_id != owner_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence page not found")
    if collection_id and str(material.collection_id) != collection_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence page not found")

    material_page = await get_material_page(material, page)
    if material_page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence page not found")

    doc_id_str = str(material.id)
    doc_name = material.original_name
    file_type = (material.file_type or "").lower()
    is_image = file_type in _IMAGE_EXTENSIONS
    raw_image_url = (
        f"{settings.api_v1_prefix}/materials/{doc_id_str}/raw?owner_id={owner_id}"
        if is_image
        else None
    )
    def _figure_url(block_id: str) -> str | None:
        from urllib.parse import quote
        return (
            f"{settings.api_v1_prefix}/evidence/figure/{doc_id_str}"
            f"?block_id={quote(block_id, safe='')}&owner_id={owner_id}"
        )

    result = EvidencePageResponse(
        doc_id=doc_id_str,
        doc_name=doc_name,
        page=page,
        source_filename=Path(material.storage_path).name,
        file_type=file_type or None,
        raw_image_url=raw_image_url,
        blocks=[
            EvidenceBlockSchema(
                block_id=block.block_id,
                block_type=block.block_type,
                page=page,
                snippet_original=block.content,
                source_language=block.language,
                bbox=BoundingBoxSchema.model_validate(block.bbox.model_dump()) if block.bbox else None,
                confidence=block.ocr_confidence,
                material_id=doc_id_str,
                doc_name=doc_name,
                figure_image_url=(
                    _figure_url(block.block_id)
                    if block.block_type == "figure" and block.extra.get("figure_image_path")
                    else None
                ),
            )
            for block in material_page.blocks
        ],
    )
    return APIResponse(success=True, message="Evidence page loaded successfully", data=result, error=None)


