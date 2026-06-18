from __future__ import annotations

import logging
from pathlib import Path

from beanie import PydanticObjectId
from pydantic import BaseModel, Field

from src.core.config import Settings
from src.models.material import BoundingBox as MaterialBoundingBox
from src.models.material import Material, MaterialBlock, MaterialPageDocument
from src.processing.types import BBox, BlockType
from src.rag.embedding_factory import build_visual_provider
from src.rag.indexer import QdrantMongoIndexer
from src.rag.types import FigureIndexItem
from src.rag.vector_store import get_qdrant_client_for_settings

logger = logging.getLogger(__name__)


class MissingVisualCrop(BaseModel):
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str = ""
    page: int
    block_id: str
    has_caption: bool = False
    has_bbox: bool = False
    has_page_image: bool = False
    reason: str


class VisualAuditResult(BaseModel):
    missing: list[MissingVisualCrop] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.missing)


class VisualBackfillResult(BaseModel):
    audited: int = 0
    cropped: int = 0
    reindexed: int = 0
    existing_indexed: int = 0
    page_indexed: int = 0
    skipped: int = 0
    missing: list[MissingVisualCrop] = Field(default_factory=list)


class VisualEvidenceBackfillService:
    """Audit and backfill figure crops for legacy indexed materials."""

    def __init__(self, *, settings: Settings, indexer: QdrantMongoIndexer | None = None) -> None:
        self.settings = settings
        self.indexer = indexer or QdrantMongoIndexer(
            settings=settings,
            qdrant_client=get_qdrant_client_for_settings(settings),
        )

    async def audit_missing_figure_images(
        self,
        *,
        owner_id: str,
        collection_id: str | None = None,
        material_ids: list[str] | None = None,
    ) -> VisualAuditResult:
        material_map = await self._material_map(owner_id=owner_id, collection_id=collection_id, material_ids=material_ids)
        if not material_map:
            return VisualAuditResult()
        pages = await self._pages_for_materials(material_map)
        missing: list[MissingVisualCrop] = []
        for page in pages:
            material = material_map.get(str(page.material_id))
            document_name = (material.original_name or material.filename) if material else ""
            for block in page.blocks:
                if not self._is_figure(block):
                    continue
                extra = block.extra or {}
                if extra.get("figure_image_path"):
                    continue
                caption = (block.content or "").strip()
                if not caption:
                    continue
                has_bbox = block.bbox is not None
                has_page_image = bool(page.image_path and Path(page.image_path).exists())
                reason = "missing_figure_image_path"
                if not has_bbox:
                    reason = "missing_bbox"
                elif not has_page_image:
                    reason = "missing_page_image"
                missing.append(
                    MissingVisualCrop(
                        owner_id=page.owner_id,
                        collection_id=str(page.collection_id),
                        material_id=str(page.material_id),
                        document_name=document_name,
                        page=page.page_number,
                        block_id=block.block_id,
                        has_caption=bool(caption),
                        has_bbox=has_bbox,
                        has_page_image=has_page_image,
                        reason=reason,
                    )
                )
        return VisualAuditResult(missing=missing)

    async def backfill_missing_crops(
        self,
        *,
        owner_id: str,
        collection_id: str | None = None,
        material_ids: list[str] | None = None,
        dry_run: bool = False,
        reindex: bool = True,
        reindex_existing: bool = True,
        index_page_fallback: bool = True,
    ) -> VisualBackfillResult:
        material_map = await self._material_map(owner_id=owner_id, collection_id=collection_id, material_ids=material_ids)
        pages = await self._pages_for_materials(material_map)
        result = VisualBackfillResult()
        figure_items: list[FigureIndexItem] = []
        cache_dir = self.settings.data_dir / "cache" / "figure_images"

        for page in pages:
            material = material_map.get(str(page.material_id))
            document_name = (material.original_name or material.filename) if material else ""
            page_changed = False
            page_has_visual_item = False
            for block in page.blocks:
                if not self._is_figure(block):
                    continue
                extra = block.extra or {}
                caption = (block.content or "").strip()
                if not caption:
                    continue
                existing_image_path = extra.get("figure_image_path")
                if existing_image_path:
                    if reindex_existing and Path(existing_image_path).exists():
                        figure_items.append(
                            self._figure_item_from_block(
                                page=page,
                                block=block,
                                document_name=document_name,
                                image_path=str(existing_image_path),
                            )
                        )
                        result.existing_indexed += 1
                        page_has_visual_item = True
                    elif reindex_existing:
                        result.skipped += 1
                        result.missing.append(
                            MissingVisualCrop(
                                owner_id=page.owner_id,
                                collection_id=str(page.collection_id),
                                material_id=str(page.material_id),
                                document_name=document_name,
                                page=page.page_number,
                                block_id=block.block_id,
                                has_caption=True,
                                has_bbox=block.bbox is not None,
                                has_page_image=False,
                                reason="missing_existing_figure_image_file",
                            )
                        )
                    continue
                result.audited += 1
                has_page_image = bool(page.image_path and Path(page.image_path).exists())
                if block.bbox is None or not has_page_image:
                    result.skipped += 1
                    result.missing.append(
                        MissingVisualCrop(
                            owner_id=page.owner_id,
                            collection_id=str(page.collection_id),
                            material_id=str(page.material_id),
                            document_name=document_name,
                            page=page.page_number,
                            block_id=block.block_id,
                            has_caption=True,
                            has_bbox=block.bbox is not None,
                            has_page_image=has_page_image,
                            reason="missing_bbox" if block.bbox is None else "missing_page_image",
                        )
                    )
                    continue
                if dry_run:
                    continue
                crop_path = self._crop_figure(
                    page_image_path=Path(page.image_path),
                    bbox=block.bbox,
                    output_dir=cache_dir,
                    material_id=str(page.material_id),
                    block_id=block.block_id,
                )
                if crop_path is None:
                    result.skipped += 1
                    continue
                block.extra["figure_image_path"] = str(crop_path)
                page_changed = True
                result.cropped += 1
                page_has_visual_item = True
                figure_items.append(
                    self._figure_item_from_block(
                        page=page,
                        block=block,
                        document_name=document_name,
                        image_path=str(crop_path),
                    )
                )
            if (
                index_page_fallback
                and not page_has_visual_item
                and material is not None
                and (material.file_type or "").lower() == "pdf"
                and page.blocks
            ):
                page_item = self._build_page_visual_fallback(
                    page=page,
                    material=material,
                    document_name=document_name,
                    dry_run=dry_run,
                )
                if page_item is not None:
                    figure_items.append(page_item)
                    result.page_indexed += 1
                    page_changed = True
            if page_changed and not dry_run:
                await page.save()

        if reindex and figure_items and not dry_run:
            visual_provider = build_visual_provider(self.settings)
            if visual_provider is None:
                logger.info("Visual crop backfill skipped reindex: visual provider unavailable")
            else:
                try:
                    await self.indexer.index_visual(figure_items=figure_items, visual_provider=visual_provider)
                    result.reindexed = len(figure_items)
                finally:
                    try:
                        visual_provider.unload()
                    except Exception:
                        pass

        return result

    async def reindex_visual_collection(
        self,
        *,
        owner_id: str,
        collection_id: str | None = None,
        material_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> VisualBackfillResult:
        """Upsert every existing figure crop for the requested scope.

        This fixes cases where Mongo blocks have `figure_image_path`, but the
        Qdrant visual collection is empty or scoped to an old collection_id.
        """
        return await self.backfill_missing_crops(
            owner_id=owner_id,
            collection_id=collection_id,
            material_ids=material_ids,
            dry_run=dry_run,
            reindex=True,
            reindex_existing=True,
            index_page_fallback=True,
        )

    async def _material_map(
        self,
        *,
        owner_id: str,
        collection_id: str | None,
        material_ids: list[str] | None,
    ) -> dict[str, Material]:
        filters: dict = {"owner_id": owner_id}
        if collection_id:
            filters["collection_id"] = PydanticObjectId(collection_id)
        ids: list[PydanticObjectId] = []
        for mid in material_ids or []:
            try:
                ids.append(PydanticObjectId(mid))
            except Exception:
                continue
        if ids:
            filters["_id"] = {"$in": ids}
        materials = await Material.find(filters).to_list()
        return {str(material.id): material for material in materials if material.id is not None}

    @staticmethod
    async def _pages_for_materials(material_map: dict[str, Material]) -> list[MaterialPageDocument]:
        ids = [material.id for material in material_map.values() if material.id is not None]
        if not ids:
            return []
        return await MaterialPageDocument.find({"material_id": {"$in": ids}}).to_list()

    @staticmethod
    def _is_figure(block) -> bool:
        return (block.block_type or "").lower() == BlockType.FIGURE.value

    @staticmethod
    def _bbox(bbox) -> BBox | None:
        if bbox is None:
            return None
        return BBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)

    def _figure_item_from_block(self, *, page, block, document_name: str, image_path: str) -> FigureIndexItem:
        return FigureIndexItem(
            owner_id=page.owner_id,
            collection_id=str(page.collection_id),
            material_id=str(page.material_id),
            document_name=document_name,
            page=page.page_number,
            block_id=block.block_id,
            block_type=block.block_type,
            caption=block.content or "",
            source_language=block.language,
            bbox=self._bbox(block.bbox),
            image_path=image_path,
        )

    def _build_page_visual_fallback(
        self,
        *,
        page: MaterialPageDocument,
        material: Material,
        document_name: str,
        dry_run: bool,
    ) -> FigureIndexItem | None:
        source_path = self._material_path(material)
        if source_path is None:
            return None
        cache_dir = self.settings.data_dir / "cache" / "figure_images"
        block_id = f"visual-page-{page.page_number}"
        image_path = None if dry_run else self._render_pdf_page(
            source_path=source_path,
            page_number=page.page_number,
            output_dir=cache_dir,
            material_id=str(page.material_id),
            block_id=block_id,
        )
        if image_path is None and not dry_run:
            return None
        caption = self._page_visual_caption(page=page, document_name=document_name)
        bbox = MaterialBoundingBox(
            x1=0,
            y1=0,
            x2=float(page.width or 0),
            y2=float(page.height or 0),
        )
        existing = next((b for b in page.blocks if b.block_id == block_id), None)
        if existing is None and not dry_run:
            max_index = max((b.block_index for b in page.blocks), default=0)
            max_order = max((b.reading_order for b in page.blocks), default=0)
            page.blocks.append(
                MaterialBlock(
                    block_id=block_id,
                    block_index=max_index + 1,
                    block_type=BlockType.FIGURE.value,
                    content=caption,
                    language=material.language,
                    bbox=bbox,
                    ocr_confidence=None,
                    reading_order=max_order + 1,
                    extra={
                        "figure_image_path": str(image_path),
                        "caption_source": "page_visual_fallback",
                        "parse_method": "visual_backfill_page",
                    },
                )
            )
        elif existing is not None and not dry_run:
            existing.block_type = BlockType.FIGURE.value
            existing.content = existing.content or caption
            existing.bbox = existing.bbox or bbox
            existing.extra = {
                **dict(existing.extra or {}),
                "figure_image_path": str(image_path),
                "caption_source": (existing.extra or {}).get("caption_source") or "page_visual_fallback",
                "parse_method": "visual_backfill_page",
            }
        return FigureIndexItem(
            owner_id=page.owner_id,
            collection_id=str(page.collection_id),
            material_id=str(page.material_id),
            document_name=document_name,
            page=page.page_number,
            block_id=block_id,
            block_type=BlockType.FIGURE.value,
            caption=caption,
            source_language=material.language,
            bbox=BBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2),
            image_path=str(image_path) if image_path is not None else None,
        )

    def _material_path(self, material: Material) -> Path | None:
        raw = Path(material.storage_path or "")
        candidates = [raw] if raw.is_absolute() else [self.settings.data_dir / raw, Path.cwd() / raw]
        return next((path for path in candidates if path.exists()), None)

    @staticmethod
    def _page_visual_caption(*, page: MaterialPageDocument, document_name: str, max_chars: int = 900) -> str:
        parts: list[str] = []
        for block in page.blocks:
            text = " ".join((block.content or "").split())
            if not text:
                continue
            if block.block_type == "heading" or len(parts) < 4:
                parts.append(text)
            if sum(len(p) for p in parts) >= max_chars:
                break
        body = " | ".join(parts)[:max_chars].strip()
        return f"Page {page.page_number} visual evidence from {document_name}: {body}".strip()

    @staticmethod
    def _render_pdf_page(
        *,
        source_path: Path,
        page_number: int,
        output_dir: Path,
        material_id: str,
        block_id: str,
    ) -> Path | None:
        try:
            import fitz

            with fitz.open(str(source_path)) as pdf:
                page_index = max(1, page_number) - 1
                if not (0 <= page_index < pdf.page_count):
                    return None
                png = pdf[page_index].get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
            if not png or len(png) < 512:
                return None
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_block = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in block_id)
            out = output_dir / f"page-{material_id}-{safe_block}.png"
            out.write_bytes(png)
            return out
        except Exception as exc:
            logger.warning("Page visual fallback render failed", extra={"path": str(source_path), "page": page_number, "error": str(exc)})
            return None

    @staticmethod
    def _crop_figure(
        *,
        page_image_path: Path,
        bbox,
        output_dir: Path,
        material_id: str,
        block_id: str,
    ) -> Path | None:
        try:
            from PIL import Image

            image = Image.open(page_image_path).convert("RGB")
            width, height = image.size
            x1 = max(0, min(width - 1, int(round(bbox.x1))))
            y1 = max(0, min(height - 1, int(round(bbox.y1))))
            x2 = max(x1 + 1, min(width, int(round(bbox.x2))))
            y2 = max(y1 + 1, min(height, int(round(bbox.y2))))
            crop = image.crop((x1, y1, x2, y2))
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_block = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in block_id)
            out = output_dir / f"backfill-{material_id}-{safe_block}.png"
            crop.save(out)
            return out
        except Exception as exc:
            logger.warning("Figure crop backfill failed", extra={"path": str(page_image_path), "error": str(exc)})
            return None
