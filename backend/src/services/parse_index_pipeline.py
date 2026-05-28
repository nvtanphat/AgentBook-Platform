from __future__ import annotations

import gc
import json
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from beanie import PydanticObjectId

from src.core.config import Settings
from src.core.model_factory import build_llm
from src.models.common import PipelineStatus, utc_now
from src.models.collection import KnowledgeCollection
from src.models.material import Material, replace_material_pages
from src.models.pipeline_job import PipelineJob
from src.processing.audio_parser import AUDIO_EXTENSIONS, AudioParser
from src.processing.chunking import AudioChunker, LayoutAwareChunker, SemanticChunker, SlideAwareChunker
from src.processing.contextual_enricher import ContextualEnricher
from src.processing.docling_parser import DoclingParser, SUPPORTED_DOCLING_EXTENSIONS
from src.processing.cross_modal_linker import CrossModalLinker
from src.processing.entity_extractor import EntityExtractor
from src.processing.entity_resolution import EntityResolver
from src.processing.semantic_relation_extractor import LLMSemanticRelationExtractor
from src.processing.event_extractor import EventExtractor
from src.processing.evidence_mapper import EvidenceMapper
from src.processing.graph_quality_gate import GraphQualityGate
from src.processing.handwriting_reader import HandwritingReader
from src.processing.layout_normalizer import LayoutNormalizer
from src.processing.language_detector import detect_block_language, detect_document_language
from src.processing.chunk_qa import run_chunk_qa
from src.processing.figure_captioner import FigureCaptioner
from src.processing.ocr_engine import EasyOCREngine, VietOCRRecognizer
from src.processing.ocr_quality_gate import score_ocr_document
from src.processing.spreadsheet_parser import SpreadsheetParser, SUPPORTED_SPREADSHEET_EXTENSIONS
from src.processing.types import BlockType, OCRQualityError, ParsedBlock, ParsedDocument, ParsedPage
from src.rag.indexer import QdrantMongoIndexer
from src.rag.types import FigureIndexItem


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}

# Heuristic threshold for "figure block is really the whole page" detection.
# Docling marks scanned-text PDF pages as a single FIGURE block with a bbox
# covering most of the page; we want full-page OCR in that case, not a crop.
_FULL_PAGE_AREA_RATIO = 0.7


def _bbox_covers_most_of_page(bbox, page_width: int, page_height: int) -> bool:
    """True when the bbox area exceeds _FULL_PAGE_AREA_RATIO of the page area.

    Accepts either a BBox dataclass / pydantic model (with x/y/width/height
    attrs) or a dict shaped like {"x":..., "y":..., "width":..., "height":...}.
    Returns False on any malformed input so the caller falls back to the
    normal crop path.
    """
    if bbox is None or page_width <= 0 or page_height <= 0:
        return False
    width = getattr(bbox, "width", None) or (bbox.get("width") if isinstance(bbox, dict) else None)
    height = getattr(bbox, "height", None) or (bbox.get("height") if isinstance(bbox, dict) else None)
    if not width or not height:
        return False
    try:
        area_ratio = (float(width) * float(height)) / (float(page_width) * float(page_height))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return area_ratio >= _FULL_PAGE_AREA_RATIO


class ParseIndexPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        docling_parser: DoclingParser | None = None,
        ocr_engine: EasyOCREngine | None = None,
        spreadsheet_parser: SpreadsheetParser | None = None,
        handwriting_reader: HandwritingReader | None = None,
        normalizer: LayoutNormalizer | None = None,
        evidence_mapper: EvidenceMapper | None = None,
        entity_extractor: EntityExtractor | None = None,
        entity_resolver: EntityResolver | None = None,
        event_extractor: EventExtractor | None = None,
        cross_modal_linker: CrossModalLinker | None = None,
        graph_quality_gate: GraphQualityGate | None = None,
        chunker: LayoutAwareChunker | SemanticChunker | None = None,
        indexer: QdrantMongoIndexer | None = None,
    ) -> None:
        self.settings = settings
        self.docling_parser = docling_parser or DoclingParser()
        self.ocr_engine = ocr_engine
        self._ocr_engines: dict[str, EasyOCREngine] = {}
        self.spreadsheet_parser = spreadsheet_parser or SpreadsheetParser()
        self.handwriting_reader = handwriting_reader or HandwritingReader(settings=settings)
        self.audio_parser = AudioParser(settings=settings)
        self.normalizer = normalizer or LayoutNormalizer()
        self.evidence_mapper = evidence_mapper or EvidenceMapper()
        _llm = build_llm(settings)
        self.entity_extractor = entity_extractor or EntityExtractor(
            llm=_llm,
            default_entity_types=settings.extraction_default_entity_types,
            few_shots=settings.extraction_few_shots,
            mode=settings.extraction_mode,
        )
        self.entity_resolver = entity_resolver or EntityResolver()
        self.event_extractor = event_extractor or EventExtractor()
        self.cross_modal_linker = cross_modal_linker or CrossModalLinker()
        # Semantic relation extraction adds an extra LLM call per material on
        # top of entity extraction. Small local models (qwen2.5:3b) saturate
        # the Ollama queue and time out, leaving 0 semantic relations and
        # wasted compute. Only activate the extractor when a cloud LLM is
        # configured (OpenAI-compatible endpoint) so latency / reliability
        # match the workload.
        _semantic_llm = _llm if (settings.llm_default_provider or "").lower() != "local" else None
        self.semantic_relation_extractor = LLMSemanticRelationExtractor(
            llm=_semantic_llm,
            max_concepts=settings.graph_semantic_relation_max_concepts,
            max_passages=settings.graph_semantic_relation_max_passages,
            max_passage_chars=settings.graph_semantic_relation_max_passage_chars,
        )
        self.graph_quality_gate = graph_quality_gate or GraphQualityGate(
            min_entity_confidence=settings.min_graph_confidence,
            min_relation_confidence=settings.min_graph_confidence,
            min_mention_count=1,
        )
        self._chunker_override = chunker
        self.indexer = indexer

    async def run(self, *, material_id: str, job_id: str) -> None:
        material = await Material.get(PydanticObjectId(material_id))
        if material is None:
            raise LookupError(f"Material not found: {material_id}")
        job = await PipelineJob.find_one(PipelineJob.job_id == job_id)
        if job is None:
            raise LookupError(f"Pipeline job not found: {job_id}")

        try:
            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.PARSING.value, stage=PipelineStatus.PARSING.value)
            parsed = await asyncio.to_thread(self._parse_material, material)
            parsed = await asyncio.to_thread(self._caption_figures, parsed, material.language)
            # Collect figure items for visual indexing before normalization may
            # drop block.extra fields.  Empty when visual embedding is disabled.
            figure_items = (
                self._collect_figure_items(parsed, material)
                if self.settings.visual_embedding_enabled
                else []
            )
            normalized = await asyncio.to_thread(self.normalizer.normalize, parsed)
            detected_language, language_counts = self._apply_language_detection(normalized, declared_language=material.language)
            material.extra_metadata["detected_language"] = detected_language
            material.extra_metadata["detected_language_counts"] = language_counts
            if material.language == "unknown" and detected_language != "unknown":
                material.language = detected_language
            material_pages = self.normalizer.to_material_pages(normalized)
            await replace_material_pages(material, material_pages)
            material.pages = []
            material.page_count = len(material_pages)
            await self._write_processed_artifacts(material, normalized)
            await self._mark(material=material, job=job, status=PipelineStatus.PARSED.value, stage=PipelineStatus.PARSED.value)
            gc.collect()  # free docling/OCR memory before loading embedding model

            await self._ensure_material_exists(material_id)
            # Initialise indexer early so SemanticChunker can reuse its embedder
            if self.indexer is None:
                from src.dependencies import get_qdrant_client

                self.indexer = QdrantMongoIndexer(settings=self.settings, qdrant_client=get_qdrant_client())

            await self._mark(material=material, job=job, status=PipelineStatus.CHUNKING.value, stage=PipelineStatus.CHUNKING.value)
            evidence_map = self.evidence_mapper.build(
                parsed=normalized,
                owner_id=material.owner_id,
                collection_id=str(material.collection_id),
                material_id=str(material.id),
                document_name=material.original_name,
            )
            chunker = self._resolve_chunker(file_type=material.file_type)
            if isinstance(chunker, SemanticChunker):
                chunks = await chunker.build_chunks_async(evidence_map)
            else:
                chunks = chunker.build_chunks(evidence_map)
            chunks = [c for c in chunks if len((c.content or "").strip()) >= self.settings.chunk_min_content_chars]
            run_chunk_qa(chunks, material_id=str(material.id))
            domain_hint = await self._fetch_domain_hint(material)
            entities = self.entity_resolver.resolve(
                await self.entity_extractor.extract_async(evidence_map, domain_hint=domain_hint)
            )
            events, relations = self.event_extractor.extract(evidence_map, entities)
            cm_entities, cm_relations = self.cross_modal_linker.link(evidence_map, entities)
            # LLM-driven typed semantic relations between concept entities
            # (uses / extends / improves / compared_with / …). Empty when
            # the LLM is unavailable or the doc has < 2 concept entities.
            try:
                semantic_relations = await self.semantic_relation_extractor.extract_async(
                    evidence_map, entities,
                )
            except Exception as exc:
                logger.warning(
                    "Semantic relation extraction failed (non-fatal)",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
                semantic_relations = []
            entities = list(entities) + cm_entities
            relations = list(relations) + cm_relations + semantic_relations

            # Apply graph quality gates
            entities = self.graph_quality_gate.prune_entities(entities)
            entities = self.graph_quality_gate.resolve_entities(entities)

            # Build valid entity ID set for relation pruning
            valid_entity_ids = {
                f"entity:{self.graph_quality_gate._slug(e.canonical_name)}"
                for e in entities
            }
            # Also include event IDs and block IDs
            for event in events:
                valid_entity_ids.add(f"event:{self.graph_quality_gate._slug(event.event_name)}")
            for block in evidence_map.blocks:
                valid_entity_ids.add(f"block:{block.block_id}")

            relations = self.graph_quality_gate.prune_relations(relations, valid_entity_ids)

            logger.info(
                "Graph extraction completed",
                extra={
                    "entities": len(entities),
                    "events": len(events),
                    "relations": len(relations),
                    "material_id": str(material.id),
                },
            )

            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.EMBEDDING.value, stage=PipelineStatus.EMBEDDING.value)
            chunks = await self._contextual_enrich(chunks, evidence_map)

            await self._ensure_material_exists(material_id)
            await self._mark(material=material, job=job, status=PipelineStatus.INDEXING.value, stage=PipelineStatus.INDEXING.value)
            await self.indexer.index(
                chunks=chunks,
                entities=entities,
                events=events,
                relations=relations,
                material_id=str(material.id),
                should_continue=lambda: self._material_exists(material_id),
            )

            # Visual embedding step — runs after text indexing so SigLIP does
            # not compete with BGE-M3 for RAM.  Graceful: failures are warned
            # and skipped; text pipeline result is not affected.
            if figure_items:
                await self._index_visual_figures(figure_items, material_id=str(material.id))

            await self._ensure_material_exists(material_id)
            await self._mark(
                material=material,
                job=job,
                status=PipelineStatus.INDEXED.value,
                stage=PipelineStatus.INDEXED.value,
                finished=True,
            )
        except MemoryError as exc:
            logger.critical(
                "Pipeline OOM - process may be unstable; reduce batch sizes or free RAM",
                extra={"material_id": material_id, "job_id": job_id, "stage": job.stage},
                exc_info=True,
            )
            await self._fail(
                material=material,
                job=job,
                failed_stage=job.stage,
                error="The processing pipeline failed. Please retry or inspect server logs.",
            )
            raise
        except Exception as exc:
            failed_stage = getattr(exc, "failed_stage", None) or job.stage
            logger.exception(
                "Parse/index pipeline failed",
                extra={"material_id": material_id, "job_id": job_id, "stage": failed_stage},
            )
            await self._fail(
                material=material,
                job=job,
                failed_stage=failed_stage,
                error="The processing pipeline failed. Please retry or inspect server logs.",
            )
            raise

    async def _fetch_domain_hint(self, material: Material) -> str | None:
        """Look up the configured hint field on the material's collection.

        Returns None when collection or field is missing — extractor handles
        that gracefully by skipping the domain hint block in the prompt.
        """
        field = self.settings.extraction_domain_hint_field or "subject"
        try:
            collection = await KnowledgeCollection.get(material.collection_id)
        except Exception:
            return None
        if collection is None:
            return None
        value = getattr(collection, field, None)
        return str(value).strip() if value else None

    @staticmethod
    async def _ensure_material_exists(material_id: str) -> None:
        if not await ParseIndexPipeline._material_exists(material_id):
            raise LookupError(f"Material was deleted while pipeline was running: {material_id}")

    @staticmethod
    async def _material_exists(material_id: str) -> bool:
        return await Material.get(PydanticObjectId(material_id)) is not None

    async def _contextual_enrich(
        self,
        chunks: list,
        evidence_map,
    ) -> list:
        from src.core.runtime_config import get_override
        enabled = get_override("contextual_retrieval_enabled", self.settings.contextual_retrieval_enabled)
        if not enabled or not chunks:
            return chunks
        try:
            from src.core.model_factory import build_llm
            llm = build_llm(self.settings)
            enricher = ContextualEnricher(llm, concurrency=self.settings.contextual_retrieval_concurrency)
            enriched = await enricher.enrich(chunks, evidence_map)
            enriched_count = sum(1 for c in enriched if c.contextualized_content)
            logger.info(
                "Contextual enrichment done",
                extra={"total": len(enriched), "enriched": enriched_count, "skipped": len(enriched) - enriched_count},
            )
            return enriched
        except Exception:
            logger.exception("Contextual enrichment failed entirely - continuing without context")
            return chunks

    # ── Visual embedding helpers ───────────────────────────────────────────────

    def _collect_figure_items(self, parsed: ParsedDocument, material) -> list[FigureIndexItem]:
        """Build FigureIndexItem list from figure blocks that have an image path."""
        items: list[FigureIndexItem] = []
        for block in parsed.blocks:
            if block.block_type != BlockType.FIGURE.value:
                continue
            img_path: str | None = block.extra.get("figure_image_path")
            if not img_path:
                continue
            items.append(
                FigureIndexItem(
                    owner_id=material.owner_id,
                    collection_id=str(material.collection_id),
                    material_id=str(material.id),
                    document_name=material.original_name,
                    page=block.page_number,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    caption=block.content or "",
                    source_language=block.language,
                    bbox=block.bbox,
                    image_path=img_path,
                )
            )
        logger.info(
            "Collected figure items for visual indexing",
            extra={"count": len(items), "material_id": str(material.id)},
        )
        return items

    async def _index_visual_figures(
        self, figure_items: list[FigureIndexItem], *, material_id: str
    ) -> None:
        """Embed and index figure images with SigLIP; always unloads provider."""
        from src.rag.embedding_factory import build_visual_provider

        visual_provider = None
        try:
            visual_provider = build_visual_provider(self.settings)
            if visual_provider is None:
                return
            await self.indexer.index_visual(
                figure_items=figure_items, visual_provider=visual_provider
            )
        except Exception as exc:
            logger.warning(
                "Visual embedding failed — skipping (text index is unaffected)",
                extra={"material_id": material_id, "error": str(exc)},
            )
        finally:
            if visual_provider is not None:
                try:
                    visual_provider.unload()
                except Exception as exc:
                    logger.debug(
                        "Visual provider unload error (non-fatal)",
                        extra={"error": str(exc)},
                    )

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _resolve_chunker(self, *, file_type: str | None = None) -> LayoutAwareChunker | SemanticChunker | SlideAwareChunker | AudioChunker:
        if self._chunker_override is not None:
            return self._chunker_override
        # PPTX: use slide-boundary-preserving chunker (1 chunk per slide).
        # Semantic chunker splits across slides → fragments lose context.
        if file_type and file_type.lower() == "pptx":
            return SlideAwareChunker(self.settings)
        # Audio: group whisper segments by time window (~45s/chunk).
        # Preserves audio_start/end_seconds for citation deep-linking.
        if file_type and file_type.lower() in AUDIO_EXTENSIONS:
            return AudioChunker(self.settings)
        if self.settings.chunk_strategy == "semantic":
            embedder = None
            if self.indexer is not None and hasattr(self.indexer, "embedder"):
                embedder = self.indexer.embedder
            return SemanticChunker(self.settings, embedder=embedder)
        return LayoutAwareChunker(self.settings)

    def _parse_material(self, material: Material) -> ParsedDocument:
        path = self._material_path(material)
        if material.file_type in SUPPORTED_DOCLING_EXTENSIONS:
            return self.docling_parser.parse(path, language=material.language)
        if material.file_type in SUPPORTED_SPREADSHEET_EXTENSIONS:
            return self.spreadsheet_parser.parse(path, language=material.language, display_name=material.original_name)
        if material.file_type in IMAGE_EXTENSIONS:
            source_type = str(material.extra_metadata.get("source_type", "")).lower()
            if "hand" in source_type or material.modality == "handwriting":
                return self.handwriting_reader.parse_image(path, language=self._declared_language(material.language))
            return self._parse_image_vlm_first(path, declared_language=material.language)
        if material.file_type in AUDIO_EXTENSIONS:
            return self.audio_parser.parse(path, language=self._declared_language(material.language))
        raise ValueError(f"No Phase 2 parser is configured for .{material.file_type}")

    def _caption_figures(self, parsed: ParsedDocument, language: str) -> ParsedDocument:
        """Fill in FIGURE block content using VLM captioning.

        Looks for blocks where extra["needs_captioning"] is True.
        For PDF: crops the bbox region from the pre-rendered page image.
        For image files and DOCX embedded pictures: the entire image or embedded image is captioned.
        Skips captioning silently if no vision model is available.
        Figures are captioned in parallel (up to 4 workers) to reduce wall-clock time.
        """
        import concurrent.futures

        figure_blocks = [
            b for b in parsed.blocks
            if b.block_type == BlockType.FIGURE.value and b.extra.get("needs_captioning")
        ]
        if not figure_blocks:
            return parsed

        captioner = FigureCaptioner(
            ollama_base_url=self.settings.ollama_base_url,
            language=language if language in {"vi", "en"} else "vi",
            timeout=self.settings.ollama_caption_timeout_seconds,
            image_max_side_px=self.settings.figure_image_max_side_px,
        )
        # Pre-check Ollama availability once before spawning threads, so threads
        # share the cached result and don't each trigger a 5s network probe.
        captioner._detect_available_model()

        # Locate rendered page images if available (PDF pipeline puts them in cache).
        page_image_dir = self.settings.data_dir / "cache" / "pdf_page_images"

        # Persistent dir for DOCX/PPTX embedded figure images (needed by visual
        # embedding step). Only created when visual_embedding is enabled.
        figure_cache_dir: Path | None = None
        if self.settings.visual_embedding_enabled:
            figure_cache_dir = self.settings.data_dir / "cache" / "figure_images"

        def _caption_block(block):
            try:
                caption, img_path = self._caption_one_figure(
                    block, captioner, page_image_dir, parsed, figure_cache_dir
                )
                return block, caption, img_path, None
            except Exception as exc:
                return block, None, None, exc

        max_workers = min(len(figure_blocks), self.settings.figure_captioner_max_workers)
        captioned_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for block, caption, img_path, exc in pool.map(_caption_block, figure_blocks):
                if exc is not None:
                    logger.warning(
                        "Figure captioning failed for block",
                        extra={"block_id": block.block_id, "page": block.page_number, "error": str(exc)},
                    )
                else:
                    if img_path:
                        block.extra["figure_image_path"] = img_path
                    if caption:
                        block.content = caption
                        block.extra.pop("needs_captioning", None)
                        caption_source = "ocr" if caption.startswith(("[Hình", "[Figure")) else "vlm"
                        block.extra["caption_source"] = caption_source
                        block.extra["parse_method"] = "figure_caption"
                        if caption_source == "vlm" and captioner._available_model:
                            block.extra["vlm_model"] = captioner._available_model
                        if caption_source == "ocr":
                            block.extra["fallback_reason"] = "vlm_unavailable_or_empty"
                        captioned_count += 1

        if captioned_count:
            logger.info(
                "Figure captioning complete",
                extra={"captioned": captioned_count, "total_figures": len(figure_blocks), "source": parsed.source_path},
            )

        # Unload captioner's OCR engine to free RAM before visual embedding loads SigLIP.
        captioner.unload()
        return parsed

    @staticmethod
    def _caption_one_figure(
        block,
        captioner: FigureCaptioner,
        page_image_dir: Path,
        parsed: ParsedDocument,
        figure_cache_dir: Path | None = None,
    ) -> tuple[str, str | None]:
        """Return (caption, image_path | None).

        image_path is the on-disk path of the figure image so the visual
        embedding pipeline can load it later.  None when the image could not
        be persisted (PDF without page render, unsupported type, etc.).
        """
        import base64
        import tempfile
        from uuid import NAMESPACE_URL, uuid5

        source_path = Path(parsed.source_path)

        # PDF: crop the bbox region from the pre-rendered page image.
        if parsed.file_type == "pdf":
            page_img_name = f"{uuid5(NAMESPACE_URL, f'{source_path}:{block.page_number}').hex}.png"
            page_img_path = page_image_dir / page_img_name
            if page_img_path.exists():
                try:
                    import cv2
                    img = cv2.imread(str(page_img_path))
                    if img is not None:
                        ph, pw = img.shape[:2]
                        # Detect "figure-only page": Docling sometimes treats a
                        # whole scanned legal page as 1 figure block with a tiny
                        # or missing bbox. Crop-then-OCR loses the page text.
                        # When bbox is missing OR covers >70% of the page area,
                        # caption the FULL page so EasyOCR sees the whole text.
                        bbox = block.bbox
                        if bbox is None or _bbox_covers_most_of_page(bbox, pw, ph):
                            caption = captioner.caption_image_path(page_img_path)
                            return caption, str(page_img_path)
                        caption, crop_path = captioner.caption_page_region_with_path(
                            page_img_path, bbox, page_width=pw, page_height=ph
                        )
                        # If crop-based captioning came back empty (typical for
                        # scanned-text figures where the VLM/OCR couldn't read
                        # the cropped region), retry on the full page image.
                        if not (caption or "").strip():
                            caption = captioner.caption_image_path(page_img_path)
                            return caption, str(page_img_path)
                        return caption, str(crop_path) if crop_path else None
                except ImportError:
                    pass

        # Standalone image: caption the whole file; the path is already on disk.
        if parsed.file_type in IMAGE_EXTENSIONS:
            return captioner.caption_image_path(source_path), str(source_path)

        # DOCX/PPTX embedded figure: decode the data-URI.
        # When visual_embedding is enabled, save to figure_cache_dir so the file
        # persists for SigLIP.  Otherwise use a temp file (legacy behaviour).
        embedded_uri = block.extra.get("embedded_image_uri")
        if isinstance(embedded_uri, str) and embedded_uri.startswith("data:image/"):
            try:
                import hashlib as _hashlib
                header, encoded = embedded_uri.split(",", 1)
                suffix = ".png"
                if "jpeg" in header or "jpg" in header:
                    suffix = ".jpg"
                elif "webp" in header:
                    suffix = ".webp"
                data = base64.b64decode(encoded)

                if figure_cache_dir is not None:
                    figure_cache_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = _hashlib.sha1(encoded[:256].encode()).hexdigest()[:12]
                    save_path = figure_cache_dir / f"fig-{img_hash}{suffix}"
                    save_path.write_bytes(data)
                    caption = captioner.caption_image_path(save_path)
                    return caption, str(save_path)

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(data)
                    tmp_path = Path(handle.name)
                try:
                    return captioner.caption_image_path(tmp_path), None
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass

        return "", None

    def _parse_image_vlm_first(self, path: Path, *, declared_language: str) -> ParsedDocument:
        """For standalone image files: try VLM captioning first, OCR fallback.

        Infographics, charts, and complex layouts are better described by a
        vision model than by OCR, which scrambles multi-column reading order.
        Falls back to _parse_printed_image transparently when no VLM is available.
        """
        language = declared_language if declared_language in {"vi", "en"} else "vi"
        captioner = FigureCaptioner(
            ollama_base_url=self.settings.ollama_base_url,
            language=language,
            ocr_fallback=False,  # we handle OCR ourselves below
            timeout=self.settings.ollama_caption_timeout_seconds,
            image_max_side_px=self.settings.figure_image_max_side_px,
        )
        vlm_model = captioner._detect_available_model()
        fallback_reason = "vlm_unavailable"
        if vlm_model:
            from src.processing.figure_captioner import _looks_like_gibberish
            caption = captioner.caption_image_path(path)
            # Reject gibberish output (VLM hallucination on stylized text)
            if caption and _looks_like_gibberish(caption):
                logger.warning(
                    "VLM produced gibberish — falling back to OCR",
                    extra={"model": vlm_model, "image": path.name, "preview": caption[:120]},
                )
                fallback_reason = "vlm_gibberish_detected"
                caption = ""
            if caption and len(caption.strip()) >= 80:
                logger.info(
                    "Standalone image parsed via VLM",
                    extra={"model": vlm_model, "image": path.name, "chars": len(caption)},
                )
                block = ParsedBlock(
                    block_id=f"blk-{path.stem[:12]}",
                    block_index=0,
                    block_type=BlockType.PARAGRAPH.value,
                    content=caption,
                    page_number=1,
                    reading_order=0,
                    language=language,
                    source="vlm",
                    extra={
                        "parse_method": "vlm",
                        "caption_source": "vlm",
                        "vlm_model": vlm_model,
                        "image_file": path.name,
                    },
                )
                return ParsedDocument(
                    source_path=str(path),
                    file_type=path.suffix.lstrip(".").lower(),
                    language=language,
                    pages=[ParsedPage(page_number=1, blocks=[block])],
                    extra={"parser": "vlm", "parse_method": "vlm", "vlm_model": vlm_model},
                )
            fallback_reason = "vlm_caption_too_short_or_empty"
        parsed = self._parse_printed_image(path, declared_language=declared_language)
        parsed.extra.setdefault("parse_method", "ocr")
        parsed.extra["fallback_reason"] = fallback_reason
        parsed.extra["vlm_attempted"] = bool(vlm_model)
        if vlm_model:
            parsed.extra["vlm_model"] = vlm_model
        return parsed

    def _parse_printed_image(self, path: Path, *, declared_language: str) -> ParsedDocument:
        runtime_language = self._ocr_runtime_language(declared_language)
        parsed = self._ocr_engine_for_language(runtime_language).parse_image(
            path,
            language=self._declared_language(declared_language),
        )
        parsed.extra.setdefault("parse_method", "ocr")
        if declared_language != "unknown":
            return self._apply_ocr_quality_gate(parsed)

        detected = detect_document_language([block.content for block in parsed.blocks], fallback="unknown")
        if detected != "vi" or runtime_language == "vi":
            parsed.extra["ocr_language_routing"] = {"initial": runtime_language, "selected": runtime_language}
            return self._apply_ocr_quality_gate(parsed)

        vi_parsed = self._ocr_engine_for_language("vi").parse_image(path, language="vi")
        vi_parsed.extra.setdefault("parse_method", "ocr")
        selected = self._select_better_ocr_result(parsed, vi_parsed)
        selected.extra["ocr_language_routing"] = {
            "initial": runtime_language,
            "candidate": "vi",
            "selected": selected.extra.get("ocr_lang", runtime_language),
            "initial_score": self._ocr_document_quality(parsed),
            "candidate_score": self._ocr_document_quality(vi_parsed),
        }
        return self._apply_ocr_quality_gate(selected)

    def _apply_ocr_quality_gate(self, parsed: ParsedDocument) -> ParsedDocument:
        report = score_ocr_document(
            parsed,
            min_score=self.settings.min_ocr_text_quality,
            warn_score=self.settings.warn_ocr_text_quality,
        )
        parsed.extra["ocr_quality"] = {
            "score": report.score,
            "valid_char_ratio": report.valid_char_ratio,
            "meaningful_word_ratio": report.meaningful_word_ratio,
            "repetition_ratio": report.repetition_ratio,
            "symbol_density": report.symbol_density,
            "total_chars": report.total_chars,
            "warnings": report.warnings,
        }
        if not report.is_acceptable(self.settings.min_ocr_text_quality):
            raise OCRQualityError(
                f"OCR quality score {report.score:.2f} is below fail threshold "
                f"{self.settings.min_ocr_text_quality:.2f}: {report.flag_summary()}",
                score=report.score,
                threshold=self.settings.min_ocr_text_quality,
            )
        if report.warnings:
            logger.warning(
                "OCR quality gate warnings",
                extra={
                    "score": report.score,
                    "flags": report.flag_summary(),
                    "source_path": parsed.source_path,
                    "stage": "ocr_quality",
                },
            )
        return parsed

    def _ocr_engine_for_language(self, language: str) -> EasyOCREngine:
        if self.ocr_engine is not None:
            return self.ocr_engine
        if language not in self._ocr_engines:
            lang = "vi" if language == "vi" else "en"
            # For Vietnamese, attach the VietOCR recognizer when configured — it
            # reads tone marks far better than EasyOCR on scanned text. EN stays
            # on EasyOCR (VietOCR is Vietnamese-only).
            recognizer = None
            if lang == "vi" and self.settings.ocr_recognition_engine == "vietocr":
                recognizer = VietOCRRecognizer(
                    device=self.settings.ocr_vietocr_device,
                    model_name=self.settings.ocr_vietocr_model_name,
                )
            self._ocr_engines[language] = EasyOCREngine(lang=lang, gpu=False, recognizer=recognizer)
        return self._ocr_engines[language]

    @staticmethod
    def _ocr_runtime_language(language: str) -> str:
        # "vi" uses latin_g2.pth which covers all Latin scripts (vi + en).
        # Only fall back to "en" when explicitly declared English.
        return "en" if language == "en" else "vi"

    @staticmethod
    def _declared_language(language: str) -> str:
        return language if language in {"vi", "en"} else "unknown"

    @staticmethod
    def _select_better_ocr_result(first: ParsedDocument, second: ParsedDocument) -> ParsedDocument:
        return second if ParseIndexPipeline._ocr_document_quality(second) > ParseIndexPipeline._ocr_document_quality(first) else first

    @staticmethod
    def _ocr_document_quality(parsed: ParsedDocument) -> float:
        text = "\n".join(block.content for block in parsed.blocks)
        confidences = [block.ocr_confidence for block in parsed.blocks if block.ocr_confidence is not None]
        confidence_score = (sum(confidences) / len(confidences)) * 10.0 if confidences else 0.0
        detected = detect_document_language([block.content for block in parsed.blocks], fallback="unknown")
        vi_bonus = 1.5 if detected == "vi" else 0.0
        suspect_penalty = sum(char in "ēūǎåīōσ" for char in text) * 0.8
        replacement_penalty = text.count("�") * 2.0
        return confidence_score + vi_bonus - suspect_penalty - replacement_penalty

    @staticmethod
    def _apply_language_detection(parsed: ParsedDocument, *, declared_language: str) -> tuple[str, dict[str, int]]:
        should_detect = declared_language in {"unknown", "mixed", ""}
        counts: dict[str, int] = {}
        for block in parsed.blocks:
            detected = detect_block_language(block.content, fallback=block.language if block.language != "unknown" else "unknown")
            if should_detect or block.language == "unknown":
                block.language = detected
            if block.language != "unknown":
                counts[block.language] = counts.get(block.language, 0) + 1
        document_language = detect_document_language([block.content for block in parsed.blocks], fallback=declared_language or "unknown")
        parsed.language = document_language
        return document_language, counts

    def _material_path(self, material: Material) -> Path:
        storage_path = Path(material.storage_path)
        if storage_path.is_absolute():
            return storage_path
        target = self.settings.data_dir / storage_path
        if not target.exists():
            target = self.settings.data_dir.parent / storage_path
        return target

    async def _write_processed_artifacts(self, material: Material, parsed: ParsedDocument) -> None:
        output_dir = self.settings.processed_data_dir / material.owner_id / str(material.collection_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{material.id}.parsed.json"
        output_path.write_text(json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        material.extra_metadata["parsed_artifact_path"] = str(output_path.relative_to(self.settings.data_dir)).replace("\\", "/")

    @staticmethod
    async def _mark(
        *,
        material: Material,
        job: PipelineJob,
        status: str,
        stage: str,
        finished: bool = False,
    ) -> None:
        material.status = status
        material.failed_stage = None
        material.error_message = None
        material.updated_at = utc_now()
        job.status = status
        job.stage = stage
        job.last_error = None
        job.failed_stage = None
        if finished:
            job.finished_at = utc_now()
        await material.save()
        await job.save()

    @staticmethod
    async def _fail(*, material: Material, job: PipelineJob, failed_stage: str | None, error: str) -> None:
        material.status = PipelineStatus.FAILED.value
        material.failed_stage = failed_stage
        material.error_message = error
        material.retry_count += 1
        material.updated_at = utc_now()
        job.status = PipelineStatus.FAILED.value
        job.failed_stage = failed_stage
        job.last_error = error
        job.retry_count += 1
        job.finished_at = utc_now()
        await material.save()
        await job.save()


