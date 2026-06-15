from __future__ import annotations

import re
import logging

from src.processing.slug import slugify
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity, ExtractedEvent, ExtractedRelation

logger = logging.getLogger(__name__)


DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.IGNORECASE,
)
EVENT_VERBS = re.compile(
    r"\b(?:reported|introduced|proposed|improved|reduced|increased|decreased|compared|evaluated|trained|tested|published|đề xuất|giảm|tăng|so sánh|đánh giá)\b",
    re.IGNORECASE,
)
MULTIMODAL_BLOCK_TYPES = {"table", "figure", "equation", "ocr_text", "handwriting"}
CAPTION_HINT = re.compile(r"\b(?:figure|fig\.?|table|equation|hình|bảng|công thức|biểu đồ)\b", re.IGNORECASE)


class EventExtractor:
    def __init__(self) -> None:
        self._relation_extractor = None

    def extract(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> tuple[list[ExtractedEvent], list[ExtractedRelation]]:
        entity_names = [entity.canonical_name for entity in entities]
        events: list[ExtractedEvent] = []
        relations: list[ExtractedRelation] = []

        for block in evidence_map.blocks:
            if not EVENT_VERBS.search(block.snippet_original):
                continue
            matched_names = [
                name for name in entity_names if re.search(rf"\b{re.escape(name)}\b", block.snippet_original, re.IGNORECASE)
            ]
            event_name = self._event_name(block.snippet_original)
            date_match = DATE_PATTERN.search(block.snippet_original)
            event = ExtractedEvent(
                event_name=event_name,
                event_time=date_match.group(0) if date_match else None,
                participants=matched_names[:8],
                evidence_refs=[block],
                temporal_status="known" if date_match else "unknown",
                confidence=0.58 if matched_names else 0.5,
            )
            events.append(event)
            event_id = f"event:{self._slug(event_name)}"
            for name in matched_names[:8]:
                relations.append(
                    ExtractedRelation(
                        source_id=f"entity:{self._slug(name)}",
                        target_id=event_id,
                        relation_type="mentioned_in_event",
                        evidence_refs=[block],
                        confidence=0.58,
                    )
                )

        # Add structural relations (block-level)
        relations.extend(self._structural_relations(evidence_map, entities))

        # Add semantic relations (entity-to-entity)
        semantic_relations = self._extract_semantic_relations(evidence_map, entities)
        relations.extend(semantic_relations)

        logger.info(
            "Relation extraction completed",
            extra={
                "structural_relations": len(relations) - len(semantic_relations),
                "semantic_relations": len(semantic_relations),
                "total_relations": len(relations),
            },
        )

        return events, self._dedupe_relations(relations)

    def _extract_semantic_relations(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """Extract semantic relations between entities using pattern matching."""
        if self._relation_extractor is None:
            try:
                from src.processing.relation_extractor import RelationExtractor
                self._relation_extractor = RelationExtractor()
            except ImportError:
                logger.warning("RelationExtractor not available, skipping semantic relation extraction")
                return []

        try:
            return self._relation_extractor.extract(evidence_map, entities)
        except Exception as exc:
            logger.exception("Semantic relation extraction failed", extra={"error": str(exc)})
            return []

    def _structural_relations(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []

        for entity in entities:
            entity_id = f"entity:{self._slug(entity.canonical_name)}"
            for ref in entity.mention_refs:
                relations.append(
                    ExtractedRelation(
                        source_id=entity_id,
                        target_id=self._block_node_id(ref.block_id),
                        relation_type="mentioned_in_block",
                        evidence_refs=[ref],
                        confidence=max(0.56, min(entity.confidence, 0.72)),
                    )
                )

        for page_blocks in self._blocks_by_page(evidence_map.blocks).values():
            current_heading: EvidenceBlock | None = None
            previous_block: EvidenceBlock | None = None
            for block in page_blocks:
                if block.block_type == "heading":
                    current_heading = block
                elif current_heading is not None:
                    relations.append(
                        ExtractedRelation(
                            source_id=self._block_node_id(current_heading.block_id),
                            target_id=self._block_node_id(block.block_id),
                            relation_type="section_contains",
                            evidence_refs=[current_heading, block],
                            confidence=0.62,
                        )
                    )

                if previous_block is not None and self._is_cross_modal_pair(previous_block, block):
                    relations.append(
                        ExtractedRelation(
                            source_id=self._block_node_id(previous_block.block_id),
                            target_id=self._block_node_id(block.block_id),
                            relation_type=self._context_relation_type(previous_block, block),
                            evidence_refs=[previous_block, block],
                            confidence=0.6,
                        )
                    )
                previous_block = block

        return relations

    @staticmethod
    def _blocks_by_page(blocks: list[EvidenceBlock]) -> dict[int, list[EvidenceBlock]]:
        pages: dict[int, list[EvidenceBlock]] = {}
        for block in blocks:
            pages.setdefault(block.page, []).append(block)
        for page_blocks in pages.values():
            page_blocks.sort(key=lambda item: item.metadata.get("reading_order", 0) if item.metadata else 0)
        return pages

    @staticmethod
    def _is_cross_modal_pair(first: EvidenceBlock, second: EvidenceBlock) -> bool:
        return first.block_type in MULTIMODAL_BLOCK_TYPES or second.block_type in MULTIMODAL_BLOCK_TYPES

    @staticmethod
    def _context_relation_type(first: EvidenceBlock, second: EvidenceBlock) -> str:
        if first.block_type in MULTIMODAL_BLOCK_TYPES and CAPTION_HINT.search(second.snippet_original):
            return "has_caption"
        if second.block_type in MULTIMODAL_BLOCK_TYPES and CAPTION_HINT.search(first.snippet_original):
            return "caption_of"
        return "adjacent_context"

    @staticmethod
    def _dedupe_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
        deduped: dict[tuple[str, str, str], ExtractedRelation] = {}
        for relation in relations:
            key = (relation.source_id, relation.target_id, relation.relation_type)
            existing = deduped.get(key)
            if existing is None or relation.confidence > existing.confidence:
                deduped[key] = relation
        return list(deduped.values())

    @staticmethod
    def _block_node_id(block_id: str | None) -> str:
        return f"block:{block_id or 'unknown'}"

    @staticmethod
    def _event_name(text: str) -> str:
        sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
        return sentence[:180]

    @staticmethod
    def _slug(value: str) -> str:
        return slugify(value)
