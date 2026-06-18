"""Structural Knowledge-Graph Linker.

Inspired by RAG-Anything (HKUDS, 2025) hierarchy preservation: connects every
content block to its enclosing section heading via a ``belongs_to`` edge, and
nests sub-sections under their parent heading. This lets graph traversal climb
the document structure (block → section → parent section) instead of treating
the document as a flat bag of blocks.

Pure positional — uses ``reading_order`` and the heading level inferred from the
Docling label (``title`` vs ``section_header``). No LLM/ML, CPU-trivial.

Two node/relation shapes are produced:
- section entity : one ``ExtractedEntity`` (entity_type="section") per heading block.
- belongs_to     : ``block:<id> → entity:<section>`` for content blocks, and
                   ``entity:<child_section> → entity:<parent_section>`` for nesting.
"""

from __future__ import annotations

import logging

from src.processing.slug import slugify
from src.processing.types import (
    BlockType,
    EvidenceBlock,
    EvidenceMap,
    ExtractedEntity,
    ExtractedRelation,
)

logger = logging.getLogger(__name__)

_HEADING_TYPE = BlockType.HEADING.value
# Docling labels that denote a top-level heading (level 0). Everything else that
# classifies as a heading is treated as a sub-section (level 1).
_TOP_LEVEL_LABELS = frozenset({"title", "document_title"})

_MAX_SECTION_NAME = 90


def _section_name(block: EvidenceBlock) -> str:
    text = (block.snippet_original or "").strip().replace("\n", " ")
    if text:
        return text[:_MAX_SECTION_NAME].strip()
    return f"Section (page {block.page})"


def _heading_level(block: EvidenceBlock) -> int:
    label = str(block.metadata.get("label") or "").lower()
    return 0 if any(tok in label for tok in _TOP_LEVEL_LABELS) else 1


class StructureLinker:
    """Builds section entities and ``belongs_to`` hierarchy relations.

    Confidence values are injected from config (no hardcoded thresholds).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        section_confidence: float = 0.9,
        belongs_to_confidence: float = 0.85,
    ) -> None:
        self.enabled = enabled
        self.section_confidence = section_confidence
        self.belongs_to_confidence = belongs_to_confidence

    def link(
        self, evidence_map: EvidenceMap
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        if not self.enabled:
            return [], []

        # Reading-order traversal across the whole document.
        blocks = sorted(
            evidence_map.blocks,
            key=lambda b: (b.page, int(b.metadata.get("reading_order", 0))),
        )

        section_entities: dict[str, ExtractedEntity] = {}  # slug → entity
        relations: list[ExtractedRelation] = []

        # Nearest preceding heading at each level, for sub-section nesting.
        # level → (slug, EvidenceBlock)
        current_by_level: dict[int, tuple[str, EvidenceBlock]] = {}
        # Nearest preceding heading of any level, for content belongs_to.
        current_section_slug: str | None = None

        for block in blocks:
            if block.block_type == _HEADING_TYPE:
                name = _section_name(block)
                slug = slugify(name)
                if not slug:
                    continue
                level = _heading_level(block)

                entity = section_entities.get(slug)
                if entity is None:
                    entity = ExtractedEntity(
                        canonical_name=name,
                        entity_type="section",
                        confidence=self.section_confidence,
                        mention_refs=[block],
                    )
                    section_entities[slug] = entity
                elif block not in entity.mention_refs:
                    entity.mention_refs.append(block)

                # Nest under the nearest preceding heading at a strictly higher
                # level (smaller level number).
                parent = next(
                    (
                        current_by_level[lvl]
                        for lvl in sorted(current_by_level)
                        if lvl < level
                    ),
                    None,
                )
                if parent is not None and parent[0] != slug:
                    relations.append(
                        ExtractedRelation(
                            source_id=f"entity:{slug}",
                            target_id=f"entity:{parent[0]}",
                            relation_type="belongs_to",
                            evidence_refs=[block],
                            confidence=self.belongs_to_confidence,
                        )
                    )

                current_by_level[level] = (slug, block)
                # Drop any deeper levels now superseded by this heading.
                for deeper in [lvl for lvl in current_by_level if lvl > level]:
                    del current_by_level[deeper]
                current_section_slug = slug
                continue

            # Content block → belongs_to its enclosing section.
            if current_section_slug is not None:
                relations.append(
                    ExtractedRelation(
                        source_id=f"block:{block.block_id}",
                        target_id=f"entity:{current_section_slug}",
                        relation_type="belongs_to",
                        evidence_refs=[block],
                        confidence=self.belongs_to_confidence,
                    )
                )

        entities = list(section_entities.values())
        logger.info(
            "StructureLinker: linked document hierarchy",
            extra={
                "material_id": evidence_map.material_id,
                "sections": len(entities),
                "belongs_to_relations": len(relations),
            },
        )
        return entities, relations
