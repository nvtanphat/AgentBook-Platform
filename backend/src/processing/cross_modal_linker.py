"""Cross-Modal Knowledge Graph Linker.

Inspired by RAG-Anything (HKUDS, 2025): builds a cross-modal graph that connects
non-text elements (tables, figures, equations) to the text-based entity graph.

Two relation types are created:
- co_located_with  : a text entity appears within PROXIMITY_WINDOW reading-order
                     positions of a table/figure/equation on the same page.
- references       : a text block explicitly mentions "Table N", "Figure N", "Eq. N"
                     and that cross-modal entity can be positionally matched.
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict

from src.processing.slug import slugify
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)

_CROSS_MODAL_TYPES = frozenset({"table", "figure", "equation"})
_PROXIMITY_WINDOW = 5  # reading_order distance within same page

# Explicit reference patterns in text
_TABLE_REF  = re.compile(r"\b(?:Table|Bảng)\s+(\d+)", re.IGNORECASE)
_FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.?|Hình)\s+(\d+)", re.IGNORECASE)
_EQ_REF     = re.compile(r"\b(?:Equation|Eq\.?|Công\s+thức)\s+(\d+)", re.IGNORECASE)

_REF_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_TABLE_REF, "table"),
    (_FIGURE_REF, "figure"),
    (_EQ_REF, "equation"),
]


def _slug(value: str) -> str:
    return slugify(value)


def _canonical_name(block: EvidenceBlock) -> str:
    """Derive a human-readable canonical name for a cross-modal block."""
    content = (block.snippet_original or "").strip()
    prefix = block.block_type.capitalize()
    if content:
        # Trim to first 80 chars so names stay manageable
        short = content[:80].replace("\n", " ").strip()
        return f"{prefix}: {short}"
    return f"{prefix} (page {block.page})"


def _dedupe_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ExtractedRelation] = []
    for rel in relations:
        key = (rel.source_id, rel.target_id, rel.relation_type)
        if key not in seen:
            seen.add(key)
            out.append(rel)
    return out


class CrossModalLinker:
    """Creates cross-modal KG entities and relations without any ML models."""

    def link(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """
        Returns:
            cm_entities  – new ExtractedEntity objects for each table/figure/equation block
            cm_relations – co_located_with and references relations
        """
        blocks = evidence_map.blocks

        # ── Step 1: Create entity nodes for cross-modal blocks ─────────────────
        cm_entities: list[ExtractedEntity] = []
        block_to_entity: dict[str, ExtractedEntity] = {}

        for block in blocks:
            if block.block_type not in _CROSS_MODAL_TYPES:
                continue
            entity = ExtractedEntity(
                canonical_name=_canonical_name(block),
                entity_type=block.block_type,
                confidence=0.88,
                mention_refs=[block],
            )
            cm_entities.append(entity)
            block_to_entity[block.block_id] = entity

        if not cm_entities:
            return [], []

        # ── Step 2: Build spatial index for proximity lookup ───────────────────
        # page → sorted list of (reading_order, block_id) for cross-modal blocks only
        cm_page_index: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for block in blocks:
            if block.block_type not in _CROSS_MODAL_TYPES:
                continue
            ro = block.metadata.get("reading_order", 0)
            cm_page_index[block.page].append((ro, block.block_id))

        # Ordered cross-modal entities per type (for positional reference matching)
        cm_by_type: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for entity in cm_entities:
            cm_by_type[entity.entity_type].append(entity)

        # Text-entity mention positions: entity_key → [(page, reading_order)]
        entity_positions: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for entity in entities:
            key = entity.canonical_name.lower()
            for ref in entity.mention_refs:
                ro = ref.metadata.get("reading_order", 0)
                entity_positions[key].append((ref.page, ro))

        # Block → its text entities (for reference relation building)
        block_to_text_entities: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            for ref in entity.mention_refs:
                block_to_text_entities[ref.block_id].append(entity)

        relations: list[ExtractedRelation] = []

        # ── Step 3: Spatial proximity relations ────────────────────────────────
        for entity in entities:
            entity_id = f"entity:{_slug(entity.canonical_name)}"
            key = entity.canonical_name.lower()
            for page, text_ro in entity_positions.get(key, []):
                for cm_ro, cm_block_id in cm_page_index.get(page, []):
                    if abs(cm_ro - text_ro) > _PROXIMITY_WINDOW:
                        continue
                    cm_entity = block_to_entity.get(cm_block_id)
                    if cm_entity is None:
                        continue
                    cm_id = f"entity:{_slug(cm_entity.canonical_name)}"
                    # Use the cross-modal block as evidence for this relation
                    cm_block = next((b for b in blocks if b.block_id == cm_block_id), None)
                    evidence_refs = [cm_block] if cm_block else []
                    relations.append(ExtractedRelation(
                        source_id=entity_id,
                        target_id=cm_id,
                        relation_type="co_located_with",
                        evidence_refs=evidence_refs,
                        confidence=0.75,
                    ))

        # ── Step 4: Explicit reference relations ("Table 1 shows…") ───────────
        for block in blocks:
            if block.block_type in _CROSS_MODAL_TYPES:
                continue  # only scan text blocks for references
            text = block.snippet_original or ""
            for pattern, cm_type in _REF_PATTERNS:
                for match in pattern.finditer(text):
                    try:
                        ref_num = int(match.group(1))
                    except (IndexError, ValueError):
                        continue
                    cm_list = cm_by_type.get(cm_type, [])
                    if ref_num < 1 or ref_num > len(cm_list):
                        continue
                    target_entity = cm_list[ref_num - 1]
                    target_id = f"entity:{_slug(target_entity.canonical_name)}"
                    for text_entity in block_to_text_entities.get(block.block_id, []):
                        relations.append(ExtractedRelation(
                            source_id=f"entity:{_slug(text_entity.canonical_name)}",
                            target_id=target_id,
                            relation_type="references",
                            evidence_refs=[block],
                            confidence=0.82,
                        ))

        logger.info(
            "CrossModalLinker: linked cross-modal KG",
            extra={
                "material_id": evidence_map.material_id,
                "cm_entities": len(cm_entities),
                "cm_relations": len(relations),
                "types": {t: len(v) for t, v in cm_by_type.items()},
            },
        )

        return cm_entities, _dedupe_relations(relations)
