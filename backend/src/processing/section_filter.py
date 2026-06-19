"""Section-aware filtering for knowledge-graph extraction.

Research papers and reports end with a References / Bibliography /
Acknowledgments section that is a dense list of author names and citations.
Running entity extraction over those sections floods the graph with hundreds
of single-mention ``person`` nodes (478/519 persons in one measured collection
were reference-list authors) that carry no domain knowledge.

This module drops blocks that fall under an excluded-section heading BEFORE
entity/relation extraction. It does NOT touch chunking or retrieval — the
reference text stays fully searchable; it is only excluded from the graph.

Domain-agnostic: heading patterns come from config (extraction_config.yaml →
``excluded_sections``) so new languages/domains need no code change.
"""

from __future__ import annotations

import logging
import re

from src.processing.types import EvidenceBlock, EvidenceMap

logger = logging.getLogger(__name__)

# Fallback patterns when config is missing. Anchored to the START of a heading
# so "References" matches but "Referenced architectures" (a real heading) does
# not. Covers EN + VI academic section names.
_DEFAULT_EXCLUDED_HEADINGS: tuple[str, ...] = (
    r"references?",
    r"bibliography",
    r"acknowledge?ments?",
    r"works\s+cited",
    r"tài\s+liệu\s+tham\s+khảo",
    r"lời\s+cảm\s+ơn",
    r"danh\s+mục\s+tài\s+liệu",
)


def _compile_excluded(patterns: tuple[str, ...]) -> re.Pattern[str]:
    # Heading must START with one of the section names (optionally numbered,
    # e.g. "6. References" / "VII. Bibliography").
    body = "|".join(patterns)
    return re.compile(
        rf"^\s*(?:[0-9ivxlcdm]+[.)]\s*)?(?:{body})\b\s*[:.]?\s*$",
        re.IGNORECASE,
    )


def filter_extraction_blocks(
    evidence_map: EvidenceMap,
    *,
    excluded_patterns: tuple[str, ...] | None = None,
) -> EvidenceMap:
    """Return a copy of evidence_map with excluded-section blocks removed.

    A block is excluded when it appears after an excluded-section heading and
    before the next heading (in reading order, per page). Headings are detected
    structurally (block_type == "heading") so the filter is layout-driven, not
    keyword-spotting inside body text.
    """
    blocks = evidence_map.blocks
    if not blocks:
        return evidence_map

    pattern = _compile_excluded(excluded_patterns or _DEFAULT_EXCLUDED_HEADINGS)

    # Order blocks globally by (page, reading_order) so a References heading on
    # page N suppresses everything until the next heading — even across pages,
    # which is how reference lists actually run.
    ordered = sorted(
        blocks,
        key=lambda b: (b.page, b.metadata.get("reading_order", 0) if b.metadata else 0),
    )

    excluded_ids: set[str] = set()
    in_excluded = False
    for block in ordered:
        # References/Bibliography/Acknowledgments are terminal back-matter: once
        # entered, stay excluded to the end. We must NOT reset on the next
        # heading, because reference entries themselves are routinely mis-parsed
        # as heading blocks ("[6] Author, Title…") — resetting there would let the
        # entire author list survive and flood the graph with citation names.
        if block.block_type == "heading" and pattern.match((block.snippet_original or "").strip()):
            in_excluded = True
        if in_excluded:
            excluded_ids.add(block.block_id)

    if not excluded_ids:
        return evidence_map

    kept = [b for b in blocks if b.block_id not in excluded_ids]
    logger.info(
        "Section-aware extraction: dropped excluded-section blocks",
        extra={
            "material_id": evidence_map.material_id,
            "dropped_blocks": len(excluded_ids),
            "kept_blocks": len(kept),
        },
    )
    return evidence_map.model_copy(update={"blocks": kept})
