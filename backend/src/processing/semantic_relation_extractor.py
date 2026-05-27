"""LLM-based semantic relation extractor.

The regex-based RelationExtractor in this folder only catches surface-level
verbal patterns and almost never produces concept-to-concept relations on
mixed VI/EN scientific text. Most relations that end up in the DB are
structural (`section_contains`, `mentioned_in_block`, `co_located_with`),
which leaves the concept graph visually rich but semantically poor.

This module asks the LLM to enumerate typed relations between *already-
extracted* concept entities, grounded in evidence passages. The output is a
list of `ExtractedRelation` records ready to be persisted alongside the
structural ones.

Design notes:
- Per-material call (not per-chunk) — one LLM invocation produces many
  relations, keeping ingestion cost bounded.
- Strictly typed vocabulary (see `_RELATION_VOCAB`) so downstream filters
  and the viz layer can rely on consistent labels.
- Skipped automatically when the LLM is unavailable or the material has
  fewer than 2 concept entities (nothing to relate).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity, ExtractedRelation

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


# Entity types whose pairwise interactions we want to capture as semantic
# edges. Authors, organisations, locations are explicitly excluded.
_CONCEPT_TYPES = frozenset({
    "concept", "model", "algorithm", "metric", "dataset",
    "framework", "method", "technology", "field",
})

# Controlled vocabulary for semantic relations. Kept short so the LLM
# is forced to commit to a clear type instead of free-form phrases.
_RELATION_VOCAB = (
    "uses",          # A applies/employs B as a component
    "extends",       # A is built on top of B
    "replaces",      # A is proposed in place of B
    "improves",      # A enhances B's behaviour or performance
    "compared_with", # A is benchmarked against B (peer baseline)
    "evaluates_on",  # A is measured on B (model on dataset)
    "depends_on",    # A requires B to function
    "part_of",       # A is a sub-component of B
    "contradicts",   # A presents a different finding from B
    "related_to",    # generic fallback when the link is real but not above
)

# Limits live in retrieval_config.yaml → graph.semantic_relation.*
# Defaults below are only used by tests / standalone runs that don't pass a
# Settings instance through the pipeline.
_DEFAULT_MAX_CONCEPTS = 25
_DEFAULT_MAX_PASSAGES = 18
_DEFAULT_MAX_PASSAGE_CHARS = 600


def _entity_slug(name: str) -> str:
    """Match graph_quality_gate._slug + endpoint _entity_slug prefix scheme."""
    body = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
    return f"entity:{body}"


_PROMPT_TEMPLATE = """Bạn là người trích quan hệ ngữ nghĩa giữa các khái niệm khoa học từ tài liệu học thuật.

Cho danh sách KHÁI NIỆM đã biết và các ĐOẠN VĂN có chứa chúng, hãy liệt kê tất cả các quan hệ có ý nghĩa giữa các khái niệm — chỉ dùng các loại quan hệ trong từ vựng dưới đây.

Từ vựng quan hệ hợp lệ (chọn đúng một loại cho mỗi cặp):
{vocab}

KHÁI NIỆM:
{entity_list}

ĐOẠN VĂN TƯỞNG ỨNG (đánh số):
{passages}

Yêu cầu đầu ra: chỉ trả về JSON array, không thêm văn bản nào khác. Mỗi phần tử có dạng:
{{"source": "<khái niệm A>", "target": "<khái niệm B>", "type": "<loại quan hệ>", "passage_index": <số đoạn văn>, "confidence": <0.0-1.0>}}

Quy tắc:
- "source" và "target" PHẢI khớp chính xác canonical_name trong danh sách KHÁI NIỆM (sao chép nguyên văn).
- Không bịa ra khái niệm hoặc quan hệ không có trong đoạn văn.
- Bỏ qua các quan hệ chỉ là "xuất hiện cùng đoạn" mà không có nghĩa rõ ràng.
- Tối đa 30 quan hệ.

JSON:"""


def _build_prompt(
    *,
    concepts: list[ExtractedEntity],
    passages: list[EvidenceBlock],
    max_passage_chars: int,
) -> str:
    entity_list = "\n".join(
        f"- {e.canonical_name} [{e.entity_type}]" for e in concepts
    )
    passage_text = "\n\n".join(
        f"[{i}] {p.snippet_original[:max_passage_chars].strip()}"
        for i, p in enumerate(passages)
    )
    vocab = ", ".join(_RELATION_VOCAB)
    return _PROMPT_TEMPLATE.format(
        vocab=vocab,
        entity_list=entity_list,
        passages=passage_text,
    )


def _select_passages(
    *, blocks: list[EvidenceBlock], concept_names_lower: set[str], max_passages: int,
) -> list[EvidenceBlock]:
    """Pick blocks that mention at least two concept entities — these are
    the only ones that can yield a relation."""
    selected: list[tuple[int, EvidenceBlock]] = []
    for block in blocks:
        text_lower = (block.snippet_original or "").lower()
        if not text_lower:
            continue
        hits = sum(1 for name in concept_names_lower if name in text_lower)
        if hits >= 2:
            selected.append((hits, block))
    selected.sort(key=lambda item: item[0], reverse=True)
    return [block for _, block in selected[:max_passages]]


def _parse_response(
    *,
    raw: str,
    concept_by_lower_name: dict[str, ExtractedEntity],
    passages: list[EvidenceBlock],
    max_passage_chars: int,
) -> list[ExtractedRelation]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        items = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.debug("Semantic relation parse failed: %s", exc)
        return []
    if not isinstance(items, list):
        return []

    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source") or "").strip()
        target_name = str(item.get("target") or "").strip()
        rel_type = str(item.get("type") or "").strip().lower()
        if not source_name or not target_name or not rel_type:
            continue
        if rel_type not in _RELATION_VOCAB:
            continue
        source = concept_by_lower_name.get(source_name.lower())
        target = concept_by_lower_name.get(target_name.lower())
        if source is None or target is None or source.canonical_name == target.canonical_name:
            continue
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        try:
            passage_idx = int(item.get("passage_index", -1))
        except (TypeError, ValueError):
            passage_idx = -1
        evidence_refs: list[EvidenceBlock] = []
        if 0 <= passage_idx < len(passages):
            evidence_refs = [passages[passage_idx]]
        key = (source.canonical_name, rel_type, target.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        evidence_text = (
            evidence_refs[0].snippet_original[:max_passage_chars]
            if evidence_refs
            else None
        )
        relations.append(
            ExtractedRelation(
                source_id=_entity_slug(source.canonical_name),
                target_id=_entity_slug(target.canonical_name),
                relation_type=rel_type,
                evidence_refs=evidence_refs,
                evidence_text_chunk=evidence_text,
                confidence=confidence,
            )
        )
    return relations


class LLMSemanticRelationExtractor:
    """LLM-driven typed-relation extractor over concept entities."""

    def __init__(
        self,
        *,
        llm: "BaseLLM | None" = None,
        max_concepts: int = _DEFAULT_MAX_CONCEPTS,
        max_passages: int = _DEFAULT_MAX_PASSAGES,
        max_passage_chars: int = _DEFAULT_MAX_PASSAGE_CHARS,
    ) -> None:
        self._llm = llm
        self._max_concepts = max_concepts
        self._max_passages = max_passages
        self._max_passage_chars = max_passage_chars

    async def extract_async(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        if self._llm is None or not entities:
            return []

        concepts = [e for e in entities if (e.entity_type or "").lower() in _CONCEPT_TYPES]
        if len(concepts) < 2:
            return []

        # Cap concept list to keep the prompt bounded; keep highest-confidence
        # entities first so the LLM works with the most reliable signals.
        concepts.sort(key=lambda e: (e.confidence, len(e.mention_refs)), reverse=True)
        concepts = concepts[: self._max_concepts]

        concept_names_lower = {e.canonical_name.lower() for e in concepts}
        concept_by_lower_name = {e.canonical_name.lower(): e for e in concepts}

        passages = _select_passages(
            blocks=evidence_map.blocks,
            concept_names_lower=concept_names_lower,
            max_passages=self._max_passages,
        )
        if not passages:
            return []

        prompt = _build_prompt(
            concepts=concepts, passages=passages, max_passage_chars=self._max_passage_chars,
        )
        try:
            raw = await self._llm.generate(prompt=prompt)
        except Exception as exc:
            logger.warning(
                "LLM semantic relation extraction failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return []

        relations = _parse_response(
            raw=raw,
            concept_by_lower_name=concept_by_lower_name,
            passages=passages,
            max_passage_chars=self._max_passage_chars,
        )
        logger.info(
            "LLM semantic relation extraction produced %d relations",
            len(relations),
            extra={"concept_count": len(concepts), "passage_count": len(passages)},
        )
        return relations
