"""LLM-based semantic relation extractor.

Implements the SOTA hybrid 3-tier architecture described in
sota_entity_relation_extraction.md:

  P1.3 — Pydantic-validated output (zero JSON retry waste, Instructor pattern)
  P2.1 — Joint relation + missed-entity gleaning in a single LLM call
  P2.2 — Ontology-guided filtering: prune relations outside valid type pairs
  P2.3 — CoDe-KG sentence decomposition before passage selection
  P3.1 — KET-RAG entity importance scoring for passage selection
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator, model_validator

from src.processing.sentence_decomposer import decompose_blocks
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity, ExtractedRelation

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


# Fallback graph entity types — used when config is missing or empty.
# Covers all 8 universal types plus common technical variants so the pipeline
# works across ML, law, history, medicine etc. without any code change.
_DEFAULT_GRAPH_ENTITY_TYPES: frozenset[str] = frozenset({
    "concept", "model", "algorithm", "metric", "dataset",
    "framework", "method", "technology", "field",
    "artifact", "person", "organization", "event", "location",
})

# Fallback relation vocabulary — used when config is missing or empty.
_DEFAULT_RELATION_VOCAB: tuple[str, ...] = (
    "uses", "extends", "replaces", "improves",
    "compared_with", "evaluates_on", "depends_on", "part_of",
    "contradicts", "governs", "applies_to", "created_by",
    "located_in", "occurred_at", "related_to",
)


@lru_cache(maxsize=1)
def _get_graph_entity_types() -> frozenset[str]:
    """Load graph-eligible entity types from config; fall back to defaults."""
    try:
        from src.core.config import get_settings
        types = get_settings().extraction_graph_entity_types
        if types:
            return frozenset(t.lower() for t in types)
    except Exception:
        pass
    return _DEFAULT_GRAPH_ENTITY_TYPES


@lru_cache(maxsize=1)
def _get_relation_vocab() -> tuple[tuple[str, ...], frozenset[str]]:
    """Load relation vocabulary from config; fall back to defaults."""
    try:
        from src.core.config import get_settings
        types = get_settings().extraction_relation_types
        if types:
            vocab = tuple(t.lower() for t in types)
            return vocab, frozenset(vocab)
    except Exception:
        pass
    return _DEFAULT_RELATION_VOCAB, frozenset(_DEFAULT_RELATION_VOCAB)

_DEFAULT_MAX_CONCEPTS = 25
_DEFAULT_MAX_PASSAGES = 18
_DEFAULT_MAX_PASSAGE_CHARS = 600


# ── P1.3: Pydantic-validated output schemas ──────────────────────────────────

class _RelationItem(BaseModel):
    source: str
    target: str
    type: str
    passage_index: int = -1
    confidence: float = 0.6

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in _get_relation_vocab()[1]:
            raise ValueError(f"Invalid relation type: {v!r}")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v) -> float:
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode="after")
    def non_empty_names(self) -> "_RelationItem":
        if not self.source.strip() or not self.target.strip():
            raise ValueError("source and target must be non-empty")
        return self


class _MissedEntity(BaseModel):
    name: str
    type: str
    confidence: float = 0.7

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v) -> float:
        return max(0.0, min(1.0, float(v)))


# ── P2.2: Ontology filter ────────────────────────────────────────────────────

def _load_ontology() -> dict[str, set[str]]:
    """Return entity-type → valid relation types from extraction_config.yaml."""
    try:
        from src.core.config import get_settings
        settings = get_settings()
        raw = getattr(settings, "extraction_ontology", {}) or {}
        return {k: set(v) for k, v in raw.items()}
    except Exception:
        return {}

_ONTOLOGY: dict[str, set[str]] | None = None


def _get_ontology() -> dict[str, set[str]]:
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = _load_ontology()
    return _ONTOLOGY


def _relation_allowed(source_type: str, rel_type: str, ontology: dict[str, set[str]]) -> bool:
    """Return True when no ontology constraint exists or the pair is explicitly allowed."""
    if not ontology:
        return True
    allowed = ontology.get(source_type.lower())
    if allowed is None:
        return True  # unknown type — permit
    return rel_type in allowed


# ── Helpers ──────────────────────────────────────────────────────────────────

from src.processing.slug import entity_node_id as _entity_slug


# ── P3.1: KET-RAG passage importance scoring ─────────────────────────────────

def _score_passage(block: EvidenceBlock, concept_names_lower: set[str]) -> tuple[int, int]:
    """Return (co_mention_count, unique_entity_count) for a block."""
    text_lower = (block.snippet_original or "").lower()
    hits = [name for name in concept_names_lower if name in text_lower]
    return len(hits), len(set(hits))


def _select_passages(
    *,
    blocks: list[EvidenceBlock],
    concept_names_lower: set[str],
    max_passages: int,
) -> list[EvidenceBlock]:
    """Pick the top-scoring passages by entity co-mention importance (KET-RAG).

    Scores each block on (co_mention_count, unique_entity_count). Blocks that
    mention only one concept cannot produce a relation and are excluded.
    This mirrors the KET-RAG skeleton approach: focus LLM attention on the
    information-dense passages only.
    """
    scored: list[tuple[int, int, EvidenceBlock]] = []
    for block in blocks:
        co, uniq = _score_passage(block, concept_names_lower)
        if co >= 2:  # at least two entity mentions → potential relation
            scored.append((co, uniq, block))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [block for _, _, block in scored[:max_passages]]


# ── Prompt ───────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """Bạn là người trích xuất quan hệ ngữ nghĩa từ tài liệu học thuật.

Cho KHÁI NIỆM đã biết và ĐOẠN VĂN, hãy:
1. Liệt kê các quan hệ có ý nghĩa giữa các khái niệm.
2. Nếu có khái niệm quan trọng bị thiếu trong danh sách, hãy bổ sung vào "missed_entities".

Từ vựng quan hệ hợp lệ: {vocab}

KHÁI NIỆM:
{entity_list}

ĐOẠN VĂN:
{passages}

Quy tắc:
- "source" và "target" PHẢI là canonical_name chính xác từ danh sách KHÁI NIỆM (sao chép nguyên văn).
- Không bịa quan hệ không có trong đoạn văn.
- Hãy TRÍCH ĐẦY ĐỦ: với mỗi cặp khái niệm có liên hệ trong đoạn văn, tạo một quan hệ.
- Tối đa 50 quan hệ, tối đa 10 missed_entities.

Trả về CHỈ JSON, không thêm văn bản nào:
{{"relations": [{{"source": "A", "target": "B", "type": "uses", "passage_index": 0, "confidence": 0.85}}], "missed_entities": [{{"name": "X", "type": "concept", "confidence": 0.8}}]}}

JSON:"""


# GraphRAG-style gleaning pass: shows what was already found and asks ONLY for
# additional relations the first pass missed. Boosts recall ~15-30%.
_GLEANING_TEMPLATE = """Bạn vừa trích xuất quan hệ ngữ nghĩa nhưng có thể đã BỎ SÓT.

Từ vựng quan hệ hợp lệ: {vocab}

KHÁI NIỆM:
{entity_list}

ĐOẠN VĂN:
{passages}

QUAN HỆ ĐÃ TÌM (đừng lặp lại):
{found}

Hãy tìm THÊM các quan hệ CÓ TRONG ĐOẠN VĂN mà danh sách trên còn thiếu.
Quy tắc:
- "source"/"target" PHẢI khớp chính xác canonical_name trong danh sách KHÁI NIỆM.
- Chỉ thêm quan hệ thật sự có căn cứ trong đoạn văn; không bịa.
- Nếu không còn quan hệ nào, trả về {{"relations": []}}.

Trả về CHỈ JSON:
{{"relations": [{{"source": "A", "target": "B", "type": "uses", "passage_index": 0, "confidence": 0.8}}], "missed_entities": []}}

JSON:"""


def _build_gleaning_prompt(
    *,
    concepts: list[ExtractedEntity],
    passages: list[EvidenceBlock],
    found: list[ExtractedRelation],
    max_passage_chars: int,
) -> str:
    entity_list = "\n".join(f"- {e.canonical_name} [{e.entity_type}]" for e in concepts)
    passage_text = "\n\n".join(
        f"[{i}] {p.snippet_original[:max_passage_chars].strip()}"
        for i, p in enumerate(passages)
    )
    found_lines = "\n".join(
        f"- {r.source_id.replace('entity:', '')} -[{r.relation_type}]-> {r.target_id.replace('entity:', '')}"
        for r in found[:60]
    ) or "(chưa có)"
    return _GLEANING_TEMPLATE.format(
        vocab=", ".join(_get_relation_vocab()[0]),
        entity_list=entity_list,
        passages=passage_text,
        found=found_lines,
    )


def _build_prompt(
    *,
    concepts: list[ExtractedEntity],
    passages: list[EvidenceBlock],
    max_passage_chars: int,
) -> str:
    entity_list = "\n".join(f"- {e.canonical_name} [{e.entity_type}]" for e in concepts)
    passage_text = "\n\n".join(
        f"[{i}] {p.snippet_original[:max_passage_chars].strip()}"
        for i, p in enumerate(passages)
    )
    return _PROMPT_TEMPLATE.format(
        vocab=", ".join(_get_relation_vocab()[0]),
        entity_list=entity_list,
        passages=passage_text,
    )


# ── P1.3: Pydantic-validated response parsing ────────────────────────────────

import json as _json
import re as _re


def _parse_response(
    *,
    raw: str,
    concept_by_lower_name: dict[str, ExtractedEntity],
    passages: list[EvidenceBlock],
    max_passage_chars: int,
    ontology: dict[str, set[str]],
) -> tuple[list[ExtractedRelation], list[tuple[str, str, float]]]:
    """Parse LLM output with Pydantic validation. Returns (relations, missed_entities).

    P1.3: Eliminates retry waste — Pydantic validates each item individually;
    invalid items are logged and dropped rather than triggering a full retry.
    """
    cleaned = raw.strip()
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=_re.MULTILINE).strip()

    # Extract the outermost {...} block
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        logger.debug("Semantic relation: no JSON object found in LLM output")
        return [], []

    try:
        payload = _json.loads(cleaned[brace_start:brace_end + 1])
    except _json.JSONDecodeError as exc:
        logger.debug("Semantic relation: JSON decode failed: %s", exc)
        return [], []

    if not isinstance(payload, dict):
        return [], []

    # ── Parse relations ──
    raw_relations = payload.get("relations") or []
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str]] = set()

    for item in raw_relations if isinstance(raw_relations, list) else []:
        try:
            rel = _RelationItem.model_validate(item)
        except Exception as exc:
            logger.debug("Dropping invalid relation item: %s — %s", item, exc)
            continue

        source = concept_by_lower_name.get(rel.source.lower())
        target = concept_by_lower_name.get(rel.target.lower())
        if source is None or target is None:
            continue
        if source.canonical_name == target.canonical_name:
            continue

        # P2.2: ontology filter
        if not _relation_allowed(source.entity_type, rel.type, ontology):
            logger.debug(
                "Dropping relation %s -[%s]-> %s: not in ontology for type %s",
                source.canonical_name, rel.type, target.canonical_name, source.entity_type,
            )
            continue

        key = (source.canonical_name, rel.type, target.canonical_name)
        if key in seen:
            continue
        seen.add(key)

        evidence_refs: list[EvidenceBlock] = []
        if 0 <= rel.passage_index < len(passages):
            evidence_refs = [passages[rel.passage_index]]

        relations.append(ExtractedRelation(
            source_id=_entity_slug(source.canonical_name),
            target_id=_entity_slug(target.canonical_name),
            relation_type=rel.type,
            evidence_refs=evidence_refs,
            evidence_text_chunk=(
                evidence_refs[0].snippet_original[:max_passage_chars] if evidence_refs else None
            ),
            confidence=rel.confidence,
        ))

    # ── P2.1: parse missed entities ──
    raw_missed = payload.get("missed_entities") or []
    missed: list[tuple[str, str, float]] = []
    for item in raw_missed if isinstance(raw_missed, list) else []:
        try:
            me = _MissedEntity.model_validate(item)
            name = me.name.strip()
            if name and name.lower() not in concept_by_lower_name:
                missed.append((name, me.type, me.confidence))
        except Exception:
            pass

    return relations, missed


# ── Main extractor class ─────────────────────────────────────────────────────

class LLMSemanticRelationExtractor:
    """LLM-driven typed-relation extractor with SOTA cost optimisations.

    Single LLM call per material. Uses Pydantic validation (P1.3), joint
    missed-entity gleaning (P2.1), ontology filtering (P2.2), sentence
    decomposition (P2.3), and KET-RAG passage scoring (P3.1).
    """

    def __init__(
        self,
        *,
        llm: "BaseLLM | None" = None,
        max_concepts: int = _DEFAULT_MAX_CONCEPTS,
        max_passages: int = _DEFAULT_MAX_PASSAGES,
        max_passage_chars: int = _DEFAULT_MAX_PASSAGE_CHARS,
        gleaning: bool = False,
    ) -> None:
        self._llm = llm
        self._max_concepts = max_concepts
        self._max_passages = max_passages
        self._max_passage_chars = max_passage_chars
        self._gleaning = gleaning

    async def extract_async(
        self,
        evidence_map: EvidenceMap,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        if self._llm is None or not entities:
            return []

        concepts = [e for e in entities if (e.entity_type or "").lower() in _get_graph_entity_types()]
        if len(concepts) < 2:
            return []

        concepts.sort(key=lambda e: (e.confidence, len(e.mention_refs)), reverse=True)
        concepts = concepts[: self._max_concepts]

        concept_names_lower = {e.canonical_name.lower() for e in concepts}
        concept_by_lower_name = {e.canonical_name.lower(): e for e in concepts}

        # P2.3: decompose compound sentences before passage selection
        decomposed_blocks = decompose_blocks(evidence_map.blocks)

        # P3.1: KET-RAG importance scoring — only pass co-mention passages
        passages = _select_passages(
            blocks=decomposed_blocks,
            concept_names_lower=concept_names_lower,
            max_passages=self._max_passages,
        )
        if not passages:
            return []

        prompt = _build_prompt(
            concepts=concepts,
            passages=passages,
            max_passage_chars=self._max_passage_chars,
        )
        try:
            raw = await self._llm.generate(prompt=prompt)
        except Exception as exc:
            logger.warning(
                "LLM semantic relation extraction failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return []

        ontology = _get_ontology()
        relations, missed = _parse_response(
            raw=raw,
            concept_by_lower_name=concept_by_lower_name,
            passages=passages,
            max_passage_chars=self._max_passage_chars,
            ontology=ontology,
        )

        # GraphRAG gleaning pass: re-prompt for relations the first pass missed.
        gleaned_count = 0
        if self._gleaning:
            try:
                glean_prompt = _build_gleaning_prompt(
                    concepts=concepts,
                    passages=passages,
                    found=relations,
                    max_passage_chars=self._max_passage_chars,
                )
                glean_raw = await self._llm.generate(prompt=glean_prompt)
                extra_relations, extra_missed = _parse_response(
                    raw=glean_raw,
                    concept_by_lower_name=concept_by_lower_name,
                    passages=passages,
                    max_passage_chars=self._max_passage_chars,
                    ontology=ontology,
                )
                seen = {(r.source_id, r.relation_type, r.target_id) for r in relations}
                for r in extra_relations:
                    key = (r.source_id, r.relation_type, r.target_id)
                    if key not in seen:
                        seen.add(key)
                        relations.append(r)
                        gleaned_count += 1
                if extra_missed:
                    missed = list(missed) + list(extra_missed)
            except Exception as exc:
                logger.warning(
                    "Relation gleaning pass failed (non-fatal)",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )

        if missed:
            logger.info(
                "Joint gleaning found %d missed entities (not yet persisted — caller must merge)",
                len(missed),
                extra={"missed": [m[0] for m in missed[:5]]},
            )

        logger.info(
            "LLM semantic relation extraction completed",
            extra={
                "relations": len(relations),
                "gleaned_relations": gleaned_count,
                "missed_entities": len(missed),
                "concepts": len(concepts),
                "passages": len(passages),
            },
        )
        return relations
