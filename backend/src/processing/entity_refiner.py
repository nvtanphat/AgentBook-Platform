"""Ontology-guided LLM entity refinement (hybrid GLiNER + LLM).

GLiNER gives high recall + exact spans but, on Vietnamese, mistypes entities
(`luật` → organization) and fragments spans (`biện 2`, `người thứ`). This pass
sends the GLiNER entities + a sample mention each to a capable (cloud) LLM and
asks it to, against the configured ontology:

  - correct each entity's type to an allowed type
  - rewrite a fragmented span to its full canonical form found in the text
  - drop structural junk
  - glean important entities GLiNER missed

Design mirrors semantic_relation_extractor.py:
  P1.3 — Pydantic-validated output (invalid items dropped, no full-retry waste)
  P2.2 — ontology-guided (types + descriptions come from extraction_config.yaml)

Single LLM call per material. Conservative: an input entity the LLM does not
echo is KEPT unchanged (never silently dropped). mention_refs are preserved
across rename/merge so the evidence trace stays intact.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from collections import OrderedDict
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


# ── Pydantic-validated output schemas ─────────────────────────────────────────

class _RefinedItem(BaseModel):
    original: str          # echo of the input canonical_name — maps result back
    name: str              # corrected / de-fragmented canonical name
    type: str              # ontology type (lower-cased)
    keep: bool = True

    @field_validator("type", mode="before")
    @classmethod
    def lower_type(cls, v) -> str:
        return str(v).strip().lower()


class _GleanedItem(BaseModel):
    name: str
    type: str
    confidence: float = 0.7

    @field_validator("type", mode="before")
    @classmethod
    def lower_type(cls, v) -> str:
        return str(v).strip().lower()

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.7


_PROMPT_TEMPLATE = """Bạn là chuyên gia chuẩn hoá thực thể cho đồ thị tri thức.

Cho DANH SÁCH THỰC THỂ (trích tự động, có thể sai loại hoặc bị cắt vụn) và ĐOẠN VĂN MẪU,
hãy với MỖI thực thể:
1. "type": gán đúng MỘT loại trong danh sách LOẠI HỢP LỆ bên dưới.
2. "name": nếu tên bị cắt vụn (vd "biện 2", "người thứ"), sửa lại thành cụm danh từ ĐẦY ĐỦ
   đúng như xuất hiện trong đoạn văn (vd "biện pháp bảo đảm", "người thứ ba"). Nếu đã đúng, giữ nguyên.
3. "keep": false nếu là rác cấu trúc (số điều/khoản, mảnh vô nghĩa, stopword); true nếu là thực thể thật.
Nếu phát hiện thực thể QUAN TRỌNG bị thiếu, thêm vào "missed".

LOẠI HỢP LỆ:
{type_block}

DANH SÁCH THỰC THỂ (original | loại hiện tại | đoạn mẫu):
{entity_list}

Quy tắc:
- "original" PHẢI sao chép nguyên văn tên ở cột original (để khớp lại).
- Chỉ dùng loại trong LOẠI HỢP LỆ. Không bịa thực thể không có trong đoạn văn.
- Tối đa {max_missed} thực thể trong "missed".

Trả về CHỈ JSON, không thêm văn bản:
{{"entities": [{{"original": "biện 2", "name": "biện pháp bảo đảm", "type": "concept", "keep": true}}], "missed": [{{"name": "X", "type": "concept", "confidence": 0.8}}]}}

JSON:"""


class EntityRefiner:
    """Single cloud-LLM ontology-guided refinement pass over GLiNER entities."""

    def __init__(
        self,
        *,
        llm: "BaseLLM | None" = None,
        allowed_types: tuple[str, ...] = (),
        type_descriptions: dict[str, str] | None = None,
        max_entities: int = 200,
        max_missed: int = 20,
        sample_chars: int = 160,
    ) -> None:
        self._llm = llm
        self._allowed = tuple(t.lower() for t in allowed_types)
        self._allowed_set = frozenset(self._allowed)
        self._descriptions = {k.lower(): v for k, v in (type_descriptions or {}).items()}
        self._max_entities = max_entities
        self._max_missed = max_missed
        self._sample_chars = sample_chars

    async def refine_async(
        self,
        entities: list[ExtractedEntity],
        evidence_map: EvidenceMap,
    ) -> list[ExtractedEntity]:
        if self._llm is None or not entities or not self._allowed_set:
            return entities

        # Cap the payload — refine the most-mentioned entities first.
        ordered = sorted(entities, key=lambda e: len(e.mention_refs), reverse=True)
        head = ordered[: self._max_entities]
        tail = ordered[self._max_entities:]  # untouched, kept as-is

        prompt = self._build_prompt(head)
        try:
            raw = await self._llm.generate(prompt=prompt)
        except Exception as exc:
            logger.warning(
                "Entity refinement LLM call failed; keeping GLiNER entities",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return entities

        refined_map, missed = self._parse_response(raw)
        result = self._apply(head, refined_map, missed, evidence_map)
        result.extend(tail)

        logger.info(
            "Entity refinement done",
            extra={
                "material_id": evidence_map.material_id,
                "in": len(entities),
                "out": len(result),
                "dropped": max(0, len(head) - sum(1 for _ in refined_map if refined_map[_].keep)),
                "gleaned": len(missed),
            },
        )
        return result

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(self, entities: list[ExtractedEntity]) -> str:
        type_block = "\n".join(
            f"- {t}: {self._descriptions.get(t, '')}".rstrip(": ").rstrip()
            for t in self._allowed
        )
        lines = []
        for e in entities:
            sample = ""
            if e.mention_refs:
                sample = (e.mention_refs[0].snippet_original or "").strip().replace("\n", " ")
                sample = sample[: self._sample_chars]
            lines.append(f"- {e.canonical_name} | {e.entity_type} | {sample}")
        return _PROMPT_TEMPLATE.format(
            type_block=type_block,
            entity_list="\n".join(lines),
            max_missed=self._max_missed,
        )

    # ── parse (P1.3) ────────────────────────────────────────────────────────────

    def _parse_response(
        self, raw: str,
    ) -> tuple[OrderedDict[str, _RefinedItem], list[_GleanedItem]]:
        cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=_re.MULTILINE).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return OrderedDict(), []
        try:
            payload = _json.loads(cleaned[start:end + 1])
        except _json.JSONDecodeError as exc:
            logger.debug("Entity refinement: JSON decode failed: %s", exc)
            return OrderedDict(), []
        if not isinstance(payload, dict):
            return OrderedDict(), []

        refined: OrderedDict[str, _RefinedItem] = OrderedDict()
        for item in payload.get("entities") or []:
            try:
                ri = _RefinedItem.model_validate(item)
            except Exception:
                continue
            if ri.type not in self._allowed_set:
                # Invalid type → don't trust the correction, keep original unchanged.
                ri = ri.model_copy(update={"keep": True, "name": ri.original, "type": ""})
            refined[ri.original.lower()] = ri

        missed: list[_GleanedItem] = []
        for item in (payload.get("missed") or [])[: self._max_missed]:
            try:
                gi = _GleanedItem.model_validate(item)
            except Exception:
                continue
            if gi.type in self._allowed_set and gi.name.strip():
                missed.append(gi)
        return refined, missed

    # ── apply ─────────────────────────────────────────────────────────────────

    def _apply(
        self,
        head: list[ExtractedEntity],
        refined_map: OrderedDict[str, _RefinedItem],
        missed: list[_GleanedItem],
        evidence_map: EvidenceMap,
    ) -> list[ExtractedEntity]:
        # Merge by corrected canonical name so de-fragmented duplicates collapse.
        merged: OrderedDict[str, ExtractedEntity] = OrderedDict()

        def upsert(name: str, etype: str, conf: float, refs: list[EvidenceBlock]) -> None:
            key = name.lower()
            existing = merged.get(key)
            if existing is None:
                merged[key] = ExtractedEntity(
                    canonical_name=name, entity_type=etype,
                    confidence=conf, mention_refs=list(refs),
                )
            else:
                seen = {b.block_id for b in existing.mention_refs}
                existing.mention_refs.extend(b for b in refs if b.block_id not in seen)
                if conf > existing.confidence:
                    merged[key] = existing.model_copy(update={"confidence": conf})

        for e in head:
            ri = refined_map.get(e.canonical_name.lower())
            if ri is None:
                # LLM didn't echo it → conservative keep, unchanged.
                upsert(e.canonical_name, e.entity_type, e.confidence, e.mention_refs)
                continue
            if not ri.keep:
                continue  # dropped as junk
            new_name = (ri.name or e.canonical_name).strip() or e.canonical_name
            new_type = ri.type if ri.type in self._allowed_set else e.entity_type
            upsert(new_name, new_type, e.confidence, e.mention_refs)

        # Gleaned entities: attach mentions by text search over the evidence map.
        existing_lower = set(merged.keys())
        for gi in missed:
            name = gi.name.strip()
            if name.lower() in existing_lower:
                continue
            refs = self._find_mentions(name, evidence_map)
            if not refs:
                continue  # no textual grounding → skip (avoid hallucinated nodes)
            upsert(name, gi.type, gi.confidence, refs)

        return list(merged.values())

    @staticmethod
    def _find_mentions(name: str, evidence_map: EvidenceMap) -> list[EvidenceBlock]:
        pattern = _re.compile(rf"(?<!\w){_re.escape(name)}(?!\w)", _re.IGNORECASE)
        return [b for b in evidence_map.blocks if pattern.search(b.snippet_original or "")]
