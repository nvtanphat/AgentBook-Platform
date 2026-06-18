"""Unified pre-generation evidence validation (product pipeline Stage 6).

Composes the existing scattered checks — `RefusalPolicy.check_evidence` plus a
modality-match check — into ONE structured verdict that the inference engine acts
on and the request trace records. Reuses RefusalPolicy verbatim, so refusal
behaviour is unchanged; this only adds a structured, measurable wrapper
(đáng tin + đo được).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.guardrails.refusal_policy import RefusalPolicy, RefusalRule
from src.rag.evidence import AudioEvidence, EvidenceBundle, EvidenceKind, TableEvidence, VisualEvidence
from src.rag.types import RetrievedChunk


class EvidenceValidationResult(BaseModel):
    sufficient: bool
    should_refuse: bool
    risk: Literal["low", "medium", "high"] = "low"
    reason: str | None = None
    rule: RefusalRule | None = None
    confidence: float = 0.0
    modality_ok: bool = True
    selected_evidence_ids: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


def _chunk_is_table(chunk: RetrievedChunk) -> bool:
    if chunk.modality == "table":
        return True
    meta = chunk.metadata or {}
    return bool(meta.get("sheet_names") or meta.get("block_kinds"))


class EvidenceValidator:
    """Single entry point for "is the evidence good enough to answer?".

    `should_refuse` stays authoritative from RefusalPolicy (no behaviour change);
    the modality check only enriches `sufficient`/`risk`/`missing` so a table
    question with no table evidence is visibly flagged in the trace.
    """

    def __init__(self, refusal_policy: RefusalPolicy | None = None) -> None:
        self.refusal_policy = refusal_policy or RefusalPolicy()

    def validate(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        evidence_bundle: EvidenceBundle | None = None,
        preferred_modality: str | None = None,
        aux_query: str = "",
    ) -> EvidenceValidationResult:
        chunks = chunks or (evidence_bundle.to_legacy_chunks() if evidence_bundle is not None else [])
        decision = self.refusal_policy.check_evidence(chunks, query, aux_query=aux_query)

        modality_ok = True
        missing: list[str] = []
        if preferred_modality and preferred_modality != "none" and chunks:
            if evidence_bundle is not None:
                if preferred_modality == "table":
                    modality_ok = any(item.kind == EvidenceKind.TABLE.value or isinstance(item, TableEvidence) for item in evidence_bundle.items)
                    missing_name = "table_evidence"
                elif preferred_modality == "figure":
                    modality_ok = any(item.kind == EvidenceKind.VISUAL.value or isinstance(item, VisualEvidence) for item in evidence_bundle.items)
                    missing_name = "visual_evidence"
                elif preferred_modality == "audio":
                    modality_ok = any(item.kind == EvidenceKind.AUDIO.value or isinstance(item, AudioEvidence) for item in evidence_bundle.items)
                    missing_name = "audio_evidence"
                else:
                    missing_name = f"{preferred_modality}_evidence"
            elif preferred_modality == "table":
                modality_ok = any(_chunk_is_table(c) for c in chunks)
                missing_name = "table_evidence"
            elif preferred_modality == "figure":
                modality_ok = any(c.modality == "figure" for c in chunks)
                missing_name = "visual_evidence"
            elif preferred_modality == "audio":
                modality_ok = any(c.modality == "audio" for c in chunks)
                missing_name = "audio_evidence"
            else:
                missing_name = f"{preferred_modality}_evidence"
            if not modality_ok:
                missing.append(missing_name)

        if decision.should_refuse:
            risk: Literal["low", "medium", "high"] = "high"
        elif decision.rule == RefusalRule.LOW_CONFIDENCE or decision.reason == "partial_confidence" or not modality_ok:
            risk = "medium"
        else:
            risk = "low"

        return EvidenceValidationResult(
            sufficient=not decision.should_refuse and modality_ok,
            should_refuse=decision.should_refuse,
            risk=risk,
            reason=decision.reason,
            rule=decision.rule,
            confidence=decision.confidence,
            modality_ok=modality_ok,
            selected_evidence_ids=(
                [item.evidence_id for item in evidence_bundle.items]
                if evidence_bundle is not None else [c.chunk_id for c in chunks]
            ),
            missing=missing,
        )
