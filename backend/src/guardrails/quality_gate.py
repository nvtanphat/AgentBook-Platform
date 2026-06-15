"""Unified post-generation quality gate (Phase 5).

Combines three independent signal groups into a single structured verdict:
  - SLEC:       sentence-level coverage (coverage_ratio, refused)
  - Citation:   citation aligner (citation_coverage, invalid_citation_count)
  - Confidence: reranker-derived confidence score

The gate does NOT override the existing `should_refuse` decision — that is
still determined by refusal_policy + SLEC as before. Instead the gate
measures the three dimensions with PASS/CAUTION/FAIL and writes them into the
RequestTrace so they are visible in QueryLog.trace.quality_stage_verdicts.

This decoupling keeps the gate non-breaking: adding it cannot increase
refusals beyond what SLEC/refusal_policy already decide.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.guardrails.citation_aligner import CitationAlignmentResult
from src.schemas.query import SentenceCoverageReport


# ── Stage verdict ──────────────────────────────────────────────────────────────


class StageVerdict(BaseModel):
    stage: str
    verdict: Literal["PASS", "CAUTION", "FAIL"]
    score: float = Field(ge=0.0, le=1.0)


# ── Gate result ────────────────────────────────────────────────────────────────


class QualityGateResult(BaseModel):
    """Single structured quality signal for a completed answer.

    Fields are designed to map 1-to-1 with QueryLog.trace so that eval scripts
    and dashboards can compute aggregate quality metrics over the log corpus.
    """

    passed: bool
    """True when all stages are PASS or CAUTION (no FAIL)."""

    should_refuse: bool = False
    """Elevated when 2+ stages FAIL (e.g. confidence low AND citation coverage poor).

    Note: SLEC FAIL and Confidence FAIL are already handled upstream by
    SentenceCoverageGate and RefusalPolicy respectively. This gate catches
    the residual case where Citation stage FAIL coincides with another FAIL.
    """

    confidence: float = Field(ge=0.0, le=1.0)

    stage_verdicts: list[StageVerdict] = Field(default_factory=list)

    unsupported_claim_count: int = 0
    """Number of SLEC-unsupported sentences (proxy for unsupported claims)."""

    invalid_citation_count: int = 0

    corrected_facts: list[str] = Field(default_factory=list)
    """Human-readable notes from citation aligner (passed through for trace)."""

    # Convenience accessor for trace serialisation
    def verdicts_dict(self) -> dict[str, dict[str, float | str]]:
        return {v.stage: {"verdict": v.verdict, "score": v.score} for v in self.stage_verdicts}


# ── Gate ───────────────────────────────────────────────────────────────────────


# Thresholds — intentionally lightweight; production thresholds live in config.
_CONF_CAUTION = 0.50
_CONF_FAIL = 0.30
_SLEC_CAUTION = 0.60
_SLEC_FAIL = 0.40
_CIT_CAUTION = 0.80
_CIT_FAIL = 0.50


def _verdict_from_score(
    score: float,
    *,
    caution_below: float,
    fail_below: float,
) -> Literal["PASS", "CAUTION", "FAIL"]:
    if score < fail_below:
        return "FAIL"
    if score < caution_below:
        return "CAUTION"
    return "PASS"


class QualityGate:
    """Compose SLEC + citation + confidence into a QualityGateResult."""

    def evaluate(
        self,
        *,
        slec_report: SentenceCoverageReport | None,
        alignment: CitationAlignmentResult,
        confidence: float,
    ) -> QualityGateResult:
        verdicts: list[StageVerdict] = []

        # ── Confidence stage ───────────────────────────────────────────────────
        conf_score = max(0.0, min(1.0, confidence))
        verdicts.append(
            StageVerdict(
                stage="confidence",
                verdict=_verdict_from_score(
                    conf_score, caution_below=_CONF_CAUTION, fail_below=_CONF_FAIL
                ),
                score=round(conf_score, 4),
            )
        )

        # ── SLEC stage ─────────────────────────────────────────────────────────
        unsupported_count = 0
        if slec_report and slec_report.enabled:
            slec_score = slec_report.coverage_ratio
            unsupported_count = slec_report.unsupported_count
            if slec_report.refused:
                slec_verdict: Literal["PASS", "CAUTION", "FAIL"] = "FAIL"
            else:
                slec_verdict = _verdict_from_score(
                    slec_score, caution_below=_SLEC_CAUTION, fail_below=_SLEC_FAIL
                )
            verdicts.append(
                StageVerdict(stage="slec", verdict=slec_verdict, score=round(slec_score, 4))
            )
        else:
            # SLEC disabled or skipped — treat as neutral
            verdicts.append(StageVerdict(stage="slec", verdict="PASS", score=1.0))

        # ── Citation stage ─────────────────────────────────────────────────────
        cit_score = alignment.citation_coverage
        verdicts.append(
            StageVerdict(
                stage="citation",
                verdict=_verdict_from_score(
                    cit_score, caution_below=_CIT_CAUTION, fail_below=_CIT_FAIL
                ),
                score=round(cit_score, 4),
            )
        )

        # ── Aggregate ──────────────────────────────────────────────────────────
        fail_count = sum(1 for v in verdicts if v.verdict == "FAIL")
        passed = fail_count == 0
        should_refuse = fail_count >= 2  # refuse when 2+ independent stages FAIL

        return QualityGateResult(
            passed=passed,
            should_refuse=should_refuse,
            confidence=round(conf_score, 4),
            stage_verdicts=verdicts,
            unsupported_claim_count=unsupported_count,
            invalid_citation_count=alignment.invalid_citation_count,
            corrected_facts=alignment.details,
        )
