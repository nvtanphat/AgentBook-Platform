# -*- coding: utf-8 -*-
"""Tests for QualityGate (Phase 5)."""
from src.guardrails.citation_aligner import CitationAlignmentResult
from src.guardrails.quality_gate import QualityGate
from src.schemas.query import SentenceCoverageReport


GATE = QualityGate()


def _alignment(coverage: float = 1.0, invalid: int = 0) -> CitationAlignmentResult:
    return CitationAlignmentResult(
        citation_coverage=coverage,
        invalid_citation_count=invalid,
        corrected_answer="",
    )


def _slec(coverage: float, refused: bool = False, total: int = 3, unsupported: int = 0) -> SentenceCoverageReport:
    return SentenceCoverageReport(
        enabled=True,
        total_sentences=total,
        supported_count=total - unsupported,
        partial_count=0,
        unsupported_count=unsupported,
        coverage_ratio=coverage,
        refused=refused,
    )


class TestPassing:
    def test_all_high_scores_passes(self):
        r = GATE.evaluate(
            slec_report=_slec(0.95),
            alignment=_alignment(1.0),
            confidence=0.85,
        )
        assert r.passed is True
        assert r.should_refuse is False
        assert all(v.verdict == "PASS" for v in r.stage_verdicts)

    def test_stage_verdicts_have_all_three_stages(self):
        r = GATE.evaluate(slec_report=_slec(0.9), alignment=_alignment(), confidence=0.8)
        stages = {v.stage for v in r.stage_verdicts}
        assert stages == {"confidence", "slec", "citation"}


class TestCaution:
    def test_medium_confidence_is_caution(self):
        r = GATE.evaluate(slec_report=_slec(0.9), alignment=_alignment(), confidence=0.45)
        conf_v = next(v for v in r.stage_verdicts if v.stage == "confidence")
        assert conf_v.verdict == "CAUTION"
        assert r.passed is True  # one caution still passes

    def test_low_slec_coverage_is_caution(self):
        r = GATE.evaluate(slec_report=_slec(0.55), alignment=_alignment(), confidence=0.8)
        slec_v = next(v for v in r.stage_verdicts if v.stage == "slec")
        assert slec_v.verdict == "CAUTION"

    def test_citation_coverage_80_is_caution(self):
        r = GATE.evaluate(slec_report=_slec(0.9), alignment=_alignment(0.75), confidence=0.8)
        cit_v = next(v for v in r.stage_verdicts if v.stage == "citation")
        assert cit_v.verdict == "CAUTION"


class TestFail:
    def test_slec_refused_marks_slec_fail(self):
        r = GATE.evaluate(slec_report=_slec(0.1, refused=True), alignment=_alignment(), confidence=0.8)
        slec_v = next(v for v in r.stage_verdicts if v.stage == "slec")
        assert slec_v.verdict == "FAIL"

    def test_all_stages_fail_triggers_should_refuse(self):
        r = GATE.evaluate(
            slec_report=_slec(0.1, refused=True),
            alignment=_alignment(0.0, invalid=5),
            confidence=0.1,
        )
        assert r.should_refuse is True
        assert r.passed is False

    def test_partial_fail_does_not_trigger_should_refuse(self):
        r = GATE.evaluate(
            slec_report=_slec(0.9),
            alignment=_alignment(0.0, invalid=5),
            confidence=0.8,
        )
        assert r.should_refuse is False
        assert r.passed is False


class TestFields:
    def test_unsupported_claim_count_from_slec(self):
        r = GATE.evaluate(slec_report=_slec(0.7, unsupported=2), alignment=_alignment(), confidence=0.7)
        assert r.unsupported_claim_count == 2

    def test_invalid_citation_count_propagated(self):
        r = GATE.evaluate(slec_report=_slec(0.9), alignment=_alignment(0.5, invalid=3), confidence=0.8)
        assert r.invalid_citation_count == 3

    def test_verdicts_dict_serializable(self):
        r = GATE.evaluate(slec_report=_slec(0.9), alignment=_alignment(), confidence=0.8)
        d = r.verdicts_dict()
        assert "confidence" in d and "slec" in d and "citation" in d
        for stage_data in d.values():
            assert "verdict" in stage_data and "score" in stage_data

    def test_no_slec_report_uses_pass(self):
        r = GATE.evaluate(slec_report=None, alignment=_alignment(), confidence=0.8)
        slec_v = next(v for v in r.stage_verdicts if v.stage == "slec")
        assert slec_v.verdict == "PASS"
        assert slec_v.score == 1.0
