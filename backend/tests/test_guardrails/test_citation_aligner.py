# -*- coding: utf-8 -*-
"""Tests for CitationAligner (Phase 5)."""
from types import SimpleNamespace

import pytest

from src.guardrails.citation_aligner import CitationAligner, _strip_invalid_markers
from src.rag.types import RetrievedChunk
from src.schemas.query import SentenceCoverageReport, SentenceSupport


def _chunk(cid: str, modality: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, owner_id="o", collection_id="c", material_id="m",
        document_name="d.pdf", content="evidence text", language="vi",
        modality=modality, metadata={}, rerank_score=0.9,
    )


def _table_chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, owner_id="o", collection_id="c", material_id="m",
        document_name="d.pdf", content="| A | B |", language="vi",
        modality="table", metadata={"sheet_names": ["Sheet1"]}, rerank_score=0.9,
    )


ALIGNER = CitationAligner()


class TestNoMarkers:
    def test_answer_without_citations_returns_pass(self):
        r = ALIGNER.align(answer="Plain answer.", chunks=[_chunk("c1")])
        assert r.citation_coverage == 1.0
        assert r.invalid_citation_count == 0
        assert r.stage == "PASS"
        assert r.corrected_answer == "Plain answer."

    def test_empty_answer_returns_default(self):
        r = ALIGNER.align(answer="", chunks=[_chunk("c1")])
        assert r.citation_coverage == 1.0


class TestOutOfRangeCitations:
    def test_citation_beyond_chunk_count_is_invalid(self):
        # Only 1 chunk, [2] is out-of-range
        r = ALIGNER.align(answer="Test [2] answer.", chunks=[_chunk("c1")])
        assert r.invalid_citation_count == 1
        assert "[2]" not in r.corrected_answer
        assert r.stage in {"CAUTION", "FAIL"}

    def test_citation_zero_is_invalid(self):
        r = ALIGNER.align(answer="Test [0].", chunks=[_chunk("c1")])
        assert r.invalid_citation_count == 1

    def test_valid_citation_passes(self):
        r = ALIGNER.align(answer="Answer [1].", chunks=[_chunk("c1")])
        assert r.invalid_citation_count == 0
        assert r.citation_coverage == 1.0
        assert r.stage == "PASS"

    def test_mixed_valid_invalid(self):
        r = ALIGNER.align(answer="A [1] and [3].", chunks=[_chunk("c1"), _chunk("c2")])
        assert r.invalid_citation_count == 1
        assert "[1]" in r.corrected_answer
        assert "[3]" not in r.corrected_answer


class TestModalityCheck:
    def test_table_citation_pointing_to_text_chunk_is_flagged(self):
        r = ALIGNER.align(
            answer="Tổng giá [1].",
            chunks=[_chunk("c1", modality="text")],
            preferred_modality="table",
        )
        assert r.invalid_citation_count >= 1
        assert r.unsupported_sentence_count >= 1

    def test_table_citation_pointing_to_table_chunk_passes(self):
        r = ALIGNER.align(
            answer="Tổng giá [1].",
            chunks=[_table_chunk("c1")],
            preferred_modality="table",
        )
        assert r.invalid_citation_count == 0
        assert r.stage == "PASS"

    def test_no_modality_constraint_skips_check(self):
        r = ALIGNER.align(
            answer="Answer [1].",
            chunks=[_chunk("c1", modality="text")],
            preferred_modality=None,
        )
        assert r.invalid_citation_count == 0

    def test_none_modality_string_skips_check(self):
        r = ALIGNER.align(
            answer="Answer [1].",
            chunks=[_chunk("c1", modality="text")],
            preferred_modality="none",
        )
        assert r.invalid_citation_count == 0


class TestStripInvalidMarkers:
    def test_removes_invalid_marker(self):
        result = _strip_invalid_markers("Hello [2] world [1].", {2})
        assert "[2]" not in result
        assert "[1]" in result

    def test_no_invalid_markers_unchanged(self):
        text = "Hello [1] world."
        assert _strip_invalid_markers(text, set()) == text


class TestCoverageStage:
    def test_all_valid_is_pass(self):
        r = ALIGNER.align(answer="A [1] B [2].", chunks=[_chunk("c1"), _chunk("c2")])
        assert r.stage == "PASS"
        assert r.citation_coverage == 1.0

    def test_half_invalid_is_caution(self):
        # 2 markers, 1 invalid → coverage 0.5
        r = ALIGNER.align(answer="A [1] B [3].", chunks=[_chunk("c1"), _chunk("c2")])
        assert r.citation_coverage == pytest.approx(0.5)
        assert r.stage == "CAUTION"

    def test_all_invalid_is_fail(self):
        r = ALIGNER.align(answer="A [5] B [6].", chunks=[_chunk("c1")])
        assert r.stage == "FAIL"
        assert r.citation_coverage == 0.0
