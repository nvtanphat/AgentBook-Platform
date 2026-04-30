from __future__ import annotations

import pytest

from src.processing.ocr_quality_gate import OCRQualityReport, score_ocr_document
from src.processing.types import ParsedBlock, ParsedDocument, ParsedPage


def _doc(text: str) -> ParsedDocument:
    block = ParsedBlock(
        block_id="b1",
        block_index=0,
        block_type="paragraph",
        content=text,
        page_number=1,
        reading_order=0,
        source="test",
    )
    page = ParsedPage(page_number=1, blocks=[block])
    return ParsedDocument(source_path="test.png", file_type="png", pages=[page])


class TestScoreOCRDocument:
    def test_empty_document_returns_zero_score(self):
        doc = _doc("")
        report = score_ocr_document(doc)
        assert report.score == 0.0
        assert report.total_chars == 0
        assert any("empty" in w for w in report.warnings)

    def test_clean_english_text_scores_high(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        report = score_ocr_document(_doc(text))
        assert report.score >= 0.55
        assert report.valid_char_ratio > 0.95
        assert report.meaningful_word_ratio > 0.7

    def test_repetition_penalty_lowers_score(self):
        text = "aaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbb ccccccccccccc"
        report = score_ocr_document(_doc(text))
        assert report.repetition_ratio > 0.0
        # A word made entirely of one letter still has >= 2 alpha chars, so
        # meaningful_word_ratio won't be zero — but score should be impacted.
        clean_text = "normal words that are meaningful and coherent " * 3
        clean_report = score_ocr_document(_doc(clean_text))
        assert clean_report.score >= report.score

    def test_replacement_chars_lower_valid_ratio(self):
        text = "word " + "word�" * 20
        report = score_ocr_document(_doc(text))
        assert report.valid_char_ratio < 1.0

    def test_symbol_density_detected_for_unusual_chars(self):
        # Characters NOT in _ALLOWED_PUNCTUATION trigger symbol_density
        text = "☃♥♦♠♣" * 10 + " normal text here"
        report = score_ocr_document(_doc(text))
        assert report.symbol_density > 0.0

    def test_is_acceptable_uses_min_score(self):
        report = OCRQualityReport(
            score=0.40,
            valid_char_ratio=0.9,
            meaningful_word_ratio=0.6,
            repetition_ratio=0.0,
            symbol_density=0.0,
            total_chars=100,
        )
        assert report.is_acceptable(0.35)
        assert not report.is_acceptable(0.55)

    def test_flag_summary_ok_when_no_warnings(self):
        report = OCRQualityReport(
            score=0.80,
            valid_char_ratio=0.98,
            meaningful_word_ratio=0.85,
            repetition_ratio=0.0,
            symbol_density=0.0,
            total_chars=200,
        )
        assert report.flag_summary() == "ok"

    def test_warn_threshold_warning_in_output(self):
        short_text = "ab cd ef gh ij kl mn"
        report = score_ocr_document(_doc(short_text), min_score=0.35, warn_score=0.99)
        # score will be below warn_score=0.99 but likely above min_score=0.35
        if report.score < 0.99:
            assert any("warning threshold" in w for w in report.warnings)

    def test_below_min_score_warning(self):
        text = "ok "  # very short, will be below 30 chars threshold
        report = score_ocr_document(_doc(text), min_score=0.99)
        assert any("fail threshold" in w for w in report.warnings)
        assert not report.is_acceptable(0.99)
