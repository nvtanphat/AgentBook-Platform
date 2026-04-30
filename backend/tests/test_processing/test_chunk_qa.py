from __future__ import annotations

import pytest

from src.processing.chunk_qa import run_chunk_qa
from src.processing.types import TextChunk


def _chunk(content: str, token_count: int | None = None) -> TextChunk:
    return TextChunk(
        owner_id="u1",
        collection_id="col1",
        material_id="mat1",
        document_name="test.pdf",
        content=content,
        language="en",
        modality="text",
        source_block_ids=["blk-1"],
        source_pages=[1],
        token_count=token_count if token_count is not None else len(content.split()),
        chunk_strategy="test",
        chunker_version="v1",
        parser_version="v1",
        embedding_model="fake",
        embedding_version="fake-v1",
        index_version="test",
    )


class TestRunChunkQA:
    def test_empty_chunks_warns(self):
        report = run_chunk_qa([], material_id="mat1")
        assert report.total_chunks == 0
        assert any("no chunks" in w for w in report.warnings)

    def test_clean_chunks_pass(self):
        chunks = [
            _chunk("This is the first paragraph with enough distinct tokens.", 10),
            _chunk("A second paragraph covering different material for testing.", 10),
            _chunk("Third paragraph with unique content that is clearly not a duplicate.", 10),
        ]
        report = run_chunk_qa(chunks, material_id="mat1")
        assert not report.has_issues
        assert report.summary() == "ok"

    def test_too_short_detected(self):
        chunks = [_chunk("hi", token_count=1), _chunk("Normal paragraph with good content here.", 10)]
        report = run_chunk_qa(chunks, material_id="mat1", min_tokens=5)
        assert 0 in report.too_short
        assert 1 not in report.too_short

    def test_too_long_detected(self):
        long_text = " ".join(["word"] * 50)
        chunks = [_chunk(long_text, token_count=700), _chunk("Short chunk.", 10)]
        report = run_chunk_qa(chunks, material_id="mat1", max_tokens=600)
        assert 0 in report.too_long
        assert 1 not in report.too_long

    def test_heading_only_detected(self):
        chunks = [_chunk("# Introduction\n## Subsection"), _chunk("Normal text here today.")]
        report = run_chunk_qa(chunks, material_id="mat1")
        assert 0 in report.heading_only
        assert 1 not in report.heading_only

    def test_duplicate_detected(self):
        text = "The same content repeated exactly here for testing purposes." * 2
        chunks = [_chunk(text), _chunk(text), _chunk("Different content entirely here.")]
        report = run_chunk_qa(chunks, material_id="mat1")
        assert len(report.duplicate) >= 1

    def test_noisy_ocr_detected(self):
        noisy = "abc\x00\x01\x02\x03\x04\x05\x06\x07\x08 " * 10
        chunks = [_chunk(noisy), _chunk("Clean normal English text here.")]
        report = run_chunk_qa(chunks, material_id="mat1", noise_char_ratio_threshold=0.05)
        assert 0 in report.noisy_ocr
        assert 1 not in report.noisy_ocr

    def test_summary_lists_issue_types(self):
        chunks = [_chunk("hi", token_count=1)]
        report = run_chunk_qa(chunks, material_id="mat1", min_tokens=5)
        assert "too-short" in report.summary()

    def test_has_issues_false_when_clean(self):
        chunks = [_chunk("This is fine content with plenty of words.", 10)]
        report = run_chunk_qa(chunks, material_id="mat1")
        assert not report.has_issues
