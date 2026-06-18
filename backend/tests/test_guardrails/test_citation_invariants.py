# -*- coding: utf-8 -*-
"""Invariant tests for _prune_to_cited citation drift prevention.

After _prune_to_cited runs, the following invariants must hold on all 3 answer paths:
1. Every [N] marker in the answer satisfies 1 ≤ N ≤ len(citations).
2. Every citation_ref in every SLEC sentence row is within {1..len(citations)}.
3. len(pruned_chunks) == len(new_citations) when chunks are provided.
"""
import re
from unittest.mock import MagicMock

from src.inference.inference_engine import InferenceEngine
from src.schemas.query import SentenceCoverageReport, SentenceSupport


# ── helpers ──────────────────────────────────────────────────────────────────


def _citation(i: int):
    c = MagicMock()
    c.chunk_id = f"chunk-{i}"
    return c


def _chunk(i: int):
    c = MagicMock()
    c.chunk_id = f"chunk-{i}"
    return c


def _slec_report(citation_refs_per_sentence: list[list[int]]) -> SentenceCoverageReport:
    sentences = [
        SentenceSupport(
            index=i - 1,
            text=f"Sentence {i}.",
            status="supported",
            score=0.9,
            supporting_block_ids=[],
            citation_refs=refs,
        )
        for i, refs in enumerate(citation_refs_per_sentence, start=1)
    ]
    return SentenceCoverageReport(
        enabled=True,
        total_sentences=len(sentences),
        supported_count=len(sentences),
        partial_count=0,
        unsupported_count=0,
        coverage_ratio=1.0,
        refused=False,
        sentences=sentences,
    )


def _extract_refs(answer: str) -> set[int]:
    return {
        int(n)
        for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer)
        for n in re.findall(r"\d+", m.group(1))
    }


# ── invariant verifier ────────────────────────────────────────────────────────


def assert_citation_invariants(
    answer: str, citations: list, slec_report, pruned_chunks, *, chunks_provided: bool = True
):
    k = len(citations)
    refs_in_answer = _extract_refs(answer)
    assert all(1 <= r <= k for r in refs_in_answer), (
        f"Answer refs out of range [1..{k}]: {refs_in_answer}"
    )
    if slec_report is not None:
        for sent in slec_report.sentences:
            for ref in sent.citation_refs:
                assert 1 <= ref <= k, (
                    f"SLEC citation_ref {ref} out of range [1..{k}] in '{sent.text}'"
                )
    # Length alignment only checked when the caller supplied a chunks list.
    # When chunks=None is passed, _prune_to_cited returns [] regardless.
    if chunks_provided and pruned_chunks is not None:
        assert len(pruned_chunks) == k, (
            f"pruned_chunks length {len(pruned_chunks)} != citations length {k}"
        )


# ── test cases ────────────────────────────────────────────────────────────────


class TestPruneToCited:
    def _run(self, answer, citations, slec_report, chunks):
        new_answer, new_citations, new_slec, pruned = InferenceEngine._prune_to_cited(
            answer, citations, slec_report, chunks=chunks,
        )
        assert_citation_invariants(new_answer, new_citations, new_slec, pruned)
        return new_answer, new_citations, new_slec, pruned

    def test_no_citations(self):
        self._run("No citations here.", [], None, None)

    def test_all_citations_referenced(self):
        cits = [_citation(i) for i in range(1, 4)]
        chunks = [_chunk(i) for i in range(1, 4)]
        answer = "Alpha [1]. Beta [2]. Gamma [3]."
        slec = _slec_report([[1], [2], [3]])
        self._run(answer, cits, slec, chunks)

    def test_partial_subset_referenced(self):
        cits = [_citation(i) for i in range(1, 6)]
        chunks = [_chunk(i) for i in range(1, 6)]
        # answer only uses [1] and [3]
        answer = "Alpha [1]. Beta [3]."
        slec = _slec_report([[1], [3]])
        new_answer, new_citations, new_slec, pruned = self._run(answer, cits, slec, chunks)
        assert len(new_citations) == 2
        assert len(pruned) == 2
        assert "[1]" in new_answer and "[2]" in new_answer  # renumbered: [1],[3] → [1],[2]
        assert "[3]" not in new_answer

    def test_slec_refs_remapped_correctly(self):
        cits = [_citation(i) for i in range(1, 5)]
        chunks = [_chunk(i) for i in range(1, 5)]
        answer = "Alpha [2]. Beta [4]."
        slec = _slec_report([[2], [4]])
        _, new_citations, new_slec, _ = self._run(answer, cits, slec, chunks)
        assert len(new_citations) == 2
        all_refs = [r for s in new_slec.sentences for r in s.citation_refs]
        assert set(all_refs) == {1, 2}

    def test_multi_citation_marker(self):
        cits = [_citation(i) for i in range(1, 5)]
        answer = "Fact [1, 3, 5]."  # [5] is out of range → dropped
        slec = _slec_report([[1, 3]])
        new_answer, new_citations, _, pruned = InferenceEngine._prune_to_cited(
            answer, cits, slec, chunks=None,
        )
        assert_citation_invariants(new_answer, new_citations, slec, pruned, chunks_provided=False)
        # only [1] and [3] are valid refs in a 4-citation set
        refs_after = _extract_refs(new_answer)
        assert all(1 <= r <= len(new_citations) for r in refs_after)

    def test_no_chunks_provided(self):
        cits = [_citation(i) for i in range(1, 4)]
        answer = "Alpha [1]."
        slec = _slec_report([[1]])
        new_answer, new_citations, new_slec, pruned = InferenceEngine._prune_to_cited(
            answer, cits, slec, chunks=None,
        )
        assert_citation_invariants(new_answer, new_citations, new_slec, pruned, chunks_provided=False)
        assert pruned == []

    def test_answer_without_any_markers(self):
        cits = [_citation(i) for i in range(1, 4)]
        answer = "No markers."
        slec = _slec_report([[1], [2], [3]])
        # No used refs → all citations kept (len(used)==0 branch)
        new_answer, new_citations, new_slec, pruned = InferenceEngine._prune_to_cited(
            answer, cits, slec, chunks=None,
        )
        assert_citation_invariants(new_answer, new_citations, new_slec, pruned, chunks_provided=False)
        # invariant: empty marker set is fine, citations unchanged
        assert len(new_citations) == len(cits)
