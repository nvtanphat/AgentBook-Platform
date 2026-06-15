"""Modality-aware citation validity checker (Phase 5).

Checks every [N] marker in the generated answer:
  1. Out-of-range: N < 1 or N > len(chunks) — reuses response_parser logic.
  2. Modality mismatch (when preferred_modality is set): a table citation must
     resolve to a table-modality chunk; a figure citation to a chunk whose
     evidence has bbox; an audio citation to evidence with a timestamp field.

Input is the SLEC SentenceCoverageReport — its per-sentence `citation_refs` and
`supporting_block_ids` are reused so we avoid a second scan of the answer.
The corrected_answer strips out invalid [N] markers from the text.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.processing.types import EvidenceBlock
from src.rag.types import RetrievedChunk
from src.schemas.query import SentenceCoverageReport


_CITATION_RE = re.compile(r"\[(\d+)\]")


class CitationAlignmentResult(BaseModel):
    """Output of the citation aligner — all fields are safe to store in the trace."""

    citation_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    """valid_citations / total_citations in the answer (1.0 when answer has none)."""

    invalid_citation_count: int = 0
    """[N] markers that are out-of-range or modality-mismatched."""

    unsupported_sentence_count: int = 0
    """Sentences that have at least one invalid citation ref."""

    corrected_answer: str = ""
    """Answer with invalid [N] markers removed."""

    stage: Literal["PASS", "CAUTION", "FAIL"] = "PASS"
    """PASS = all citations valid; CAUTION = some invalid but answer survives;
    FAIL = majority of citations invalid (coverage < 0.5)."""

    details: list[str] = Field(default_factory=list)
    """Human-readable notes for trace debugging (max 5 entries)."""


# ── Helpers ────────────────────────────────────────────────────────────────────


def _has_bbox(blocks: list[EvidenceBlock]) -> bool:
    return any(getattr(b, "bbox", None) for b in blocks)


def _has_timestamp(blocks: list[EvidenceBlock]) -> bool:
    for b in blocks:
        meta = getattr(b, "metadata", None) or {}
        if meta.get("start_sec") is not None or meta.get("timestamp") is not None:
            return True
    return False


def _chunk_matches_modality(chunk: RetrievedChunk, modality: str) -> bool:
    """Return True if chunk satisfies the modality contract for a citation."""
    if modality == "table":
        if chunk.modality == "table":
            return True
        meta = chunk.metadata or {}
        return bool(meta.get("sheet_names") or meta.get("block_kinds"))

    if modality == "figure":
        blocks: list[EvidenceBlock] = list(chunk.evidence) if chunk.evidence else []
        return _has_bbox(blocks)

    if modality == "audio":
        blocks = list(chunk.evidence) if chunk.evidence else []
        return _has_timestamp(blocks)

    return True  # no constraint for text / NONE


def _strip_invalid_markers(answer: str, invalid: set[int]) -> str:
    """Remove [N] markers whose N is in the invalid set."""
    if not invalid:
        return answer

    def _replacer(m: re.Match) -> str:  # type: ignore[type-arg]
        n = int(m.group(1))
        return "" if n in invalid else m.group(0)

    cleaned = _CITATION_RE.sub(_replacer, answer)
    # Collapse double-spaces left by removed markers
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


# ── Aligner ────────────────────────────────────────────────────────────────────


class CitationAligner:
    """Align every [N] in the answer against the retrieved chunks."""

    def align(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk],
        slec_report: SentenceCoverageReport | None = None,
        preferred_modality: str | None = None,
    ) -> CitationAlignmentResult:
        if not answer or not chunks:
            return CitationAlignmentResult(corrected_answer=answer or "")

        chunk_count = len(chunks)
        all_markers = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
        if not all_markers:
            return CitationAlignmentResult(corrected_answer=answer)

        # Step 1: out-of-range markers
        invalid: set[int] = {n for n in all_markers if n < 1 or n > chunk_count}

        # Step 2: modality-mismatch check (only when preferred_modality is set)
        details: list[str] = []
        mismatched_sentences = 0

        if preferred_modality and preferred_modality != "none":
            # Use SLEC sentence-level citation_refs when available (avoids re-parsing)
            sentences_to_check: list[tuple[list[int], str]] = []
            if slec_report and slec_report.sentences:
                for s in slec_report.sentences:
                    if s.citation_refs:
                        sentences_to_check.append((s.citation_refs, s.text[:60]))
            else:
                # Fallback: scan the raw answer sentence by sentence
                for sent in re.split(r"(?<=[.!?])\s+", answer):
                    refs = [int(m.group(1)) for m in _CITATION_RE.finditer(sent)]
                    if refs:
                        sentences_to_check.append((refs, sent[:60]))

            for refs, sent_snippet in sentences_to_check:
                has_mismatch = False
                for ref in refs:
                    if ref < 1 or ref > chunk_count:
                        continue  # already flagged as out-of-range
                    chunk = chunks[ref - 1]
                    if not _chunk_matches_modality(chunk, preferred_modality):
                        invalid.add(ref)
                        has_mismatch = True
                        if len(details) < 5:
                            details.append(
                                f"[{ref}] modality={chunk.modality!r} "
                                f"expected={preferred_modality!r}: …{sent_snippet}…"
                            )
                if has_mismatch:
                    mismatched_sentences += 1

        total = len(all_markers)
        invalid_count = len(invalid)
        valid_count = total - invalid_count
        coverage = valid_count / total if total > 0 else 1.0

        corrected = _strip_invalid_markers(answer, invalid)

        if coverage >= 0.9:
            stage: Literal["PASS", "CAUTION", "FAIL"] = "PASS"
        elif coverage >= 0.5:
            stage = "CAUTION"
        else:
            stage = "FAIL"

        return CitationAlignmentResult(
            citation_coverage=round(coverage, 4),
            invalid_citation_count=invalid_count,
            unsupported_sentence_count=mismatched_sentences,
            corrected_answer=corrected,
            stage=stage,
            details=details,
        )
