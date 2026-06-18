"""Sentence-level Evidence Coverage (SLEC) gate.

Centerpiece of the Adaptive Evidence-Guided RAG architecture: after the LLM
generates an answer, every sentence is independently scored against the
retrieved evidence blocks using the BGE reranker (treated as an evidence-support
classifier). Three verdicts are possible per sentence:

  SUPPORTED   — score ≥ supported_threshold; keep as-is and attach citation
  PARTIAL     — partial_threshold ≤ score < supported_threshold; keep but flag
  UNSUPPORTED — score < partial_threshold; drop (or flag) the sentence

A whole-answer refusal triggers when the weighted coverage ratio falls below
`refuse_below`. The gate is bypassed for routes that already use a specialised
verifier (e.g., CLAIM_CHECK uses claim_verifier directly).

This module is intentionally self-contained and depends only on:
  - the settings object (for thresholds)
  - a cross-encoder reranker (for scoring)
  - retrieved chunks + their evidence blocks
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass

from src.core.config import Settings
from src.processing.types import EvidenceBlock
from src.rag.evidence import EvidenceBundle
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievedChunk
from src.schemas.query import SentenceCoverageReport, SentenceSupport

logger = logging.getLogger(__name__)


# ── Sentence splitter ──────────────────────────────────────────────────────────
# Vietnamese-aware: splits on ., !, ? and Vietnamese paragraph breaks.
# Avoids splitting on common patterns: numbers (3.14), abbreviations (vd., tức.),
# inline citation tags ([1]), markdown headers, and bullet markers.

_ABBREV = {
    "vd", "v.d", "tr", "tt", "tnhh", "hcm", "hn", "tp",
    "etc", "e.g", "i.e", "fig", "no", "vs", "approx",
}
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?…])\s+(?=[A-ZÀ-Ỹ0-9])")
_TRAILING_CITATION = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]\s*$")


def _split_sentences(text: str) -> list[str]:
    """Split answer text into sentences using Vietnamese-aware rules.

    Markdown blocks (code fences, tables, blockquotes, images) are preserved as
    single units — they are not real prose so we skip scoring them.
    """
    if not text or not text.strip():
        return []

    # Strip markdown image/table/code blocks first; they are non-prose.
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)         # inline images
    cleaned = re.sub(r"^\s*\|.+\|\s*$", " ", cleaned, flags=re.MULTILINE)  # table rows
    cleaned = re.sub(r"^\s*>\s.*$", " ", cleaned, flags=re.MULTILINE)      # blockquotes
    cleaned = re.sub(r"^\s*#{1,6}\s.+$", " ", cleaned, flags=re.MULTILINE) # headings
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return []

    raw = _SENT_SPLIT.split(cleaned)
    sentences: list[str] = []
    buffer = ""
    for piece in raw:
        candidate = (buffer + " " + piece).strip() if buffer else piece.strip()
        last_token = candidate.split()[-1].rstrip(".,").lower() if candidate.split() else ""
        if last_token in _ABBREV:
            # Looks like an abbreviation — merge with next piece.
            buffer = candidate
            continue
        buffer = ""
        if candidate:
            sentences.append(candidate)
    if buffer:
        sentences.append(buffer)
    return sentences


def _extract_trailing_citation(sentence: str) -> tuple[str, list[int]]:
    """Pull `[1]`, `[1, 2]` markers off the end of a sentence."""
    match = _TRAILING_CITATION.search(sentence)
    if not match:
        return sentence, []
    refs_raw = match.group(1)
    refs = [int(r.strip()) for r in refs_raw.split(",") if r.strip().isdigit()]
    stripped = sentence[: match.start()].rstrip()
    return stripped, refs


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class _EvidenceEntry:
    block_id: str
    chunk_id: str
    material_id: str
    text: str


# ── Gate ───────────────────────────────────────────────────────────────────────


class SentenceCoverageGate:
    """Run per-sentence entailment-style scoring over retrieved evidence."""

    def __init__(self, *, settings: Settings, reranker: CrossEncoderReranker) -> None:
        self.settings = settings
        self.reranker = reranker

    async def verify(
        self,
        *,
        answer: str,
        chunks: list[RetrievedChunk] | None = None,
        evidence_bundle: EvidenceBundle | None = None,
        route_type: str | None = None,
    ) -> tuple[str, SentenceCoverageReport]:
        """Score the answer and (optionally) rewrite it.

        Returns `(final_answer, report)`. When `drop_unsupported=true`, sentences
        flagged UNSUPPORTED are removed from the returned answer. When the
        weighted coverage ratio is below `refuse_below`, the gate signals a full
        refusal by setting `report.refused = True` and leaving the answer empty.
        """
        if not self.settings.slec_enabled:
            return answer, SentenceCoverageReport(enabled=False)

        if route_type and route_type.lower() in {r.lower() for r in self.settings.slec_skip_routes}:
            return answer, SentenceCoverageReport(enabled=False)

        # Reranker is required for scoring. When disabled (e.g., low-resource
        # deployments), short-circuit cleanly rather than refuse every answer.
        if not getattr(self.settings, "reranker_enabled", True):
            return answer, SentenceCoverageReport(enabled=False)

        chunks = chunks or []
        evidence = (
            self._collect_bundle_evidence(evidence_bundle)
            if evidence_bundle is not None
            else self._collect_evidence(chunks)
        )
        sentences_raw = _split_sentences(answer)
        if not sentences_raw or not evidence:
            return answer, SentenceCoverageReport(
                enabled=True,
                total_sentences=len(sentences_raw),
                supported_count=0,
                partial_count=0,
                unsupported_count=len(sentences_raw),
                coverage_ratio=0.0,
            )

        sentences_raw = sentences_raw[: self.settings.slec_max_sentences]

        scored: list[SentenceSupport] = []
        for idx, sent_raw in enumerate(sentences_raw):
            sent_stripped, refs = _extract_trailing_citation(sent_raw)
            if len(sent_stripped) < self.settings.slec_min_sentence_chars:
                # Treat tiny tokens (headings, residual bullets) as supported by default
                # — they carry no factual claim, so SLEC shouldn't punish them.
                scored.append(
                    SentenceSupport(
                        index=idx,
                        text=sent_raw,
                        status="supported",
                        score=1.0,
                        supporting_block_ids=[],
                        citation_refs=refs,
                    )
                )
                continue

            best_score, supporting_ids = await self._score_sentence(sent_stripped, evidence)
            status = self._classify(best_score)
            scored.append(
                SentenceSupport(
                    index=idx,
                    text=sent_raw,
                    status=status,
                    score=float(best_score),
                    supporting_block_ids=supporting_ids[:2],
                    citation_refs=refs,
                )
            )

        weighted = sum(self._weight(s.status) for s in scored) / max(1, len(scored))
        report = SentenceCoverageReport(
            enabled=True,
            total_sentences=len(scored),
            supported_count=sum(1 for s in scored if s.status == "supported"),
            partial_count=sum(1 for s in scored if s.status == "partial"),
            unsupported_count=sum(1 for s in scored if s.status == "unsupported"),
            coverage_ratio=float(weighted),
            sentences=scored,
        )

        if weighted < self.settings.slec_refuse_below:
            report.refused = True
            logger.info(
                "SLEC: refusing answer below coverage floor",
                extra={
                    "coverage_ratio": round(weighted, 3),
                    "floor": self.settings.slec_refuse_below,
                    "supported": report.supported_count,
                    "total": report.total_sentences,
                },
            )
            return "", report

        if self.settings.slec_drop_unsupported:
            kept_sentences = [s.text for s in scored if s.status != "unsupported"]
            dropped = report.unsupported_count
            final_answer = " ".join(kept_sentences).strip()
            report.dropped_count = dropped
            # If everything was dropped, treat as refusal.
            if not final_answer.strip():
                report.refused = True
                return "", report
            return final_answer, report

        return answer, report

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _classify(self, score: float) -> str:
        if score >= self.settings.slec_supported_threshold:
            return "supported"
        if score >= self.settings.slec_partial_threshold:
            return "partial"
        return "unsupported"

    @staticmethod
    def _weight(status: str) -> float:
        # Partial sentences count as half-credit, mirroring soft-coverage metrics.
        return 1.0 if status == "supported" else (0.5 if status == "partial" else 0.0)

    @staticmethod
    def _collect_evidence(
        chunks: list[RetrievedChunk], max_chunks: int = 5, max_total_blocks: int = 24
    ) -> list[_EvidenceEntry]:
        """Pull evidence text from the top-ranked chunks only.

        Capping is critical for latency: SLEC scores (n_sentences × n_evidence)
        pairs through the cross-encoder. v21 ablation tried max_total_blocks=12
        which dropped scoring cost but also lost faith from 0.979 → 0.935 —
        smaller evidence pool meant SLEC couldn't find supporting matches and
        dropped legitimate sentences. Reverted to 24 (the v17 setting) which
        keeps cost manageable while preserving SLEC's ability to support all
        sentences in multi-paragraph answers.
        """
        entries: list[_EvidenceEntry] = []
        for chunk in chunks[:max_chunks]:
            blocks: list[EvidenceBlock] = list(chunk.evidence) if chunk.evidence else []
            if blocks:
                for blk in blocks:
                    text = (blk.snippet_original or "").strip()
                    if not text:
                        continue
                    entries.append(
                        _EvidenceEntry(
                            block_id=blk.block_id or "",
                            chunk_id=chunk.chunk_id,
                            material_id=chunk.material_id,
                            text=text,
                        )
                    )
                    if len(entries) >= max_total_blocks:
                        return entries
            else:
                text = (chunk.content or "").strip()
                if text:
                    entries.append(
                        _EvidenceEntry(
                            block_id="",
                            chunk_id=chunk.chunk_id,
                            material_id=chunk.material_id,
                            text=text,
                        )
                    )
                    if len(entries) >= max_total_blocks:
                        return entries
        return entries

    @staticmethod
    def _collect_bundle_evidence(
        bundle: EvidenceBundle, max_items: int = 5, max_total_blocks: int = 24
    ) -> list[_EvidenceEntry]:
        entries: list[_EvidenceEntry] = []
        for item in bundle.items[:max_items]:
            blocks: list[EvidenceBlock] = list(item.evidence_blocks or [])
            if blocks:
                for blk in blocks:
                    text = (blk.snippet_original or "").strip()
                    if not text:
                        continue
                    entries.append(
                        _EvidenceEntry(
                            block_id=blk.block_id or item.block_id or "",
                            chunk_id=item.evidence_id,
                            material_id=item.material_id,
                            text=text,
                        )
                    )
                    if len(entries) >= max_total_blocks:
                        return entries
            else:
                text = (item.prompt_text() or item.snippet or "").strip()
                if text:
                    entries.append(
                        _EvidenceEntry(
                            block_id=item.block_id or "",
                            chunk_id=item.evidence_id,
                            material_id=item.material_id,
                            text=text,
                        )
                    )
                    if len(entries) >= max_total_blocks:
                        return entries
        return entries

    async def _score_sentence(
        self, sentence: str, evidence: list[_EvidenceEntry]
    ) -> tuple[float, list[str]]:
        """Reuse the cross-encoder to compute relevance(sentence, evidence_block).

        Returns the *normalised* best score in [0,1] (sigmoid of the raw logit)
        and the block_ids of up to two evidence entries with the highest scores.
        """
        if not evidence:
            return 0.0, []

        model = await self.reranker._aload_model()
        pairs = [(sentence, e.text) for e in evidence]
        try:
            raw_scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            logger.warning("SLEC reranker scoring failed", extra={"error": str(exc)})
            return 0.0, []

        normalised = [_sigmoid(float(s)) for s in raw_scores]
        ranked = sorted(zip(evidence, normalised), key=lambda p: p[1], reverse=True)
        best_score = ranked[0][1] if ranked else 0.0
        supporting_ids = [entry.block_id for entry, _ in ranked[:2] if entry.block_id]
        return best_score, supporting_ids
