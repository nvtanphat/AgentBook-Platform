"""Quality finalization chain — extracted from InferenceEngine for maintainability.

Owns the post-generation pipeline:
  SLEC → refine citation blocks → prune to cited → citation aligner → quality gate

Shared by all three answer paths (direct text, direct visual, agentic) so every
QueryResponse carries the same quality_stage_verdicts / citation_error_count /
claim_count signals for consistent LNFCG gate inputs.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.core.trace import RequestTrace
    from src.guardrails.citation_aligner import CitationAligner
    from src.guardrails.quality_gate import QualityGate
    from src.guardrails.sentence_coverage import SentenceCoverageGate
    from src.inference.response_parser import ResponseParser
    from src.rag.evidence import EvidenceBundle
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class QualityFinalizeResult:
    """Output of QualityFinalizer.finalize — carries unified post-gen signals."""

    answer: str
    citations: list = field(default_factory=list)
    slec_report: object | None = None
    pruned_chunks: list = field(default_factory=list)
    should_refuse: bool = False
    refusal_reason: str | None = None
    gate_result: object | None = None
    alignment: object | None = None


class QualityFinalizer:
    """Stateful helper that owns the full post-generation quality chain.

    Constructed once per InferenceEngine instance and reused across all answer paths.
    """

    def __init__(
        self,
        *,
        sentence_coverage_gate: "SentenceCoverageGate",
        citation_aligner: "CitationAligner",
        quality_gate: "QualityGate",
        response_parser: "ResponseParser",
        settings: "Settings",
        refusal_answer: str = "",
    ) -> None:
        self.sentence_coverage_gate = sentence_coverage_gate
        self.citation_aligner = citation_aligner
        self.quality_gate = quality_gate
        self.response_parser = response_parser
        self.settings = settings
        self._refusal_answer = refusal_answer

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def prune_to_cited(
        answer: str,
        citations: list,
        coverage_report=None,
        *,
        chunks: "list[RetrievedChunk] | None" = None,
    ):
        """Keep only citations the answer actually references, renumbered 1..k.

        Reranked context often includes off-topic low-rank chunks the LLM never
        cites; surfacing them as "citations" is misleading. We keep exactly the
        referenced ones, rewrite the answer's [N] markers, and remap SLEC sentence
        citation_refs so every consumer stays consistent.

        Optional `chunks`: when provided, the same 1-based index filter is applied
        so the returned chunk list stays aligned with the returned citations.
        Returns (answer, citations, coverage_report, pruned_chunks).
        """
        if not citations:
            return answer, citations, coverage_report, chunks or []
        used = sorted({
            n
            for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer or "")
            for n in (int(x) for x in re.findall(r"\d+", m.group(1)))
            if 1 <= n <= len(citations)
        })
        if not used or len(used) == len(citations):
            return answer, citations, coverage_report, chunks or []
        remap = {old: new for new, old in enumerate(used, start=1)}
        new_citations = [citations[old - 1] for old in used]
        new_chunks = [chunks[old - 1] for old in used] if chunks is not None else []

        def _repl(m):
            kept = [str(remap[n]) for n in (int(x) for x in re.findall(r"\d+", m.group(1))) if n in remap]
            return "[" + ", ".join(kept) + "]" if kept else ""

        new_answer = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _repl, answer or "")
        if coverage_report is not None and getattr(coverage_report, "sentences", None):
            for s in coverage_report.sentences:
                refs = getattr(s, "citation_refs", None)
                if refs:
                    s.citation_refs = [remap[r] for r in refs if r in remap]
                text = getattr(s, "text", None)
                if text:
                    s.text = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _repl, text)
        return new_answer, new_citations, coverage_report, new_chunks

    @staticmethod
    def refine_citation_blocks(
        citations: list,
        chunks: "list[RetrievedChunk]",
        coverage_report,
    ) -> list:
        """Override each citation's primary block with the block SLEC identified
        as actually supporting the sentences that cite [N].

        citations_from_chunks() picks the primary evidence block by query-token
        overlap — "best match with query" ≠ "best support for the answer sentence".
        SLEC's supporting_block_ids are scored against the *generated* sentences,
        so they are a stronger signal for which block to show in the citation.

        Only block_id and snippet_original are overridden; page, doc_id, and all
        provenance fields are unchanged to preserve evidence trace integrity.
        citations[i] ↔ chunks[i] — same ordering as the LLM prompt.
        """
        if not coverage_report or not getattr(coverage_report, "sentences", None):
            return citations

        citation_sup: dict[int, list[str]] = defaultdict(list)
        for sent in coverage_report.sentences:
            refs: list[int] = getattr(sent, "citation_refs", []) or []
            sup_ids: list[str] = getattr(sent, "supporting_block_ids", []) or []
            for ref in refs:
                zero_idx = ref - 1
                if 0 <= zero_idx < len(citations) and sup_ids:
                    citation_sup[zero_idx].extend(sup_ids)

        refined = list(citations)
        for cit_idx, sup_block_ids in citation_sup.items():
            if cit_idx >= len(chunks):
                continue
            chunk = chunks[cit_idx]
            ev_by_id = {(ev.block_id or ""): ev for ev in (chunk.evidence or []) if ev.block_id}
            current_bid = getattr(refined[cit_idx], "block_id", None) or ""
            for bid in sup_block_ids:
                if not bid or bid not in ev_by_id or bid == current_bid:
                    continue
                ev = ev_by_id[bid]
                snippet = (ev.snippet_original or "").strip()
                if len(snippet) < 30:
                    continue
                bbox = None
                if ev.bbox is not None:
                    from src.schemas.evidence import BoundingBoxSchema
                    bbox = BoundingBoxSchema.model_validate(ev.bbox.model_dump())
                refined[cit_idx] = refined[cit_idx].model_copy(
                    update={
                        "doc_id": ev.material_id,
                        "doc_name": ev.document_name,
                        "page": ev.page,
                        "block_id": bid,
                        "block_type": ev.block_type,
                        "snippet_original": snippet,
                        "bbox": bbox,
                        "source_language": ev.source_language,
                    }
                )
                break
        return refined

    # ── Main chain ─────────────────────────────────────────────────────────────

    async def finalize(
        self,
        *,
        answer: str,
        citations: list,
        confidence: float,
        evidence_bundle: "EvidenceBundle | None",
        context_chunks: list,
        route_decision=None,
        trace: "RequestTrace | None" = None,
        run_slec: bool = True,
        multimodal: bool = False,
        modality_str: str | None = None,
    ) -> QualityFinalizeResult:
        """SLEC → refine blocks → prune → aligner → quality gate.

        Args:
            run_slec: False to skip SLEC (e.g. chitchat, routes in slec_skip_routes).
            multimodal: True for visual/audio paths; SLEC uses caption text from
                        evidence_bundle instead of chunk.content.
            modality_str: preferred modality string (pre-computed by caller).
        """
        from src.schemas.query import SentenceCoverageReport

        should_refuse = False
        refusal_reason: str | None = None
        sentence_coverage_report = None
        refusal_answer = self._refusal_answer

        route_type_str = (
            route_decision.route_type.value if route_decision is not None else None
        )

        # ── 1. SLEC ───────────────────────────────────────────────────────────
        if run_slec and self.settings.slec_enabled and answer and answer != refusal_answer:
            try:
                slec_answer, sentence_coverage_report = await self.sentence_coverage_gate.verify(
                    answer=answer,
                    chunks=context_chunks if not multimodal else [],
                    evidence_bundle=evidence_bundle,
                    route_type=route_type_str,
                )
                if sentence_coverage_report and sentence_coverage_report.refused:
                    should_refuse = True
                    refusal_reason = "slec_coverage_below_floor"
                    answer = refusal_answer
                elif sentence_coverage_report and getattr(sentence_coverage_report, "dropped_count", 0) > 0:
                    answer = self.response_parser.inject_citations(answer, context_chunks)
                    answer = slec_answer
            except Exception as exc:
                logger.warning(
                    "QualityFinalizer: SLEC failed — keeping original answer",
                    extra={"error": str(exc)},
                )
                sentence_coverage_report = SentenceCoverageReport(enabled=False)
        else:
            sentence_coverage_report = SentenceCoverageReport(enabled=False)

        # ── 2. Refine citation blocks from SLEC supporting_block_ids ─────────
        if (
            not should_refuse
            and sentence_coverage_report is not None
            and context_chunks
            and not multimodal
        ):
            citations = self.refine_citation_blocks(citations, context_chunks, sentence_coverage_report)

        # ── 3. Prune to cited (renumbers [N] markers + SLEC refs in lockstep) ─
        answer, citations, sentence_coverage_report, pruned_chunks = self.prune_to_cited(
            answer, citations, sentence_coverage_report,
            chunks=context_chunks if not multimodal else None,
        )

        # ── 4. Citation aligner ───────────────────────────────────────────────
        alignment = None
        gate_result = None
        if not should_refuse and answer:
            try:
                if multimodal and evidence_bundle is not None:
                    alignment = self.citation_aligner.align(
                        answer=answer,
                        evidence_bundle=evidence_bundle,
                        preferred_modality=modality_str or "figure",
                    )
                else:
                    alignment = self.citation_aligner.align(
                        answer=answer,
                        chunks=pruned_chunks,
                        slec_report=sentence_coverage_report,
                        preferred_modality=modality_str,
                    )
                if alignment.invalid_citation_count > 0:
                    answer = alignment.corrected_answer

                # ── 5. Quality gate ───────────────────────────────────────────
                gate_result = self.quality_gate.evaluate(
                    slec_report=sentence_coverage_report,
                    alignment=alignment,
                    confidence=confidence,
                    evidence_bundle=evidence_bundle,
                )
                if trace is not None:
                    trace.set("quality_stage_verdicts", gate_result.verdicts_dict())
                    trace.set("citation_error_count", alignment.invalid_citation_count)
                    trace.set(
                        "claim_count",
                        sentence_coverage_report.total_sentences if sentence_coverage_report else 0,
                    )
                if gate_result.should_refuse:
                    should_refuse = True
                    refusal_reason = "quality_gate_multi_stage_fail"
                    answer = refusal_answer
            except Exception as exc:
                logger.warning(
                    "QualityFinalizer: aligner/gate failed — keeping answer",
                    extra={"error": str(exc)},
                )

        return QualityFinalizeResult(
            answer=answer,
            citations=citations,
            slec_report=sentence_coverage_report,
            pruned_chunks=pruned_chunks,
            should_refuse=should_refuse,
            refusal_reason=refusal_reason,
            gate_result=gate_result,
            alignment=alignment,
        )
