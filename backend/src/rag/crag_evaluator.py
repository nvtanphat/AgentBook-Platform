from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_LLM_CRAG_PROMPT = """\
You are an evidence quality judge. Given a query and a retrieved evidence chunk, classify the chunk:
- CORRECT: the chunk directly answers or strongly supports the query
- AMBIGUOUS: the chunk is related but incomplete or only partially relevant
- INCORRECT: the chunk is irrelevant or contradicts the query

Query: {query}
Evidence (first 400 chars): {evidence}

Output ONLY one word: CORRECT, AMBIGUOUS, or INCORRECT\
"""


class CRAGEvaluator:
    """Score-based CRAG evaluator (Yan et al., ICML 2025).

    Classifies retrieved chunks as CORRECT / AMBIGUOUS / INCORRECT using
    reranker/fused scores — no extra LLM calls needed.

    Decision logic:
    - score >= correct_threshold  → CORRECT
    - score >= incorrect_threshold → AMBIGUOUS
    - score <  incorrect_threshold → INCORRECT

    If >50% of chunks are INCORRECT, restricts to CORRECT-only to reduce
    hallucination from noise evidence.
    """

    # Configured thresholds (correct_threshold / incorrect_threshold) are
    # tuned for the reranker's sigmoid output (0–1). RRF fused scores live
    # on a far smaller scale (typically 0.02–0.5), so filtering on fused
    # scores here pre-empts the reranker's job. When chunks lack a
    # rerank_score, this evaluator passes everything through unchanged and
    # lets the cross-encoder downstream do the actual selection.

    def __init__(
        self,
        correct_threshold: float = 0.55,
        incorrect_threshold: float = 0.25,
    ) -> None:
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

    def evaluate(
        self,
        *,
        chunks: list["RetrievedChunk"],
    ) -> list["RetrievedChunk"]:
        """Return filtered chunk list based on CRAG decision."""
        if not chunks:
            return chunks

        # Pre-rerank pass-through: RRF fused scores are too noisy to filter
        # on reliably (observed gold chunk at fused rank #26 with score
        # 0.077, then promoted to rerank rank #1 with score 0.58). Drop
        # nothing here — let the cross-encoder do the actual selection
        # downstream. Verdicts are still computed for the iterative
        # planner's "needs more evidence" signal.
        has_rerank = any(c.rerank_score is not None for c in chunks)
        if not has_rerank:
            return list(chunks)

        correct: list[RetrievedChunk] = []
        ambiguous: list[RetrievedChunk] = []
        incorrect: list[RetrievedChunk] = []
        correct_t, incorrect_t = self.correct_threshold, self.incorrect_threshold

        for chunk in chunks:
            score = chunk.rerank_score if chunk.rerank_score is not None else (chunk.fused_score or 0.0)
            if score >= correct_t:
                correct.append(chunk)
            elif score >= incorrect_t:
                ambiguous.append(chunk)
            else:
                incorrect.append(chunk)

        total = len(chunks)
        incorrect_ratio = len(incorrect) / total

        logger.info(
            "CRAG evaluation",
            extra={
                "correct": len(correct),
                "ambiguous": len(ambiguous),
                "incorrect": len(incorrect),
                "incorrect_ratio": round(incorrect_ratio, 3),
            },
        )

        if incorrect_ratio > 0.5:
            logger.warning(
                "CRAG: majority INCORRECT chunks — restricting to CORRECT only",
                extra={"incorrect_ratio": round(incorrect_ratio, 3)},
            )
            return correct

        return correct + ambiguous


class LLMCRAGEvaluator:
    """LLM-based CRAG evaluator (Yan et al., ICML 2025 — true correctness check).

    Uses the LLM to judge whether a chunk *actually answers* the query, not
    just whether its reranker score exceeds a threshold. Only fires for
    AMBIGUOUS chunks (those in the grey zone between correct/incorrect
    thresholds) to keep LLM calls minimal.

    Enabled when `crag.llm_enabled: true` in retrieval_config.yaml.
    Falls back silently to the original score-based label on any LLM error.
    """

    _LABEL_MAP = {"correct": "CORRECT", "ambiguous": "AMBIGUOUS", "incorrect": "INCORRECT"}

    def __init__(self, llm: "BaseLLM") -> None:
        self.llm = llm

    async def re_evaluate_ambiguous(
        self,
        *,
        query: str,
        chunks: "list[RetrievedChunk]",
        ambiguous_indices: list[int],
    ) -> "list[RetrievedChunk]":
        """Re-classify AMBIGUOUS chunks via LLM; INCORRECT are dropped."""
        if not ambiguous_indices or not query:
            return chunks

        results: list[RetrievedChunk] = list(chunks)
        for idx in ambiguous_indices:
            chunk = chunks[idx]
            prompt = _LLM_CRAG_PROMPT.format(
                query=query,
                evidence=chunk.content[:400],
            )
            try:
                raw = (await self.llm.generate(prompt=prompt)).strip().upper()
                if "INCORRECT" in raw:
                    results[idx] = None  # type: ignore[assignment]
                    logger.info(
                        "LLM CRAG: downgraded AMBIGUOUS → INCORRECT",
                        extra={"chunk_id": chunk.chunk_id, "raw": raw[:40]},
                    )
                elif "CORRECT" in raw:
                    logger.info(
                        "LLM CRAG: upgraded AMBIGUOUS → CORRECT",
                        extra={"chunk_id": chunk.chunk_id},
                    )
                # AMBIGUOUS or unrecognised → keep as-is
            except Exception as exc:
                logger.warning(
                    "LLM CRAG: evaluation failed for chunk — keeping AMBIGUOUS",
                    extra={"chunk_id": chunk.chunk_id, "error": str(exc)},
                )
        return [c for c in results if c is not None]
