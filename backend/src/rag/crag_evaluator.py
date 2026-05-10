from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


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

        correct: list[RetrievedChunk] = []
        ambiguous: list[RetrievedChunk] = []
        incorrect: list[RetrievedChunk] = []

        for chunk in chunks:
            score = chunk.reranker_score if getattr(chunk, "reranker_score", None) is not None else (chunk.fused_score or 0.0)
            if score >= self.correct_threshold:
                correct.append(chunk)
            elif score >= self.incorrect_threshold:
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
