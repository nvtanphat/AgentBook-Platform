from __future__ import annotations

import logging
import math

from src.core.config import Settings
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid for any real logit value
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class ConfidenceScorer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def score(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0
        rerank_scores = [chunk.rerank_score for chunk in chunks if chunk.rerank_score is not None]
        if rerank_scores:
            # BGE-M3 reranker returns unbounded logits — sigmoid maps them to (0, 1)
            normalized = [_sigmoid(s) for s in rerank_scores]
            return round(sum(normalized) / len(normalized), 4)
        fused = [chunk.fused_score for chunk in chunks]
        max_score = max(fused) or 1.0
        return round(sum(score / max_score for score in fused) / len(fused), 4)

    def should_refuse(self, *, chunks: list[RetrievedChunk], confidence: float) -> tuple[bool, str | None]:
        if not chunks:
            return True, "no relevant evidence was found in the scoped materials"
        top_chunk = chunks[0]
        normalized_top = _sigmoid(top_chunk.rerank_score) if top_chunk.rerank_score is not None else confidence
        threshold = self.settings.min_evidence_confidence
        soft_threshold = threshold * 0.6

        # Tier 1: Confident → answer normally
        if normalized_top >= threshold:
            return False, None

        # Tier 2: Partial confidence → answer with warning, don't refuse
        if normalized_top >= soft_threshold:
            logger.info(
                "Partial confidence: %.4f (soft threshold %.4f)",
                normalized_top, soft_threshold,
            )
            return False, "partial_confidence"

        # Tier 3: Too low → refuse
        logger.info(
            "Refusing: score %.4f below soft threshold %.4f",
            normalized_top, soft_threshold,
        )
        return True, "retrieved evidence confidence is below the configured threshold"
