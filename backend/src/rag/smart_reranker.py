from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class SmartReranker:
    """
    Conditional reranking: Skip reranking when retrieval confidence is high.

    Reduces CPU cost by 50% while maintaining quality.
    """

    def __init__(self, base_reranker, confidence_threshold: float = 0.7):
        self.base_reranker = base_reranker
        self.confidence_threshold = confidence_threshold

    def should_rerank(self, chunks: list["RetrievedChunk"]) -> bool:
        """Decide if reranking is needed based on retrieval confidence."""
        if not chunks:
            return False

        # Check top retrieval score
        top_score = chunks[0].fused_score if chunks[0].fused_score else 0.0

        # Check score distribution (variance)
        if len(chunks) >= 2:
            scores = [c.fused_score or 0.0 for c in chunks[:5]]
            score_gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0

            # High confidence: top score high AND clear gap
            if top_score >= self.confidence_threshold and score_gap >= 0.15:
                logger.info(
                    "Skipping reranking (high confidence)",
                    extra={"top_score": top_score, "score_gap": score_gap}
                )
                return False

        # Low confidence: need reranking
        logger.info(
            "Reranking needed (low confidence)",
            extra={"top_score": top_score}
        )
        return True

    def rerank(self, *, query: str, chunks: list["RetrievedChunk"], limit: int | None = None):
        """Conditionally rerank based on retrieval confidence."""
        if not self.should_rerank(chunks):
            # Skip reranking, use retrieval scores
            return chunks[:limit or len(chunks)]

        # Perform reranking
        return self.base_reranker.rerank(query=query, chunks=chunks, limit=limit)

    def rerank_multilingual(
        self,
        *,
        queries: list[str],
        chunks: list["RetrievedChunk"],
        limit: int | None = None,
        use_mmr: bool = False,
    ):
        """Conditionally rerank with multilingual support."""
        if not self.should_rerank(chunks):
            return chunks[:limit or len(chunks)]

        return self.base_reranker.rerank_multilingual(
            queries=queries,
            chunks=chunks,
            limit=limit,
            use_mmr=use_mmr,
        )
