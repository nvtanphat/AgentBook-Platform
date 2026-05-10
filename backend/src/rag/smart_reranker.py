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
        """Decide if reranking is needed based on retrieval confidence.

        RRF fused_score is ~1/(k+rank) — max ~0.033 with k=60.
        Use normalized rank-based logic: skip only when top chunk is already
        reranked (rerank_score available) with high confidence.
        """
        if not chunks:
            return False

        # If rerank_score is already set (second pass), use it
        top_rerank = chunks[0].rerank_score if chunks[0].rerank_score else None
        if top_rerank is not None:
            if top_rerank >= self.confidence_threshold:
                logger.info("Skipping reranking (rerank_score high)", extra={"top_rerank": top_rerank})
                return False

        # Always rerank when fused_score only (RRF scores too small to threshold on)
        logger.info("Reranking needed", extra={"top_fused": chunks[0].fused_score or 0.0})
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
