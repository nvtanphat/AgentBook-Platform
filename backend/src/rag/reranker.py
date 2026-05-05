from __future__ import annotations

import logging
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

from src.core.config import Settings
from src.processing.types import DependencyUnavailableError
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:
                raise DependencyUnavailableError(
                    f"sentence-transformers could not be imported: {exc}"
                ) from exc
            self._model = CrossEncoder(self.settings.reranker_model_name, device=self.settings.reranker_device)
        return self._model

    def rerank(self, *, query: str, chunks: list[RetrievedChunk], limit: int | None = None) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        if not self.settings.reranker_enabled:
            return self._fallback(candidates, limit)
        try:
            pairs = [(query, chunk.content) for chunk in candidates]
            scores = self.model.predict(pairs)
            scored: list[RetrievedChunk] = []
            for chunk, score in zip(candidates, scores, strict=True):
                scored.append(chunk.model_copy(update={"rerank_score": float(score)}))
            scored.sort(key=self._rank_key, reverse=True)
            return scored[: limit or self.settings.final_top_k]
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed, falling back to fused scores",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return self._fallback(candidates, limit)

    def rerank_multilingual(
        self,
        *,
        queries: list[str],
        chunks: list[RetrievedChunk],
        limit: int | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        candidates = chunks[: self.settings.rerank_input_k]
        if not candidates:
            return []
        unique_queries = list(dict.fromkeys(q for q in queries if q.strip()))
        if not unique_queries:
            unique_queries = [""]

        # Cap number of query variants to stay within reranker_max_pairs budget.
        max_queries = max(1, self.settings.reranker_max_pairs // max(1, len(candidates)))
        if len(unique_queries) > max_queries:
            logger.info(
                "Truncating reranker queries to stay under pair cap",
                extra={"query_count": len(unique_queries), "max_queries": max_queries, "candidate_count": len(candidates)},
            )
            unique_queries = unique_queries[:max_queries]

        if not self.settings.reranker_enabled:
            scored = self._fallback(candidates, limit=None)
        else:
            try:
                pairs = [(query, chunk.content) for chunk in candidates for query in unique_queries]
                scores = [float(s) for s in self.model.predict(pairs)]
                query_count = len(unique_queries)
                scored = []
                for index, chunk in enumerate(candidates):
                    start = index * query_count
                    best_score = max(scores[start: start + query_count])
                    scored.append(chunk.model_copy(update={"rerank_score": best_score}))
                scored.sort(key=self._rank_key, reverse=True)
            except Exception as exc:
                logger.warning(
                    "Multilingual CrossEncoder reranking failed, falling back to fused scores",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )
                scored = self._fallback(candidates, limit=None)

        final_limit = limit or self.settings.final_top_k
        if use_mmr and len(scored) > 1:
            return self.apply_mmr(scored, limit=final_limit)
        return scored[:final_limit]

    def apply_mmr(
        self,
        chunks: list[RetrievedChunk],
        *,
        limit: int | None = None,
        lambda_: float = 0.7,
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance: select top-k chunks that balance relevance with diversity.

        lambda_=1.0 → pure relevance ranking (identical to score sort)
        lambda_=0.0 → pure diversity (greedy maximum coverage)
        Default 0.7 favours relevance while still penalising near-duplicate chunks.
        """
        n = limit or len(chunks)
        if len(chunks) <= 1 or n <= 1:
            return chunks[:n]

        selected: list[RetrievedChunk] = [chunks[0]]
        remaining = list(chunks[1:])

        while remaining and len(selected) < n:
            best_mmr = float("-inf")
            best_chunk = remaining[0]
            for candidate in remaining:
                rel = (
                    candidate.rerank_score
                    if candidate.rerank_score is not None
                    else (candidate.fused_score or 0.0)
                )
                max_sim = max(self._jaccard(candidate.content, sel.content) for sel in selected)
                mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_chunk = candidate
            selected.append(best_chunk)
            remaining.remove(best_chunk)

        return selected

    def _fallback(self, chunks: list[RetrievedChunk], limit: int | None) -> list[RetrievedChunk]:
        fallback = sorted(chunks, key=lambda c: c.fused_score, reverse=True)
        return fallback[: limit or self.settings.final_top_k]

    @staticmethod
    def _rank_key(chunk: RetrievedChunk) -> float:
        rerank = chunk.rerank_score if chunk.rerank_score is not None else 0.0
        # fused_score is RRF-normalized (0–1); blend at 10% weight as tiebreaker only
        return rerank + 0.1 * (chunk.fused_score or 0.0)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-level Jaccard similarity for MMR redundancy measurement."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
