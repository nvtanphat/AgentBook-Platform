from __future__ import annotations

import logging
import math
import re

from src.core.config import Settings
from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_VI_STOPWORDS = {
    # Vietnamese function words
    "là", "gì", "có", "và", "hay", "hoặc", "của", "trong", "trên", "về",
    "cho", "với", "tại", "bởi", "từ", "đến", "khi", "nếu", "thì", "mà",
    "cũng", "đã", "sẽ", "đang", "được", "bị", "các", "một", "những",
    "này", "đó", "thế", "nào", "sao", "như", "vậy", "nêu", "hãy",
    "tóm", "tắt", "liệt", "kê", "trình", "bày", "giải", "thích",
    "ra", "đi", "lên", "về", "theo", "vì", "để", "mà", "bằng",
    "so", "sánh", "ở", "ta", "họ",
    # English function words (2+ chars)
    "the", "and", "for", "are", "was", "how", "what", "why", "when",
    "does", "can", "not", "this", "that", "with", "from", "into",
    "is", "in", "of", "to", "be", "it", "an", "at", "by", "do",
    "if", "or", "as", "on", "up",
}

_WORD_RE = re.compile(r"[\w]{2,}", re.UNICODE)  # include short terms like L1, L2, AI


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

    @staticmethod
    def _topic_coverage(query: str, chunks: list[RetrievedChunk], min_overlap: int = 1) -> bool:
        """Return True if at least one chunk covers ≥ min_overlap key query terms.

        Prevents hallucination when the reranker scores domain-adjacent but
        off-topic chunks highly (e.g. 'ML training steps' → scikit-learn resources).
        """
        tokens = {t for t in _WORD_RE.findall(query.lower()) if t not in _VI_STOPWORDS}
        if len(tokens) < 2:
            return True  # query too short to judge
        for chunk in chunks:
            text = chunk.content.lower()
            if sum(1 for t in tokens if t in text) >= min_overlap:
                return True
        return False

    def should_refuse(self, *, chunks: list[RetrievedChunk], confidence: float, query: str = "") -> tuple[bool, str | None]:
        if not chunks:
            return True, "no relevant evidence was found in the scoped materials"
        top_chunk = chunks[0]
        normalized_top = _sigmoid(top_chunk.rerank_score) if top_chunk.rerank_score is not None else confidence
        threshold = self.settings.min_evidence_confidence
        soft_threshold = threshold * 0.6

        # Tier 1: Confident reranker score — but still check topic coverage
        if normalized_top >= threshold:
            if query and not self._topic_coverage(query, chunks):
                logger.info("Refusing: high reranker score but low topic coverage for query: %.55s", query)
                return True, "retrieved evidence does not cover the query topic"
            return False, None

        # Tier 2: Partial confidence → answer with warning, don't refuse
        if normalized_top >= soft_threshold:
            logger.info("Partial confidence: %.4f (soft threshold %.4f)", normalized_top, soft_threshold)
            return False, "partial_confidence"

        # Tier 3: Too low → refuse
        logger.info("Refusing: score %.4f below soft threshold %.4f", normalized_top, soft_threshold)
        return True, "retrieved evidence confidence is below the configured threshold"
