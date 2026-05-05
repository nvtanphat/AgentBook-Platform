from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@k: Proportion of relevant documents in top-k results.

    Args:
        retrieved: List of retrieved document IDs in ranked order
        relevant: List of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@k score in [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(relevant)) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@k: Proportion of top-k results that are relevant.

    Args:
        retrieved: List of retrieved document IDs in ranked order
        relevant: List of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        Precision@k score in [0, 1]
    """
    if not retrieved[:k]:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(relevant)) / k


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first relevant document.

    Args:
        retrieved: List of retrieved document IDs in ranked order
        relevant: List of ground-truth relevant document IDs

    Returns:
        MRR score in [0, 1]
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain@k.

    Assumes binary relevance (relevant or not). For graded relevance,
    pass relevance scores instead of binary list.

    Args:
        retrieved: List of retrieved document IDs in ranked order
        relevant: List of ground-truth relevant document IDs
        k: Cutoff rank

    Returns:
        nDCG@k score in [0, 1]
    """
    if not relevant:
        return 0.0

    # DCG: sum of (relevance / log2(rank + 1))
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc in enumerate(retrieved[:k])
        if doc in relevant
    )

    # IDCG: DCG of perfect ranking
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Average Precision: Mean of precision values at each relevant document.

    Args:
        retrieved: List of retrieved document IDs in ranked order
        relevant: List of ground-truth relevant document IDs

    Returns:
        AP score in [0, 1]
    """
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    precisions = []
    num_relevant_seen = 0

    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            num_relevant_seen += 1
            precisions.append(num_relevant_seen / i)

    return sum(precisions) / len(relevant) if precisions else 0.0


def mean_average_precision(results: List[tuple[List[str], List[str]]]) -> float:
    """
    Mean Average Precision across multiple queries.

    Args:
        results: List of (retrieved, relevant) tuples for each query

    Returns:
        MAP score in [0, 1]
    """
    if not results:
        return 0.0
    return sum(average_precision(ret, rel) for ret, rel in results) / len(results)


class RetrievalEvaluator:
    """
    Evaluate retrieval quality against ground truth.

    Usage:
        evaluator = RetrievalEvaluator()
        evaluator.add_query(retrieved_ids, relevant_ids)
        metrics = evaluator.compute_metrics(k=5)
    """

    def __init__(self):
        self.queries: List[tuple[List[str], List[str]]] = []

    def add_query(self, retrieved: List[str], relevant: List[str]) -> None:
        """Add a query result for evaluation."""
        self.queries.append((retrieved, relevant))

    def compute_metrics(self, k: int = 5) -> dict[str, float]:
        """
        Compute all metrics for accumulated queries.

        Args:
            k: Cutoff rank for @k metrics

        Returns:
            Dictionary of metric names to scores
        """
        if not self.queries:
            return {}

        metrics = {
            f"recall@{k}": np.mean([recall_at_k(ret, rel, k) for ret, rel in self.queries]),
            f"precision@{k}": np.mean([precision_at_k(ret, rel, k) for ret, rel in self.queries]),
            f"ndcg@{k}": np.mean([ndcg_at_k(ret, rel, k) for ret, rel in self.queries]),
            "mrr": np.mean([mean_reciprocal_rank(ret, rel) for ret, rel in self.queries]),
            "map": mean_average_precision(self.queries),
        }

        logger.info("Retrieval evaluation metrics", extra={"metrics": metrics, "num_queries": len(self.queries)})
        return metrics

    def reset(self) -> None:
        """Clear accumulated queries."""
        self.queries.clear()
