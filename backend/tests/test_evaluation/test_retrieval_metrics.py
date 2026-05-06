from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.core.config import get_settings
from src.evaluation.retrieval_metrics import RetrievalEvaluator, recall_at_k, mean_reciprocal_rank, ndcg_at_k
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings

logger = logging.getLogger(__name__)


@pytest.fixture
def eval_queries():
    """Load ground truth queries from YAML."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "eval_queries.yaml"
    with fixture_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["queries"]


@pytest.fixture
def retriever():
    """Create retriever instance."""
    settings = get_settings()
    qdrant_client = get_qdrant_client_for_settings(settings)
    return HybridRetriever(settings=settings, qdrant_client=qdrant_client)


class TestRetrievalMetrics:
    """Test retrieval metrics computation."""

    def test_recall_at_k(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc2", "doc4", "doc6"]

        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1/3)  # Found 1 of 3
        assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2/3)  # Found 2 of 3
        assert recall_at_k(retrieved, relevant, k=10) == pytest.approx(2/3)  # Still 2 of 3

    def test_mrr(self):
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]

        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1/2)  # Found at rank 2

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4"]

        assert mean_reciprocal_rank(retrieved, relevant) == 0.0  # Not found

    def test_ndcg_at_k(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = ["doc2", "doc4"]

        # Perfect ranking would be ["doc2", "doc4", ...]
        # Actual ranking has them at positions 2 and 4
        score = ndcg_at_k(retrieved, relevant, k=4)
        assert 0 < score < 1  # Not perfect, but not zero

    def test_evaluator(self):
        evaluator = RetrievalEvaluator()

        # Add multiple queries
        evaluator.add_query(
            retrieved=["doc1", "doc2", "doc3"],
            relevant=["doc2", "doc3"]
        )
        evaluator.add_query(
            retrieved=["doc4", "doc5", "doc6"],
            relevant=["doc5"]
        )

        metrics = evaluator.compute_metrics(k=3)

        assert "recall@3" in metrics
        assert "precision@3" in metrics
        assert "ndcg@3" in metrics
        assert "mrr" in metrics
        assert "map" in metrics

        # All metrics should be in [0, 1]
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} = {value} out of range"


@pytest.mark.asyncio
@pytest.mark.integration
class TestRetrievalEvaluation:
    """Integration test: evaluate retrieval on ground truth queries."""

    async def test_evaluate_retrieval(self, retriever, eval_queries):
        """
        Evaluate retrieval quality on ground truth queries.

        This test requires:
        1. Sample documents indexed in Qdrant
        2. Ground truth queries with known relevant chunks

        Skip if no test data available.
        """
        if not eval_queries:
            pytest.skip("No evaluation queries available")

        evaluator = RetrievalEvaluator()
        scope = RetrievalScope(owner_id="test_user", collection_id="test_collection")
        saw_retrieved_chunks = False

        for query_data in eval_queries[:5]:  # Test first 5 queries
            query = query_data["query"]
            relevant_chunks = query_data["relevant_chunks"]

            # Retrieve
            retrieved = await retriever.retrieve(query=query, scope=scope, limit=10)
            retrieved_ids = [chunk.chunk_id for chunk in retrieved]
            saw_retrieved_chunks = saw_retrieved_chunks or bool(retrieved_ids)

            # Evaluate
            evaluator.add_query(retrieved_ids, relevant_chunks)

            logger.info(
                "Query evaluated",
                extra={
                    "query": query,
                    "retrieved_count": len(retrieved_ids),
                    "relevant_count": len(relevant_chunks),
                }
            )

        if not saw_retrieved_chunks:
            pytest.skip("No indexed evaluation corpus available in Qdrant for test_user/test_collection")

        # Compute metrics
        metrics = evaluator.compute_metrics(k=5)

        # Log results
        logger.info("Retrieval evaluation complete", extra={"metrics": metrics})

        # Assert minimum quality thresholds
        assert metrics["recall@5"] >= 0.3, "Recall@5 too low"
        assert metrics["mrr"] >= 0.2, "MRR too low"

        return metrics
