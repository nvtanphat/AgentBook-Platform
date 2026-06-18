from __future__ import annotations

from src.core.config import Settings
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievedChunk


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.1, 0.9]


class FakeSingleCrossEncoder:
    def predict(self, pairs):
        return [0.1 for _ in pairs]


class FakeMultilingualCrossEncoder:
    def predict(self, pairs):
        scores = {
            ("vi query", "weak"): 0.2,
            ("en query", "weak"): 0.1,
            ("vi query", "strong"): 0.1,
            ("en query", "strong"): 0.95,
        }
        return [scores[pair] for pair in pairs]


class CountingCrossEncoder:
    def __init__(self) -> None:
        self.pair_count = 0

    def predict(self, pairs):
        self.pair_count = len(pairs)
        return [0.1 for _ in pairs]


def test_cross_encoder_reranker_orders_by_score() -> None:
    reranker = CrossEncoderReranker(Settings(testing=True))
    reranker._model = FakeCrossEncoder()
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="a.pdf",
            content="weak",
            language="en",
            modality="text",
        ),
        RetrievedChunk(
            chunk_id="b",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="b.pdf",
            content="strong",
            language="en",
            modality="text",
        ),
    ]

    results = reranker.rerank(query="dropout", chunks=chunks, limit=2)

    assert [chunk.chunk_id for chunk in results] == ["b", "a"]
    assert results[0].rerank_score == 0.9


def test_cross_encoder_reranker_preserves_visual_fusion_score() -> None:
    reranker = CrossEncoderReranker(Settings(testing=True))
    reranker._model = FakeSingleCrossEncoder()
    chunks = [
        RetrievedChunk(
            chunk_id="visual",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="a.pdf",
            content="caption text",
            language="en",
            modality="figure",
            fused_score=0.8,
        ),
        RetrievedChunk(
            chunk_id="text",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="b.pdf",
            content="weak",
            language="en",
            modality="text",
            fused_score=0.1,
        ),
    ]

    results = reranker.rerank(query="Figure 1", chunks=chunks, limit=2)

    assert [chunk.chunk_id for chunk in results] == ["visual", "text"]
    assert results[0].rerank_score == 0.8


def test_cross_encoder_reranker_multilingual_uses_best_query_score() -> None:
    reranker = CrossEncoderReranker(Settings(testing=True))
    reranker._model = FakeMultilingualCrossEncoder()
    chunks = [
        RetrievedChunk(
            chunk_id="a",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="a.pdf",
            content="weak",
            language="vi",
            modality="text",
        ),
        RetrievedChunk(
            chunk_id="b",
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="b.pdf",
            content="strong",
            language="en",
            modality="text",
        ),
    ]

    results = reranker.rerank_multilingual(queries=["vi query", "en query"], chunks=chunks, limit=2)

    assert [chunk.chunk_id for chunk in results] == ["b", "a"]
    assert results[0].rerank_score == 0.95


def test_cross_encoder_reranker_multilingual_caps_pair_count() -> None:
    model = CountingCrossEncoder()
    reranker = CrossEncoderReranker(Settings(testing=True, reranker_max_pairs=4))
    reranker._model = model
    chunks = [
        RetrievedChunk(
            chunk_id=str(index),
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="a.pdf",
            content=f"chunk {index}",
            language="en",
            modality="text",
        )
        for index in range(2)
    ]

    reranker.rerank_multilingual(queries=["q1", "q2", "q3"], chunks=chunks, limit=2)

    assert model.pair_count == 4
