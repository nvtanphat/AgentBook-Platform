from __future__ import annotations

from qdrant_client import models

from src.core.config import Settings
from src.rag.embedder import EmbeddedText, SparseEmbedding
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievalScope, RetrievedChunk


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        return [EmbeddedText(dense=[0.1, 0.2], sparse=SparseEmbedding(indices=[1, 2], values=[0.7, 0.3]))]


class DenseOnlyEmbedder:
    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        return [EmbeddedText(dense=[0.1, 0.2], sparse=SparseEmbedding(indices=[], values=[]))]


class FakeQdrant:
    def __init__(self) -> None:
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "FakeQueryResponse",
            (),
            {
                "points": [
                    models.ScoredPoint(id="p1", version=1, score=0.95, payload={"chunk_id": "chunk-b"}),
                    models.ScoredPoint(id="p2", version=1, score=0.90, payload={"chunk_id": "chunk-a"}),
                ]
            },
        )()

    def scroll(self, **kwargs):
        self.calls.append({"scroll": kwargs})
        return [
            models.Record(id="p3", payload={"chunk_id": "chunk-text"}),
        ], None


def make_chunk(chunk_id: str, fused_score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        content=f"content {chunk_id}",
        language="en",
        modality="text",
        fused_score=fused_score,
    )


def test_hybrid_retriever_enforces_scope_filter_and_rrf(monkeypatch) -> None:
    qdrant = FakeQdrant()
    retriever = HybridRetriever(settings=Settings(testing=True), qdrant_client=qdrant, embedder=FakeEmbedder())

    async def fake_hydrate_points(points):
        return [make_chunk(str(point.payload["chunk_id"]), fused_score=float(point.score)) for point in points]

    monkeypatch.setattr(retriever, "_hydrate_points", fake_hydrate_points)

    import asyncio

    results = asyncio.run(
        retriever.retrieve(
            query="dropout",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert [chunk.chunk_id for chunk in results][:2] == ["chunk-b", "chunk-a"]
    assert len(qdrant.calls) == 1
    assert isinstance(qdrant.calls[0]["query"], models.FusionQuery)
    for prefetch in qdrant.calls[0]["prefetch"]:
        conditions = prefetch.filter.must
        assert any(condition.key == "owner_id" and condition.match.value == "user_demo" for condition in conditions)
        assert any(
            condition.key == "collection_id" and condition.match.value == "65f000000000000000000002"
            for condition in conditions
        )


def test_retrieval_scope_rejects_unscoped_queries() -> None:
    scope = RetrievalScope(owner_id="user_demo")
    try:
        scope.ensure_scoped()
    except ValueError as exc:
        assert "collection_id or material_ids" in str(exc)
    else:
        raise AssertionError("unscoped retrieval was not rejected")


def test_hybrid_retriever_uses_text_fallback_when_sparse_signal_is_empty(monkeypatch) -> None:
    qdrant = FakeQdrant()
    retriever = HybridRetriever(settings=Settings(testing=True), qdrant_client=qdrant, embedder=DenseOnlyEmbedder())

    async def fake_hydrate_points(points):
        return [make_chunk(str(point.payload["chunk_id"]), fused_score=float(point.score)) for point in points]

    monkeypatch.setattr(retriever, "_hydrate_points", fake_hydrate_points)

    import asyncio

    results = asyncio.run(
        retriever.retrieve(
            query="ngắn",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert "chunk-text" in [chunk.chunk_id for chunk in results]
    assert any("scroll" in call for call in qdrant.calls)
