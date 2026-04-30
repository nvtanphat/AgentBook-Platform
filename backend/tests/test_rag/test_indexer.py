from __future__ import annotations

import asyncio
import pytest
from types import SimpleNamespace

from beanie import PydanticObjectId
from qdrant_client import QdrantClient

from src.core.config import Settings
from src.processing.types import EvidenceBlock, TextChunk
from src.rag.embedder import EmbeddedText, SparseEmbedding
from src.rag.indexer import QdrantMongoIndexer


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        self.calls.append(list(texts))
        return [
            EmbeddedText(
                dense=[float(index + 1), float(index + 2), float(index + 3), float(index + 4)],
                sparse=SparseEmbedding(indices=[1, 7, 42], values=[0.1, 0.2, 0.3]),
            )
            for index, _ in enumerate(texts)
        ]


class InMemoryIndexer(QdrantMongoIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cleanup_calls: list[set[str]] = []

    async def _cleanup_existing_material_artifacts(self, material_ids: set[str]) -> None:
        self.cleanup_calls.append(set(material_ids))

    async def _store_chunks(self, chunks: list[TextChunk]):
        return [SimpleNamespace(id=PydanticObjectId()) for _ in chunks]

    async def _store_graph(self, *, entities, events, relations) -> None:
        return None


def make_chunk(index: int) -> TextChunk:
    owner_id = "user_demo"
    collection_id = "65f000000000000000000002"
    material_id = "65f000000000000000000001"
    block_id = f"blk-{index}"
    evidence = [
        EvidenceBlock(
            owner_id=owner_id,
            collection_id=collection_id,
            material_id=material_id,
            document_name="doc.pdf",
            page=index + 1,
            block_id=block_id,
            block_type="paragraph",
            snippet_original=f"content block {index}",
            source_language="vi",
        )
    ]
    return TextChunk(
        owner_id=owner_id,
        collection_id=collection_id,
        material_id=material_id,
        document_name="doc.pdf",
        content=f"chunk {index} has enough content for embedding",
        language="vi",
        modality="text",
        source_block_ids=[block_id],
        source_pages=[index + 1],
        token_count=7,
        chunk_strategy="layout_heading_parent_child",
        chunker_version="test",
        parser_version="test",
        embedding_model="fake",
        embedding_version="fake-v1",
        index_version="test-index",
        evidence=evidence,
    )


def test_indexer_batches_dense_sparse_vectors_and_payloads() -> None:
    asyncio.run(run_indexer_smoke())


async def run_indexer_smoke() -> None:
    settings = Settings(
        testing=True,
        qdrant_url=":memory:",
        qdrant_collection_name="idx_smoke",
        embedding_dense_size=4,
        index_batch_size=2,
    )
    qdrant_client = QdrantClient(location=":memory:")
    embedder = FakeEmbedder()
    indexer = InMemoryIndexer(settings=settings, qdrant_client=qdrant_client, embedder=embedder)

    stored = await indexer.index(
        chunks=[make_chunk(index) for index in range(5)],
        entities=[],
        events=[],
        relations=[],
    )

    points, _ = qdrant_client.scroll(
        collection_name="idx_smoke",
        limit=10,
        with_payload=True,
        with_vectors=True,
    )

    assert len(stored) == 5
    assert qdrant_client.count(collection_name="idx_smoke", exact=True).count == 5
    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    assert all(len(point.vector["dense"]) == 4 for point in points)
    assert all(point.vector["bge_m3_sparse"].indices == [1, 7, 42] for point in points)
    assert {tuple(point.payload["page_numbers"]) for point in points} == {(1,), (2,), (3,), (4,), (5,)}
    assert {tuple(point.payload["block_types"]) for point in points} == {("paragraph",)}
    assert all(point.payload["index_version"] == "test-index" for point in points)
    assert all(point.payload["content_text"].startswith("chunk ") for point in points)
    assert indexer.cleanup_calls == [{"65f000000000000000000001"}]


def test_indexer_aborts_when_material_deleted_before_batch() -> None:
    asyncio.run(run_indexer_abort_smoke())


async def run_indexer_abort_smoke() -> None:
    settings = Settings(
        testing=True,
        qdrant_url=":memory:",
        qdrant_collection_name="idx_abort",
        embedding_dense_size=4,
        index_batch_size=2,
    )
    qdrant_client = QdrantClient(location=":memory:")
    indexer = InMemoryIndexer(settings=settings, qdrant_client=qdrant_client, embedder=FakeEmbedder())

    async def should_continue() -> bool:
        return False

    with pytest.raises(LookupError):
        await indexer.index(
            chunks=[make_chunk(0)],
            entities=[],
            events=[],
            relations=[],
            should_continue=should_continue,
        )
