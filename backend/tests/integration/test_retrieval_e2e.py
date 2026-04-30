from __future__ import annotations

import os
import time

import pytest
from beanie import PydanticObjectId
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer
from testcontainers.mongodb import MongoDbContainer

from src.core.config import Settings
from src.database import close_database, init_database
from src.models.collection import KnowledgeCollection
from src.models.common import PipelineStatus
from src.models.material import BoundingBox, Material, MaterialBlock, MaterialPage, replace_material_pages
from src.processing.types import EvidenceBlock, TextChunk
from src.rag.embedder import EmbeddedText, SparseEmbedding
from src.rag.indexer import QdrantMongoIndexer
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievalScope

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTBOOK_RUN_INTEGRATION") != "true",
    reason="Set AGENTBOOK_RUN_INTEGRATION=true to run Docker-backed MongoDB/Qdrant integration tests.",
)


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        return [
            EmbeddedText(
                dense=[0.1, 0.2, 0.3, 0.4],
                sparse=SparseEmbedding(indices=[7, 42], values=[0.3, 0.7]),
            )
            for _ in texts
        ]


@pytest.mark.asyncio
async def test_index_and_retrieve_against_real_mongodb_and_qdrant() -> None:
    with MongoDbContainer("mongo:7.0") as mongo, DockerContainer("qdrant/qdrant:v1.12.1").with_exposed_ports(6333) as qdrant:
        qdrant_url = f"http://{qdrant.get_container_host_ip()}:{qdrant.get_exposed_port(6333)}"
        settings = Settings(
            testing=False,
            mongodb_uri=mongo.get_connection_url(),
            mongodb_database="prism_integration",
            qdrant_url=qdrant_url,
            qdrant_collection_name="integration_chunks",
            embedding_dense_size=4,
            reranker_enabled=False,
            query_rewriter_enabled=False,
        )
        await init_database(settings)
        try:
            collection = KnowledgeCollection(name="Integration", owner_id="user_demo")
            await collection.insert()
            material = Material(
                owner_id="user_demo",
                collection_id=collection.id,
                filename="integration.pdf",
                original_name="integration.pdf",
                file_type="pdf",
                checksum_sha256="a" * 64,
                file_size_bytes=128,
                storage_path="raw/user_demo/integration.pdf",
                status=PipelineStatus.PARSED.value,
                parse_version=settings.parse_version,
                chunk_version=settings.chunk_version,
                embedding_version=settings.embedding_version,
                index_version=settings.index_version,
            )
            await material.insert()
            await replace_material_pages(
                material,
                [
                    MaterialPage(
                        page_number=1,
                        blocks=[
                            MaterialBlock(
                                block_id="blk-1",
                                block_index=0,
                                block_type="paragraph",
                                content="Dropout reduces overfitting in neural networks.",
                                language="en",
                                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                                ocr_confidence=0.99,
                                reading_order=0,
                            )
                        ],
                    )
                ],
            )
            evidence = EvidenceBlock(
                owner_id="user_demo",
                collection_id=str(collection.id),
                material_id=str(material.id),
                document_name=material.original_name,
                page=1,
                block_id="blk-1",
                block_type="paragraph",
                snippet_original="Dropout reduces overfitting in neural networks.",
                source_language="en",
                confidence=0.99,
            )
            chunk = TextChunk(
                owner_id="user_demo",
                collection_id=str(collection.id),
                material_id=str(material.id),
                document_name=material.original_name,
                content="Dropout reduces overfitting in neural networks.",
                language="en",
                modality="text",
                source_block_ids=["blk-1"],
                source_pages=[1],
                token_count=8,
                chunk_strategy="integration",
                chunker_version="test",
                parser_version="test",
                embedding_model="fake",
                embedding_version="fake-v1",
                index_version="test-index",
                evidence=[evidence],
            )
            qdrant_client = QdrantClient(url=qdrant_url)
            indexer = QdrantMongoIndexer(settings=settings, qdrant_client=qdrant_client, embedder=FakeEmbedder())
            await indexer.index(chunks=[chunk], entities=[], events=[], relations=[])

            # Qdrant may need a short moment to expose freshly upserted points through query_points.
            time.sleep(0.2)

            retriever = HybridRetriever(settings=settings, qdrant_client=qdrant_client, embedder=FakeEmbedder())
            results = await retriever.retrieve(
                query="dropout overfitting",
                scope=RetrievalScope(owner_id="user_demo", collection_id=str(collection.id)),
                limit=3,
            )

            assert results
            assert results[0].document_name == "integration.pdf"
            assert results[0].evidence[0].block_id == "blk-1"
        finally:
            await close_database()


@pytest.mark.asyncio
async def test_multi_chunk_retrieval_ranks_relevant_first_and_citation_accuracy() -> None:
    """Index multiple chunks (one highly relevant, several irrelevant) and verify
    the retrieval pipeline returns the relevant chunk first with correct evidence."""
    with MongoDbContainer("mongo:7.0") as mongo, DockerContainer("qdrant/qdrant:v1.12.1").with_exposed_ports(6333) as qdrant:
        qdrant_url = f"http://{qdrant.get_container_host_ip()}:{qdrant.get_exposed_port(6333)}"
        settings = Settings(
            testing=False,
            mongodb_uri=mongo.get_connection_url(),
            mongodb_database="prism_multichunk",
            qdrant_url=qdrant_url,
            qdrant_collection_name="multichunk_test",
            embedding_dense_size=4,
            reranker_enabled=False,
            query_rewriter_enabled=False,
            final_top_k=5,
        )
        await init_database(settings)
        try:
            collection = KnowledgeCollection(name="MultiChunk", owner_id="user_mc")
            await collection.insert()
            material = Material(
                owner_id="user_mc",
                collection_id=collection.id,
                filename="multichunk.pdf",
                original_name="multichunk.pdf",
                file_type="pdf",
                checksum_sha256="b" * 64,
                file_size_bytes=256,
                storage_path="raw/user_mc/multichunk.pdf",
                status=PipelineStatus.PARSED.value,
                parse_version=settings.parse_version,
                chunk_version=settings.chunk_version,
                embedding_version=settings.embedding_version,
                index_version=settings.index_version,
            )
            await material.insert()

            def _make_chunk(block_id: str, content: str, page: int) -> TextChunk:
                ev = EvidenceBlock(
                    owner_id="user_mc",
                    collection_id=str(collection.id),
                    material_id=str(material.id),
                    document_name="multichunk.pdf",
                    page=page,
                    block_id=block_id,
                    block_type="paragraph",
                    snippet_original=content,
                    source_language="en",
                    confidence=0.9,
                )
                return TextChunk(
                    owner_id="user_mc",
                    collection_id=str(collection.id),
                    material_id=str(material.id),
                    document_name="multichunk.pdf",
                    content=content,
                    language="en",
                    modality="text",
                    source_block_ids=[block_id],
                    source_pages=[page],
                    token_count=len(content.split()),
                    chunk_strategy="test",
                    chunker_version="test",
                    parser_version="test",
                    embedding_model="fake",
                    embedding_version="fake-v1",
                    index_version="test-index",
                    evidence=[ev],
                )

            target_chunk = _make_chunk("blk-target", "Backpropagation computes gradients via the chain rule.", 1)
            irrelevant_chunks = [
                _make_chunk("blk-irr-1", "The history of ancient Rome spans many centuries.", 2),
                _make_chunk("blk-irr-2", "Photosynthesis converts sunlight into chemical energy.", 3),
                _make_chunk("blk-irr-3", "The Amazon river flows through South America.", 4),
            ]
            all_chunks = [target_chunk] + irrelevant_chunks

            qdrant_client = QdrantClient(url=qdrant_url)

            class DifferentiatingEmbedder:
                """Assigns distinct dense vectors so the target chunk scores differently."""
                _embeddings = {
                    "Backpropagation computes gradients via the chain rule.": [0.9, 0.8, 0.7, 0.6],
                    "backpropagation gradients chain rule": [0.9, 0.8, 0.7, 0.6],
                }
                _default = [0.1, 0.1, 0.1, 0.1]

                def encode(self, texts: list[str]) -> list[EmbeddedText]:
                    return [
                        EmbeddedText(
                            dense=self._embeddings.get(text, self._default),
                            sparse=SparseEmbedding(
                                indices=[1, 2],
                                values=[0.5 if "backpropagation" in text.lower() or "gradients" in text.lower() else 0.1, 0.3],
                            ),
                        )
                        for text in texts
                    ]

            embedder = DifferentiatingEmbedder()
            indexer = QdrantMongoIndexer(settings=settings, qdrant_client=qdrant_client, embedder=embedder)
            await indexer.index(chunks=all_chunks, entities=[], events=[], relations=[])
            time.sleep(0.2)

            retriever = HybridRetriever(settings=settings, qdrant_client=qdrant_client, embedder=embedder)
            results = await retriever.retrieve(
                query="backpropagation gradients chain rule",
                scope=RetrievalScope(owner_id="user_mc", collection_id=str(collection.id)),
                limit=5,
            )

            assert results, "No results returned"
            # Evidence trace: every result must carry at least one evidence block
            for result in results:
                assert result.evidence, f"Missing evidence on chunk {result.chunk_id}"
                assert result.evidence[0].owner_id == "user_mc"
                assert result.evidence[0].collection_id == str(collection.id)
                assert result.evidence[0].material_id == str(material.id)
                assert result.evidence[0].block_id, "block_id must not be empty"

            # Citation accuracy: the target chunk must appear in results
            retrieved_block_ids = {ev.block_id for r in results for ev in r.evidence}
            assert "blk-target" in retrieved_block_ids, (
                f"Target chunk not retrieved. Got block_ids: {retrieved_block_ids}"
            )

            # Scope isolation: no chunks from other owners leak through
            for result in results:
                assert result.owner_id == "user_mc", f"Scope leak: got owner {result.owner_id}"
                assert result.collection_id == str(collection.id)
        finally:
            await close_database()
