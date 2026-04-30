from __future__ import annotations

import asyncio
import unicodedata

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.inference.inference_engine import InferenceEngine
from src.processing.types import BBox, EvidenceBlock
from src.rag.types import RetrievalScope, RetrievedChunk


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None):
        self.calls.append({"query": query, "limit": limit})
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                owner_id=scope.owner_id,
                collection_id=scope.collection_id or "c",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                content="Dropout randomly disables activations to reduce co-adaptation.",
                language="en",
                modality="text",
                source_block_ids=["blk-1"],
                source_pages=[4],
                evidence=[
                    EvidenceBlock(
                        owner_id=scope.owner_id,
                        collection_id=scope.collection_id or "c",
                        material_id="65f000000000000000000001",
                        document_name="lecture.pdf",
                        page=4,
                        block_id="blk-1",
                        block_type="paragraph",
                        snippet_original="Dropout randomly disables activations to reduce co-adaptation.",
                        source_language="en",
                        bbox=BBox(x1=1, y1=2, x2=3, y2=4),
                        confidence=0.95,
                    )
                ],
                fused_score=0.8,
            )
        ]


class EmptyRetriever:
    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None):
        return []


class FakeGraphRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve_paths(self, *, query: str, scope: RetrievalScope, max_hops: int | None = None):
        self.calls.append({"query": query, "max_hops": max_hops})
        return []


class FakeReranker:
    def rerank(self, *, query: str, chunks, limit: int | None = None):
        return [chunk.model_copy(update={"rerank_score": 0.9}) for chunk in chunks][: limit or 5]


class FakeLLM(BaseLLM):
    async def generate(self, *, prompt: str) -> str:
        assert "Dropout randomly disables activations" in prompt
        return "Dropout giảm overfitting bằng cách vô hiệu hóa ngẫu nhiên activation [Nguồn: lecture.pdf, trang 4, block blk-1]."


def test_inference_engine_returns_grounded_answer_with_citation() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=FakeRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Dropout giúp giảm overfitting như thế nào?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert response.answer_language == "vi"
    assert response.citations[0].doc_name == "lecture.pdf"
    assert response.citations[0].block_id == "blk-1"
    assert response.citations[0].bbox.x1 == 1


def test_inference_engine_respects_requested_answer_language() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=FakeRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="How does dropout reduce overfitting?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
            answer_language="vi",
        )
    )

    assert response.answer_language == "vi"


def test_inference_engine_refuses_without_evidence() -> None:
    engine = InferenceEngine(
        settings=Settings(testing=True),
        retriever=EmptyRetriever(),
        graph_retriever=FakeGraphRetriever(),
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    # Use a knowledge query (domain signal present) so intent classifier routes to RAG,
    # but EmptyRetriever returns nothing → engine should refuse for lack of evidence.
    response = asyncio.run(
        engine.answer(query="Giải thích khái niệm này", scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"))
    )

    assert response.was_refused is True
    assert response.citations == []
    assert "du bang chung" in strip_accents(response.answer.lower())


def test_inference_engine_routes_factual_without_graph_and_scaled_limit() -> None:
    retriever = FakeRetriever()
    graph = FakeGraphRetriever()
    engine = InferenceEngine(
        settings=Settings(testing=True, rerank_input_k=16, final_top_k=5),
        retriever=retriever,
        graph_retriever=graph,
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Dropout là gì?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert retriever.calls[0]["limit"] == 12
    assert graph.calls == []


def test_inference_engine_routes_graph_relation_with_graph() -> None:
    retriever = FakeRetriever()
    graph = FakeGraphRetriever()
    engine = InferenceEngine(
        settings=Settings(testing=True, rerank_input_k=16, final_top_k=5),
        retriever=retriever,
        graph_retriever=graph,
        reranker=FakeReranker(),
        llm=FakeLLM(),
    )

    response = asyncio.run(
        engine.answer(
            query="Quan hệ giữa dropout và overfitting là gì?",
            scope=RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002"),
        )
    )

    assert response.was_refused is False
    assert retriever.calls[0]["limit"] == 16
    assert len(graph.calls) == 1
