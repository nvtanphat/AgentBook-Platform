from __future__ import annotations

import asyncio

from src.core.config import Settings
from src.inference.response_parser import ResponseParser
from src.processing.types import EvidenceBlock
from src.rag.types import RetrievalScope, RetrievedChunk
from src.schemas.query import CompareRequest
from src.services.query_service import QueryService


def _chunk(material_id: str, dimension: str) -> RetrievedChunk:
    content = f"{dimension} evidence for dropout in {material_id}."
    return RetrievedChunk(
        chunk_id=f"{material_id}-{dimension}",
        owner_id="user_demo",
        collection_id="collection_demo",
        material_id=material_id,
        document_name=f"{material_id}.pdf",
        content=content,
        language="en",
        modality="text",
        source_block_ids=[f"block-{material_id}-{dimension}"],
        source_pages=[1],
        evidence=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="collection_demo",
                material_id=material_id,
                document_name=f"{material_id}.pdf",
                page=1,
                block_id=f"block-{material_id}-{dimension}",
                block_type="paragraph",
                snippet_original=content,
                source_language="en",
                confidence=0.9,
            )
        ],
        fused_score=0.8,
    )


class FakeRetriever:
    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None):
        material_id = scope.material_ids[0]
        if material_id == "source_b" and "hạn chế" in query:
            return []
        dimension = query.rsplit(": ", 1)[-1]
        return [_chunk(material_id, dimension)]


class OSErrorRetriever:
    async def retrieve(self, *, query: str, scope: RetrievalScope, limit: int | None = None):
        raise OSError(22, "Invalid argument")


class FakeConfidenceScorer:
    def score(self, chunks) -> float:
        return 0.85 if chunks else 0.0


class FakeLLM:
    async def generate(self, *, prompt: str) -> str:
        return "Bằng chứng cho thấy nội dung này có liên quan trực tiếp."


class FakeInferenceEngine:
    def __init__(self) -> None:
        self.llm = FakeLLM()


def test_compare_v2_returns_matrix_cell_citations_and_missing_evidence() -> None:
    service = QueryService(
        settings=Settings(),
        retriever=FakeRetriever(),  # type: ignore[arg-type]
        response_parser=ResponseParser(),
        confidence_scorer=FakeConfidenceScorer(),  # type: ignore[arg-type]
        inference_engine=FakeInferenceEngine(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        service.compare(
            CompareRequest(
                owner_id="user_demo",
                collection_id="collection_demo",
                material_ids=["source_a", "source_b"],
                topic="So sánh dropout",
                dimensions=["ý chính", "hạn chế"],
            )
        )
    )

    assert len(response.sources) == 2
    assert response.matrix["source_a"]["ý chính"].missing_evidence is False
    assert response.matrix["source_a"]["ý chính"].citation_ids
    assert response.matrix["source_b"]["hạn chế"].missing_evidence is True
    assert response.dimension_coverage[1].dimension == "hạn chế"
    assert response.dimension_coverage[1].covered_count == 1
    assert response.coverage is not None
    assert response.coverage.covered_count == 2


def test_compare_v2_treats_cell_retrieval_oserror_as_missing_evidence() -> None:
    service = QueryService(
        settings=Settings(),
        retriever=OSErrorRetriever(),  # type: ignore[arg-type]
        response_parser=ResponseParser(),
        confidence_scorer=FakeConfidenceScorer(),  # type: ignore[arg-type]
        inference_engine=FakeInferenceEngine(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        service.compare(
            CompareRequest(
                owner_id="user_demo",
                collection_id="collection_demo",
                material_ids=["source_a", "source_b"],
                topic="So sánh dropout",
                dimensions=["ý chính"],
            )
        )
    )

    assert response.matrix["source_a"]["ý chính"].missing_evidence is True
    assert response.matrix["source_b"]["ý chính"].missing_evidence is True
    assert response.dimension_coverage[0].covered_count == 0
    assert response.coverage is not None
    assert response.coverage.covered_count == 0
