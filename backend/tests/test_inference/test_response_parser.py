from __future__ import annotations

from src.inference.response_parser import ResponseParser
from src.rag.types import RetrievedChunk


def make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        owner_id="user_demo",
        collection_id="col",
        material_id="mat",
        document_name="doc.pdf",
        content=content,
        language="en",
        modality="text",
        fused_score=0.8,
    )


def test_inject_citations_does_not_cite_unmatched_hallucinated_sentence() -> None:
    answer = "The moon is made of cheese."

    result = ResponseParser().inject_citations(answer, [make_chunk("Dropout reduces overfitting.")])

    assert result == answer


def test_inject_citations_does_not_cite_refusal_sentence() -> None:
    answer = "Tôi không tìm thấy đủ bằng chứng trong tài liệu được cung cấp."

    result = ResponseParser().inject_citations(answer, [make_chunk("Dropout reduces overfitting.")])

    assert result == answer


def test_inject_citations_cites_sentence_with_overlap() -> None:
    answer = "Dropout reduces overfitting."

    result = ResponseParser().inject_citations(answer, [make_chunk("Dropout reduces overfitting.")])

    assert result == "Dropout reduces overfitting[1]."
