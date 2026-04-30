from __future__ import annotations

import asyncio

from src.rag.query_processor import QueryProcessor


class FailingRewriter:
    async def rewrite(self, query: str):
        return None


def test_query_processor_detects_vietnamese_and_adds_english_retrieval_query() -> None:
    processed = QueryProcessor().process("Dropout giúp giảm overfitting như thế nào?")

    assert processed.query_language == "vi"
    assert processed.answer_language == "vi"
    assert processed.translated_query is not None
    assert len(processed.retrieval_queries) == 2
    assert processed.retrieval_queries[0] == "Dropout giúp giảm overfitting như thế nào?"
    assert "dropout" in processed.retrieval_queries[1]


def test_query_processor_respects_requested_answer_language() -> None:
    processed = QueryProcessor().process("How does dropout reduce overfitting?", answer_language="vi")

    assert processed.query_language == "en"
    assert processed.answer_language == "vi"


def test_query_processor_keeps_english_single_query() -> None:
    processed = QueryProcessor().process("How does dropout reduce overfitting?")

    assert processed.query_language == "en"
    assert processed.translated_query is None
    assert processed.retrieval_queries == ["How does dropout reduce overfitting?"]


def test_query_processor_async_rewriter_failure_uses_original_query_only() -> None:
    processed = asyncio.run(
        QueryProcessor().process_async(
            "Dropout giúp giảm overfitting như thế nào?",
            rewriter=FailingRewriter(),
        )
    )

    assert processed.query_language == "vi"
    assert processed.translated_query is None
    assert processed.retrieval_queries == ["Dropout giúp giảm overfitting như thế nào?"]
