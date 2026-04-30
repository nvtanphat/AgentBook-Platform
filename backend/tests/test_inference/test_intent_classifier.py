from __future__ import annotations

import pytest
from src.inference.intent_classifier import IntentClassifier, QueryIntent


@pytest.fixture()
def classifier() -> IntentClassifier:
    # No LLM — tests only cover Tier 1 and Tier 2.
    return IntentClassifier(llm=None)


# ---------------------------------------------------------------------------
# Chitchat (Tier 1 — regex patterns)
# ---------------------------------------------------------------------------
CHITCHAT_QUERIES = [
    "xin chào",
    "Chào bạn!",
    "hello",
    "cảm ơn bạn nhé",
    "thanks",
    "bạn là ai?",
    "bạn có khỏe không",
    "tạm biệt",
    "ok",
    "được rồi",
]

@pytest.mark.asyncio
@pytest.mark.parametrize("query", CHITCHAT_QUERIES)
async def test_chitchat_classified(classifier: IntentClassifier, query: str) -> None:
    assert await classifier.classify(query) == QueryIntent.CHITCHAT


# ---------------------------------------------------------------------------
# Off-topic (Tier 2 — heuristics: ≤2 tokens, no domain signal, no ?)
# ---------------------------------------------------------------------------
OFF_TOPIC_QUERIES = [
    "tào lao",
    "abcxyz",
    "blah blah",
    "asdf",
    "xyz123",
    "lol lol",
]

@pytest.mark.asyncio
@pytest.mark.parametrize("query", OFF_TOPIC_QUERIES)
async def test_off_topic_classified(classifier: IntentClassifier, query: str) -> None:
    assert await classifier.classify(query) == QueryIntent.OFF_TOPIC


# ---------------------------------------------------------------------------
# Knowledge — domain signals → Tier 2 short-circuits to KNOWLEDGE
# ---------------------------------------------------------------------------
KNOWLEDGE_QUERIES = [
    "Dropout là gì?",
    "Giải thích overfitting",
    "Tại sao gradient vanishing xảy ra?",
    "So sánh L1 và L2 regularization",
    "How does backpropagation work?",
    "Định nghĩa entropy trong information theory",
    "What is the difference between precision and recall?",
    "tóm tắt chương 3",
    "ví dụ về regularization",
]

@pytest.mark.asyncio
@pytest.mark.parametrize("query", KNOWLEDGE_QUERIES)
async def test_knowledge_classified(classifier: IntentClassifier, query: str) -> None:
    assert await classifier.classify(query) == QueryIntent.KNOWLEDGE


# ---------------------------------------------------------------------------
# Fallback: ambiguous without LLM → KNOWLEDGE (safe default)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ambiguous_falls_back_to_knowledge(classifier: IntentClassifier) -> None:
    # 3+ tokens, no domain signal, no question mark → goes to Tier 3 (LLM).
    # With no LLM wired, should fall back to KNOWLEDGE.
    result = await classifier.classify("transformer hoạt động thế nào đó")
    assert result == QueryIntent.KNOWLEDGE
