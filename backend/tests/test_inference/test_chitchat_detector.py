from __future__ import annotations

import pytest
from src.inference.chitchat_detector import is_chitchat

CHITCHAT_CASES = [
    "xin chào",
    "Chào bạn!",
    "hello",
    "Hi there",
    "good morning",
    "tạm biệt",
    "bye",
    "cảm ơn",
    "Cám ơn bạn nhé",
    "thanks",
    "thank you",
    "bạn là ai?",
    "bạn tên gì",
    "bạn có khỏe không",
    "how are you?",
    "bạn là gì",
    "ok",
    "okay",
    "được rồi",
    "kể chuyện cười đi",
]

KNOWLEDGE_CASES = [
    "Dropout là gì?",
    "Giải thích overfitting",
    "Tại sao gradient vanishing xảy ra?",
    "How does backpropagation work?",
    "So sánh L1 và L2 regularization",
    "Định nghĩa của entropy trong information theory là gì?",
    "What is the difference between precision and recall?",
    "Transformer architecture hoạt động như thế nào?",
]


@pytest.mark.parametrize("query", CHITCHAT_CASES)
def test_chitchat_detected(query: str) -> None:
    assert is_chitchat(query), f"Expected chitchat but not detected: {query!r}"


@pytest.mark.parametrize("query", KNOWLEDGE_CASES)
def test_knowledge_not_detected_as_chitchat(query: str) -> None:
    assert not is_chitchat(query), f"Knowledge query wrongly detected as chitchat: {query!r}"
