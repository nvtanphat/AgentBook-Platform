from __future__ import annotations

import re
from enum import Enum

from src.core.base_llm import BaseLLM
from src.inference.chitchat_detector import is_chitchat


class QueryIntent(str, Enum):
    KNOWLEDGE = "knowledge"   # needs full RAG pipeline
    CHITCHAT = "chitchat"     # conversational, fast LLM response
    OFF_TOPIC = "off_topic"   # irrelevant/nonsensical, quick refusal


# Keywords that signal an educational or document-related knowledge query.
_DOMAIN_SIGNALS = re.compile(
    r"\b("
    r"là gì|nghĩa là|định nghĩa|giải thích|tại sao|vì sao|như thế nào|ảnh hưởng|"
    r"so sánh|phân biệt|khác nhau|liên quan|ví dụ|trình bày|phân tích|tóm tắt|"
    r"tổng quan|tổng hợp|tóm lược|khái quát|overview|"
    r"nêu|mô tả|chứng minh|tính|tính chất|công thức|thuật toán|phương pháp|"
    r"what|why|how|when|where|which|define|explain|compare|describe|analyze|"
    r"difference|relationship|example|summarize|outline|prove|calculate|"
    r"tài liệu|chương|phần|trang|bài|slide|sách|giáo trình|đề cương|bài giảng|"
    r"tom tat|tong quan|tong hop|khai quat|trinh bay|phan tich|giai thich|"
    r"so sanh|phan biet|lien quan|dinh nghia|huong dan|mo ta|tai lieu|bai giang"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_MARK = re.compile(r"[?？]")

# Prompt kept deliberately short to minimise token cost.
_CLASSIFY_PROMPT = """\
Bạn là classifier cho hệ thống hỏi đáp tài liệu học tập AgentBook.

Phân loại câu hỏi vào đúng một nhóm:
- knowledge  : hỏi về nội dung tài liệu, khái niệm học thuật, tính năng hệ thống (upload, tóm tắt, so sánh, mindmap, trích dẫn), hoặc bất kỳ điều cần tra cứu/phân tích
- chitchat   : chào hỏi, cảm ơn, trò chuyện xã giao thuần túy
- off_topic  : văn bản vô nghĩa, spam, hoặc rõ ràng không liên quan đến học tập hay hệ thống

Ví dụ:
Câu: "Neural network là gì?" → knowledge
Câu: "Tóm tắt chương 3 cho tôi" → knowledge
Câu: "So sánh hai tài liệu này" → knowledge
Câu: "Xin chào bạn" → chitchat
Câu: "Cảm ơn nhé!" → chitchat
Câu: "asdfghjkl 123" → off_topic
Câu: "Mua bán bất động sản ở đâu?" → off_topic

Khi nghi ngờ, ưu tiên chọn knowledge thay vì off_topic.

Câu hỏi: {query}

Chỉ trả lời đúng một từ: knowledge, chitchat, hoặc off_topic"""


class IntentClassifier:
    """
    Three-tier intent classifier:
      Tier 1 — regex patterns   (instant, catches clear chitchat)
      Tier 2 — heuristics       (instant, catches obvious off-topic)
      Tier 3 — LLM              (async, only for ambiguous queries)
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm

    async def classify(self, query: str) -> QueryIntent:
        # Tier 1: known chitchat patterns (0 ms)
        if is_chitchat(query):
            return QueryIntent.CHITCHAT

        # Tier 2: heuristic gates (0 ms)
        heuristic = self._heuristic(query)
        if heuristic is not None:
            return heuristic

        # Tier 3: LLM for ambiguous queries
        if self._llm is not None:
            return await self._llm_classify(query)

        # Safe fallback: assume knowledge so we never silently block real queries.
        return QueryIntent.KNOWLEDGE

    @staticmethod
    def _heuristic(query: str) -> QueryIntent | None:
        text = query.strip()
        tokens = text.split()
        n = len(tokens)
        has_question = _QUESTION_MARK.search(text) is not None
        has_domain = _DOMAIN_SIGNALS.search(text) is not None

        # Clear domain signal → treat as knowledge immediately.
        if has_domain:
            return QueryIntent.KNOWLEDGE

        # Very short + no signal + no question mark → almost certainly off-topic.
        # Threshold kept conservative (≤2) to avoid false positives on short
        # but valid queries like "Dropout là gì" → caught by domain signal above.
        if has_question and n >= 3:
            return QueryIntent.KNOWLEDGE

        if n <= 2 and not has_question:
            return QueryIntent.OFF_TOPIC

        # Longer queries without domain signal → let LLM decide.
        return None

    async def _llm_classify(self, query: str) -> QueryIntent:
        prompt = _CLASSIFY_PROMPT.format(query=query[:300])  # cap length
        try:
            raw = await self._llm.generate(prompt=prompt)
            label = raw.strip().lower().split()[0].rstrip(".,;:")
            return QueryIntent(label)
        except (ValueError, IndexError, Exception):
            # Unknown label or LLM failure → fall through to RAG rather than blocking.
            return QueryIntent.KNOWLEDGE
