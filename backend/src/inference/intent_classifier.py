from __future__ import annotations

import re
import unicodedata
from enum import Enum

from src.core.base_llm import BaseLLM
from src.inference.chitchat_detector import is_chitchat


class QueryIntent(str, Enum):
    KNOWLEDGE = "knowledge"   # needs full RAG pipeline
    CHITCHAT = "chitchat"     # conversational, fast LLM response
    OFF_TOPIC = "off_topic"   # irrelevant/nonsensical, quick refusal


_DOMAIN_SIGNALS = re.compile(
    r"\b("
    r"là gì|nghĩa là|định nghĩa|giải thích|tại sao|vì sao|như thế nào|ảnh hưởng|"
    r"so sánh|phân biệt|khác nhau|khác với|so với|liên quan|ví dụ|trình bày|phân tích|tóm tắt|"
    r"tổng quan|tổng hợp|tóm lược|khái quát|overview|"
    r"nêu|mô tả|chứng minh|tính|tính chất|công thức|thuật toán|phương pháp|"
    r"what|why|how|when|where|which|define|explain|compare|describe|analyze|"
    r"difference|relationship|example|summarize|outline|prove|calculate|"
    r"tài liệu|chương|phần|trang|bài|slide|sách|giáo trình|đề cương|bài giảng|"
    r"tom tat|tong quan|tong hop|khai quat|trinh bay|phan tich|giai thich|"
    r"so sanh|phan biet|khac voi|so voi|lien quan|dinh nghia|huong dan|mo ta|tai lieu|bai giang|"
    # Data / table / figure lookup intents — multi-domain (product, finance, legal…),
    # not just academic. A "how much / how many / which row" question over an
    # uploaded table is a knowledge query, not off-topic.
    r"bao nhiêu|bao nhieu|giá|gia tien|gia ban|số lượng|so luong|danh sách|danh sach|liệt kê|liet ke|"
    r"bảng|bang|cột|cot|hàng nào|dòng nào|giá trị|gia tri|"
    r"tổng|tong|trung bình|trung binh|lớn nhất|lon nhat|nhỏ nhất|nho nhat|cao nhất|cao nhat|thấp nhất|thap nhat|"
    r"how much|how many|list|total|average|sum|maximum|minimum|highest|lowest|price|cost|value|column|row"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_MARK = re.compile(r"[?？]")
_REAL_TIME_OFF_TOPIC_RE = re.compile(
    r"\b("
    r"weather|forecast|temperature|rain|raining|sunny|"
    r"thoi tiet|du bao thoi tiet|nhiet do|troi mua|"
    r"stock price|gia vang|ty gia|exchange rate|"
    r"current news|tin tuc|thoi su"
    r")\b",
    re.IGNORECASE,
)

_CLASSIFY_PROMPT = """\
Bạn là classifier cho hệ thống hỏi đáp tài liệu học tập Noelys.

Phân loại câu hỏi vào đúng một nhóm:
- knowledge  : hỏi về nội dung tài liệu, khái niệm học thuật, tính năng hệ thống (upload, tóm tắt, so sánh, mindmap, trích dẫn), hoặc bất kỳ điều cần tra cứu/phân tích
- chitchat   : chào hỏi, cảm ơn, trò chuyện xã giao thuần túy
- off_topic  : văn bản vô nghĩa, spam, hoặc rõ ràng không liên quan đến học tập hay hệ thống

Ví dụ:
Câu: "Neural network là gì?" -> knowledge
Câu: "Tóm tắt chương 3 cho tôi" -> knowledge
Câu: "So sánh hai tài liệu này" -> knowledge
Câu: "Xin chào bạn" -> chitchat
Câu: "Cảm ơn nhé!" -> chitchat
Câu: "asdfghjkl 123" -> off_topic
Câu: "Mua bán bất động sản ở đâu?" -> off_topic

Khi nghi ngờ, ưu tiên chọn knowledge thay vì off_topic.

Câu hỏi: {query}

Chỉ trả lời đúng một từ: knowledge, chitchat, hoặc off_topic"""


class IntentClassifier:
    """
    Three-tier intent classifier:
      Tier 1: regex patterns   (instant, catches clear chitchat)
      Tier 2: heuristics       (instant, catches obvious off-topic)
      Tier 3: LLM              (async, only for ambiguous queries)
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm

    async def classify(self, query: str) -> QueryIntent:
        if is_chitchat(query):
            return QueryIntent.CHITCHAT

        heuristic = self._heuristic(query)
        if heuristic is not None:
            return heuristic

        if self._llm is not None:
            return await self._llm_classify(query)

        return QueryIntent.KNOWLEDGE

    @staticmethod
    def _heuristic(query: str) -> QueryIntent | None:
        text = query.strip()
        normalized = IntentClassifier._ascii_fold(text)
        if _REAL_TIME_OFF_TOPIC_RE.search(normalized):
            return QueryIntent.OFF_TOPIC
        tokens = text.split()
        n = len(tokens)
        has_question = _QUESTION_MARK.search(text) is not None
        has_domain = _DOMAIN_SIGNALS.search(text) is not None

        if has_domain:
            return QueryIntent.KNOWLEDGE

        if n <= 2 and not has_question:
            return QueryIntent.OFF_TOPIC

        # No domain signal — defer to LLM classifier rather than defaulting to
        # KNOWLEDGE. A blanket "question mark + ≥3 tokens = knowledge" rule
        # let chitchat like "Hôm nay nên ăn món gì?" slip through the
        # retrieval pipeline.
        return None

    @staticmethod
    def _ascii_fold(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value.lower())
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")

    async def _llm_classify(self, query: str) -> QueryIntent:
        prompt = _CLASSIFY_PROMPT.format(query=query[:300])
        try:
            raw = await self._llm.generate(prompt=prompt)
            label = raw.strip().lower().split()[0].rstrip(".,;:")
            return QueryIntent(label)
        except (ValueError, IndexError, Exception):
            return QueryIntent.KNOWLEDGE
