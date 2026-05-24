from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """\
You are a query classifier for a multilingual document Q&A system. Classify the query into exactly one route.

Routes:
- "factual": asks for a definition, explanation, or a specific isolated fact
- "summarization": asks to summarize, outline, or list main points
- "comparison": asks to compare, contrast, or differentiate items
- "graph_relation": asks how entities are related, how one thing affects/influences/causes/depends-on another
- "claim_check": asks to verify, fact-check, or validate a statement (contains "đúng không", "có phải", "true or false", etc.)
- "general": anything else

Examples:
"F1-score liên quan đến Precision và Recall như thế nào?" → graph_relation
"Dữ liệu training ảnh hưởng thế nào đến hiệu suất mô hình?" → graph_relation
"Overfitting ảnh hưởng đến kết quả dự đoán như thế nào?" → graph_relation
"Supervised learning và unsupervised learning khác nhau như thế nào?" → comparison
"Machine learning là gì?" → factual
"Tóm tắt các bước xây dựng mô hình" → summarization
"F1-score là trung bình cộng của Precision và Recall đúng không?" → claim_check

Query: {query}

Output ONLY a JSON object (no markdown, no prose):
{{"route": "factual", "use_multi_query": true, "use_mmr": false, "use_graph": false, "top_k_multiplier": 0.75}}

JSON:\
"""


class RouteType(str, Enum):
    FACTUAL = "factual"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    GRAPH_RELATION = "graph_relation"
    CLAIM_CHECK = "claim_check"
    GENERAL = "general"


class RouteDecision(BaseModel):
    route_type: RouteType
    top_k_multiplier: float = 1.0
    use_graph: bool = False
    graph_priority: bool = False
    use_multi_query: bool = False
    use_mmr: bool = False


_FACTUAL_RE = re.compile(
    r"\b("
    r"là gì|la gi|định nghĩa|dinh nghia|khái niệm|khai niem|"
    r"nghĩa là|nghia la|có ý nghĩa|co y nghia|"
    r"hiểu thế nào|hieu the nao|thế nào là|the nao la|"
    r"là loại gì|la loai gi|được hiểu là|duoc hieu la|"
    r"what is|what are|define|definition|meaning of|concept of"
    r")\b",
    re.IGNORECASE,
)

_SUMMARIZATION_RE = re.compile(
    r"\b("
    r"tổng quan|tong quan|tóm tắt|tom tat|tóm lược|tom luoc|"
    r"khái quát|khai quat|trình bày|trinh bay|nêu các|neu cac|"
    r"ý chính|y chinh|nội dung chính|noi dung chinh|ý nghĩa chính|"
    r"liệt kê|liet ke|nêu rõ|neu ro|"
    r"overview|summari[sz]e|summary|outline|main (point|idea|content|concept)|"
    r"list (the|all|key)"
    r")\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b("
    r"so sánh|so sanh|khác nhau|khac nhau|phân biệt|phan biet|"
    r"điểm khác|diem khac|điểm giống|diem giong|"
    r"giống nhau|giong nhau|tương đồng|tuong dong|"
    r"ưu nhược|uu nhuoc|ưu điểm|nhược điểm|"
    r"compare|comparison|different|difference|versus|vs\.?|"
    r"similarities|pros.{0,5}cons|advantages.{0,5}disadvantages"
    r")\b",
    re.IGNORECASE,
)

_GRAPH_RELATION_RE = re.compile(
    r"\b("
    r"quan hệ|quan he|liên kết|lien ket|liên quan|lien quan|kết nối|ket noi|"
    r"ảnh hưởng|anh huong|tác động|tac dong|gây ra|gay ra|"
    r"dẫn đến|dan den|phụ thuộc|phu thuoc|chi phối|chi phoi|"
    r"relationship|relation|related|link|connect|connection|"
    r"impact|affect|cause|depend|influence"
    r")\b",
    re.IGNORECASE,
)

_CLAIM_CHECK_RE = re.compile(
    r"\b("
    r"có đúng không|co dung khong|đúng không|dung khong|"
    r"kiểm chứng|kiem chung|xác minh|xac minh|"
    r"verify|is it true|true or false|fact.?check"
    r")\b",
    re.IGNORECASE,
)


class QueryRouter:
    """Adaptive routing for knowledge queries — LLM-powered with regex fallback."""

    async def route_with_llm(self, query: str, *, llm: "BaseLLM") -> RouteDecision:
        """LLM-powered routing. Falls back to deterministic regex on any failure."""
        try:
            prompt = _ROUTER_PROMPT.format(query=query)
            raw = await llm.generate(prompt=prompt)
            decision = self._parse_llm_route(raw)
            if decision is not None:
                logger.info("LLM router decision", extra={"route": decision.route_type.value})
                return decision
        except Exception as exc:
            logger.warning("LLM router failed — falling back to regex", extra={"error": str(exc)})
        return self.route(query)

    def _parse_llm_route(self, raw: str) -> RouteDecision | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        route_str = str(data.get("route", "general")).lower()
        try:
            route_type = RouteType(route_str)
        except ValueError:
            return None

        return RouteDecision(
            route_type=route_type,
            top_k_multiplier=float(data.get("top_k_multiplier", 1.0)),
            use_graph=bool(data.get("use_graph", route_type == RouteType.GRAPH_RELATION)),
            graph_priority=route_type == RouteType.GRAPH_RELATION,
            use_multi_query=bool(data.get("use_multi_query", True)),
            use_mmr=bool(data.get("use_mmr", route_type in (RouteType.COMPARISON, RouteType.SUMMARIZATION))),
        )

    def route(self, query: str) -> RouteDecision:
        text = " ".join(query.split())

        # Multipliers reverted to v12-baseline values after v13 regression.
        # Aggressive multipliers (2.5-3.0) bloated rerank candidate pool and
        # caused 4 false refusals on synthesis queries — keep tight here.
        if _CLAIM_CHECK_RE.search(text):
            return RouteDecision(
                route_type=RouteType.CLAIM_CHECK,
                top_k_multiplier=1.25,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )

        if _GRAPH_RELATION_RE.search(text):
            return RouteDecision(
                route_type=RouteType.GRAPH_RELATION,
                top_k_multiplier=1.5,
                use_graph=True,
                graph_priority=True,
                use_multi_query=True,
                use_mmr=False,
            )

        if _COMPARISON_RE.search(text):
            return RouteDecision(
                route_type=RouteType.COMPARISON,
                top_k_multiplier=1.5,
                use_graph=False,
                use_multi_query=True,
                use_mmr=True,
            )

        if _SUMMARIZATION_RE.search(text):
            return RouteDecision(
                route_type=RouteType.SUMMARIZATION,
                top_k_multiplier=2.0,
                use_graph=False,
                use_multi_query=True,
                use_mmr=True,
            )

        if _FACTUAL_RE.search(text):
            return RouteDecision(
                route_type=RouteType.FACTUAL,
                top_k_multiplier=0.75,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )

        return RouteDecision(
            route_type=RouteType.GENERAL,
            top_k_multiplier=1.0,
            use_graph=False,
            use_multi_query=True,
            use_mmr=False,
        )
