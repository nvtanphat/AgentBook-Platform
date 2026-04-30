from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel


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
    """Rule-based adaptive routing for knowledge queries."""

    def route(self, query: str) -> RouteDecision:
        text = " ".join(query.split())

        if _CLAIM_CHECK_RE.search(text):
            return RouteDecision(
                route_type=RouteType.CLAIM_CHECK,
                top_k_multiplier=1.0,
                use_graph=False,
                use_multi_query=False,
                use_mmr=False,
            )

        if _GRAPH_RELATION_RE.search(text):
            return RouteDecision(
                route_type=RouteType.GRAPH_RELATION,
                top_k_multiplier=1.0,
                use_graph=True,
                graph_priority=True,
                use_multi_query=False,
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
            # Multi-query enabled: VI queries benefit from English paraphrase
            # for cross-lingual matching against English source documents.
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
