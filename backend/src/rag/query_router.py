from __future__ import annotations

import json
import logging
import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")

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


class PreferredModality(str, Enum):
    """Modality the query is asking about — orthogonal to RouteType.

    A "compare the two tables" query is COMPARISON route + TABLE modality. NONE
    means no modality signal (plain text QA, the default).
    """

    NONE = "none"
    TABLE = "table"
    FIGURE = "figure"
    AUDIO = "audio"


class Difficulty(str, Enum):
    """How much work the query needs — drives agentic activation."""

    SIMPLE = "simple"        # one fact / one hop
    MULTI_HOP = "multi_hop"  # 2-3 hops, cross-doc, or multi-question
    COMPLEX = "complex"      # 4+ factors / synthesis-heavy


class TableQueryType(str, Enum):
    """Sub-intent for TABLE-modality queries — drives the table executor."""

    LOOKUP = "lookup"            # value of column C where A = X
    AGGREGATION = "aggregation"  # sum/avg/max/min/count over a column
    FILTER = "filter"            # rows matching a condition
    COMPARISON = "comparison"    # compare specific rows
    SORT = "sort"                # rank/sort by a column


class RouteDecision(BaseModel):
    route_type: RouteType
    top_k_multiplier: float = 1.0
    use_graph: bool = False
    graph_priority: bool = False
    use_multi_query: bool = False
    use_mmr: bool = False
    preferred_modality: PreferredModality = PreferredModality.NONE
    # Structured product-router signals (Phase 2). Defaults keep prior behaviour.
    difficulty: Difficulty = Difficulty.SIMPLE
    confidence: float = 0.6
    should_use_agentic: bool = False
    table_query_type: TableQueryType | None = None


# Fallback modality keyword seeds — used when config lists are empty. VI + EN.
_DEFAULT_TABLE_KEYWORDS = [
    "bảng", "cột", "hàng", "dòng", "ô", "giá trị", "bao nhiêu", "tổng", "trung bình",
    "table", "column", "row", "cell", "value", "sum", "average", "mean", "total",
]
_DEFAULT_FIGURE_KEYWORDS = [
    "biểu đồ", "đồ thị", "sơ đồ", "hình", "ảnh", "hình vẽ", "hình ảnh",
    "chart", "charts", "figure", "figures", "diagram", "diagrams",
    "graph", "graphs", "plot", "plots", "image", "images", "visualization", "visualizations",
]
_DEFAULT_AUDIO_KEYWORDS = [
    "ghi âm", "đoạn ghi", "băng ghi", "phút", "giây", "nói rằng",
    "audio", "recording", "timestamp", "minute", "transcript",
]


def _compile_keyword_re(keywords: list[str]) -> re.Pattern[str]:
    # ASCII-fold keywords so a query typed without Vietnamese diacritics
    # ("bao nhieu") still matches a diacritic keyword ("bao nhiêu"). Drop folded
    # keywords shorter than 2 chars (e.g. "ô"→"o") — they would match any letter —
    # and require word boundaries so we match whole words, not substrings.
    from src.processing.slug import ascii_fold
    folded = {f for k in keywords if k and len(f := ascii_fold(k).lower().strip()) >= 2}
    body = "|".join(re.escape(k) for k in sorted(folded, key=len, reverse=True))
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


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
    r"có đúng là.{0,120}không|co dung la.{0,120}khong|"
    r"có đúng không|co dung khong|đúng không|dung khong|"
    r"kiểm chứng|kiem chung|xác minh|xac minh|"
    r"verify|is it true|true or false|fact.?check"
    r")\b",
    re.IGNORECASE,
)

_CONTEXT_FACTUAL_RE = re.compile(
    r"\b("
    r"theo doan trich|trong doan trich|doan trich|passage|excerpt|"
    r"duoc mo ta|duoc nhac|xuat hien|cong cu nao|nhan nao|ten .* la gi|"
    r"which|what .* listed|what .* shown|who .* shown"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_FACT_RE = re.compile(
    r"\b(nao|gi|ai|o dau|bao nhieu|which|what|who|where|how many)\b",
    re.IGNORECASE,
)


# ── Structured-signal patterns (Phase 2) ─────────────────────────────────────
# Matched on ASCII-folded text. Table sub-intent for the deterministic executor.
_AGG_RE = re.compile(
    r"\b(tong|trung binh|lon nhat|nho nhat|cao nhat|thap nhat|dem so|bao nhieu (san pham|dong|hang|muc)|"
    r"sum|total|average|mean|max|min|maximum|minimum|count|highest|lowest)\b",
    re.IGNORECASE,
)
_SORT_RE = re.compile(r"\b(sap xep|xep hang|thu tu|sort|rank|ranked|descending|ascending|top \d+)\b", re.IGNORECASE)
_FILTER_RE = re.compile(r"\b(nhung (san pham|dong|hang|muc) (co|ma)|loc|where|filter|matching|co dieu kien)\b", re.IGNORECASE)
_TABLE_COMPARE_RE = re.compile(r"\b(so sanh|cao hon|thap hon|nhieu hon|it hon|compare|versus|\bvs\b|hon)\b", re.IGNORECASE)

# Difficulty factors — each hit is one complexity point. Deliberately excludes
# bare "và"/"and" (ubiquitous conjunctions) so simple compound questions are NOT
# mis-flagged multi-hop; only genuine chaining/relational phrases count.
_MULTIHOP_RE = re.compile(
    r"\b(sau do|truoc do|dan den|keo theo|moi quan he|lien quan den|tu do suy ra|"
    r"and then|because of|leads to|relationship between|step by step)\b",
    re.IGNORECASE,
)
_CROSSDOC_RE = re.compile(r"\b(tat ca tai lieu|cac tai lieu|nhieu nguon|across documents|all sources|moi nguon)\b", re.IGNORECASE)
_TEMPORAL_RE = re.compile(
    r"\b(theo thoi gian|tien trinh|lich su|truoc do|sau do|"
    r"before that|after that|over time|timeline|evolution)\b",
    re.IGNORECASE,
)
# Year-range or trend pattern: "từ 2020 đến 2024", "xu hướng", "2020-2024", etc.
# More specific than _TEMPORAL_RE so it doesn't false-fire on "sau thuế".
_TREND_RE = re.compile(
    r"(\d{4}\s*(?:den|đến|to|through|-|–)\s*\d{4}"
    r"|tu\s+\d{4}|xu\s+h[uư][oơ]ng|xu\s+huong"
    r"|from\s+\d{4}|over\s+the\s+years|year.?over.?year|yoy"
    r"|tang\s+lien\s+tuc|giam\s+lien\s+tuc|bien\s+dong"
    r"|(?:\d{4}[\s,]+){2,}\d{4}"           # 3+ years: "2022 2023 2024" or "2022, 2023, 2024"
    r"|qua\s+(?:cac\s+)?nam\s+\d{4}"       # "qua cac nam 2022"
    r"|thay\s+doi\s+(?:the\s+nao\s+)?qua)" # "thay doi the nao qua"
    ,
    re.IGNORECASE,
)

# route_type → base confidence (well-defined intents score higher).
_ROUTE_CONFIDENCE: dict = {
    RouteType.FACTUAL: 0.85,
    RouteType.CLAIM_CHECK: 0.80,
    RouteType.GENERAL: 0.62,
    RouteType.COMPARISON: 0.55,
    RouteType.SUMMARIZATION: 0.58,
    RouteType.GRAPH_RELATION: 0.50,
}


class QueryRouter:
    """Adaptive routing for knowledge queries — LLM-powered with regex fallback."""

    def __init__(self, settings=None) -> None:
        if settings is None:
            from src.core.config import get_settings
            settings = get_settings()
        self._modality_enabled = getattr(settings, "modality_routing_enabled", True)
        self._table_re = _compile_keyword_re(
            getattr(settings, "modality_table_keywords", None) or _DEFAULT_TABLE_KEYWORDS
        )
        self._figure_re = _compile_keyword_re(
            getattr(settings, "modality_figure_keywords", None) or _DEFAULT_FIGURE_KEYWORDS
        )
        self._audio_re = _compile_keyword_re(
            getattr(settings, "modality_audio_keywords", None) or _DEFAULT_AUDIO_KEYWORDS
        )
        self._agentic_conf_threshold = float(
            getattr(settings, "router_agentic_confidence_threshold", 0.55)
        )

    def _detect_modality(self, text: str) -> PreferredModality:
        """Orthogonal modality detection — does NOT change route_type/multipliers."""
        if not self._modality_enabled:
            return PreferredModality.NONE
        from src.processing.slug import ascii_fold
        text = ascii_fold(text)
        # Table wins over figure/audio when multiple hit: table QA is the highest-value
        # dedicated path and table keywords ("giá trị", "bao nhiêu") are most specific.
        if self._table_re.search(text):
            return PreferredModality.TABLE
        if self._figure_re.search(text):
            return PreferredModality.FIGURE
        if self._audio_re.search(text):
            return PreferredModality.AUDIO
        return PreferredModality.NONE

    # ── Structured product-router signals ────────────────────────────────────
    def _enrich(self, decision: RouteDecision, query: str) -> RouteDecision:
        """Attach difficulty / confidence / should_use_agentic / table_query_type.

        Orthogonal to route_type & multipliers — never changes existing retrieval
        behaviour; these signals drive the table executor and agentic activation.
        """
        from src.processing.slug import ascii_fold
        text = ascii_fold(query).lower()

        if decision.preferred_modality == PreferredModality.TABLE:
            decision.table_query_type = self._detect_table_subtype(text)

        factors = 0
        factors += 1 if _MULTIHOP_RE.search(text) else 0
        factors += 1 if _CROSSDOC_RE.search(text) else 0
        factors += 1 if _TEMPORAL_RE.search(text) else 0
        has_trend = bool(_TREND_RE.search(text))
        factors += 1 if has_trend else 0
        if decision.route_type in (RouteType.COMPARISON, RouteType.GRAPH_RELATION):
            factors += 1  # relational/synthesis routes are inherently multi-hop
        if factors >= 3:
            decision.difficulty = Difficulty.COMPLEX
        elif factors >= 1:
            decision.difficulty = Difficulty.MULTI_HOP
        else:
            decision.difficulty = Difficulty.SIMPLE

        base = _ROUTE_CONFIDENCE.get(decision.route_type, 0.6)
        decision.confidence = round(max(0.0, base - 0.1 * factors), 3)
        decision.should_use_agentic = (
            decision.confidence < self._agentic_conf_threshold
            or decision.difficulty == Difficulty.COMPLEX
        )
        # Enable graph retrieval for multi-hop / complex queries that did not
        # already set use_graph (e.g. GRAPH_RELATION route). Trend / year-range
        # queries benefit from graph traversal even when the route is GENERAL.
        if not decision.use_graph and decision.difficulty != Difficulty.SIMPLE:
            decision.use_graph = True
        return decision

    def _detect_table_subtype(self, text: str) -> "TableQueryType | None":
        if _AGG_RE.search(text):
            return TableQueryType.AGGREGATION
        if _SORT_RE.search(text):
            return TableQueryType.SORT
        if _FILTER_RE.search(text):
            return TableQueryType.FILTER
        if _TABLE_COMPARE_RE.search(text):
            return TableQueryType.COMPARISON
        # Plain LOOKUP drives the table-cell executor, which returns a
        # high-confidence single-cell answer. Only take that fast path when the
        # query actually carries a table lexical signal ("giá trị", "cột", "bao
        # nhiêu", "value"…). Otherwise an LLM router that over-eagerly tags a
        # conceptual question as TABLE would hijack it into a wrong cell answer
        # (e.g. a yes/no "does X use recurrence?" answered with a stray F1 cell).
        # Without a signal, return None so synthesis answers it normally.
        if self._table_re.search(text):
            return TableQueryType.LOOKUP
        return None

    async def route_with_llm(self, query: str, *, llm: "BaseLLM") -> RouteDecision:
        """LLM-powered routing. Falls back to deterministic regex on any failure."""
        try:
            prompt = _ROUTER_PROMPT.format(query=query)
            raw = await llm.generate(prompt=prompt)
            decision = self._parse_llm_route(raw, query=query)
            if decision is not None:
                logger.info("LLM router decision", extra={"route": decision.route_type.value})
                return decision
        except Exception as exc:
            logger.warning("LLM router failed — falling back to regex", extra={"error": str(exc)})
        return self.route(query)

    def _parse_llm_route(self, raw: str, *, query: str = "") -> RouteDecision | None:
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

        # Modality: trust an explicit LLM "modality" key, else regex-detect from query.
        modality = PreferredModality.NONE
        raw_modality = str(data.get("modality", "")).lower().strip()
        if raw_modality in {m.value for m in PreferredModality}:
            modality = PreferredModality(raw_modality)
        elif query:
            modality = self._detect_modality(" ".join(query.split()))

        decision = RouteDecision(
            route_type=route_type,
            top_k_multiplier=float(data.get("top_k_multiplier", 1.0)),
            use_graph=bool(data.get("use_graph", route_type == RouteType.GRAPH_RELATION)),
            graph_priority=route_type == RouteType.GRAPH_RELATION,
            use_multi_query=bool(data.get("use_multi_query", True)),
            use_mmr=bool(data.get("use_mmr", route_type in (RouteType.COMPARISON, RouteType.SUMMARIZATION))),
            preferred_modality=modality,
        )
        return self._enrich(decision, query) if query else decision

    def route(self, query: str) -> RouteDecision:
        text = " ".join(query.split())
        folded_text = _fold_text(text).lower()

        # Multipliers reverted to v12-baseline values after v13 regression.
        # Aggressive multipliers (2.5-3.0) bloated rerank candidate pool and
        # caused 4 false refusals on synthesis queries — keep tight here.
        if _CLAIM_CHECK_RE.search(text):
            decision = RouteDecision(
                route_type=RouteType.CLAIM_CHECK,
                top_k_multiplier=1.25,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )
        elif _GRAPH_RELATION_RE.search(text):
            decision = RouteDecision(
                route_type=RouteType.GRAPH_RELATION,
                top_k_multiplier=1.5,
                use_graph=True,
                graph_priority=True,
                use_multi_query=True,
                use_mmr=False,
            )
        elif _COMPARISON_RE.search(text):
            decision = RouteDecision(
                route_type=RouteType.COMPARISON,
                top_k_multiplier=1.5,
                use_graph=False,
                use_multi_query=True,
                use_mmr=True,
            )
        elif _CONTEXT_FACTUAL_RE.search(folded_text) and _QUESTION_FACT_RE.search(folded_text):
            decision = RouteDecision(
                route_type=RouteType.FACTUAL,
                top_k_multiplier=0.75,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )
        elif _SUMMARIZATION_RE.search(text):
            decision = RouteDecision(
                route_type=RouteType.SUMMARIZATION,
                top_k_multiplier=2.0,
                use_graph=False,
                use_multi_query=True,
                use_mmr=True,
            )
        elif _FACTUAL_RE.search(text):
            decision = RouteDecision(
                route_type=RouteType.FACTUAL,
                top_k_multiplier=0.75,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )
        else:
            decision = RouteDecision(
                route_type=RouteType.GENERAL,
                top_k_multiplier=1.0,
                use_graph=False,
                use_multi_query=True,
                use_mmr=False,
            )

        # Orthogonal modality dimension — leaves route_type/multipliers untouched.
        decision.preferred_modality = self._detect_modality(text)
        return self._enrich(decision, text)
