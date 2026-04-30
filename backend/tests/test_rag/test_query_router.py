from __future__ import annotations

from src.rag.query_router import QueryRouter, RouteType


def test_query_router_classifies_route_types() -> None:
    router = QueryRouter()

    assert router.route("Dropout là gì?").route_type == RouteType.FACTUAL
    assert router.route("Tóm tắt tài liệu này").route_type == RouteType.SUMMARIZATION
    assert router.route("So sánh KAN và MLP").route_type == RouteType.COMPARISON
    assert router.route("Quan hệ giữa regularization và overfitting").route_type == RouteType.GRAPH_RELATION
    assert router.route("Kết luận này có đúng không?").route_type == RouteType.CLAIM_CHECK
    # "nội dung chính" is a summarization signal — GENERAL only for truly unclassified queries.
    assert router.route("Phân tích nội dung chính").route_type == RouteType.SUMMARIZATION
    assert router.route("Tell me something").route_type == RouteType.GENERAL


def test_query_router_returns_adaptive_decision_flags() -> None:
    router = QueryRouter()

    factual = router.route("What is dropout?")
    assert factual.top_k_multiplier == 0.75
    assert factual.use_graph is False
    # Multi-query enabled for FACTUAL: Vietnamese queries benefit from English paraphrase.
    assert factual.use_multi_query is True

    summary = router.route("overview of this collection")
    assert summary.top_k_multiplier == 2.0
    assert summary.use_multi_query is True

    graph = router.route("relationship between concepts")
    assert graph.use_graph is True
    assert graph.graph_priority is True
