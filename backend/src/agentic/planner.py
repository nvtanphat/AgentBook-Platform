from __future__ import annotations

from pydantic import BaseModel, Field

from src.rag.query_router import RouteDecision, RouteType


class AgenticPlan(BaseModel):
    plan_type: str
    steps: list[str] = Field(default_factory=list)
    use_graph: bool = False
    use_multi_query: bool = False
    use_per_source: bool = False
    requires_coverage: bool = False


class AgenticPlanner:
    """Deterministic planner for the Agentic RAG MVP."""

    def build(self, *, route: RouteDecision, material_count: int) -> AgenticPlan:
        steps: list[str] = ["retrieve_multi_query" if route.use_multi_query else "retrieve_text"]
        use_per_source = material_count > 1
        requires_coverage = material_count > 1

        if use_per_source:
            steps.append("retrieve_per_source")
        if route.use_graph:
            steps.append("trace_graph")
        if requires_coverage:
            steps.append("verify_coverage")
            steps.append("repair_retrieval")
        steps.extend(["synthesize_answer", "verify_claims"])

        plan_type = route.route_type.value
        if route.route_type == RouteType.GENERAL and requires_coverage:
            plan_type = "multi_source_general"
        if route.route_type == RouteType.GRAPH_RELATION:
            plan_type = "relation_trace"
        if route.route_type == RouteType.CLAIM_CHECK:
            plan_type = "claim_check"

        return AgenticPlan(
            plan_type=plan_type,
            steps=steps,
            use_graph=route.use_graph,
            use_multi_query=route.use_multi_query,
            use_per_source=use_per_source,
            requires_coverage=requires_coverage,
        )

