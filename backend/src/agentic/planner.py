from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.rag.query_router import RouteDecision, RouteType

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_PLANNER_PROMPT = """\
You are a retrieval planner for a document Q&A system. Given a user query, output a minimal JSON plan.

Available tools for sub_questions:
- "retrieve_text": hybrid vector search on document chunks (default)
- "retrieve_per_source": retrieve from each source document separately
- "trace_graph": traverse entity relationship graph

Inputs:
  query: {query}
  detected_route: {route_type}
  document_count: {material_count}

Output ONLY a JSON object (no markdown, no prose):
{{
  "plan_type": "short label",
  "use_graph": false,
  "use_multi_query": true,
  "use_per_source": false,
  "requires_coverage": false,
  "sub_questions": [
    {{"text": "specific retrieval question", "tool": "retrieve_text", "critical": true, "depends_on": null}}
  ]
}}

Rules:
- use_graph: true only if query asks about relationships, causes, connections, or impact between entities
- use_per_source: true if comparing multiple documents or if per-source coverage is needed
- requires_coverage: must be true when use_per_source is true
- sub_questions: 1-4 specific decomposed questions using concrete terms; omit if query is simple
- critical: false for nice-to-have sub-questions; true for those whose absence makes the answer incomplete
- depends_on: null (independent) OR 0-based index of the prerequisite sub-question.
  Use ONLY for true sequential multi-hop: when question B needs question A's answer first.
  Example: Q0="What is regularization?", Q1={{"text":"How does it affect the loss function?","depends_on":0}}
  NEVER set depends_on if both questions can be retrieved in parallel.

JSON:\
"""


class AgenticSubQuestion(BaseModel):
    text: str
    tool: str = "retrieve_text"
    critical: bool = True
    depends_on: int | None = None


class AgenticPlan(BaseModel):
    plan_type: str
    steps: list[str] = Field(default_factory=list)
    sub_questions: list[AgenticSubQuestion] = Field(default_factory=list)
    use_graph: bool = False
    use_multi_query: bool = False
    use_per_source: bool = False
    requires_coverage: bool = False


class AgenticPlanner:
    """Deterministic planner with optional LLM-powered override."""

    async def build_with_llm(
        self,
        *,
        query: str,
        route: RouteDecision,
        material_count: int,
        llm: "BaseLLM",
    ) -> AgenticPlan:
        """LLM-powered planner that reasons about optimal retrieval strategy.
        Falls back to deterministic plan on any parse/validation failure.
        """
        prompt = _PLANNER_PROMPT.format(
            query=query,
            route_type=route.route_type.value,
            material_count=material_count,
        )
        try:
            raw = await llm.generate(prompt=prompt)
            plan = self._parse_llm_plan(raw, route=route, material_count=material_count)
            if plan is not None:
                logger.info(
                    "LLM planner produced plan",
                    extra={"plan_type": plan.plan_type, "sub_question_count": len(plan.sub_questions)},
                )
                return plan
        except Exception as exc:
            logger.warning(
                "LLM planner failed — falling back to deterministic",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
        return self.build(query=query, route=route, material_count=material_count)

    def _parse_llm_plan(self, raw: str, *, route: RouteDecision, material_count: int) -> AgenticPlan | None:
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

        # Default to False — LLM should explicitly opt in to per-source retrieval.
        # Falling back to material_count > 1 caused per-source on every factual query.
        use_per_source = bool(data.get("use_per_source", False))
        requires_coverage = bool(data.get("requires_coverage", use_per_source))
        use_graph = bool(data.get("use_graph", route.use_graph))
        use_multi_query = bool(data.get("use_multi_query", route.use_multi_query))
        plan_type = str(data.get("plan_type", route.route_type.value))[:64]

        raw_sqs = data.get("sub_questions", [])
        sub_questions: list[AgenticSubQuestion] = []
        valid_tools = {"retrieve_text", "retrieve_per_source", "trace_graph"}
        if isinstance(raw_sqs, list):
            for item in raw_sqs[:6]:
                if not isinstance(item, dict):
                    continue
                text_val = str(item.get("text", "")).strip()
                tool_val = str(item.get("tool", "retrieve_text"))
                critical_val = bool(item.get("critical", True))
                raw_dep = item.get("depends_on")
                depends_on_val: int | None = None
                if isinstance(raw_dep, int) and 0 <= raw_dep < len(raw_sqs):
                    depends_on_val = raw_dep
                if text_val and tool_val in valid_tools:
                    sub_questions.append(AgenticSubQuestion(text=text_val, tool=tool_val, critical=critical_val, depends_on=depends_on_val))

        steps: list[str] = ["retrieve_multi_query" if use_multi_query else "retrieve_text"]
        if use_per_source:
            steps.append("retrieve_per_source")
        if sub_questions:
            steps.append("retrieve_sub_questions")
        if use_graph:
            steps.append("trace_graph")
        if requires_coverage:
            steps.extend(["verify_coverage", "repair_retrieval"])
        steps.extend(["verify_evidence_quality", "synthesize_answer", "verify_claims"])

        return AgenticPlan(
            plan_type=plan_type,
            steps=steps,
            sub_questions=sub_questions,
            use_graph=use_graph,
            use_multi_query=use_multi_query,
            use_per_source=use_per_source,
            requires_coverage=requires_coverage,
        )

    def build(self, *, query: str = "", route: RouteDecision, material_count: int) -> AgenticPlan:
        steps: list[str] = ["retrieve_multi_query" if route.use_multi_query else "retrieve_text"]
        # Only use per-source retrieval for routes that genuinely need cross-doc
        # coverage (COMPARISON, SUMMARIZATION). FACTUAL / GENERAL queries already
        # get good results from hybrid text search; per-source adds noise + latency.
        use_per_source = (
            material_count > 1
            and route.route_type in {RouteType.COMPARISON, RouteType.SUMMARIZATION}
        )
        requires_coverage = use_per_source
        sub_questions = self._sub_questions(query=query, route=route, material_count=material_count)

        if use_per_source:
            steps.append("retrieve_per_source")
        if sub_questions:
            steps.append("retrieve_sub_questions")
        if route.use_graph:
            steps.append("trace_graph")
        if requires_coverage:
            steps.append("verify_coverage")
            steps.append("repair_retrieval")
        steps.extend(["verify_evidence_quality", "synthesize_answer", "verify_claims"])

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
            sub_questions=sub_questions,
            use_graph=route.use_graph,
            use_multi_query=route.use_multi_query,
            use_per_source=use_per_source,
            requires_coverage=requires_coverage,
        )

    def _sub_questions(self, *, query: str, route: RouteDecision, material_count: int) -> list[AgenticSubQuestion]:
        text = " ".join(query.split())
        if not text:
            return []

        # Sub-questions use the original query text directly — no English prefix
        # templates. Prefixes like "What evidence directly defines or explains:"
        # pollute the embedding when the query is in Vietnamese, degrading retrieval.
        if route.route_type == RouteType.COMPARISON:
            return self._cap(
                [
                    AgenticSubQuestion(text=text, tool="retrieve_per_source"),
                    AgenticSubQuestion(text=f"điểm giống nhau: {text}"),
                    AgenticSubQuestion(text=f"điểm khác nhau: {text}"),
                ]
            )
        if route.route_type == RouteType.GRAPH_RELATION:
            return self._cap(
                [AgenticSubQuestion(text=text)]
            )
        if route.route_type == RouteType.CLAIM_CHECK:
            return self._cap(
                [
                    AgenticSubQuestion(text=f"supporting evidence / bằng chứng ủng hộ: {text}"),
                    AgenticSubQuestion(text=f"evidence that contradicts / bằng chứng phản bác: {text}"),
                ]
            )
        if route.route_type == RouteType.SUMMARIZATION:
            return self._cap(
                [
                    AgenticSubQuestion(text=f"ý chính về: {text}"),
                    AgenticSubQuestion(text=f"định nghĩa và ví dụ cho: {text}"),
                    AgenticSubQuestion(
                        text=text, tool="retrieve_per_source",
                        critical=material_count > 1,
                    ),
                ]
            )
        if route.route_type == RouteType.GENERAL and material_count > 1:
            return self._cap(
                [
                    AgenticSubQuestion(text=text),
                    AgenticSubQuestion(text=f"bối cảnh và căn cứ cho: {text}", critical=False),
                ]
            )
        if route.route_type == RouteType.FACTUAL:
            # FACTUAL = direct lookup; the main retrieval pass covers it.
            # Adding a duplicate sub-question just causes a redundant 33s embedding.
            return []
        return []

    @staticmethod
    def _cap(items: list[AgenticSubQuestion], limit: int = 6) -> list[AgenticSubQuestion]:
        return items[:limit]
