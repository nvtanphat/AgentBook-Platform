"""PlannerAgent — query decomposition with persona-driven LLM prompt.

Wraps the existing `AgenticPlanner.build_with_llm` but with explicit persona
framing and stronger fallback to the deterministic rule-based planner. The
persona makes the LLM reason like a "retrieval strategist" rather than a
generic Q&A assistant — produces tighter sub-question lists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent
from src.agentic.planner import AgenticPlan, AgenticPlanner, AgenticSubQuestion

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.core.base_llm import BaseLLM
    from src.rag.query_router import RouteDecision

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self, *, llm: "BaseLLM", planner: AgenticPlanner | None = None) -> None:
        super().__init__(llm=llm)
        self._planner = planner or AgenticPlanner()

    async def run(
        self,
        *,
        query: str,
        route: "RouteDecision",
        material_count: int,
        use_llm: bool = True,
    ) -> AgenticPlan:
        """Build a plan. When `use_llm=False` or LLM fails, falls back to the
        deterministic rule-based plan that ships with `AgenticPlanner.build()`.
        """
        if use_llm and self.llm is not None:
            try:
                plan = await self._planner.build_with_llm(
                    query=query,
                    route=route,
                    material_count=material_count,
                    llm=self.llm,
                )
                logger.info(
                    "PlannerAgent: LLM plan built",
                    extra={
                        "plan_type": plan.plan_type,
                        "sub_qs": len(plan.sub_questions),
                        "use_graph": plan.use_graph,
                        "use_per_source": plan.use_per_source,
                    },
                )
                return plan
            except Exception as exc:
                logger.info(
                    "PlannerAgent: LLM failed → deterministic fallback",
                    extra={"error": str(exc)},
                )
        return self._planner.build(query=query, route=route, material_count=material_count)

    async def act(self, state: "AgentState", *, use_llm: bool = True) -> "AgentState":
        """Blackboard entry point. Mutates `state` in place with planning fields.

        On replan loops (state.current_iteration > 0 with critic warnings),
        the planner extends sub-questions using `critic_warnings` so the next
        retrieval pass targets the identified gaps.
        """
        if state.route is None:
            logger.info("PlannerAgent.act called without route — skipping")
            return state

        # Initial plan
        if not state.sub_questions and state.current_iteration == 0:
            plan = await self.run(
                query=state.resolved_query or state.query,
                route=state.route,
                material_count=len(state.expected_material_ids) or 0,
                use_llm=use_llm,
            )
            state.plan_type = plan.plan_type
            state.sub_questions = list(plan.sub_questions)
            state.use_graph = plan.use_graph
            state.use_per_source = plan.use_per_source
            state.use_multi_query = plan.use_multi_query
            state.requires_coverage = plan.requires_coverage
            return state

        # Replan from critic warnings — only use domain-content gaps, NOT
        # internal system messages. System warnings like "All evidence rated
        # below the CRAG correct threshold" are diagnostics, not queries;
        # feeding them into the retriever produces completely off-topic results.
        _SYSTEM_WARNING_PATTERNS = (
            "crag correct threshold",
            "replanning may help",
            "fewer than 25%",
            "consider broader queries",
            "no evidence retrieved",
        )
        if state.critic_warnings:
            for gap in state.critic_warnings[-3:]:
                text = gap.strip()
                if not text:
                    continue
                # Skip internal diagnostic messages — not valid retrieval queries.
                if any(pattern in text.lower() for pattern in _SYSTEM_WARNING_PATTERNS):
                    continue
                if any(text.lower() in sq.text.lower() for sq in state.sub_questions):
                    continue
                state.sub_questions.append(AgenticSubQuestion(text=text, tool="retrieve_text", critical=False))
        return state
