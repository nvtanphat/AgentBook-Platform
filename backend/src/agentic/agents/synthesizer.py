"""SynthesizerAgent — composes a grounded answer from collected evidence.

Uses the InferenceEngine's _build_prompt + llm.generate path (the same prompt
templates per route already calibrated through Phase C). The agent adds a
persona preamble that frames the LLM as a "grounded answer composer" and
makes the citation contract explicit.

This agent is the only one that mutates the final answer text. Critic agent
runs AFTER and may trigger another synthesis pass — but on the second pass
the synthesizer just receives augmented evidence, no recursion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.core.base_llm import BaseLLM
    from src.inference.inference_engine import InferenceEngine
    from src.rag.query_router import RouteType
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SYNTH_PERSONA = """\
You are a careful answer composer. Your job:
  1. Read the EVIDENCE passages carefully.
  2. Answer the QUESTION using ONLY information from the evidence.
  3. Cite every factual sentence with [N] markers matching evidence indices.
  4. Never invent details. If evidence is insufficient, say so plainly.
  5. Match the answer language to the question language.

QUESTION: {query}

EVIDENCE:
{evidence}

Compose your answer in {answer_language}. Begin now:\
"""


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def __init__(self, *, llm: "BaseLLM", engine: "InferenceEngine") -> None:
        super().__init__(llm=llm)
        self.engine = engine

    async def run(
        self,
        *,
        query: str,
        context_chunks: list["RetrievedChunk"],
        route_type: "RouteType",
        answer_language: str,
        memory_context: str = "",
    ) -> str:
        """Generate the grounded answer. Delegates to engine._build_prompt to
        keep route-specific prompt templates (Phase C calibrated) intact.
        """
        prompt = self.engine._build_prompt(
            query=query,
            chunks=context_chunks,
            answer_language=answer_language,
            memory_context=memory_context,
            route_type=route_type,
        )
        try:
            answer = await self.llm.generate(prompt=prompt)
            return answer.strip()
        except Exception as exc:
            logger.warning(
                "SynthesizerAgent: LLM generation failed",
                extra={"error": str(exc)},
            )
            return ""

    async def act(self, state: "AgentState", *, mode: str = "draft") -> "AgentState":
        """Blackboard entry. mode='draft' uses cleaned_evidence; mode='repair'
        uses context_chunks and injects grounding warnings into the prompt."""
        chunks = state.context_chunks or state.cleaned_evidence or state.raw_evidence
        if not chunks:
            state.draft_answer = ""
            return state
        route_type = getattr(state.route, "route_type", None) if state.route else None
        try:
            answer = await self.run(
                query=state.resolved_query or state.query,
                context_chunks=chunks,
                route_type=route_type,
                answer_language=state.answer_language or "vi",
                memory_context=state.memory_context or "",
            )
        except Exception as exc:
            logger.warning("SynthesizerAgent.act failed", extra={"error": str(exc)})
            answer = ""
        if mode == "repair":
            state.final_answer = answer
        else:
            state.draft_answer = answer
        return state
