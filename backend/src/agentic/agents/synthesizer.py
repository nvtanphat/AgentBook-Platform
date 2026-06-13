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

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.core.base_llm import BaseLLM
    from src.inference.inference_engine import InferenceEngine
    from src.rag.query_router import RouteType
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def __init__(
        self,
        *,
        llm: "BaseLLM",
        engine: "InferenceEngine",
        consistency_n: int = 1,
        consistency_threshold: float = 0.65,
    ) -> None:
        super().__init__(llm=llm)
        self.engine = engine
        self.consistency_n = max(1, consistency_n)
        self.consistency_threshold = consistency_threshold

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
        uses context_chunks and injects grounding warnings into the prompt.

        When consistency_n > 1 and confidence is below threshold, generates
        N candidates and selects the one with the most cross-answer support.
        """
        chunks = state.context_chunks or state.cleaned_evidence or state.raw_evidence
        if not chunks:
            state.draft_answer = ""
            return state
        route_type = getattr(state.route, "route_type", None) if state.route else None
        confidence = getattr(state, "confidence_score", None)
        use_consistency = (
            mode == "draft"
            and self.consistency_n > 1
            and (confidence is None or confidence < self.consistency_threshold)
        )
        try:
            if use_consistency:
                answer = await self._consistent_run(
                    query=state.resolved_query or state.query,
                    context_chunks=chunks,
                    route_type=route_type,
                    answer_language=state.answer_language or "vi",
                    memory_context=state.memory_context or "",
                )
            else:
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

    async def _consistent_run(
        self,
        *,
        query: str,
        context_chunks: "list[RetrievedChunk]",
        route_type: "RouteType | None",
        answer_language: str,
        memory_context: str,
    ) -> str:
        """Generate N candidates in parallel and return the one with most cross-answer sentence support."""
        prompt = self.engine._build_prompt(
            query=query,
            chunks=context_chunks,
            answer_language=answer_language,
            memory_context=memory_context,
            route_type=route_type,
        )
        tasks = [
            self.llm.generate(prompt=prompt, temperature=0.0 if i == 0 else 0.25)
            for i in range(self.consistency_n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        candidates = [r.strip() for r in results if isinstance(r, str) and r.strip()]
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        best = self._majority_vote(candidates)
        logger.info(
            "SynthesizerAgent: self-consistency vote from %d candidates",
            len(candidates),
            extra={"query_prefix": query[:60]},
        )
        return best

    @staticmethod
    def _majority_vote(answers: list[str]) -> str:
        """Return the answer whose sentences have most cross-answer lexical support.

        Splits each answer into sentences, then for each candidate counts how
        many of its sentences appear (Jaccard ≥ 0.5) in at least half the other
        candidates. The candidate with the highest supported-sentence count wins.
        """
        def _tokenset(text: str) -> frozenset[str]:
            return frozenset(re.findall(r"[\w]{3,}", text.lower()))

        def _sentences(text: str) -> list[str]:
            return [s.strip() for s in re.split(r"(?<=[.!?。\n])\s+", text) if len(s.strip()) > 15]

        threshold = len(answers) / 2
        all_sents = [_sentences(a) for a in answers]

        best_score, best_answer = -1, answers[0]
        for i, (ans, sents) in enumerate(zip(answers, all_sents)):
            score = 0
            for s in sents:
                stok = _tokenset(s)
                if not stok:
                    continue
                votes = sum(
                    1 for j, other_sents in enumerate(all_sents)
                    if j != i and any(
                        (lambda ut=stok | _tokenset(os): len(stok & _tokenset(os)) / max(len(ut), 1) >= 0.5)()
                        for os in other_sents
                    )
                )
                if votes >= threshold:
                    score += 1
            if score > best_score:
                best_score, best_answer = score, ans
        return best_answer
