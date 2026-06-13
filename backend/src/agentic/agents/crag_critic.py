"""CRAGCriticAgent — Corrective RAG evidence triage.

Labels each retrieved chunk as CORRECT / AMBIGUOUS / INCORRECT using the
existing score-based `CRAGEvaluator`, optionally cleans the chunk text via
the `TextCleanerTool`, and writes the survivors into
`state.cleaned_evidence`. If too many INCORRECT chunks dominate the
evidence pool, it raises a `critic_warning` so the coordinator can loop
back to the planner for replanning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent
from src.agentic.state import CRAGEvidenceVerdict, CRAGLabel

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.agentic.tools import TextCleanerTool
    from src.rag.crag_evaluator import CRAGEvaluator, LLMCRAGEvaluator

logger = logging.getLogger(__name__)


class CRAGCriticAgent(BaseAgent):
    name = "crag_critic"

    def __init__(
        self,
        *,
        evaluator: "CRAGEvaluator",
        cleaner: "TextCleanerTool | None" = None,
        correct_threshold: float = 0.55,
        incorrect_threshold: float = 0.25,
        llm_evaluator: "LLMCRAGEvaluator | None" = None,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.cleaner = cleaner
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold
        self.llm_evaluator = llm_evaluator

    async def act(self, state: "AgentState") -> "AgentState":
        chunks = state.raw_evidence or []
        if not chunks:
            state.cleaned_evidence = []
            state.crag_verdicts = []
            state.add_warning("No evidence retrieved.")
            return state

        verdicts: list[CRAGEvidenceVerdict] = []
        for chunk in chunks:
            score = chunk.rerank_score if chunk.rerank_score is not None else (chunk.fused_score or 0.0)
            if score >= self.correct_threshold:
                label = CRAGLabel.CORRECT
                reason = "score>=correct_threshold"
            elif score >= self.incorrect_threshold:
                label = CRAGLabel.AMBIGUOUS
                reason = "score in (incorrect,correct]"
            else:
                label = CRAGLabel.INCORRECT
                reason = "score<incorrect_threshold"
            verdicts.append(
                CRAGEvidenceVerdict(chunk_id=chunk.chunk_id, label=label, score=float(score), reason=reason)
            )
        state.crag_verdicts = verdicts

        # Keep CORRECT + AMBIGUOUS by default; the evaluator decides whether
        # to drop ambiguous chunks when INCORRECT dominates.
        filtered = self.evaluator.evaluate(chunks=chunks)
        # Cleaner pass: strip boilerplate without touching evidence trace.
        if self.cleaner and filtered:
            try:
                result = await self.cleaner.run(chunks=filtered, query=state.resolved_query or state.query)
                if result.success and result.data:
                    filtered = result.data
            except Exception as exc:
                logger.info("TextCleanerTool failed", extra={"error": str(exc)})

        # LLM re-evaluation of AMBIGUOUS chunks: promotes CORRECT or drops INCORRECT.
        # Only runs when llm_evaluator is configured (crag.llm_enabled in config).
        if self.llm_evaluator and filtered:
            # Use chunk_id lookup — `filtered` is a subset of `chunks` so positional
            # indices from `verdicts` do NOT align with indices in `filtered`.
            ambiguous_ids = {v.chunk_id for v in verdicts if v.label == CRAGLabel.AMBIGUOUS}
            ambiguous_indices = [i for i, c in enumerate(filtered) if c.chunk_id in ambiguous_ids]
            if ambiguous_indices:
                try:
                    filtered = await self.llm_evaluator.re_evaluate_ambiguous(
                        query=state.resolved_query or state.query,
                        chunks=filtered,
                        ambiguous_indices=ambiguous_indices,
                    )
                except Exception as exc:
                    logger.warning("LLM CRAG re-evaluation failed", extra={"error": str(exc)})

        state.cleaned_evidence = filtered

        correct_count = sum(1 for v in verdicts if v.label == CRAGLabel.CORRECT)
        if correct_count == 0 and verdicts:
            state.add_warning("All evidence rated below the CRAG correct threshold — replanning may help.")
        elif correct_count / max(1, len(verdicts)) < 0.25:
            state.add_warning("Fewer than 25% of retrieved chunks are CORRECT — consider broader queries.")

        logger.info(
            "CRAGCritic: triage done",
            extra={
                "owner_id": state.scope.owner_id,
                "collection_id": state.scope.collection_id,
                "raw_count": len(chunks),
                "kept_count": len(filtered),
                "correct": correct_count,
            },
        )
        return state
