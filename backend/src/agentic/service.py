"""AgenticCoordinatingEngine — multi-agent, multi-tool RAG orchestrator.

This module implements the blackboard / shared-state coordination pattern
described in the AgentBook plan. The previous `AgenticRagService` ran a
linear pipeline and only fell back to a critic on low confidence; this
engine instead drives a bounded async loop over specialist agents, each
mutating a single shared `AgentState`.

Loop sketch:
  1. PlannerAgent          — decompose / re-decompose the query.
  2. RetrieverDirectorAgent — dispatch sub-questions to the right tools.
  3. CRAGCriticAgent       — triage evidence (CORRECT/AMBIGUOUS/INCORRECT).
  4. (loop) if evidence weak and iteration budget left → planner replans.
  5. SynthesizerAgent      — produce a grounded draft answer.
  6. GuardrailsAgent       — NLI verification.
  7. (loop) if guardrails fail → synthesizer repairs once.
  8. Build QueryResponse + AgentTrace.

Backward compatibility: `AgenticRagService` is kept as a thin wrapper that
delegates `.answer(...)` to the engine. Existing callers
(`QueryService.agentic_rag`) continue to work without changes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from beanie import PydanticObjectId

from src.agentic.agents import (
    CRAGCriticAgent,
    CriticAgent,
    GuardrailsAgent,
    PlannerAgent,
    RetrieverDirectorAgent,
    SynthesizerAgent,
)
from src.agentic.planner import AgenticPlanner
from src.agentic.state import AgentState
from src.rag.crag_evaluator import LLMCRAGEvaluator
from src.agentic.tools import (
    GraphRelationSearchTool,
    HybridTextSearchTool,
    NLIVerifierTool,
    TextCleanerTool,
    VisualImageSearchTool,
)
from src.guardrails.claim_verifier import ClaimVerdict
from src.inference.inference_engine import (
    PUBLIC_GENERATION_ERROR,
    PUBLIC_RETRIEVAL_ERROR,
    REFUSAL_ANSWER,
    InferenceEngine,
)
from src.inference.intent_classifier import QueryIntent
from src.inference.reasoning_path_builder import build_reasoning_path
from src.models.common import PipelineStatus
from src.models.material import Material
from src.rag.query_router import RouteType
from src.rag.retriever import dedupe_retrieved_chunks
from src.rag.types import RetrievalScope, RetrievedChunk
from src.schemas.query import (
    AgentTrace,
    AgentTraceStep,
    AgentVerification,
    CoverageReport,
    CoverageSource,
    QueryResponse,
)

logger = logging.getLogger(__name__)

AgentStepCallback = Callable[[AgentTraceStep], Awaitable[None] | None]


@dataclass(frozen=True)
class _CoordinatorConfig:
    """Snapshot of orchestration-level settings, sourced from engine.settings.

    All thresholds and limits come from configuration — never hardcode here.
    """

    max_iterations: int
    critic_activation_confidence: float
    crag_correct_threshold: float
    crag_incorrect_threshold: float
    enable_visual_tool: bool
    anaphora_resolution_enabled: bool
    planner_llm_enabled: bool


class AgenticCoordinatingEngine:
    """Dynamic coordinator that drives the specialist agents over `AgentState`."""

    def __init__(self, *, engine: InferenceEngine) -> None:
        self.engine = engine
        self.planner = AgenticPlanner()
        self._rerank_fallback_semaphore = asyncio.Semaphore(1)
        _pattern = getattr(engine.settings, "extraction_anaphora_pattern", None) or (
            r"\b(nó|chúng|họ|điều này|điều đó|vấn đề này|khái niệm này|cái này|cái đó|"
            r"it|they|them|this|that|these|those|"
            r"the concept|the topic|the above|the same|the latter|the former)\b"
        )
        self._anaphora_re = re.compile(_pattern, re.IGNORECASE)

        # ── Tool layer ────────────────────────────────────────────────────
        self.text_tool = HybridTextSearchTool(retriever=engine.retriever)
        self.graph_tool = GraphRelationSearchTool(
            graph_retriever=engine.graph_retriever, engine=engine,
        )
        self.visual_tool = VisualImageSearchTool(
            retriever=engine.retriever, visual_provider=getattr(engine, "visual_provider", None),
        )
        self.cleaner_tool = TextCleanerTool()
        self.nli_tool = NLIVerifierTool(verifier=engine.claim_verifier)

        # ── Agent layer ───────────────────────────────────────────────────
        self.agent_planner = PlannerAgent(llm=engine.llm, planner=self.planner)
        self.agent_director = RetrieverDirectorAgent(
            text_tool=self.text_tool,
            graph_tool=self.graph_tool,
            budget_phase2_enabled=getattr(engine.settings, "budget_phase2_enabled", True),
            budget_min_strong_score=float(getattr(engine.settings, "budget_min_strong_score", 0.045)),
            budget_min_strong_count=int(getattr(engine.settings, "budget_min_strong_count", 3)),
            semantic_dedup_threshold=float(getattr(engine.settings, "retrieval_semantic_dedup_threshold", 0.85)),
        )
        _llm_crag_evaluator: LLMCRAGEvaluator | None = None
        if getattr(engine.settings, "crag_llm_enabled", False) and engine.llm is not None:
            _llm_crag_evaluator = LLMCRAGEvaluator(llm=engine.llm)
        self.agent_crag = CRAGCriticAgent(
            evaluator=engine.crag_evaluator,
            cleaner=self.cleaner_tool,
            correct_threshold=engine.settings.crag_correct_threshold,
            incorrect_threshold=engine.settings.crag_incorrect_threshold,
            llm_evaluator=_llm_crag_evaluator,
        )
        self.agent_synthesizer = SynthesizerAgent(
            llm=engine.llm,
            engine=engine,
            consistency_n=int(getattr(engine.settings, "agentic_self_consistency_n", 1)),
            consistency_threshold=float(getattr(engine.settings, "agentic_self_consistency_threshold", 0.65)),
        )
        self.agent_guardrails = GuardrailsAgent(verifier_tool=self.nli_tool)
        self.agent_critic = CriticAgent(llm=engine.llm)

        # ── Coordinator config ────────────────────────────────────────────
        self.config = _CoordinatorConfig(
            max_iterations=max(1, getattr(engine.settings, "agentic_max_retrieval_iterations", 2) + 1),
            critic_activation_confidence=float(
                getattr(engine.settings, "agentic_critic_activation_confidence", 0.65)
            ),
            crag_correct_threshold=engine.settings.crag_correct_threshold,
            crag_incorrect_threshold=engine.settings.crag_incorrect_threshold,
            enable_visual_tool=bool(getattr(engine.settings, "visual_embedding_enabled", False)),
            anaphora_resolution_enabled=bool(getattr(engine.settings, "agentic_anaphora_resolution_enabled", True)),
            planner_llm_enabled=bool(getattr(engine.settings, "agentic_planner_llm_enabled", False)),
        )

    def _clean_answer_text(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        answer_language: str | None,
    ) -> str:
        """Apply optional response-parser cleanup hooks when available."""
        strip_acronyms = getattr(self.engine.response_parser, "strip_unverified_acronym_expansions", None)
        if callable(strip_acronyms):
            answer = strip_acronyms(answer, chunks)
        strip_language_drift = getattr(self.engine.response_parser, "strip_language_drift", None)
        if callable(strip_language_drift):
            answer = strip_language_drift(answer, answer_language)
        return answer

    # ── Public entrypoint ────────────────────────────────────────────────
    async def answer(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        top_k: int | None = None,
        answer_language: str | None = None,
        memory_context: str | None = None,
        on_step: AgentStepCallback | None = None,
    ) -> QueryResponse:
        # 0) Intent classification — chitchat / off-topic short-circuits.
        intent = await self.engine.intent_classifier.classify(query)
        if intent == QueryIntent.CHITCHAT:
            response = await self.engine._answer_chitchat(query)
            response.agent_trace = AgentTrace(
                plan_type="chitchat",
                steps=[AgentTraceStep(name="classify_intent", status="completed", warning="chitchat")],
            )
            return response
        if intent == QueryIntent.OFF_TOPIC:
            response = self.engine._refuse_off_topic()
            response.agent_trace = AgentTrace(
                plan_type="off_topic",
                steps=[AgentTraceStep(name="classify_intent", status="completed", warning="off_topic")],
            )
            return response

        # 1) Build initial state.
        resolved_query = await self._maybe_resolve_anaphora(query=query, memory_context=memory_context, scope=scope)
        route = self.engine.query_router.route(resolved_query)
        expected_material_ids = scope.material_ids or await self._indexed_material_ids_for_scope(scope)
        processed = await self.engine.query_processor.process_async(
            resolved_query, answer_language=answer_language,
        )
        use_multi_query = route.use_multi_query and bool(getattr(self.engine.settings, "multi_query_enabled", False))
        original_query = getattr(processed, "original_query", resolved_query)
        retrieval_queries = processed.retrieval_queries if use_multi_query else [original_query]

        state = AgentState(
            query=query,
            resolved_query=resolved_query,
            scope=scope,
            memory_context=memory_context,
            answer_language=processed.answer_language,
            top_k=top_k,
            expected_material_ids=list(expected_material_ids),
            route=route,
            processed_query=processed,
            retrieval_queries=list(retrieval_queries),
            use_multi_query=use_multi_query,
        )

        retrieval_limit = self.engine._scaled_limit(self.engine.settings.rerank_input_k, route)
        # Only inflate final_limit for coverage when the user explicitly
        # passed material_ids (= "answer using THESE docs"). Treating the
        # full indexed-material set as a coverage floor bloated answers
        # with 9+ chunks even when the user asked top_k=5.
        user_pinned_materials = bool(scope.material_ids)
        coverage_floor = len(expected_material_ids) if user_pinned_materials else 1
        final_limit = max(
            self.engine._scaled_limit(top_k or self.engine.settings.final_top_k, route),
            coverage_floor,
        )

        # ── Iterative retrieval loop ────────────────────────────────────
        for iteration in range(self.config.max_iterations):
            state.current_iteration = iteration
            # 1.x) Plan / replan
            try:
                await self.agent_planner.act(state, use_llm=self.config.planner_llm_enabled)
            except Exception as exc:
                logger.warning("Planner failed — deterministic fallback used", extra={"error": str(exc)})
                if state.route is not None:
                    plan = self.planner.build(
                        query=state.resolved_query or state.query,
                        route=state.route,
                        material_count=len(state.expected_material_ids),
                    )
                    state.plan_type = plan.plan_type
                    state.sub_questions = list(plan.sub_questions)
                    state.use_graph = plan.use_graph
                    state.use_per_source = plan.use_per_source
                    state.use_multi_query = plan.use_multi_query
                    state.requires_coverage = plan.requires_coverage
            self._record_planning_step(state)
            await self._emit_step(state.steps_history[-1], on_step)

            # 2.x) Retrieve via routed tools
            try:
                await self.agent_director.act(state, limit=retrieval_limit)
            except Exception as exc:
                logger.error(
                    "Director retrieval failed",
                    exc_info=True,
                    extra={"owner_id": scope.owner_id, "error": str(exc)},
                )
                state.last_error = str(exc)
                state.record_step(AgentTraceStep(name="retrieve_text", status="failed", warning="retrieval_failed"))
                await self._emit_step(state.steps_history[-1], on_step)
                return self._build_refusal_response(
                    state, processed=processed, reason=PUBLIC_RETRIEVAL_ERROR,
                )
            state.record_step(
                AgentTraceStep(
                    name="retrieve_evidence",
                    status="completed" if state.raw_evidence else "skipped",
                    tool="retriever_director",
                    evidence_count=len(state.raw_evidence),
                    metadata={"iteration": iteration, "queries": len(state.retrieval_queries)},
                )
            )
            await self._emit_step(state.steps_history[-1], on_step)

            # 3.x) CRAG triage
            await self.agent_crag.act(state)
            state.record_step(
                AgentTraceStep(
                    name="crag_triage",
                    status="completed" if state.cleaned_evidence else "skipped",
                    tool="crag_critic",
                    evidence_count=len(state.cleaned_evidence),
                    warning="weak_evidence" if state.needs_more_evidence() else None,
                    metadata={
                        "correct": sum(1 for v in state.crag_verdicts if v.label.value == "correct"),
                        "ambiguous": sum(1 for v in state.crag_verdicts if v.label.value == "ambiguous"),
                        "incorrect": sum(1 for v in state.crag_verdicts if v.label.value == "incorrect"),
                    },
                )
            )
            await self._emit_step(state.steps_history[-1], on_step)

            # Coverage report for tracing
            state.coverage = await self._coverage_report(
                expected_material_ids=state.expected_material_ids, chunks=state.cleaned_evidence,
            )

            if not state.needs_more_evidence() or iteration + 1 >= self.config.max_iterations:
                break
            # Otherwise: loop back; PlannerAgent.act will read critic_warnings
            # and append targeted sub-questions on the next iteration.
            state.repair_attempted = True

        # ── Rerank cleaned evidence + finalise context ──────────────────
        candidates = state.cleaned_evidence or state.raw_evidence
        if candidates:
            # Always rerank: HybridTextSearchTool only calls retriever.retrieve()
            # (RRF fusion, no cross-encoder). Skipping here means raw fused-score
            # order becomes final context — reranker is the main quality signal.
            reranked = await self._arerank(
                query=state.resolved_query,
                queries=state.retrieval_queries,
                chunks=candidates,
                limit=final_limit,
                use_mmr=route.use_mmr,
            )
            reranked = self._ensure_context_coverage(
                selected=reranked, candidates=candidates,
                expected_material_ids=state.expected_material_ids, limit=final_limit,
            )
            state.context_chunks = self.engine._pack_context_chunks(reranked)
        else:
            state.context_chunks = []
        state.coverage = await self._coverage_report(
            expected_material_ids=state.expected_material_ids, chunks=state.context_chunks,
        )
        state.record_step(
            AgentTraceStep(name="rerank_evidence", status="completed", evidence_count=len(state.context_chunks))
        )
        await self._emit_step(state.steps_history[-1], on_step)

        # ── Confidence + early refusal ──────────────────────────────────
        state.confidence_score = self.engine.confidence_scorer.score(state.context_chunks)
        should_refuse, refusal_reason = self.engine.confidence_scorer.should_refuse(
            chunks=state.context_chunks, confidence=state.confidence_score,
        )
        if route.route_type == RouteType.SUMMARIZATION and state.context_chunks:
            should_refuse = False
            if refusal_reason not in (None, "partial_confidence"):
                refusal_reason = None
        state.citations = self.engine.response_parser.citations_from_chunks(
            state.context_chunks, focus_text=state.resolved_query,
        )
        if should_refuse:
            state.was_refused = True
            state.refusal_reason = refusal_reason
            return await self._build_query_response(state, processed=processed, route=route, answer=REFUSAL_ANSWER)

        # ── Synthesize draft ────────────────────────────────────────────
        try:
            await self.agent_synthesizer.act(state, mode="draft")
        except Exception as exc:
            logger.error(
                "Synthesizer failed",
                exc_info=True,
                extra={"owner_id": scope.owner_id, "error": str(exc)},
            )
            state.last_error = str(exc)
            return self._build_refusal_response(state, processed=processed, reason=PUBLIC_GENERATION_ERROR)

        answer = state.draft_answer.strip()
        if not answer:
            state.was_refused = True
            state.refusal_reason = "LLM returned an empty grounded answer"
            return await self._build_query_response(state, processed=processed, route=route, answer=REFUSAL_ANSWER)
        answer = self._clean_answer_text(
            answer,
            state.context_chunks,
            state.answer_language or processed.answer_language,
        )
        answer = self.engine.response_parser.inject_citations(answer, state.context_chunks)
        state.draft_answer = answer
        state.final_answer = answer
        state.record_step(
            AgentTraceStep(
                name="synthesize_answer", status="completed", tool="synthesizer",
                evidence_count=len(state.context_chunks),
            )
        )
        await self._emit_step(state.steps_history[-1], on_step)

        # ── Guardrails / NLI verification ───────────────────────────────
        await self.agent_guardrails.act(state)
        guard = state.guardrail_report
        grounding_failed = (guard.unsupported_sentence_count > 0) or (guard.invalid_citation_count > 0)
        verdict_failed = guard.verdict in {ClaimVerdict.CONTRADICTED.value, ClaimVerdict.NOT_ENOUGH_EVIDENCE.value}
        if state.context_chunks and (grounding_failed or verdict_failed):
            state.answer_repair_attempted = True
            repaired = await self._repair_answer(
                query=state.resolved_query,
                answer=state.final_answer,
                chunks=state.context_chunks,
                answer_language=processed.answer_language,
                warning=", ".join(
                    item for item in [
                        f"{guard.unsupported_sentence_count} unsupported sentences" if guard.unsupported_sentence_count else "",
                        f"{guard.invalid_citation_count} invalid citations" if guard.invalid_citation_count else "",
                        guard.verdict,
                    ] if item
                ),
            )
            if repaired.strip():
                repaired = self._clean_answer_text(
                    repaired,
                    state.context_chunks,
                    state.answer_language or processed.answer_language,
                )
                state.final_answer = self.engine.response_parser.inject_citations(repaired, state.context_chunks)
                await self.agent_guardrails.act(state)
                guard = state.guardrail_report
            state.record_step(
                AgentTraceStep(
                    name="repair_answer", status="completed", tool="synthesizer",
                    evidence_count=len(state.context_chunks),
                    warning=None if not (guard.unsupported_sentence_count or guard.invalid_citation_count)
                    else "Grounding issues remain after repair.",
                    metadata={
                        "unsupported_sentence_count": guard.unsupported_sentence_count,
                        "invalid_citation_count": guard.invalid_citation_count,
                    },
                )
            )
            await self._emit_step(state.steps_history[-1], on_step)

        # ── Final adjudication based on guardrail verdict ───────────────
        warning: str | None = guard.warning
        if guard.verdict == ClaimVerdict.CONTRADICTED.value:
            should_refuse = True
            refusal_reason = f"claim_verification_{guard.verdict}"
            state.final_answer = REFUSAL_ANSWER
            warning = "Answer appears to conflict with retrieved evidence."
        elif guard.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE.value:
            if route.route_type == RouteType.CLAIM_CHECK:
                should_refuse = True
                refusal_reason = f"claim_verification_{guard.verdict}"
                state.final_answer = REFUSAL_ANSWER
            else:
                _lang = getattr(state, "answer_language", None) or self.engine.settings.inference_default_answer_language
                state.final_answer = state.final_answer + self.engine.settings.messages_low_confidence_warning.get(_lang, self.engine.settings.messages_low_confidence_warning.get("vi", ""))
                warning = "Evidence may not directly support every claim."
        elif refusal_reason == "partial_confidence":
            _lang = getattr(state, "answer_language", None) or self.engine.settings.inference_default_answer_language
            state.final_answer = state.final_answer + self.engine.settings.messages_partial_confidence_warning.get(_lang, self.engine.settings.messages_partial_confidence_warning.get("vi", ""))
            warning = "Answer is based on limited-confidence evidence."

        if (
            not should_refuse and state.citations and state.answer_repair_attempted
            and (guard.unsupported_sentence_count or guard.invalid_citation_count)
            and guard.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE.value
        ):
            should_refuse = True
            _lang = getattr(state, "answer_language", None) or self.engine.settings.inference_default_answer_language
            refusal_reason = self.engine.settings.messages_insufficient_evidence_refusal.get(_lang, self.engine.settings.messages_insufficient_evidence_refusal.get("vi", ""))
            state.final_answer = REFUSAL_ANSWER

        state.record_step(
            AgentTraceStep(
                name="verify_claims", status="completed", tool="guardrails",
                evidence_count=sum(len(chunk.evidence) for chunk in state.context_chunks),
                warning=warning,
                metadata={
                    "verdict": guard.verdict,
                    "unsupported_sentence_count": guard.unsupported_sentence_count,
                    "invalid_citation_count": guard.invalid_citation_count,
                    "answer_repair_attempted": state.answer_repair_attempted,
                },
            )
        )
        await self._emit_step(state.steps_history[-1], on_step)

        # ── Legacy critic loop (refine via follow-up retrieval) ─────────
        if (
            getattr(self.engine.settings, "agentic_critic_enabled", True)
            and not should_refuse
            and state.final_answer.strip()
            and state.final_answer != REFUSAL_ANSWER
            and self.agent_critic.should_fire(confidence=state.confidence_score, route_type=route.route_type.value)
        ):
            critic_verdict = await self.agent_critic.run(
                query=state.resolved_query, answer=state.final_answer, context_chunks=state.context_chunks,
            )
            state.record_step(
                AgentTraceStep(
                    name="critic_review", status="completed", tool="critic",
                    warning=critic_verdict.verdict,
                    metadata={
                        "verdict": critic_verdict.verdict,
                        "reason": critic_verdict.reason,
                        "follow_ups": critic_verdict.follow_up_queries,
                    },
                )
            )
            await self._emit_step(state.steps_history[-1], on_step)
            if critic_verdict.verdict == "refine" and critic_verdict.follow_up_queries:
                extras: list[RetrievedChunk] = []
                max_follow_ups = int(getattr(self.engine.settings, "agentic_critic_max_follow_ups", 2))
                for fq in critic_verdict.follow_up_queries[:max_follow_ups]:
                    result = await self.text_tool.run(query=fq, scope=scope, limit=4)
                    if result.success and result.data:
                        extras.extend(result.data)
                if extras:
                    context_buffer = int(getattr(self.engine.settings, "agentic_critic_context_buffer", 4))
                    augmented = dedupe_retrieved_chunks(state.context_chunks + extras)[: final_limit + context_buffer]
                    state.context_chunks = augmented
                    await self.agent_synthesizer.act(state, mode="repair")
                    if state.final_answer.strip():
                        state.final_answer = self._clean_answer_text(
                            state.final_answer,
                            augmented,
                            state.answer_language or processed.answer_language,
                        )
                        state.final_answer = self.engine.response_parser.inject_citations(
                            state.final_answer, augmented,
                        )
                        state.record_step(
                            AgentTraceStep(
                                name="critic_refined_synthesis", status="completed",
                                tool="synthesizer", evidence_count=len(augmented),
                                metadata={"added_chunks": len(extras)},
                            )
                        )
                        await self._emit_step(state.steps_history[-1], on_step)

        # ── Sentence-level evidence coverage (SLEC) ─────────────────────
        # Drops sentences whose rerank-vs-chunk score falls below
        # `slec_partial_threshold` and refuses the whole answer when the
        # weighted coverage ratio is below `slec_refuse_below`. This is the
        # only safeguard against the LLM stitching a fluent answer from
        # chunks that merely mention the query topic without grounding it.
        if (
            not should_refuse
            and bool(getattr(self.engine.settings, "slec_enabled", False))
            and state.context_chunks
            and state.final_answer.strip()
            and state.final_answer != REFUSAL_ANSWER
        ):
            try:
                slec_answer, slec_report = await self.engine.sentence_coverage_gate.verify(
                    answer=state.final_answer,
                    chunks=state.context_chunks,
                    route_type=route.route_type.value,
                )
                state.sentence_coverage_report = slec_report
                if slec_report and slec_report.refused:
                    should_refuse = True
                    refusal_reason = "slec_coverage_below_floor"
                    state.final_answer = REFUSAL_ANSWER
                elif slec_report and slec_report.dropped_count > 0:
                    state.final_answer = self.engine.response_parser.inject_citations(
                        slec_answer, state.context_chunks,
                    )
                else:
                    state.final_answer = slec_answer
            except Exception as exc:
                logger.warning("SLEC gate failed in agentic — keeping original answer", extra={"error": str(exc)})

        state.was_refused = should_refuse
        state.refusal_reason = refusal_reason
        return await self._build_query_response(
            state, processed=processed, route=route,
            answer=state.final_answer if state.final_answer.strip() else REFUSAL_ANSWER,
            warning=warning,
        )

    # ── Helpers ─────────────────────────────────────────────────────────
    def _record_planning_step(self, state: AgentState) -> None:
        state.record_step(
            AgentTraceStep(
                name="plan_query",
                status="completed",
                query=state.query,
                tool="planner",
                sources_requested=len(state.expected_material_ids) or None,
                warning=state.plan_type,
                metadata={
                    "sub_question_count": len(state.sub_questions),
                    "iteration": state.current_iteration,
                    "llm_planner_enabled": self.config.planner_llm_enabled,
                    "sub_questions": [item.model_dump() for item in state.sub_questions],
                },
            )
        )

    async def _maybe_resolve_anaphora(self, *, query: str, memory_context: str | None, scope: RetrievalScope) -> str:
        if not self.config.anaphora_resolution_enabled or not memory_context:
            return query
        if not memory_context.strip() or not self._anaphora_re.search(query):
            return query
        prompt = (
            "Resolve any pronouns or vague references in the current question using the conversation history.\n"
            "Return ONLY the rewritten, self-contained question. If nothing needs resolving, return the original exactly.\n\n"
            f"Conversation history:\n{memory_context[:600]}\n\n"
            f"Current question: {query}\n\n"
            "Rewritten question:"
        )
        try:
            resolved = (await self.engine.llm.generate(prompt=prompt)).strip()
            if resolved and len(resolved) <= len(query) * 3:
                logger.info(
                    "Anaphora resolved",
                    extra={
                        "owner_id": scope.owner_id,
                        "original": query[:80],
                        "resolved": resolved[:80],
                    },
                )
                return resolved
        except Exception as exc:
            logger.debug("Anaphora resolution failed", extra={"error": str(exc)})
        return query

    async def _arerank(
        self, *, query: str, queries: list[str], chunks: list[RetrievedChunk],
        limit: int, use_mmr: bool,
    ) -> list[RetrievedChunk]:
        if hasattr(self.engine.reranker, "arerank_multilingual"):
            return await self.engine.reranker.arerank_multilingual(
                queries=queries, chunks=chunks, limit=limit, use_mmr=use_mmr,
            )
        if hasattr(self.engine.reranker, "rerank_multilingual"):
            async with self._rerank_fallback_semaphore:
                return await asyncio.to_thread(
                    self.engine.reranker.rerank_multilingual,
                    queries=queries, chunks=chunks, limit=limit, use_mmr=use_mmr,
                )
        if hasattr(self.engine.reranker, "arerank"):
            return await self.engine.reranker.arerank(query=query, chunks=chunks, limit=limit)
        async with self._rerank_fallback_semaphore:
            return await asyncio.to_thread(self.engine.reranker.rerank, query=query, chunks=chunks, limit=limit)

    async def _repair_answer(
        self, *, query: str, answer: str, chunks: list[RetrievedChunk],
        answer_language: str, warning: str,
    ) -> str:
        evidence_text = self.engine.response_parser.format_evidence_for_prompt(chunks)
        prompt = (
            "Rewrite the answer so it is strictly grounded in the evidence.\n"
            f"Answer language: {answer_language}\n"
            f"Question: {query}\n"
            f"Grounding issue: {warning}\n\n"
            "Rules:\n"
            "- Use only the evidence below.\n"
            "- Keep valid citation markers like [1], [2].\n"
            "- Remove any claim that is not directly supported.\n"
            "- If the evidence is insufficient, say that the documents do not provide enough evidence.\n\n"
            f"Current answer:\n{answer}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        try:
            return await self.engine.llm.generate(prompt=prompt)
        except Exception:
            logger.warning("Agentic answer repair failed", exc_info=True)
            return ""

    @staticmethod
    def _ensure_context_coverage(
        *, selected: list[RetrievedChunk], candidates: list[RetrievedChunk],
        expected_material_ids: list[str], limit: int,
    ) -> list[RetrievedChunk]:
        if not expected_material_ids:
            return selected[:limit]
        result: list[RetrievedChunk] = []
        seen_chunks: set[str] = set()
        seen_materials: set[str] = set()
        for chunk in selected:
            if chunk.chunk_id in seen_chunks:
                continue
            result.append(chunk)
            seen_chunks.add(chunk.chunk_id)
            seen_materials.add(chunk.material_id)
        for material_id in expected_material_ids:
            if len(result) >= limit or material_id in seen_materials:
                continue
            candidate = next(
                (c for c in candidates if c.material_id == material_id and c.chunk_id not in seen_chunks),
                None,
            )
            if candidate is None:
                continue
            result.append(candidate)
            seen_chunks.add(candidate.chunk_id)
            seen_materials.add(candidate.material_id)
        return result[:limit]

    async def _coverage_report(self, *, expected_material_ids: list[str], chunks: list[RetrievedChunk]) -> CoverageReport:
        expected = list(dict.fromkeys(mid for mid in expected_material_ids if mid))
        covered = {chunk.material_id for chunk in chunks if chunk.material_id}
        names = await self._material_names(expected)
        sources = [
            CoverageSource(material_id=mid, name=names.get(mid, mid), covered=mid in covered)
            for mid in expected
        ]
        return CoverageReport(
            requested_count=len(sources),
            covered_count=sum(1 for s in sources if s.covered),
            sources=sources,
        )

    async def _indexed_material_ids_for_scope(self, scope: RetrievalScope) -> list[str]:
        if not scope.collection_id:
            return []
        try:
            collection_oid = PydanticObjectId(scope.collection_id)
        except Exception:
            return []
        try:
            materials = await Material.find(
                Material.owner_id == scope.owner_id,
                Material.collection_id == collection_oid,
                Material.status == PipelineStatus.INDEXED.value,
            ).sort("created_at").to_list()
        except Exception:
            return []
        return [str(material.id) for material in materials if material.id is not None]

    @staticmethod
    async def _material_names(material_ids: list[str]) -> dict[str, str]:
        object_ids: list[PydanticObjectId] = []
        for material_id in material_ids:
            try:
                object_ids.append(PydanticObjectId(material_id))
            except Exception:
                continue
        if not object_ids:
            return {}
        try:
            materials = await Material.find({"_id": {"$in": object_ids}}).to_list()
        except Exception:
            return {}
        return {
            str(material.id): material.original_name or material.filename or str(material.id)
            for material in materials if material.id is not None
        }

    @staticmethod
    async def _emit_step(step: AgentTraceStep, on_step: AgentStepCallback | None) -> None:
        if on_step is None:
            return
        result = on_step(step)
        if inspect.isawaitable(result):
            await result

    # ── Response builders ───────────────────────────────────────────────
    async def _build_query_response(
        self, state: AgentState, *, processed, route, answer: str, warning: str | None = None,
    ) -> QueryResponse:
        trace = AgentTrace(
            plan_type=state.plan_type,
            steps=list(state.steps_history),
            repair_attempted=state.repair_attempted,
            verification=AgentVerification(
                verdict=state.guardrail_report.verdict,
                confidence=state.guardrail_report.confidence,
                warning=warning,
                unsupported_sentence_count=state.guardrail_report.unsupported_sentence_count,
                invalid_citation_count=state.guardrail_report.invalid_citation_count,
                repair_attempted=state.answer_repair_attempted,
            ) if state.guardrail_report.verdict != "not_run" else None,
        )
        # Keep only citations the answer cites (same policy as the direct engine
        # path) — drops off-topic low-rank context, renumbers markers + SLEC refs.
        answer, state.citations, state.sentence_coverage_report, _ = InferenceEngine._prune_to_cited(
            answer, state.citations, state.sentence_coverage_report, chunks=state.context_chunks
        )
        return QueryResponse(
            answer=answer,
            answer_language=processed.answer_language,
            query_language=processed.query_language,
            translated_query=processed.translated_query,
            source_languages=sorted({c.source_language for c in state.citations}),
            citations=state.citations,
            confidence=state.confidence_score,
            was_refused=state.was_refused,
            refusal_reason=state.refusal_reason,
            reasoning_path=await build_reasoning_path(
                query=state.resolved_query,
                retrieved_chunks=state.raw_evidence,
                graph_chunks=state.graph_evidence,
                reranked_chunks=state.context_chunks,
                use_graph=route.use_graph,
            ),
            coverage=state.coverage,
            sentence_coverage=state.sentence_coverage_report,
            agent_trace=trace,
        )

    def _build_refusal_response(self, state: AgentState, *, processed, reason: str) -> QueryResponse:
        state.was_refused = True
        state.refusal_reason = reason
        trace = AgentTrace(
            plan_type=state.plan_type,
            steps=list(state.steps_history),
            repair_attempted=state.repair_attempted,
            verification=None,
        )
        return QueryResponse(
            answer=REFUSAL_ANSWER,
            answer_language=processed.answer_language,
            query_language=processed.query_language,
            translated_query=processed.translated_query,
            source_languages=[],
            citations=[],
            confidence=0.0,
            was_refused=True,
            refusal_reason=reason,
            agent_trace=trace,
        )


# ── Backward-compatible wrapper ────────────────────────────────────────
class AgenticRagService:
    """Thin wrapper preserving the previous public API.

    Existing callers (`QueryService.agentic_rag`) instantiate this with an
    `InferenceEngine` and invoke `.answer(...)` — both behaviours are kept.
    All real work happens in `AgenticCoordinatingEngine`.
    """

    def __init__(self, *, engine: InferenceEngine) -> None:
        self.engine = engine
        self.coordinator = AgenticCoordinatingEngine(engine=engine)
        # Expose individual agents for callers that introspected the old service.
        self.planner = self.coordinator.planner
        self.agent_planner = self.coordinator.agent_planner
        self.agent_director = self.coordinator.agent_director
        self.agent_synthesizer = self.coordinator.agent_synthesizer
        self.agent_critic = self.coordinator.agent_critic
        self.agent_crag = self.coordinator.agent_crag
        self.agent_guardrails = self.coordinator.agent_guardrails

    async def answer(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        top_k: int | None = None,
        answer_language: str | None = None,
        memory_context: str | None = None,
        on_step: AgentStepCallback | None = None,
    ) -> QueryResponse:
        return await self.coordinator.answer(
            query=query, scope=scope, top_k=top_k, answer_language=answer_language,
            memory_context=memory_context, on_step=on_step,
        )


# Module-level constant exports retained for backward-compatibility with
# imports like `from src.agentic.service import REFUSAL_ANSWER`.
__all__ = [
    "AgenticCoordinatingEngine",
    "AgenticRagService",
    "AgentStepCallback",
    "PUBLIC_GENERATION_ERROR",
    "PUBLIC_RETRIEVAL_ERROR",
    "REFUSAL_ANSWER",
]
