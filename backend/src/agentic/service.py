from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from beanie import PydanticObjectId

from src.agentic.planner import AgenticPlanner, AgenticSubQuestion
from src.guardrails.claim_verifier import ClaimVerdict
from src.inference.inference_engine import PUBLIC_GENERATION_ERROR, PUBLIC_RETRIEVAL_ERROR, REFUSAL_ANSWER, InferenceEngine
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
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass(frozen=True)
class EvidenceQualityReport:
    sub_questions_requested: int
    sub_questions_covered: int
    sources_requested: int
    sources_covered: int
    evidence_count: int
    missing_sub_questions: list[str]


@dataclass(frozen=True)
class AnswerGroundingReport:
    unsupported_sentence_count: int
    invalid_citation_count: int

    @property
    def passed(self) -> bool:
        return self.unsupported_sentence_count == 0 and self.invalid_citation_count == 0


class AgenticRagService:
    """MVP agentic orchestration around the existing RAG engine."""

    def __init__(self, *, engine: InferenceEngine) -> None:
        self.engine = engine
        self.planner = AgenticPlanner()
        self._rerank_fallback_semaphore = asyncio.Semaphore(1)

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

        # Anaphora resolution: rewrite pronouns/vague references using conversation context
        resolved_query = query
        if getattr(self.engine.settings, "agentic_anaphora_resolution_enabled", True) and memory_context:
            resolved_query = await self._resolve_anaphora(query=query, memory_context=memory_context)
            if resolved_query != query:
                logger.info(
                    "Anaphora resolved",
                    extra={"original": query[:80], "resolved": resolved_query[:80], "owner_id": scope.owner_id},
                )

        route = self.engine.query_router.route(resolved_query)
        expected_material_ids = scope.material_ids or await self._indexed_material_ids_for_scope(scope)
        material_count = len(expected_material_ids)
        if getattr(self.engine.settings, "agentic_planner_llm_enabled", False):
            plan = await self.planner.build_with_llm(
                query=resolved_query, route=route, material_count=material_count, llm=self.engine.llm
            )
        else:
            plan = self.planner.build(query=resolved_query, route=route, material_count=material_count)
        steps: list[AgentTraceStep] = []
        await self._record_step(
            steps,
            AgentTraceStep(
                name="plan_query",
                status="completed",
                query=query,
                tool="planner",
                sources_requested=material_count or None,
                warning=plan.plan_type,
                metadata={
                    "sub_question_count": len(plan.sub_questions),
                    "llm_planner_enabled": bool(getattr(self.engine.settings, "agentic_planner_llm_enabled", False)),
                    "sub_questions": [item.model_dump() for item in plan.sub_questions],
                },
            ),
            on_step,
        )

        retrieval_limit = self.engine._scaled_limit(self.engine.settings.rerank_input_k, route)
        final_limit = max(self.engine._scaled_limit(top_k or self.engine.settings.final_top_k, route), material_count or 1)
        processed = await self.engine.query_processor.process_async(
            resolved_query,
            answer_language=answer_language,
        )
        original_query = getattr(processed, "original_query", resolved_query)
        use_multi_query = route.use_multi_query and bool(getattr(self.engine.settings, "multi_query_enabled", False))
        retrieval_queries = processed.retrieval_queries if use_multi_query else [original_query]

        retrieved: list[RetrievedChunk] = []
        graph_chunks: list[RetrievedChunk] = []
        try:
            retrieved = await self._retrieve_multi_query(
                queries=retrieval_queries,
                scope=scope,
                limit=retrieval_limit,
                steps=steps,
                step_name="retrieve_multi_query" if use_multi_query else "retrieve_text",
                on_step=on_step,
            )
            if plan.use_per_source and expected_material_ids:
                retrieved.extend(
                    await self._retrieve_per_source(
                        query=resolved_query,
                        scope=scope,
                        material_ids=expected_material_ids,
                        limit=max(1, min(3, retrieval_limit)),
                        steps=steps,
                        on_step=on_step,
                    )
                )
            if plan.sub_questions:
                retrieved.extend(
                    await self._retrieve_sub_questions(
                        sub_questions=plan.sub_questions,
                        scope=scope,
                        limit=max(1, min(3, retrieval_limit)),
                        steps=steps,
                        on_step=on_step,
                    )
                )
            if plan.use_graph:
                graph_chunks = await self._trace_graph(query=resolved_query, scope=scope, steps=steps, priority=route.graph_priority, on_step=on_step)
        except Exception as exc:
            logger.error("Agentic retrieval failed", exc_info=True, extra={"owner_id": scope.owner_id, "error": str(exc)})
            failure_step = AgentTraceStep(name="retrieve_text", status="failed", warning="retrieval_failed")
            trace = AgentTrace(
                plan_type=plan.plan_type,
                steps=[*steps, failure_step],
                repair_attempted=False,
                verification=None,
            )
            await self._emit_step(failure_step, on_step)
            return QueryResponse(
                answer=REFUSAL_ANSWER,
                answer_language=processed.answer_language,
                query_language=processed.query_language,
                translated_query=processed.translated_query,
                source_languages=[],
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason=PUBLIC_RETRIEVAL_ERROR,
                agent_trace=trace,
            )

        candidates = dedupe_retrieved_chunks((graph_chunks + retrieved) if route.graph_priority else (retrieved + graph_chunks))
        coverage = await self._coverage_report(expected_material_ids=expected_material_ids, chunks=candidates)
        repair_attempted = False
        if plan.requires_coverage and coverage.covered_count < coverage.requested_count:
            repair_attempted = True
            missing = [source.material_id for source in coverage.sources if not source.covered]
            repaired = await self._retrieve_per_source(
                query=resolved_query,
                scope=scope,
                material_ids=missing,
                limit=max(2, min(4, retrieval_limit)),
                steps=steps,
                step_name="repair_retrieval",
                on_step=on_step,
            )
            candidates = dedupe_retrieved_chunks([*candidates, *repaired])
            coverage = await self._coverage_report(expected_material_ids=expected_material_ids, chunks=candidates)
        elif plan.requires_coverage:
            await self._record_step(
                steps,
                AgentTraceStep(
                    name="verify_coverage",
                    status="completed",
                    sources_requested=coverage.requested_count,
                    sources_covered=coverage.covered_count,
                    evidence_count=len(candidates),
                ),
                on_step,
            )

        # Iterative retrieval with reflection: retry up to N times when sub-questions are still uncovered
        max_iterations = max(1, getattr(self.engine.settings, "agentic_max_retrieval_iterations", 2))
        evidence_quality = self._evidence_quality_report(
            sub_questions=plan.sub_questions,
            expected_material_ids=expected_material_ids,
            chunks=candidates,
        )
        for iteration in range(max_iterations):
            if not evidence_quality.missing_sub_questions:
                break
            if iteration == 0:
                # First pass: re-retrieve the missing sub-questions directly
                repair_attempted = True
                repaired = await self._retrieve_sub_questions(
                    sub_questions=[
                        AgenticSubQuestion(text=text, tool="retrieve_text")
                        for text in evidence_quality.missing_sub_questions[:3]
                    ],
                    scope=scope,
                    limit=max(1, min(3, retrieval_limit)),
                    steps=steps,
                    step_name="repair_retrieval",
                    on_step=on_step,
                )
            else:
                # Subsequent passes: LLM reflection → refined queries
                refined = await self._reflect_on_gaps(
                    query=resolved_query,
                    missing_sub_questions=evidence_quality.missing_sub_questions,
                    chunks=candidates,
                )
                if not refined:
                    break
                repair_attempted = True
                repaired = await self._retrieve_multi_query(
                    queries=refined,
                    scope=scope,
                    limit=max(1, min(3, retrieval_limit)),
                    steps=steps,
                    step_name=f"iterative_retrieval_{iteration + 1}",
                    on_step=on_step,
                )
            candidates = dedupe_retrieved_chunks([*candidates, *repaired])
            coverage = await self._coverage_report(expected_material_ids=expected_material_ids, chunks=candidates)
            evidence_quality = self._evidence_quality_report(
                sub_questions=plan.sub_questions,
                expected_material_ids=expected_material_ids,
                chunks=candidates,
            )
        await self._record_step(
            steps,
            AgentTraceStep(
                name="verify_evidence_quality",
                status="completed",
                tool="quality_gate",
                sources_requested=evidence_quality.sources_requested or None,
                sources_covered=evidence_quality.sources_covered or None,
                evidence_count=evidence_quality.evidence_count,
                warning="Some sub-questions have weak evidence." if evidence_quality.missing_sub_questions else None,
                metadata={
                    "sub_questions_requested": evidence_quality.sub_questions_requested,
                    "sub_questions_covered": evidence_quality.sub_questions_covered,
                    "missing_sub_questions": evidence_quality.missing_sub_questions,
                },
            ),
            on_step,
        )

        reranked = await self._arerank(
            query=resolved_query,
            queries=retrieval_queries,
            chunks=candidates,
            limit=final_limit,
            use_mmr=route.use_mmr,
        )
        reranked = self._ensure_context_coverage(selected=reranked, candidates=candidates, expected_material_ids=expected_material_ids, limit=final_limit)
        context_chunks = self.engine._pack_context_chunks(reranked)
        coverage = await self._coverage_report(expected_material_ids=expected_material_ids, chunks=context_chunks)
        await self._record_step(
            steps,
            AgentTraceStep(name="rerank_evidence", status="completed", evidence_count=len(context_chunks)),
            on_step,
        )
        confidence = self.engine.confidence_scorer.score(reranked)
        should_refuse, refusal_reason = self.engine.confidence_scorer.should_refuse(chunks=reranked, confidence=confidence)

        if route.route_type == RouteType.SUMMARIZATION and reranked:
            should_refuse = False
            if refusal_reason not in (None, "partial_confidence"):
                refusal_reason = None

        citations = self.engine.response_parser.citations_from_chunks(context_chunks, focus_text=resolved_query)
        if should_refuse:
            trace = AgentTrace(plan_type=plan.plan_type, steps=steps, repair_attempted=repair_attempted)
            return QueryResponse(
                answer=REFUSAL_ANSWER,
                answer_language=processed.answer_language,
                query_language=processed.query_language,
                translated_query=processed.translated_query,
                source_languages=sorted({citation.source_language for citation in citations}),
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=refusal_reason,
                reasoning_path=build_reasoning_path(query=resolved_query, retrieved_chunks=retrieved, graph_chunks=graph_chunks, reranked_chunks=reranked, use_graph=route.use_graph),
                coverage=coverage,
                agent_trace=trace,
            )

        prompt = self.engine._build_prompt(
            query=resolved_query,
            chunks=context_chunks,
            answer_language=processed.answer_language,
            memory_context=memory_context or "",
            route_type=route.route_type,
            plan_type=plan.plan_type,
        )
        try:
            answer = await self.engine.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.error("Agentic LLM generation failed", exc_info=True, extra={"owner_id": scope.owner_id, "error": str(exc)})
            trace = AgentTrace(plan_type=plan.plan_type, steps=steps, repair_attempted=repair_attempted)
            return QueryResponse(
                answer=REFUSAL_ANSWER,
                answer_language=processed.answer_language,
                query_language=processed.query_language,
                translated_query=processed.translated_query,
                source_languages=sorted({citation.source_language for citation in citations}),
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=PUBLIC_GENERATION_ERROR,
                reasoning_path=build_reasoning_path(query=resolved_query, retrieved_chunks=retrieved, graph_chunks=graph_chunks, reranked_chunks=reranked, use_graph=route.use_graph),
                coverage=coverage,
                agent_trace=trace,
            )

        if not answer.strip():
            answer = REFUSAL_ANSWER
            should_refuse = True
            refusal_reason = "LLM returned an empty grounded answer"
        else:
            answer = self.engine.response_parser.inject_citations(answer, context_chunks)

        await self._record_step(
            steps,
            AgentTraceStep(name="synthesize_answer", status="completed", tool="llm", evidence_count=len(context_chunks)),
            on_step,
        )
        grounding = self._grounding_report(answer=answer, citation_count=len(citations))
        verification = await self._verify_claim(
            claim=answer,
            evidence=[evidence for chunk in context_chunks for evidence in chunk.evidence],
        )
        answer_repair_attempted = False
        if (
            context_chunks
            and not should_refuse
            and (not grounding.passed or verification.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.NOT_ENOUGH_EVIDENCE})
        ):
            answer_repair_attempted = True
            repaired_answer = await self._repair_answer(
                query=resolved_query,
                answer=answer,
                chunks=context_chunks,
                answer_language=processed.answer_language,
                warning=", ".join(
                    item
                    for item in [
                        f"{grounding.unsupported_sentence_count} unsupported sentences" if grounding.unsupported_sentence_count else "",
                        f"{grounding.invalid_citation_count} invalid citations" if grounding.invalid_citation_count else "",
                        verification.verdict.value,
                    ]
                    if item
                ),
            )
            if repaired_answer.strip():
                answer = self.engine.response_parser.inject_citations(repaired_answer, context_chunks)
                grounding = self._grounding_report(answer=answer, citation_count=len(citations))
                verification = await self._verify_claim(
                    claim=answer,
                    evidence=[evidence for chunk in context_chunks for evidence in chunk.evidence],
                )
            await self._record_step(
                steps,
                AgentTraceStep(
                    name="repair_answer",
                    status="completed",
                    tool="llm",
                    evidence_count=len(context_chunks),
                    warning=None if grounding.passed else "Grounding issues remain after repair.",
                    metadata={
                        "unsupported_sentence_count": grounding.unsupported_sentence_count,
                        "invalid_citation_count": grounding.invalid_citation_count,
                    },
                ),
                on_step,
            )

        warning = None
        if verification.verdict == ClaimVerdict.CONTRADICTED:
            warning = "Answer appears to conflict with retrieved evidence."
            answer = answer + "\n\n> Cảnh báo: Phát hiện mâu thuẫn giữa câu trả lời và bằng chứng gốc."
        elif verification.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE:
            warning = "Evidence may not directly support every claim."
            answer = answer + "\n\n> Cảnh báo: Một số nhận định chưa được bằng chứng hỗ trợ trực tiếp."
        elif refusal_reason == "partial_confidence":
            warning = "Answer is based on limited-confidence evidence."
            answer = answer + "\n\n> Cảnh báo: Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra nguồn gốc."

        if verification.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.NOT_ENOUGH_EVIDENCE}:
            should_refuse = True
            refusal_reason = f"claim_verification_{verification.verdict.value}"
            answer = REFUSAL_ANSWER

        if not should_refuse and citations and answer_repair_attempted and not grounding.passed and verification.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE:
            should_refuse = True
            refusal_reason = "Không đủ bằng chứng đáng tin cậy để tạo câu trả lời có citation hợp lệ."
            answer = REFUSAL_ANSWER

        await self._record_step(
            steps,
            AgentTraceStep(
                name="verify_claims",
                status="completed",
                tool="claim_verifier",
                evidence_count=sum(len(chunk.evidence) for chunk in context_chunks),
                warning=warning,
                metadata={
                    "unsupported_sentence_count": grounding.unsupported_sentence_count,
                    "invalid_citation_count": grounding.invalid_citation_count,
                    "answer_repair_attempted": answer_repair_attempted,
                },
            ),
            on_step,
        )
        trace = AgentTrace(
            plan_type=plan.plan_type,
            steps=steps,
            repair_attempted=repair_attempted,
            verification=AgentVerification(
                verdict=verification.verdict.value,
                confidence=verification.confidence,
                warning=warning,
                unsupported_sentence_count=grounding.unsupported_sentence_count,
                invalid_citation_count=grounding.invalid_citation_count,
                repair_attempted=answer_repair_attempted,
            ),
        )

        return QueryResponse(
            answer=answer,
            answer_language=processed.answer_language,
            query_language=processed.query_language,
            translated_query=processed.translated_query,
            source_languages=sorted({citation.source_language for citation in citations}),
            citations=citations,
            confidence=confidence,
            was_refused=should_refuse,
            refusal_reason=refusal_reason,
            reasoning_path=build_reasoning_path(query=resolved_query, retrieved_chunks=retrieved, graph_chunks=graph_chunks, reranked_chunks=reranked, use_graph=route.use_graph),
            coverage=coverage,
            agent_trace=trace,
        )

    async def _retrieve_multi_query(
        self,
        *,
        queries: list[str],
        scope: RetrievalScope,
        limit: int,
        steps: list[AgentTraceStep],
        step_name: str,
        on_step: AgentStepCallback | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        tasks = [self.engine.retriever.retrieve(query=item, scope=scope, limit=limit) for item in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        chunks: list[RetrievedChunk] = []
        warnings: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                warnings.append("retrieval_failed")
                continue
            chunks.extend(result)
        await self._record_step(
            steps,
            AgentTraceStep(
                name=step_name,
                status="completed" if chunks else "failed",
                query=" | ".join(queries[:3]),
                tool="retriever",
                duration_ms=self._elapsed_ms(started),
                evidence_count=len(chunks),
                warning=", ".join(warnings) or None,
                metadata={"query_count": len(queries)},
            ),
            on_step,
        )
        return chunks

    async def _retrieve_per_source(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        material_ids: list[str],
        limit: int,
        steps: list[AgentTraceStep],
        step_name: str = "retrieve_per_source",
        on_step: AgentStepCallback | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        scopes = [
            RetrievalScope(owner_id=scope.owner_id, collection_id=scope.collection_id, material_ids=[mid])
            for mid in material_ids
        ]
        results = await asyncio.gather(
            *[self.engine.retriever.retrieve(query=query, scope=s, limit=limit) for s in scopes],
            return_exceptions=True,
        )
        chunks: list[RetrievedChunk] = []
        for result in results:
            if not isinstance(result, Exception):
                chunks.extend(result)
        covered = len({chunk.material_id for chunk in chunks})
        await self._record_step(
            steps,
            AgentTraceStep(
                name=step_name,
                status="completed" if chunks else "skipped",
                query=query,
                tool="retriever",
                duration_ms=self._elapsed_ms(started),
                sources_requested=len(material_ids),
                sources_covered=covered,
                evidence_count=len(chunks),
                warning=None if covered == len(material_ids) else "Some sources returned no evidence.",
            ),
            on_step,
        )
        return chunks

    async def _retrieve_sub_questions(
        self,
        *,
        sub_questions: list[AgenticSubQuestion],
        scope: RetrievalScope,
        limit: int,
        steps: list[AgentTraceStep],
        step_name: str = "retrieve_sub_questions",
        on_step: AgentStepCallback | None = None,
    ) -> list[RetrievedChunk]:
        # Only route text-retrieval sub-questions here; per-source and graph are
        # handled by dedicated steps (_retrieve_per_source / _trace_graph).
        text_only = [item for item in sub_questions if item.tool == "retrieve_text"]
        if not text_only:
            await self._record_step(
                steps,
                AgentTraceStep(name=step_name, status="skipped", tool="retriever", evidence_count=0,
                               warning="All sub-questions use dedicated retrieval steps."),
                on_step,
            )
            return []
        started = time.perf_counter()
        tasks = [self.engine.retriever.retrieve(query=item.text, scope=scope, limit=limit) for item in text_only]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        chunks: list[RetrievedChunk] = []
        covered: list[str] = []
        warnings: list[str] = []
        for sub_question, result in zip(text_only, results, strict=False):
            if isinstance(result, Exception):
                warnings.append(f"{sub_question.text}: retrieval_failed")
                continue
            if result:
                covered.append(sub_question.text)
                chunks.extend(result)
        await self._record_step(
            steps,
            AgentTraceStep(
                name=step_name,
                status="completed" if chunks else "skipped",
                tool="retriever",
                duration_ms=self._elapsed_ms(started),
                evidence_count=len(chunks),
                warning=", ".join(warnings) or None,
                metadata={
                    "sub_questions_requested": len(text_only),
                    "sub_questions_covered": len(covered),
                    "covered_sub_questions": covered,
                },
            ),
            on_step,
        )
        return chunks

    async def _trace_graph(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        steps: list[AgentTraceStep],
        priority: bool,
        on_step: AgentStepCallback | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        try:
            graph_paths = await self.engine.graph_retriever.retrieve_paths(query=query, scope=scope)
            chunks = self.engine._chunks_from_graph_paths(graph_paths, scope=scope, priority=priority)
        except Exception as exc:
            await self._record_step(
                steps,
                AgentTraceStep(name="trace_graph", status="failed", query=query, tool="graph_retriever", duration_ms=self._elapsed_ms(started), warning="graph_retrieval_failed"),
                on_step,
            )
            return []
        await self._record_step(
            steps,
            AgentTraceStep(name="trace_graph", status="completed", query=query, tool="graph_retriever", duration_ms=self._elapsed_ms(started), evidence_count=len(chunks)),
            on_step,
        )
        return chunks

    @classmethod
    async def _record_step(cls, steps: list[AgentTraceStep], step: AgentTraceStep, on_step: AgentStepCallback | None) -> None:
        steps.append(step)
        await cls._emit_step(step, on_step)

    @staticmethod
    async def _emit_step(step: AgentTraceStep, on_step: AgentStepCallback | None) -> None:
        if on_step is None:
            return
        result = on_step(step)
        if inspect.isawaitable(result):
            await result

    async def _repair_answer(
        self,
        *,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        answer_language: str,
        warning: str,
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

    async def _verify_claim(self, *, claim: str, evidence: list) -> object:
        verifier = self.engine.claim_verifier
        if hasattr(verifier, "averify"):
            return await verifier.averify(claim=claim, evidence=evidence)
        result = verifier.verify(claim=claim, evidence=evidence)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _evidence_quality_report(
        *,
        sub_questions: list[AgenticSubQuestion],
        expected_material_ids: list[str],
        chunks: list[RetrievedChunk],
    ) -> EvidenceQualityReport:
        covered_materials = {chunk.material_id for chunk in chunks if chunk.material_id}
        missing_sub_questions: list[str] = []
        covered_sub_questions = 0
        for sub_question in sub_questions:
            if not sub_question.critical:
                continue
            if AgenticRagService._has_text_overlap(sub_question.text, chunks):
                covered_sub_questions += 1
            else:
                missing_sub_questions.append(sub_question.text)
        critical_count = sum(1 for item in sub_questions if item.critical)
        return EvidenceQualityReport(
            sub_questions_requested=critical_count,
            sub_questions_covered=covered_sub_questions,
            sources_requested=len(expected_material_ids),
            sources_covered=len([material_id for material_id in expected_material_ids if material_id in covered_materials]),
            evidence_count=len(chunks),
            missing_sub_questions=missing_sub_questions,
        )

    @staticmethod
    def _grounding_report(*, answer: str, citation_count: int) -> AnswerGroundingReport:
        if citation_count <= 0 or not answer.strip() or answer == REFUSAL_ANSWER:
            return AnswerGroundingReport(unsupported_sentence_count=0, invalid_citation_count=0)
        markers = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
        invalid = sum(1 for m in markers if m < 1 or m > citation_count)
        # Paragraph-level grounding: LLMs write multi-sentence blocks ending with [N].
        # Flagging each sentence individually causes too many false positives.
        # Only count a paragraph as "unsupported" if it has ≥2 substantive sentences
        # AND no citation marker anywhere in the paragraph.
        paragraphs = re.split(r"\n\s*\n", answer)
        unsupported = 0
        for para in paragraphs:
            if _CITATION_RE.search(para):
                continue
            sentences = [
                s.strip() for s in _SENTENCE_RE.findall(para)
                if len(s.strip()) >= 12 and not s.strip().startswith(">")
            ]
            if len(sentences) >= 2:
                unsupported += 1
        return AnswerGroundingReport(unsupported_sentence_count=unsupported, invalid_citation_count=invalid)

    @staticmethod
    def _has_text_overlap(query: str, chunks: list[RetrievedChunk]) -> bool:
        # Use rerank_score as semantic coverage signal (set by reranker).
        # Fall back to lexical only when rerank_score unavailable.
        for chunk in chunks:
            if chunk.rerank_score is not None:
                if chunk.rerank_score >= 0.4:
                    return True
            else:
                # Lexical fallback: require at least 2 matching terms to reduce false positives
                query_terms = {term.lower() for term in re.findall(r"[\wÀ-ỹ]{4,}", query, flags=re.UNICODE)}
                if not query_terms:
                    return bool(chunks)
                text = chunk.content
                if chunk.evidence:
                    text += " " + " ".join(e.snippet_original for e in chunk.evidence)
                chunk_terms = {term.lower() for term in re.findall(r"[\wÀ-ỹ]{4,}", text, flags=re.UNICODE)}
                if len(query_terms & chunk_terms) >= 2:
                    return True
        return False

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int((time.perf_counter() - started) * 1000))

    _ANAPHORA_RE = re.compile(
        r"\b(nó|chúng|họ|điều này|điều đó|vấn đề này|khái niệm này|cái này|cái đó|"
        r"it\b|they\b|them\b|this\b|that\b|these\b|those\b|"
        r"the concept|the topic|the above|the same|the latter|the former)\b",
        re.IGNORECASE,
    )

    async def _resolve_anaphora(self, *, query: str, memory_context: str) -> str:
        """Resolve pronouns/vague references in query using conversation history.
        Skips LLM call when no anaphoric patterns are detected.
        """
        if not memory_context.strip() or not self._ANAPHORA_RE.search(query):
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
                return resolved
        except Exception as exc:
            logger.debug("Anaphora resolution failed", extra={"error": str(exc)})
        return query

    async def _reflect_on_gaps(
        self,
        *,
        query: str,
        missing_sub_questions: list[str],
        chunks: list[RetrievedChunk],
    ) -> list[str]:
        """Ask LLM to generate refined search queries for uncovered sub-questions."""
        if not missing_sub_questions:
            return []
        evidence_preview = " | ".join(c.content[:100] for c in chunks[:5])
        prompt = (
            f"Retrieval planner. The following sub-questions are not yet covered by the evidence found.\n"
            f"Original query: {query}\n"
            f"Evidence found so far (preview): {evidence_preview}\n"
            f"Missing sub-questions: {'; '.join(missing_sub_questions[:3])}\n\n"
            "Generate 2 specific, distinct search queries (different phrasings) that would help find the missing evidence.\n"
            'Output JSON only: {"queries": ["...", "..."]}'
        )
        try:
            raw = await self.engine.llm.generate(prompt=prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]
        except Exception as exc:
            logger.debug("Gap reflection failed", extra={"error": str(exc)})
        return []

    async def _arerank(self, *, query: str, queries: list[str], chunks: list[RetrievedChunk], limit: int, use_mmr: bool) -> list[RetrievedChunk]:
        if hasattr(self.engine.reranker, "arerank_multilingual"):
            return await self.engine.reranker.arerank_multilingual(queries=queries, chunks=chunks, limit=limit, use_mmr=use_mmr)
        if hasattr(self.engine.reranker, "rerank_multilingual"):
            async with self._rerank_fallback_semaphore:
                return await asyncio.to_thread(
                    self.engine.reranker.rerank_multilingual,
                    queries=queries,
                    chunks=chunks,
                    limit=limit,
                    use_mmr=use_mmr,
                )
        if hasattr(self.engine.reranker, "arerank"):
            return await self.engine.reranker.arerank(query=query, chunks=chunks, limit=limit)
        async with self._rerank_fallback_semaphore:
            return await asyncio.to_thread(self.engine.reranker.rerank, query=query, chunks=chunks, limit=limit)

    @staticmethod
    def _ensure_context_coverage(
        *,
        selected: list[RetrievedChunk],
        candidates: list[RetrievedChunk],
        expected_material_ids: list[str],
        limit: int,
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
            candidate = next((chunk for chunk in candidates if chunk.material_id == material_id and chunk.chunk_id not in seen_chunks), None)
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
            CoverageSource(material_id=material_id, name=names.get(material_id, material_id), covered=material_id in covered)
            for material_id in expected
        ]
        return CoverageReport(requested_count=len(sources), covered_count=sum(1 for source in sources if source.covered), sources=sources)

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
            for material in materials
            if material.id is not None
        }
