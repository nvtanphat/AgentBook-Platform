from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from beanie import PydanticObjectId

from src.agentic.planner import AgenticPlanner
from src.guardrails.claim_verifier import ClaimVerdict
from src.inference.inference_engine import REFUSAL_ANSWER, InferenceEngine
from src.inference.intent_classifier import QueryIntent
from src.inference.reasoning_path_builder import build_reasoning_path
from src.models.common import PipelineStatus
from src.models.material import Material
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


class AgenticRagService:
    """MVP agentic orchestration around the existing RAG engine."""

    def __init__(self, *, engine: InferenceEngine) -> None:
        self.engine = engine
        self.planner = AgenticPlanner()

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

        route = self.engine.query_router.route(query)
        expected_material_ids = scope.material_ids or await self._indexed_material_ids_for_scope(scope)
        material_count = len(expected_material_ids)
        plan = self.planner.build(route=route, material_count=material_count)
        steps: list[AgentTraceStep] = []
        await self._record_step(
            steps,
            AgentTraceStep(
                name="plan_query",
                status="completed",
                query=query,
                sources_requested=material_count or None,
                warning=plan.plan_type,
            ),
            on_step,
        )

        retrieval_limit = self.engine._scaled_limit(self.engine.settings.rerank_input_k, route)
        final_limit = max(self.engine._scaled_limit(top_k or self.engine.settings.final_top_k, route), material_count or 1)
        query_rewriter = self.engine.query_rewriter if route.use_multi_query else None
        processed = await self.engine.query_processor.process_async(
            query,
            answer_language=answer_language,
            rewriter=query_rewriter,
        )

        retrieved: list[RetrievedChunk] = []
        graph_chunks: list[RetrievedChunk] = []
        try:
            retrieved = await self._retrieve_multi_query(
                queries=processed.retrieval_queries,
                scope=scope,
                limit=retrieval_limit,
                steps=steps,
                step_name="retrieve_multi_query" if route.use_multi_query else "retrieve_text",
                on_step=on_step,
            )
            if plan.use_per_source and expected_material_ids:
                retrieved.extend(
                    await self._retrieve_per_source(
                        query=query,
                        scope=scope,
                        material_ids=expected_material_ids,
                        limit=max(1, min(3, retrieval_limit)),
                        steps=steps,
                        on_step=on_step,
                    )
                )
            if plan.use_graph:
                graph_chunks = await self._trace_graph(query=query, scope=scope, steps=steps, priority=route.graph_priority, on_step=on_step)
        except Exception as exc:
            logger.error("Agentic retrieval failed", exc_info=True, extra={"owner_id": scope.owner_id, "error": str(exc)})
            failure_step = AgentTraceStep(name="retrieve_text", status="failed", warning=type(exc).__name__)
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
                refusal_reason=f"Retrieval failed: {type(exc).__name__}",
                agent_trace=trace,
            )

        candidates = dedupe_retrieved_chunks((graph_chunks + retrieved) if route.graph_priority else (retrieved + graph_chunks))
        coverage = await self._coverage_report(expected_material_ids=expected_material_ids, chunks=candidates)
        repair_attempted = False
        if plan.requires_coverage and coverage.covered_count < coverage.requested_count:
            repair_attempted = True
            missing = [source.material_id for source in coverage.sources if not source.covered]
            repaired = await self._retrieve_per_source(
                query=query,
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

        reranked = self._rerank(
            query=query,
            queries=processed.retrieval_queries,
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

        if route.route_type.value == "summarization" and reranked:
            should_refuse = False
            if refusal_reason not in (None, "partial_confidence"):
                refusal_reason = None

        citations = self.engine.response_parser.citations_from_chunks(context_chunks)
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
                reasoning_path=build_reasoning_path(query=query, retrieved_chunks=retrieved, graph_chunks=graph_chunks, reranked_chunks=reranked, use_graph=route.use_graph),
                coverage=coverage,
                agent_trace=trace,
            )

        prompt = self.engine._build_prompt(
            query=query,
            chunks=context_chunks,
            answer_language=processed.answer_language,
            memory_context=memory_context or "",
            route_type=route.route_type,
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
                refusal_reason=f"Answer generation failed: {type(exc).__name__}",
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
            AgentTraceStep(name="synthesize_answer", status="completed", evidence_count=len(context_chunks)),
            on_step,
        )
        verification = self.engine.claim_verifier.verify(
            claim=answer,
            evidence=[evidence for chunk in context_chunks for evidence in chunk.evidence],
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

        await self._record_step(
            steps,
            AgentTraceStep(
                name="verify_claims",
                status="completed",
                evidence_count=sum(len(chunk.evidence) for chunk in context_chunks),
                warning=warning,
            ),
            on_step,
        )
        trace = AgentTrace(
            plan_type=plan.plan_type,
            steps=steps,
            repair_attempted=repair_attempted,
            verification=AgentVerification(verdict=verification.verdict.value, confidence=verification.confidence, warning=warning),
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
            reasoning_path=build_reasoning_path(query=query, retrieved_chunks=retrieved, graph_chunks=graph_chunks, reranked_chunks=reranked, use_graph=route.use_graph),
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
        tasks = [self.engine.retriever.retrieve(query=item, scope=scope, limit=limit) for item in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        chunks: list[RetrievedChunk] = []
        warnings: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                warnings.append(type(result).__name__)
                continue
            chunks.extend(result)
        await self._record_step(
            steps,
            AgentTraceStep(
                name=step_name,
                status="completed" if chunks else "failed",
                query=" | ".join(queries[:3]),
                evidence_count=len(chunks),
                warning=", ".join(warnings) or None,
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
        chunks: list[RetrievedChunk] = []
        for material_id in material_ids:
            material_scope = RetrievalScope(owner_id=scope.owner_id, collection_id=scope.collection_id, material_ids=[material_id])
            chunks.extend(await self.engine.retriever.retrieve(query=query, scope=material_scope, limit=limit))
        covered = len({chunk.material_id for chunk in chunks})
        await self._record_step(
            steps,
            AgentTraceStep(
                name=step_name,
                status="completed" if chunks else "skipped",
                query=query,
                sources_requested=len(material_ids),
                sources_covered=covered,
                evidence_count=len(chunks),
                warning=None if covered == len(material_ids) else "Some sources returned no evidence.",
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
        try:
            graph_paths = await self.engine.graph_retriever.retrieve_paths(query=query, scope=scope)
            chunks = self.engine._chunks_from_graph_paths(graph_paths, scope=scope, priority=priority)
        except Exception as exc:
            await self._record_step(steps, AgentTraceStep(name="trace_graph", status="failed", query=query, warning=type(exc).__name__), on_step)
            return []
        await self._record_step(steps, AgentTraceStep(name="trace_graph", status="completed", query=query, evidence_count=len(chunks)), on_step)
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

    def _rerank(self, *, query: str, queries: list[str], chunks: list[RetrievedChunk], limit: int, use_mmr: bool) -> list[RetrievedChunk]:
        if hasattr(self.engine.reranker, "rerank_multilingual"):
            return self.engine.reranker.rerank_multilingual(queries=queries, chunks=chunks, limit=limit, use_mmr=use_mmr)
        return self.engine.reranker.rerank(query=query, chunks=chunks, limit=limit)

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
