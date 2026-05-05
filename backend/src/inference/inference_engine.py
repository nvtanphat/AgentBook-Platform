from __future__ import annotations

import asyncio
import json
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncGenerator

from src.core.base_llm import BaseLLM
from src.core.config import Settings, project_root
from src.core.model_factory import build_llm
from src.guardrails.claim_verifier import ClaimVerdict, ClaimVerifier
from src.inference.chitchat_detector import get_instant_reply
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.intent_classifier import IntentClassifier, QueryIntent
from src.inference.reasoning_path_builder import build_reasoning_path
from src.inference.response_parser import ResponseParser
from src.rag.graph_retriever import GraphRetriever
from src.rag.query_processor import QueryProcessor
from src.rag.query_rewriter import LLMQueryRewriter
from src.rag.query_router import QueryRouter, RouteDecision, RouteType
from src.rag.retriever import HybridRetriever, dedupe_retrieved_chunks
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievalScope, RetrievedChunk
from src.schemas.query import QueryResponse

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = "Tôi không tìm thấy đủ bằng chứng trong tài liệu được cung cấp để trả lời câu hỏi này."


class InferenceEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        graph_retriever: GraphRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: BaseLLM | None = None,
        response_parser: ResponseParser | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        query_processor: QueryProcessor | None = None,
        query_router: QueryRouter | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.graph_retriever = graph_retriever or GraphRetriever(settings)
        self.reranker = reranker or CrossEncoderReranker(settings)
        self.llm = llm or build_llm(settings)
        self.response_parser = response_parser or ResponseParser()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(settings)
        self.query_processor = query_processor or QueryProcessor()
        self.query_router = query_router or QueryRouter()
        self.claim_verifier = claim_verifier or ClaimVerifier()
        self.query_rewriter: LLMQueryRewriter | None = (
            LLMQueryRewriter(self.llm) if settings.query_rewriter_enabled else None
        )
        self.intent_classifier = IntentClassifier(llm=self.llm)

    async def answer(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        top_k: int | None = None,
        answer_language: str | None = None,
        memory_context: str | None = None,
    ) -> QueryResponse:
        intent = await self.intent_classifier.classify(query)
        if intent == QueryIntent.CHITCHAT:
            return await self._answer_chitchat(query)
        if intent == QueryIntent.OFF_TOPIC:
            return self._refuse_off_topic()

        route_decision = self.query_router.route(query)
        retrieval_limit = self._scaled_limit(self.settings.rerank_input_k, route_decision)
        final_limit = self._scaled_limit(top_k or self.settings.final_top_k, route_decision)
        query_rewriter = self.query_rewriter if route_decision.use_multi_query else None

        processed = await self.query_processor.process_async(
            query, answer_language=answer_language, rewriter=query_rewriter
        )

        try:
            retrieval_tasks = [
                self.retriever.retrieve(query=retrieval_query, scope=scope, limit=retrieval_limit)
                for retrieval_query in processed.retrieval_queries
            ]
            graph_task = self.graph_retriever.retrieve_paths(query=query, scope=scope) if route_decision.use_graph else None
            tasks = [*retrieval_tasks, graph_task] if graph_task is not None else retrieval_tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            retrieved: list[RetrievedChunk] = []
            retrieval_results = results[:-1] if graph_task is not None else results
            for result in retrieval_results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Retrieval query failed",
                        extra={"owner_id": scope.owner_id, "error": str(result), "error_type": type(result).__name__},
                    )
                    continue
                retrieved.extend(result)
            if graph_task is None:
                graph_chunks = []
            else:
                graph_result = results[-1]
                if isinstance(graph_result, Exception):
                    logger.warning(
                        "Graph retrieval failed",
                        extra={"owner_id": scope.owner_id, "error": str(graph_result), "error_type": type(graph_result).__name__},
                    )
                    graph_paths = []
                else:
                    graph_paths = graph_result
                graph_chunks = self._chunks_from_graph_paths(graph_paths, scope=scope, priority=route_decision.graph_priority)
        except Exception as exc:
            logger.error(
                "Retrieval pipeline failed",
                exc_info=True,
                extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id, "error": str(exc)},
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
                refusal_reason=f"Retrieval failed: {type(exc).__name__}. Check that Qdrant and embedding service are running.",
            )

        candidates = dedupe_retrieved_chunks((graph_chunks + retrieved) if route_decision.graph_priority else (retrieved + graph_chunks))
        if hasattr(self.reranker, "rerank_multilingual"):
            reranked = self.reranker.rerank_multilingual(
                queries=processed.retrieval_queries,
                chunks=candidates,
                limit=final_limit,
                use_mmr=route_decision.use_mmr,
            )
        else:
            reranked = self.reranker.rerank(query=query, chunks=candidates, limit=final_limit)
        context_chunks = self._pack_context_chunks(reranked)
        confidence = self.confidence_scorer.score(reranked)
        should_refuse, refusal_reason = self.confidence_scorer.should_refuse(chunks=reranked, confidence=confidence)

        # Summarization: cross-encoder inherently scores low (query = instruction, not content match).
        # Any retrieved evidence is sufficient — only refuse when nothing was found at all.
        if route_decision.route_type == RouteType.SUMMARIZATION and reranked:
            should_refuse = False
            if refusal_reason not in (None, "partial_confidence"):
                refusal_reason = None

        citations = self.response_parser.citations_from_chunks(context_chunks)
        if should_refuse:
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
            )

        prompt = self._build_prompt(
            query=query,
            chunks=context_chunks,
            answer_language=processed.answer_language,
            memory_context=memory_context or "",
            route_type=route_decision.route_type,
        )
        try:
            answer = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.error(
                "LLM generation failed",
                exc_info=True,
                extra={"owner_id": scope.owner_id, "error": str(exc)},
            )
            return QueryResponse(
                answer=REFUSAL_ANSWER,
                answer_language=processed.answer_language,
                query_language=processed.query_language,
                translated_query=processed.translated_query,
                source_languages=sorted({citation.source_language for citation in citations}),
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=f"Answer generation failed: {type(exc).__name__}. Check that the LLM service (Ollama/OpenAI) is running.",
            )

        if not answer.strip():
            answer = REFUSAL_ANSWER
            should_refuse = True
            refusal_reason = "LLM returned an empty grounded answer"
        else:
            answer = self.response_parser.inject_citations(answer, context_chunks)
            verification = self.claim_verifier.verify(
                claim=answer,
                evidence=[evidence for chunk in context_chunks for evidence in chunk.evidence],
            )
            if verification.verdict == ClaimVerdict.CONTRADICTED:
                answer = answer + "\n\n> Canh bao: Phat hien mau thuan giua cau tra loi va bang chung goc."
            if refusal_reason == "partial_confidence":
                answer = answer + "\n\n> ⚠️ Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra lại nguồn gốc."

        # Build reasoning path for transparency
        reasoning_path = build_reasoning_path(
            query=query,
            retrieved_chunks=retrieved,
            graph_chunks=graph_chunks,
            reranked_chunks=reranked,
            use_graph=route_decision.use_graph,
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
            reasoning_path=reasoning_path,
        )

    async def answer_stream(
        self,
        *,
        query: str,
        scope: RetrievalScope,
        top_k: int | None = None,
        answer_language: str | None = None,
        memory_context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield SSE-formatted lines.  Events:
          token  – {"token": "..."}  for each LLM output token
          done   – full QueryResponse JSON on completion
          error  – {"message": "..."} on failure
        """
        # Fast paths that don't need streaming
        intent = await self.intent_classifier.classify(query)
        if intent == QueryIntent.CHITCHAT:
            response = await self._answer_chitchat(query)
            yield f"event: done\ndata: {response.model_dump_json()}\n\n"
            return
        if intent == QueryIntent.OFF_TOPIC:
            response = self._refuse_off_topic()
            yield f"event: done\ndata: {response.model_dump_json()}\n\n"
            return

        route_decision = self.query_router.route(query)
        retrieval_limit = self._scaled_limit(self.settings.rerank_input_k, route_decision)
        final_limit = self._scaled_limit(top_k or self.settings.final_top_k, route_decision)
        query_rewriter = self.query_rewriter if route_decision.use_multi_query else None

        processed = await self.query_processor.process_async(
            query, answer_language=answer_language, rewriter=query_rewriter
        )

        # ── Retrieval ────────────────────────────────────────────────────────
        try:
            retrieval_tasks = [
                self.retriever.retrieve(query=rq, scope=scope, limit=retrieval_limit)
                for rq in processed.retrieval_queries
            ]
            graph_task = (
                self.graph_retriever.retrieve_paths(query=query, scope=scope)
                if route_decision.use_graph else None
            )
            tasks = [*retrieval_tasks, graph_task] if graph_task is not None else retrieval_tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            retrieved: list[RetrievedChunk] = []
            retrieval_results = results[:-1] if graph_task is not None else results
            for result in retrieval_results:
                if isinstance(result, Exception):
                    logger.warning("Retrieval query failed", extra={"error": str(result)})
                    continue
                retrieved.extend(result)
            graph_chunks = []
            if graph_task is not None:
                graph_result = results[-1]
                if not isinstance(graph_result, Exception):
                    graph_chunks = self._chunks_from_graph_paths(graph_result, scope=scope, priority=route_decision.graph_priority)
        except Exception as exc:
            err = json.dumps({"message": f"Retrieval failed: {type(exc).__name__}"})
            yield f"event: error\ndata: {err}\n\n"
            return

        candidates = dedupe_retrieved_chunks(
            (graph_chunks + retrieved) if route_decision.graph_priority else (retrieved + graph_chunks)
        )
        if hasattr(self.reranker, "rerank_multilingual"):
            reranked = self.reranker.rerank_multilingual(
                queries=processed.retrieval_queries,
                chunks=candidates,
                limit=final_limit,
                use_mmr=route_decision.use_mmr,
            )
        else:
            reranked = self.reranker.rerank(query=query, chunks=candidates, limit=final_limit)

        context_chunks = self._pack_context_chunks(reranked)
        confidence = self.confidence_scorer.score(reranked)
        should_refuse, refusal_reason = self.confidence_scorer.should_refuse(chunks=reranked, confidence=confidence)
        if route_decision.route_type == RouteType.SUMMARIZATION and reranked:
            should_refuse = False
            if refusal_reason not in (None, "partial_confidence"):
                refusal_reason = None

        citations = self.response_parser.citations_from_chunks(context_chunks)

        if should_refuse:
            response = QueryResponse(
                answer=REFUSAL_ANSWER,
                answer_language=processed.answer_language,
                query_language=processed.query_language,
                translated_query=processed.translated_query,
                source_languages=sorted({c.source_language for c in citations}),
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=refusal_reason,
            )
            yield f"event: done\ndata: {response.model_dump_json()}\n\n"
            return

        # ── Stream LLM tokens ────────────────────────────────────────────────
        prompt = self._build_prompt(
            query=query,
            chunks=context_chunks,
            answer_language=processed.answer_language,
            memory_context=memory_context or "",
            route_type=route_decision.route_type,
        )
        accumulated = ""
        try:
            async for token in self.llm.stream(prompt=prompt):
                if token:
                    accumulated += token
                    yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("LLM stream failed", exc_info=True, extra={"owner_id": scope.owner_id})
            err = json.dumps({"message": f"LLM generation failed: {type(exc).__name__}"})
            yield f"event: error\ndata: {err}\n\n"
            return

        # ── Post-process and send done ────────────────────────────────────────
        answer = accumulated.strip() or REFUSAL_ANSWER
        if accumulated.strip():
            answer = self.response_parser.inject_citations(answer, context_chunks)
            verification = self.claim_verifier.verify(
                claim=answer,
                evidence=[ev for chunk in context_chunks for ev in chunk.evidence],
            )
            if verification.verdict == ClaimVerdict.CONTRADICTED:
                answer += "\n\n> Canh bao: Phat hien mau thuan giua cau tra loi va bang chung goc."
            if refusal_reason == "partial_confidence":
                answer += "\n\n> ⚠️ Câu trả lời dựa trên bằng chứng có độ tin cậy hạn chế. Vui lòng kiểm tra lại nguồn gốc."

        response = QueryResponse(
            answer=answer,
            answer_language=processed.answer_language,
            query_language=processed.query_language,
            translated_query=processed.translated_query,
            source_languages=sorted({c.source_language for c in citations}),
            citations=citations,
            confidence=confidence,
            was_refused=not accumulated.strip(),
            refusal_reason=refusal_reason if not accumulated.strip() else None,
        )
        yield f"event: done\ndata: {response.model_dump_json()}\n\n"

    async def _answer_chitchat(self, query: str) -> QueryResponse:
        # Fast path: pre-written reply for common unambiguous patterns (no LLM cost)
        answer = get_instant_reply(query)
        if answer is None:
            # Slow path: LLM for novel/complex chitchat
            template_path = project_root() / "backend" / "src" / "prompts" / "chitchat.txt"
            prompt = template_path.read_text(encoding="utf-8").format(query=query)
            try:
                answer = (await self.llm.generate(prompt=prompt)).strip()
            except Exception as exc:
                logger.warning("Chitchat LLM call failed", extra={"error": str(exc)})
            if not answer:
                answer = "Xin chào! Tôi có thể giúp gì cho bạn?"
        return QueryResponse(
            answer=answer,
            answer_language="vi",
            query_language="vi",
            translated_query=None,
            source_languages=[],
            citations=[],
            confidence=1.0,
            was_refused=False,
            refusal_reason=None,
        )

    def _refuse_off_topic(self) -> QueryResponse:
        template_path = project_root() / "backend" / "src" / "prompts" / "off_topic.txt"
        answer = template_path.read_text(encoding="utf-8").strip()
        return QueryResponse(
            answer=answer,
            answer_language="vi",
            query_language="vi",
            translated_query=None,
            source_languages=[],
            citations=[],
            confidence=0.0,
            was_refused=True,
            refusal_reason="off_topic",
        )

    _LANGUAGE_NAMES: dict[str, str] = {
        "vi": "tiếng Việt",
        "en": "English",
        "zh": "Chinese",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
    }

    def _build_prompt(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        answer_language: str,
        memory_context: str = "",
        route_type: RouteType = RouteType.GENERAL,
    ) -> str:
        prompt_file = (
            "summarization.txt"
            if route_type == RouteType.SUMMARIZATION
            else "qa_grounded.txt"
        )
        template_path = project_root() / "backend" / "src" / "prompts" / prompt_file
        template = template_path.read_text(encoding="utf-8")
        lang_name = self._LANGUAGE_NAMES.get(answer_language, answer_language)
        memory_ctx = memory_context.strip()
        formatted_memory = f"\nLỊCH SỬ LIÊN QUAN:\n{memory_ctx}\n\n---\n" if memory_ctx else ""
        values = defaultdict(
            str,
            evidence=self.response_parser.format_evidence_for_prompt(chunks),
            memory_context=formatted_memory,
            query=query,
            answer_language=lang_name,
        )
        return template.format_map(values)

    @staticmethod
    def _scaled_limit(base: int, decision: RouteDecision) -> int:
        return max(1, math.ceil(base * decision.top_k_multiplier))

    @staticmethod
    def _pack_context_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep strongest evidence at the prompt edges to reduce middle-position loss."""
        if len(chunks) <= 2:
            return chunks
        return [chunks[0], *chunks[2:], chunks[1]]

    @staticmethod
    def _chunks_from_graph_paths(graph_paths, *, scope: RetrievalScope, priority: bool = False) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for index, path in enumerate(graph_paths):
            if not path.evidence_refs:
                continue
            first = path.evidence_refs[0]
            content = "\n".join(ref.snippet_original for ref in path.evidence_refs)
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph-path-{index}",
                    owner_id=scope.owner_id,
                    collection_id=scope.collection_id or first.collection_id,
                    material_id=first.material_id,
                    document_name=first.document_name,
                    content=content,
                    language=first.source_language,
                    modality="graph",
                    source_block_ids=[ref.block_id for ref in path.evidence_refs],
                    source_pages=sorted({ref.page for ref in path.evidence_refs}),
                    bboxes=[ref.bbox for ref in path.evidence_refs if ref.bbox is not None],
                    evidence=path.evidence_refs,
                    graph_score=path.confidence,
                    fused_score=min(1.0, path.confidence + 0.25) if priority else path.confidence,
                )
            )
        return chunks
