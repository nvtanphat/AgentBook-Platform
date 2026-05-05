from __future__ import annotations

import asyncio
from time import perf_counter
from typing import AsyncGenerator

from beanie import PydanticObjectId

from src.core.config import Settings
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.inference_engine import InferenceEngine
from src.inference.response_parser import ResponseParser
from src.models.material import BoundingBox
from src.models.query_log import QueryCitation, QueryLog
from src.services.memory_service import MemoryService
from src.rag.graph_retriever import GraphRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from src.guardrails.contradiction_detector import ContradictionDetector
from src.schemas.query import CompareRequest, CompareResponse, ComparisonCell, QueryRequest, QueryResponse


class QueryService:
    def __init__(
        self,
        *,
        settings: Settings,
        inference_engine: InferenceEngine | None = None,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        response_parser: ResponseParser | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.settings = settings
        self.response_parser = response_parser or ResponseParser()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(settings)
        self.memory_service = memory_service or MemoryService()
        self.retriever = retriever or HybridRetriever(settings=settings, qdrant_client=get_qdrant_client_for_settings(settings))
        self.reranker = reranker or CrossEncoderReranker(settings)
        self.inference_engine = inference_engine or InferenceEngine(
            settings=settings,
            retriever=self.retriever,
            graph_retriever=GraphRetriever(settings),
            reranker=self.reranker,
            response_parser=self.response_parser,
            confidence_scorer=self.confidence_scorer,
        )

    async def ask(self, request: QueryRequest) -> QueryResponse:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=scope, conversation_id=conversation_id)
        started = perf_counter()
        response = await self.inference_engine.answer(
            query=request.query,
            scope=scope,
            top_k=request.top_k,
            answer_language=request.answer_language,
            memory_context=memory_context,
        )
        latency_ms = int((perf_counter() - started) * 1000)
        await self._log_query(request=request, response=response, latency_ms=latency_ms)
        await self.memory_service.update_after_query(scope=scope, conversation_id=conversation_id)
        return response

    async def ask_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=scope, conversation_id=conversation_id)
        async for chunk in self.inference_engine.answer_stream(
            query=request.query,
            scope=scope,
            top_k=request.top_k,
            answer_language=request.answer_language,
            memory_context=memory_context,
        ):
            yield chunk
        await self.memory_service.update_after_query(scope=scope, conversation_id=conversation_id)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()

        async def _process_dimension(dimension: str) -> tuple[ComparisonCell, list, list]:
            query = f"{request.topic} — {dimension}"
            retrieved = await self.retriever.retrieve(query=query, scope=scope, limit=self.settings.rerank_input_k)
            reranked = self.reranker.rerank(query=query, chunks=retrieved, limit=request.top_k or self.settings.final_top_k)
            confidence = self.confidence_scorer.score(reranked)
            citations = self.response_parser.citations_from_chunks(reranked)
            evidence = [ev for chunk in reranked for ev in chunk.evidence]
            if not reranked or not citations:
                return (
                    ComparisonCell(dimension=dimension, value="Không tìm thấy evidence cho chiều này.", source="—", citation=None, confidence=0.0),
                    [],
                    evidence,
                )
            value = await self._synthesize_dimension(
                topic=request.topic, dimension=dimension, chunks=reranked,
                answer_language=request.answer_language,
            )
            cell = ComparisonCell(dimension=dimension, value=value, source=reranked[0].document_name, citation=citations[0], confidence=confidence)
            return cell, citations, evidence

        results = await asyncio.gather(*[_process_dimension(dim) for dim in request.dimensions])

        cells: list[ComparisonCell] = []
        all_citations: list = []
        all_evidence: list = []
        for cell, citations, evidence in results:
            cells.append(cell)
            all_citations.extend(citations)
            all_evidence.extend(evidence)

        deduped = {f"{c.doc_id}:{c.page}:{c.block_id}": c for c in all_citations}
        contradictions = ContradictionDetector().detect(all_evidence)
        conflicts = [c.description for c in contradictions]
        return CompareResponse(topic=request.topic, comparison_table=cells, citations=list(deduped.values()), conflicts=conflicts)

    async def _synthesize_dimension(self, *, topic: str, dimension: str, chunks: list, answer_language: str = "vi") -> str:
        """Synthesize a concise 1-3 sentence answer for one comparison dimension via LLM."""
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(answer_language, answer_language)
        evidence = self.response_parser.format_evidence_for_prompt(chunks)
        prompt = (
            f"Bạn là Prism — trợ lý tri thức học tập của AgentBook.\n"
            f"Dựa trên BẰNG CHỨNG bên dưới, trả lời ngắn gọn (1–3 câu) bằng {lang_name}:\n"
            f"Về chủ đề '{topic}', khía cạnh '{dimension}' là gì?\n\n"
            f"QUY TẮC: Chỉ dùng thông tin có trong BẰNG CHỨNG. Không suy diễn.\n"
            f"Nếu BẰNG CHỨNG không đủ, viết: 'Không tìm thấy thông tin về khía cạnh này.'\n\n"
            f"BẰNG CHỨNG:\n{evidence}\n\nTRẢ LỜI NGẮN GỌN:"
        )
        try:
            answer = await self.inference_engine.llm.generate(prompt=prompt)
            return answer.strip() or chunks[0].content
        except Exception:
            return chunks[0].content

    async def _log_query(self, *, request: QueryRequest, response: QueryResponse, latency_ms: int) -> None:
        citations: list[QueryCitation] = []
        for citation in response.citations:
            try:
                material_id = PydanticObjectId(citation.doc_id)
            except Exception:
                continue
            citations.append(
                QueryCitation(
                    material_id=material_id,
                    doc_name=citation.doc_name,
                    page=citation.page,
                    block_id=citation.block_id,
                    block_type=citation.block_type,
                    content_snippet=citation.snippet_original,
                    bbox=self._to_query_log_bbox(citation.bbox),
                    role=citation.role,
                    source_language=citation.source_language,
                    confidence=citation.confidence,
                )
            )
        collection_id = None
        if request.collection_id:
            collection_id = PydanticObjectId(request.collection_id)
        query_log = QueryLog(
            owner_id=request.owner_id,
            collection_id=collection_id,
            conversation_id=self._conversation_id(request.conversation_id),
            query=request.query,
            query_language=response.query_language,
            answer=response.answer,
            citations=citations,
            confidence=response.confidence,
            was_refused=response.was_refused,
            refusal_reason=response.refusal_reason,
            retrieval_trace={
                "top_k": request.top_k or self.settings.final_top_k,
                "sources_used_count": len(response.source_languages),
                "source_languages": response.source_languages,
                "retrieval_time_ms": latency_ms,
            },
            latency_ms=latency_ms,
        )
        await query_log.insert()

    @staticmethod
    def _conversation_id(value: str | None) -> str:
        text = " ".join((value or "default").split())
        return text[:128] or "default"

    @staticmethod
    def _to_query_log_bbox(bbox) -> BoundingBox | None:
        if bbox is None:
            return None
        return BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)
