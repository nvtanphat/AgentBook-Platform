from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from time import perf_counter
from typing import AsyncGenerator, cast

from src.agentic.service import AgenticRagService
from beanie import PydanticObjectId

from src.core.background import spawn_background_task
from src.core.config import Settings
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.inference_engine import InferenceEngine
from src.inference.response_parser import ResponseParser
from src.models.common import PipelineStatus
from src.models.material import BoundingBox, Material
from src.models.query_log import QueryCitation, QueryLog
from src.services.memory_service import MemoryService
from src.rag.graph_retriever import GraphRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from src.guardrails.claim_verifier import ClaimVerdict
from src.guardrails.contradiction_detector import ContradictionDetector
from src.schemas.query import (
    CompareMatrixCell,
    CompareRequest,
    CompareResponse,
    CompareSource,
    ComparisonCell,
    CoverageReport,
    CoverageSource,
    DimensionCoverage,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


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
        self.agentic_rag = AgenticRagService(engine=self.inference_engine)

    async def ask(self, request: QueryRequest) -> QueryResponse:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=scope, conversation_id=conversation_id)
        effective_query = self._resolve_lightweight_anaphora(request.query, memory_context)
        started = perf_counter()
        flags = request.rag_flags
        use_agentic = flags.get("agentic_rag_enabled", self.settings.agentic_rag_enabled)
        if use_agentic:
            response = await self.agentic_rag.answer(
                query=effective_query,
                scope=scope,
                top_k=request.top_k,
                answer_language=request.answer_language,
                memory_context=memory_context,
            )
        else:
            response = await self.inference_engine.answer(
                query=effective_query,
                scope=scope,
                top_k=request.top_k,
                answer_language=request.answer_language,
                memory_context=memory_context,
                rag_flags=flags,
            )
        latency_ms = int((perf_counter() - started) * 1000)
        await self._log_query(request=request, response=response, latency_ms=latency_ms)
        spawn_background_task(
            self.memory_service.update_after_query(scope=scope, conversation_id=conversation_id),
            name="memory-update-after-query",
        )
        return response

    async def ask_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=scope, conversation_id=conversation_id)
        effective_query = self._resolve_lightweight_anaphora(request.query, memory_context)
        started = perf_counter()
        final_response: QueryResponse | None = None

        if self.settings.agentic_rag_enabled:
            queue: asyncio.Queue[object] = asyncio.Queue()

            async def publish_step(step) -> None:
                await queue.put(("agent_step", step.model_dump()))

            async def run_agentic() -> None:
                try:
                    response = await self.agentic_rag.answer(
                        query=effective_query,
                        scope=scope,
                        top_k=request.top_k,
                        answer_language=request.answer_language,
                        memory_context=memory_context,
                        on_step=publish_step,
                    )
                    await queue.put(("done", response))
                except Exception as exc:
                    logger.exception("Agentic streaming query failed", extra={"owner_id": request.owner_id})
                    await queue.put(("error", "Query pipeline failed. Please retry later."))

            task = spawn_background_task(run_agentic(), name="agentic-stream-answer")
            try:
                while True:
                    event_type, payload = await queue.get()
                    if event_type == "agent_step":
                        yield f"event: agent_step\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        continue
                    if event_type == "error":
                        yield f"event: error\ndata: {json.dumps({'message': payload}, ensure_ascii=False)}\n\n"
                        break
                    if event_type == "done":
                        final_response = cast(QueryResponse, payload)
                        yield f"event: done\ndata: {payload.model_dump_json()}\n\n"
                        break
            finally:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        else:
            _DONE_PREFIX = "event: done\ndata: "
            async for chunk in self.inference_engine.answer_stream(
                query=effective_query,
                scope=scope,
                top_k=request.top_k,
                answer_language=request.answer_language,
                memory_context=memory_context,
            ):
                yield chunk
                if final_response is None and chunk.startswith(_DONE_PREFIX):
                    raw = chunk[len(_DONE_PREFIX):].rstrip("\n")
                    try:
                        final_response = QueryResponse.model_validate_json(raw)
                    except Exception:
                        pass

        latency_ms = int((perf_counter() - started) * 1000)
        if final_response is not None:
            spawn_background_task(
                self._log_query(request=request, response=final_response, latency_ms=latency_ms),
                name="stream-query-log",
            )
        spawn_background_task(
            self.memory_service.update_after_query(scope=scope, conversation_id=conversation_id),
            name="stream-memory-update",
        )

    async def compare(self, request: CompareRequest) -> CompareResponse:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()

        dimensions = [dim.strip() for dim in request.dimensions if dim.strip()]
        if not dimensions:
            raise ValueError("At least one comparison dimension is required")

        material_ids = request.material_ids or await self._indexed_material_ids_for_collection(request)
        if len(material_ids) >= 2:
            return await self._compare_by_source(request=request, dimensions=dimensions, material_ids=material_ids)

        query = f"{request.topic}. Compare by: {', '.join(dimensions)}"
        limit = max(request.top_k or self.settings.final_top_k, min(12, len(dimensions) * 3))
        try:
            retrieved = await self.retriever.retrieve(query=query, scope=scope, limit=limit)
        except OSError as exc:
            logger.warning("Compare retrieval failed", exc_info=True, extra={"owner_id": request.owner_id, "error": str(exc)})
            retrieved = []
        citations_by_chunk = self.response_parser.citations_from_chunks(retrieved)
        citation_lookup = {chunk.chunk_id: citation for chunk, citation in zip(retrieved, citations_by_chunk)}
        results = [
            self._build_comparison_cell(
                topic=request.topic,
                dimension=dimension,
                chunks=retrieved,
                citation_lookup=citation_lookup,
            )
            for dimension in dimensions
        ]

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
        coverage = await self._coverage_report(expected_material_ids=material_ids, covered_material_ids=[chunk.material_id for chunk in retrieved])
        return CompareResponse(topic=request.topic, comparison_table=cells, citations=list(deduped.values()), conflicts=conflicts, coverage=coverage)

    async def _compare_by_source(self, *, request: CompareRequest, dimensions: list[str], material_ids: list[str]) -> CompareResponse:
        chunks_by_material_dimension: dict[str, dict[str, list]] = {material_id: {} for material_id in material_ids}
        all_chunks: list = []
        per_cell_limit = max(2, min(4, request.top_k or self.settings.final_top_k))

        for material_id in material_ids:
            material_scope = RetrievalScope(
                owner_id=request.owner_id,
                collection_id=request.collection_id,
                material_ids=[material_id],
            )
            for dimension in dimensions:
                query = f"{request.topic}. Khía cạnh so sánh: {dimension}"
                try:
                    chunks = await self.retriever.retrieve(query=query, scope=material_scope, limit=per_cell_limit)
                except OSError as exc:
                    logger.warning(
                        "Compare cell retrieval failed",
                        exc_info=True,
                        extra={
                            "owner_id": request.owner_id,
                            "material_id": material_id,
                            "dimension": dimension,
                            "error": str(exc),
                        },
                    )
                    chunks = []
                ranked = self._rank_chunks_for_dimension(topic=request.topic, dimension=dimension, chunks=chunks)
                selected = ranked[:2]
                chunks_by_material_dimension[material_id][dimension] = selected
                all_chunks.extend(selected)

        citation_lookup = {
            chunk.chunk_id: citation
            for chunk, citation in zip(all_chunks, self.response_parser.citations_from_chunks(all_chunks))
        }
        source_names = await self._material_names(material_ids)
        compare_sources = [
            CompareSource(source_id=material_id, name=source_names.get(material_id, material_id))
            for material_id in material_ids
        ]

        cells: list[ComparisonCell] = []
        all_citations: list = []
        all_evidence: list = []
        matrix: dict[str, dict[str, CompareMatrixCell]] = {source.source_id: {} for source in compare_sources}
        cell_citations: dict[str, list[str]] = {}
        dimension_coverage: list[DimensionCoverage] = []
        for dimension in dimensions:
            dimension_selected: list = []
            missing_source_ids: list[str] = []
            for material_id in material_ids:
                chunks = chunks_by_material_dimension.get(material_id, {}).get(dimension) or []
                source_name = source_names.get(material_id, material_id)
                if not chunks:
                    missing_source_ids.append(material_id)
                    value = "Không đủ bằng chứng cho khía cạnh này."
                    matrix[material_id][dimension] = CompareMatrixCell(value=value, confidence=0.0, citation_ids=[], missing_evidence=True)
                    cells.append(
                        ComparisonCell(
                            dimension=dimension,
                            value=value,
                            source=source_name,
                            citation=None,
                            confidence=0.0,
                            source_id=material_id,
                            citation_ids=[],
                            missing_evidence=True,
                        )
                    )
                    continue

                answer = await self._synthesize_dimension(
                    topic=request.topic,
                    dimension=dimension,
                    chunks=chunks,
                    answer_language=request.answer_language,
                )
                answer = self.response_parser.inject_citations(answer, chunks)
                citations = [citation_lookup[chunk.chunk_id] for chunk in chunks if chunk.chunk_id in citation_lookup]
                citation_ids = [self._citation_key(citation) for citation in citations]
                confidence = self.confidence_scorer.score(chunks)
                all_citations.extend(citations)
                all_evidence.extend(ev for chunk in chunks for ev in chunk.evidence)
                dimension_selected.extend(chunks)
                matrix[material_id][dimension] = CompareMatrixCell(
                    value=answer,
                    confidence=confidence,
                    citation_ids=citation_ids,
                    missing_evidence=False,
                )
                cell_citations[f"{material_id}::{dimension}"] = citation_ids
                cells.append(
                    ComparisonCell(
                        dimension=dimension,
                        value=answer,
                        source=source_name,
                        citation=citations[0] if citations else None,
                        confidence=confidence,
                        source_id=material_id,
                        citation_ids=citation_ids,
                        missing_evidence=False,
                    )
                )

            dimension_coverage.append(
                DimensionCoverage(
                    dimension=dimension,
                    requested_count=len(material_ids),
                    covered_count=len(material_ids) - len(missing_source_ids),
                    missing_source_ids=missing_source_ids,
                )
            )
            confidence = self.confidence_scorer.score(dimension_selected)
            dimension_citations = [citation_lookup[chunk.chunk_id] for chunk in dimension_selected if chunk.chunk_id in citation_lookup]
            lines = [f"{dimension}:"]
            for material_id in material_ids:
                source_name = source_names.get(material_id, material_id)
                value = matrix[material_id][dimension].value
                lines.append(f"- {source_name}: {self._snippet(value, max_chars=260)}")
            cells.append(
                ComparisonCell(
                    dimension=dimension,
                    value="\n".join(lines),
                    source=f"{len(dimension_selected)} evidence chunks",
                    citation=dimension_citations[0] if dimension_citations else None,
                    confidence=confidence,
                    citation_ids=[self._citation_key(citation) for citation in dimension_citations],
                    missing_evidence=bool(missing_source_ids),
                )
            )

        deduped = {f"{c.doc_id}:{c.page}:{c.block_id}": c for c in all_citations}
        contradictions = ContradictionDetector().detect(all_evidence)
        conflicts = [c.description for c in contradictions]
        covered_material_ids = [
            material_id
            for material_id, by_dimension in chunks_by_material_dimension.items()
            if any(by_dimension.get(dimension) for dimension in dimensions)
        ]
        coverage = await self._coverage_report(expected_material_ids=material_ids, covered_material_ids=covered_material_ids)
        return CompareResponse(
            topic=request.topic,
            comparison_table=cells,
            citations=list(deduped.values()),
            conflicts=conflicts,
            coverage=coverage,
            sources=compare_sources,
            matrix=matrix,
            cell_citations=cell_citations,
            dimension_coverage=dimension_coverage,
        )

    async def _indexed_material_ids_for_collection(self, request: CompareRequest) -> list[str]:
        if not request.collection_id:
            return []
        try:
            collection_oid = PydanticObjectId(request.collection_id)
        except Exception:
            return []
        try:
            materials = await Material.find(
                Material.owner_id == request.owner_id,
                Material.collection_id == collection_oid,
                Material.status == PipelineStatus.INDEXED.value,
            ).sort("created_at").to_list()
        except Exception:
            return []
        return [str(material.id) for material in materials if material.id is not None]

    async def _coverage_report(self, *, expected_material_ids: list[str], covered_material_ids: list[str]) -> CoverageReport:
        expected = list(dict.fromkeys(mid for mid in expected_material_ids if mid))
        covered = set(mid for mid in covered_material_ids if mid)
        names = await self._material_names(expected)
        sources = [
            CoverageSource(material_id=material_id, name=names.get(material_id, material_id), covered=material_id in covered)
            for material_id in expected
        ]
        return CoverageReport(
            requested_count=len(sources),
            covered_count=sum(1 for source in sources if source.covered),
            sources=sources,
        )

    @staticmethod
    async def _material_names(material_ids: list[str]) -> dict[str, str]:
        names: dict[str, str] = {}
        object_ids: list[PydanticObjectId] = []
        for material_id in material_ids:
            try:
                object_ids.append(PydanticObjectId(material_id))
            except Exception:
                continue
        if not object_ids:
            return names
        try:
            materials = await Material.find({"_id": {"$in": object_ids}}).to_list()
        except Exception:
            return names
        for material in materials:
            if material.id is not None:
                names[str(material.id)] = material.original_name or material.filename or str(material.id)
        return names

    def _build_comparison_cell(self, *, topic: str, dimension: str, chunks: list, citation_lookup: dict) -> tuple[ComparisonCell, list, list]:
        ranked = self._rank_chunks_for_dimension(topic=topic, dimension=dimension, chunks=chunks)
        if not ranked:
            return (
                ComparisonCell(
                    dimension=dimension,
                    value="Không tìm thấy bằng chứng cho khía cạnh này.",
                    source="-",
                    citation=None,
                    confidence=0.0,
                ),
                [],
                [],
            )

        primary = ranked[0]
        selected = ranked[:3]
        citations = [citation_lookup[chunk.chunk_id] for chunk in selected if chunk.chunk_id in citation_lookup]
        evidence = [ev for chunk in selected for ev in chunk.evidence]
        confidence = self.confidence_scorer.score(selected)
        value = self._extractive_compare_answer(dimension=dimension, chunks=selected)
        return (
            ComparisonCell(
                dimension=dimension,
                value=value,
                source=primary.document_name,
                citation=citations[0] if citations else None,
                confidence=confidence,
            ),
            citations,
            evidence,
        )

    def _rank_chunks_for_dimension(self, *, topic: str, dimension: str, chunks: list) -> list:
        query_terms = self._compare_terms(f"{topic} {dimension}")

        def score(chunk) -> tuple[float, float]:
            content_terms = self._compare_terms(chunk.content)
            overlap = len(query_terms & content_terms)
            base_score = chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score
            return float(overlap), float(base_score or 0.0)

        return sorted(chunks, key=score, reverse=True)

    @staticmethod
    def _compare_terms(text: str) -> set[str]:
        return {term for term in re.findall(r"[\wÀ-ỹ]{3,}", text.lower(), flags=re.UNICODE)}

    @staticmethod
    def _extractive_compare_answer(*, dimension: str, chunks: list) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            snippet = " ".join(chunk.content.split())
            if len(snippet) > 360:
                snippet = snippet[:357].rstrip() + "..."
            source = chunk.document_name
            pages = sorted(set(chunk.source_pages))
            page_text = f", trang {pages[0]}" if pages else ""
            lines.append(f"{index}. {snippet} (Nguồn: {source}{page_text})")
        heading = f"{dimension}:"
        return f"{heading}\n" + "\n".join(lines)

    @staticmethod
    def _snippet(content: str, *, max_chars: int) -> str:
        snippet = " ".join(content.split())
        if len(snippet) > max_chars:
            return snippet[: max_chars - 3].rstrip() + "..."
        return snippet

    @staticmethod
    def _citation_key(citation) -> str:
        return f"{citation.doc_id}:{citation.page}:{citation.block_id}"

    async def _synthesize_dimension(self, *, topic: str, dimension: str, chunks: list, answer_language: str = "vi") -> str:
        """Synthesize a concise 1-3 sentence answer for one comparison dimension via LLM."""
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(answer_language, answer_language)
        evidence = self.response_parser.format_evidence_for_prompt(chunks)
        evidence_safety = self.inference_engine._evidence_safety_rules()
        prompt = (
            f"{evidence_safety}\n\n"
            f"Bạn là Noelys, trợ lý tri thức học tập của Noelys.\n"
            f"Dựa trên BẰNG CHỨNG bên dưới, trả lời ngắn gọn (1-3 câu) bằng {lang_name}:\n"
            f"Về chủ đề '{topic}', khía cạnh '{dimension}' là gì?\n\n"
            f"QUY TẮC: Chỉ dùng thông tin có trong BẰNG CHỨNG. Không suy diễn.\n"
            f"Nếu BẰNG CHỨNG không đủ, viết: 'Không tìm thấy thông tin về khía cạnh này.'\n\n"
            f"BẰNG CHỨNG:\n{evidence}\n\nTRẢ LỜI NGẮN GỌN:"
        )
        try:
            answer = await self.inference_engine.llm.generate(prompt=prompt)
            answer = self.response_parser.inject_citations(answer.strip(), chunks)
            verification = await self.inference_engine.claim_verifier.averify(
                claim=answer,
                evidence=[evidence for chunk in chunks for evidence in chunk.evidence],
            )
            if verification.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.NOT_ENOUGH_EVIDENCE}:
                return "KhÃ´ng tÃ¬m tháº¥y thÃ´ng tin vá» khÃ­a cáº¡nh nÃ y."
            return answer or chunks[0].content
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

    _ANAPHORA_START_RE = re.compile(
        r"^(it|this|that|they|them|the former|the latter|"
        r"no|chung|ho|day|do|dieu nay|dieu do|cai nay|cai do)\b",
        re.IGNORECASE,
    )
    _RECENT_USER_RE = re.compile(r"-\s*(?:Người dùng|Nguoi dung|User):\s*(.+?)(?=\s+-\s*(?:Trợ lý|Tro ly|Assistant):|$)", re.IGNORECASE)

    @classmethod
    def _resolve_lightweight_anaphora(cls, query: str, memory_context: str) -> str:
        """Make short follow-up questions retrievable without an extra LLM call."""
        normalized_query = cls._ascii_fold(query.strip())
        if not memory_context.strip() or not cls._ANAPHORA_START_RE.search(normalized_query):
            return query
        previous_queries = [item.strip() for item in cls._RECENT_USER_RE.findall(memory_context) if item.strip()]
        if not previous_queries:
            return query
        previous = previous_queries[-1]
        if cls._ascii_fold(previous) == normalized_query:
            return query
        return f"{previous}\nFollow-up question: {query}"

    @staticmethod
    def _ascii_fold(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value.lower())
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")

    @staticmethod
    def _to_query_log_bbox(bbox) -> BoundingBox | None:
        if bbox is None:
            return None
        return BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2)
