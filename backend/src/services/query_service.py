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
from src.rag.query_processor import QueryProcessor
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
from src.schemas.evidence import BoundingBoxSchema, CitationSchema
from src.schemas.query import (
    CompareMatrixCell,
    CompareRequest,
    CompareResponse,
    CompareSource,
    ComparisonCell,
    CoverageReport,
    CoverageSource,
    DimensionCoverage,
    QueryByGraphRequest,
    QueryByImageRequest,
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
        self._query_processor = QueryProcessor()

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
        session_id = self.memory_service._session_id(request.owner_id, conversation_id)
        spawn_background_task(
            self.memory_service.update(
                session_id,
                request.query,
                response.answer,
                request.collection_id or "",
                owner_id=request.owner_id,
            ),
            name="memory-update-after-query",
        )
        return response

    async def ask_with_graph_anchor(self, request: QueryByGraphRequest) -> QueryResponse:
        """GraphRAG anchored query — user selected node(s)/edge(s) on the
        knowledge graph; backend uses them as primary retrieval anchor.

        Flow:
          1. Resolve entity slugs → Mongo Entity docs in scope.
          2. K-hop expansion to gather neighbour entities + relations.
          3. Collect chunk_ids referenced by those entities/relations.
          4. Restrict `RetrievalScope` to the material_ids of those chunks so
             standard hybrid retrieval focuses on the right docs.
          5. Run the usual InferenceEngine.answer pipeline (SLEC + reranker +
             route pipelines still apply — Phase A/B/C carry over).
          6. Annotate the response with `used_entity_ids` + `used_relation_ids`
             so the frontend can highlight what backed the answer.
        """

        scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=request.material_ids,
        )
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(
            scope=scope, conversation_id=conversation_id
        )

        graph_retriever = self.inference_engine.graph_retriever
        anchor_chunk_ids, anchor_entities, anchor_relations = (
            await graph_retriever.retrieve_around_entities(
                entity_slugs=request.entity_ids,
                scope=scope,
                hops=request.hops,
            )
        )
        if not anchor_entities:
            return QueryResponse(
                answer="Không tìm thấy entity nào trong knowledge graph khớp với lựa chọn. Hãy thử chọn lại node khác hoặc đặt câu hỏi tổng quát.",
                answer_language=request.answer_language or "vi",
                query_language="vi",
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason="no_graph_entities_resolved",
            )

        # Restrict retrieval scope to materials touched by these chunks so the
        # rest of the pipeline focuses on graph-anchored docs.
        material_ids_from_graph: list[str] = []
        seen_mat: set[str] = set()
        for e in anchor_entities:
            for ref in e.mention_refs:
                mid = str(ref.material_id)
                if mid not in seen_mat:
                    seen_mat.add(mid)
                    material_ids_from_graph.append(mid)
        for r in anchor_relations:
            for ref in r.evidence_refs:
                mid = str(ref.material_id)
                if mid not in seen_mat:
                    seen_mat.add(mid)
                    material_ids_from_graph.append(mid)

        anchored_scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=material_ids_from_graph or request.material_ids,
        )

        # Augment the user query with the anchor entity names so the embedder
        # gives them weight — but keep the user's original query intent first.
        anchor_names = ", ".join(
            sorted({e.canonical_name for e in anchor_entities[:5]})
        )
        effective_query = (
            f"{request.query.strip()} (về: {anchor_names})"
            if anchor_names else request.query.strip()
        )

        started = perf_counter()
        response = await self.inference_engine.answer(
            query=effective_query,
            scope=anchored_scope,
            top_k=request.top_k,
            answer_language=request.answer_language or "vi",
            memory_context=memory_context,
            rag_flags={},
        )
        latency_ms = int((perf_counter() - started) * 1000)

        # Populate provenance — every anchor entity is "used" by definition,
        # every relation in the subgraph that links two anchored entities too.
        anchor_entity_slugs = list(dict.fromkeys(
            f"entity:{graph_retriever._slug(e.canonical_name)}"
            for e in anchor_entities
        ))
        anchor_relation_ids = list(dict.fromkeys(
            str(r.id) for r in anchor_relations if r.id
        ))
        response.used_entity_ids = anchor_entity_slugs
        response.used_relation_ids = anchor_relation_ids

        logger.info(
            "GraphRAG anchored query answered",
            extra={
                "owner_id": request.owner_id,
                "collection_id": request.collection_id,
                "anchor_entity_count": len(anchor_entities),
                "anchor_relation_count": len(anchor_relations),
                "anchored_material_count": len(material_ids_from_graph),
                "latency_ms": latency_ms,
            },
        )

        # Mirror ask(): persist log + memory turn in background.
        log_request = QueryRequest(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=request.material_ids,
            conversation_id=request.conversation_id,
            query=f"[graph:{','.join(request.entity_ids[:3])}] {request.query}",
            top_k=request.top_k,
            answer_language=request.answer_language,
        )
        await self._log_query(request=log_request, response=response, latency_ms=latency_ms)
        session_id = self.memory_service._session_id(request.owner_id, conversation_id)
        spawn_background_task(
            self.memory_service.update(
                session_id,
                log_request.query,
                response.answer,
                request.collection_id or "",
                owner_id=request.owner_id,
            ),
            name="memory-update-after-graph-query",
        )
        return response

    async def ask_with_image(
        self,
        *,
        request: QueryByImageRequest,
        image_bytes: bytes,
        image_filename: str,
    ) -> QueryResponse:
        """Image-as-query: SigLIP-embed the upload, retrieve top-K visual hits,
        then route to the standard inference pipeline using the matched materials
        as scope and a synthesised text query (user text + figure captions)."""
        if not self.settings.visual_embedding_enabled:
            return QueryResponse(
                answer="Tính năng truy vấn bằng hình ảnh chưa được bật. Vui lòng kích hoạt visual_embedding trong config và đánh chỉ mục lại tài liệu.",
                answer_language="vi",
                query_language="vi",
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason="visual_embedding_disabled",
            )

        from src.rag.visual_embedder import SigLIPProvider
        visual_provider = SigLIPProvider(self.settings)
        try:
            image_vec = await asyncio.to_thread(visual_provider.embed_image_bytes, image_bytes)
        except Exception as exc:
            logger.warning("SigLIP encode failed", extra={"owner_id": request.owner_id, "error": str(exc)})
            return QueryResponse(
                answer="Không đọc được file ảnh bạn vừa tải lên. Hãy thử lại với ảnh PNG/JPG hợp lệ.",
                answer_language="vi",
                query_language="vi",
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason="image_decode_failed",
            )

        search_scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=request.material_ids,
        )
        search_scope.ensure_scoped()

        visual_hits = await self.retriever.retrieve_visual_with_vector(
            vector=image_vec,
            scope=search_scope,
            limit=max(5, request.top_k or self.settings.final_top_k),
        )
        if not visual_hits:
            return QueryResponse(
                answer="Không tìm thấy hình ảnh tương tự nào trong bộ tài liệu hiện tại.",
                answer_language=request.answer_language or "vi",
                query_language="vi",
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason="no_visual_matches",
            )

        matched_material_ids = list(dict.fromkeys(h.material_id for h in visual_hits if h.material_id))

        # Compose a text query for the standard text pipeline.
        # Top captions become the semantic seed; user text (if any) is appended verbatim.
        caption_seeds = " ".join(
            (h.caption or h.document_name or "").strip()
            for h in visual_hits[:3]
            if (h.caption or h.document_name)
        ).strip()
        user_text = (request.query_text or "").strip()
        if user_text and caption_seeds:
            seed_query = f"{user_text}. Bối cảnh hình ảnh: {caption_seeds}"
        elif user_text:
            seed_query = user_text
        elif caption_seeds:
            seed_query = f"Giải thích nội dung liên quan đến: {caption_seeds}"
        else:
            seed_query = "Mô tả nội dung tương tự với hình ảnh được tải lên."

        # Restrict downstream retrieval to materials that actually contained
        # a visually-similar figure — keeps the answer grounded in the right docs.
        downstream_scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=request.material_ids or matched_material_ids,
        )

        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=downstream_scope, conversation_id=conversation_id)
        started = perf_counter()
        response = await self.inference_engine.answer(
            query=seed_query,
            scope=downstream_scope,
            top_k=request.top_k,
            answer_language=request.answer_language or "vi",
            memory_context=memory_context,
            rag_flags={},
        )
        latency_ms = int((perf_counter() - started) * 1000)

        # Prepend visual citations so the frontend VisualCitationStrip renders thumbnails.
        visual_citations = [self._visual_hit_to_citation(h) for h in visual_hits[:4]]
        existing_keys = {(c.doc_id, c.page, c.block_id) for c in response.citations}
        for vc in visual_citations:
            if (vc.doc_id, vc.page, vc.block_id) not in existing_keys:
                response.citations.insert(0, vc)

        logger.info(
            "Image-as-query answered",
            extra={
                "owner_id": request.owner_id,
                "collection_id": request.collection_id,
                "image_filename": image_filename,
                "visual_hits": len(visual_hits),
                "latency_ms": latency_ms,
            },
        )

        # Mirror ask(): persist the query log + memory turn in the background.
        log_request = QueryRequest(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=request.material_ids,
            conversation_id=request.conversation_id,
            query=f"[image:{image_filename}] {user_text}".strip(),
            top_k=request.top_k,
            answer_language=request.answer_language,
        )
        await self._log_query(request=log_request, response=response, latency_ms=latency_ms)
        session_id = self.memory_service._session_id(request.owner_id, conversation_id)
        spawn_background_task(
            self.memory_service.update(
                session_id,
                log_request.query,
                response.answer,
                request.collection_id or "",
                owner_id=request.owner_id,
            ),
            name="memory-update-after-image-query",
        )
        return response

    def _visual_hit_to_citation(self, hit) -> CitationSchema:
        bbox_schema: BoundingBoxSchema | None = None
        if hit.bbox is not None:
            bbox_schema = BoundingBoxSchema(
                x1=hit.bbox.x1, y1=hit.bbox.y1, x2=hit.bbox.x2, y2=hit.bbox.y2,
            )
        snippet = (hit.caption or hit.document_name or "Hình ảnh tương tự").strip()
        return CitationSchema(
            doc_id=hit.material_id,
            doc_name=hit.document_name,
            page=hit.page or None,
            pages=[hit.page] if hit.page else [],
            block_id=hit.block_id or None,
            block_type=hit.block_type or "figure",
            snippet_original=snippet,
            snippet_translated=None,
            bbox=bbox_schema,
            role="visual_match",
            source_language=hit.source_language or "unknown",
            confidence=float(min(max(hit.score, 0.0), 1.0)),
        )

    async def ask_stream(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        scope = RetrievalScope(owner_id=request.owner_id, collection_id=request.collection_id, material_ids=request.material_ids)
        scope.ensure_scoped()
        conversation_id = self._conversation_id(request.conversation_id)
        memory_context = await self.memory_service.build_context(scope=scope, conversation_id=conversation_id)
        effective_query = self._resolve_lightweight_anaphora(request.query, memory_context)
        started = perf_counter()
        final_response: QueryResponse | None = None

        flags = request.rag_flags
        use_agentic = flags.get("agentic_rag_enabled", self.settings.agentic_rag_enabled)
        if use_agentic:
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
                except Exception:
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
            session_id = self.memory_service._session_id(request.owner_id, conversation_id)
            spawn_background_task(
                self.memory_service.update(
                    session_id,
                    request.query,
                    final_response.answer,
                    request.collection_id or "",
                    owner_id=request.owner_id,
                ),
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
        compare_start = perf_counter()
        per_cell_limit = max(2, min(4, request.top_k or self.settings.final_top_k))

        # ── Phase 1: parallel retrieval for all (material, dimension) pairs ────
        async def _retrieve_cell(material_id: str, dimension: str) -> tuple[str, str, list]:
            material_scope = RetrievalScope(
                owner_id=request.owner_id,
                collection_id=request.collection_id,
                material_ids=[material_id],
            )
            base_query = f"{request.topic}. Khía cạnh so sánh: {dimension}"
            # Multi-query expansion: translate VI→EN if applicable for dual retrieval
            processed = self._query_processor.process(base_query)
            queries = list(dict.fromkeys(filter(None, processed.retrieval_queries)))
            if not queries:
                queries = [base_query]
            # Retrieve with the primary query (BGE-M3 handles bilingual natively)
            try:
                chunks = await self.retriever.retrieve(query=queries[0], scope=material_scope, limit=per_cell_limit)
                # Merge with translated query results if a translation exists
                if len(queries) > 1:
                    try:
                        extra = await self.retriever.retrieve(query=queries[1], scope=material_scope, limit=per_cell_limit)
                        seen = {c.chunk_id for c in chunks}
                        chunks = chunks + [c for c in extra if c.chunk_id not in seen]
                    except OSError:
                        pass
            except OSError as exc:
                logger.warning(
                    "Compare cell retrieval failed",
                    extra={"owner_id": request.owner_id, "material_id": material_id, "dimension": dimension, "error": str(exc)},
                )
                chunks = []
            ranked = self._rank_chunks_for_dimension(topic=request.topic, dimension=dimension, chunks=chunks)
            return material_id, dimension, ranked[:2]

        retrieval_tasks = [
            _retrieve_cell(material_id, dimension)
            for material_id in material_ids
            for dimension in dimensions
        ]
        retrieval_results = await asyncio.gather(*retrieval_tasks)

        chunks_by_material_dimension: dict[str, dict[str, list]] = {mid: {} for mid in material_ids}
        all_chunks: list = []
        for material_id, dimension, selected in retrieval_results:
            chunks_by_material_dimension[material_id][dimension] = selected
            all_chunks.extend(selected)

        retrieval_ms = int((perf_counter() - compare_start) * 1000)
        logger.info(
            "Compare retrieval complete",
            extra={
                "owner_id": request.owner_id,
                "materials": len(material_ids),
                "dimensions": len(dimensions),
                "retrieval_ms": retrieval_ms,
            },
        )

        citation_lookup = {
            chunk.chunk_id: citation
            for chunk, citation in zip(all_chunks, self.response_parser.citations_from_chunks(all_chunks))
        }
        source_names, _ = await asyncio.gather(
            self._material_names(material_ids),
            asyncio.sleep(0),  # yield to event loop
        )
        source_names = source_names  # type: ignore[assignment]
        compare_sources = [
            CompareSource(source_id=material_id, name=source_names.get(material_id, material_id))
            for material_id in material_ids
        ]

        # ── Phase 2: parallel synthesis — ONE LLM call per material ───────────
        synth_start = perf_counter()
        synthesis_tasks = [
            self._synthesize_all_dimensions_for_material(
                topic=request.topic,
                material_name=source_names.get(material_id, material_id),
                dimensions=dimensions,
                chunks_by_dimension=chunks_by_material_dimension[material_id],
                answer_language=request.answer_language,
            )
            for material_id in material_ids
        ]
        synthesis_results: list[dict[str, str]] = list(await asyncio.gather(*synthesis_tasks))
        synth_ms = int((perf_counter() - synth_start) * 1000)
        logger.info(
            "Compare synthesis complete",
            extra={
                "owner_id": request.owner_id,
                "materials": len(material_ids),
                "synth_ms": synth_ms,
                "llm_calls": len(material_ids),
            },
        )

        # ── Phase 3: assemble cells from synthesis results ────────────────────
        answers_by_material: dict[str, dict[str, str]] = {
            material_id: synthesis_results[i]
            for i, material_id in enumerate(material_ids)
        }

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

                answer = answers_by_material[material_id].get(dimension, "")
                if not answer:
                    answer = self._extractive_compare_answer(dimension=dimension, chunks=chunks)
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

        total_ms = int((perf_counter() - compare_start) * 1000)
        logger.info(
            "Compare complete",
            extra={
                "owner_id": request.owner_id,
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "synth_ms": synth_ms,
            },
        )

        deduped = {f"{c.doc_id}:{c.page}:{c.block_id}": c for c in all_citations}
        contradictions = ContradictionDetector().detect(all_evidence)
        conflicts = [c.description for c in contradictions]
        covered_material_ids = [
            material_id
            for material_id, by_dimension in chunks_by_material_dimension.items()
            if any(by_dimension.get(dim) for dim in dimensions)
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

    async def _synthesize_all_dimensions_for_material(
        self,
        *,
        topic: str,
        material_name: str,
        dimensions: list[str],
        chunks_by_dimension: dict[str, list],
        answer_language: str = "vi",
    ) -> dict[str, str]:
        """One LLM call per material: returns {dimension: answer} for all active dimensions."""
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(answer_language, answer_language)
        evidence_safety = InferenceEngine._evidence_safety_rules()

        active_dims = [d for d in dimensions if chunks_by_dimension.get(d)]
        if not active_dims:
            return {}

        seen_ids: set[str] = set()
        all_chunks: list = []
        for dim in active_dims:
            for chunk in chunks_by_dimension.get(dim, []):
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        evidence = self.response_parser.format_evidence_for_prompt(all_chunks)
        dim_lines = "\n".join(f'  - "{dim}"' for dim in active_dims)
        dim_keys = ", ".join(f'"{dim}"' for dim in active_dims)

        synth_start = perf_counter()
        prompt = (
            f"{evidence_safety}\n\n"
            f"Bạn là Noelys, trợ lý tri thức học tập.\n"
            f"Dựa trên các trích đoạn từ tài liệu '{material_name}', "
            f"trả lời từng khía cạnh bằng {lang_name} (2-3 câu mỗi khía cạnh).\n\n"
            f"CHỦ ĐỀ: {topic}\n"
            f"CÁC KHÍA CẠNH CẦN TRẢ LỜI:\n{dim_lines}\n\n"
            f"QUY TẮC:\n"
            f"- Chỉ dùng thông tin có trong TRÍCH ĐOẠN bên dưới.\n"
            f"- Nếu không có đủ thông tin, viết: \"Không đủ dữ liệu\".\n"
            f"- Trả lời dưới dạng JSON hợp lệ, không giải thích thêm.\n\n"
            f"TRÍCH ĐOẠN:\n{evidence}\n\n"
            f"PHẢN HỒI JSON với các key: {dim_keys}"
        )

        try:
            raw = await self.inference_engine.llm.generate(prompt=prompt)
            result = self._parse_dimensions_json(raw, active_dims)
        except Exception as exc:
            logger.warning(
                "Compare synthesis LLM failed",
                extra={"material_name": material_name, "error": str(exc)},
            )
            result = {}

        if not result:
            try:
                strict_prompt = (
                    f"Return ONLY a JSON object with these exact keys: {dim_keys}.\n"
                    f"Each value: 1-2 sentence answer in {lang_name} based on:\n{evidence}\n"
                    f"Topic: {topic}\nJSON only, no markdown:"
                )
                raw2 = await self.inference_engine.llm.generate(prompt=strict_prompt)
                result = self._parse_dimensions_json(raw2, active_dims)
            except Exception:
                result = {}

        for dim in active_dims:
            if not result.get(dim):
                chunks = chunks_by_dimension.get(dim, [])
                result[dim] = chunks[0].content[:300] if chunks else "Không đủ dữ liệu"

        logger.debug(
            "Per-material synthesis done",
            extra={"material_name": material_name, "dims": len(active_dims), "synth_ms": int((perf_counter() - synth_start) * 1000)},
        )
        return result

    @staticmethod
    def _parse_dimensions_json(raw: str, dimensions: list[str]) -> dict[str, str]:
        """Extract {dimension: answer} from LLM JSON response; returns {} on any parse error."""
        if not raw:
            return {}
        text = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {dim: str(v).strip() for dim in dimensions if (v := parsed.get(dim)) and str(v).strip()}

    async def _synthesize_dimension(self, *, topic: str, dimension: str, chunks: list, answer_language: str = "vi") -> str:
        """Synthesize a concise 1-3 sentence answer for one comparison dimension via LLM."""
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(answer_language, answer_language)
        evidence = self.response_parser.format_evidence_for_prompt(chunks)
        evidence_safety = InferenceEngine._evidence_safety_rules()
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
                evidence=[ev for chunk in chunks for ev in chunk.evidence],
            )
            if verification.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.NOT_ENOUGH_EVIDENCE}:
                return "Không tìm thấy thông tin về khía cạnh này."
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
    _RECENT_USER_RE = re.compile(r"^User:\s*(.+?)(?=\nAssistant:|$)", re.IGNORECASE | re.MULTILINE)

    @classmethod
    def _resolve_lightweight_anaphora(cls, query: str, memory_context: str) -> str:
        """Make short follow-up questions retrievable without an extra LLM call."""
        normalized_query = cls._ascii_fold(query.strip())
        if not memory_context.strip() or not cls._ANAPHORA_START_RE.search(normalized_query):
            return query
        # Queries with >5 tokens already carry enough context — skip expansion to avoid misrouting
        if len(query.split()) > 5:
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
