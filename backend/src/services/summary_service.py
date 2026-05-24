from __future__ import annotations

import asyncio
import logging

from beanie import PydanticObjectId

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.core.model_factory import build_llm
from src.guardrails.claim_verifier import ClaimVerdict, ClaimVerifier
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.inference_engine import InferenceEngine
from src.inference.response_parser import ResponseParser
from src.models.common import PipelineStatus
from src.models.material import Material
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from src.schemas.query import CoverageReport, CoverageSource, SummaryRequest, SummaryResponse

logger = logging.getLogger(__name__)

_REFUSAL_TEXT = "Tôi không tìm thấy đủ bằng chứng trong tài liệu để tóm tắt phạm vi này."


class SummaryService:
    _rerank_semaphore = asyncio.Semaphore(1)

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: BaseLLM | None = None,
        response_parser: ResponseParser | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings=settings, qdrant_client=get_qdrant_client_for_settings(settings))
        self.reranker = reranker or CrossEncoderReranker(settings)
        self.llm = llm or build_llm(settings)
        self.response_parser = response_parser or ResponseParser()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(settings)
        self.claim_verifier = claim_verifier or ClaimVerifier()

    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        material_ids = self._request_material_ids(request)
        scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=material_ids,
        )
        scope.ensure_scoped()
        # Multi-lingual query — covers VN audio/docs and EN docs equally well.
        # BGE-M3 is cross-lingual so mixing both terms broadens recall.
        query = (
            f"tóm tắt nội dung chính ý chính khái niệm tổng quan "
            f"summarize {request.scope} key concepts overview main ideas"
        )
        try:
            retrieved = await self._retrieve_summary_chunks(query=query, request=request, scope=scope)
        except Exception as exc:
            logger.exception("Retrieval failed in SummaryService", extra={"owner_id": request.owner_id})
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason="Summary pipeline failed. Please retry later.",
            )
        retrieved_material_ids = list({chunk.material_id for chunk in retrieved})
        expected_material_ids = material_ids or await self._indexed_material_ids_for_collection(request) or retrieved_material_ids
        coverage = await self._coverage_report(expected_material_ids=expected_material_ids, covered_material_ids=retrieved_material_ids)
        target_limit = self._summary_target_limit(request=request, material_ids=expected_material_ids)
        if len({chunk.material_id for chunk in retrieved}) > 1:
            reranked = self._ensure_material_coverage(chunks=retrieved, selected=retrieved, limit=target_limit)
        else:
            async with self._rerank_semaphore:
                reranked = await asyncio.to_thread(self.reranker.rerank, query=query, chunks=retrieved, limit=target_limit)
            reranked = self._ensure_material_coverage(chunks=retrieved, selected=reranked, limit=target_limit)
        confidence = self.confidence_scorer.score(reranked)
        # Summary is a broad task — if we have ANY chunks from the requested scope,
        # we should summarize what's there rather than refuse. Refusal only applies
        # when truly nothing relevant was retrieved (empty `reranked`).
        should_refuse = len(reranked) == 0
        refusal_reason = "no relevant evidence was found in the scoped materials" if should_refuse else None
        citations = self.response_parser.citations_from_chunks(reranked)
        covered_after_rerank = list({chunk.material_id for chunk in reranked})
        coverage = await self._coverage_report(expected_material_ids=expected_material_ids, covered_material_ids=covered_after_rerank)
        if should_refuse:
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=refusal_reason,
                coverage=coverage,
            )
        if len({chunk.material_id for chunk in reranked}) > 1:
            return SummaryResponse(
                summary=self._extractive_collection_summary(reranked),
                citations=citations,
                confidence=confidence,
                coverage=coverage,
            )
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(request.answer_language, request.answer_language)
        evidence_text = self.response_parser.format_evidence_for_prompt(reranked)
        evidence_safety = InferenceEngine._evidence_safety_rules()
        prompt = (
            f"{evidence_safety}\n\n"
            f"Bạn là Noelys, trợ lý tri thức học tập của Noelys.\n"
            f"Tóm tắt nội dung bên dưới bằng {lang_name}, CHỈ dựa trên BẰNG CHỨNG được cung cấp.\n\n"
            f"QUY TẮC:\n"
            f"- Viết 3-5 câu thành đoạn văn liền mạch, dễ hiểu, nắm bắt ý chính.\n"
            f"- Không thêm kiến thức ngoài BẰNG CHỨNG, không suy diễn.\n"
            f"- Không liệt kê chi tiết vụn vặt; ưu tiên tổng hợp ý lớn.\n\n"
            f"BẰNG CHỨNG:\n{self._format_summary_evidence(reranked)}\n\n"
            f"TÓM TẮT:"
        )
        prompt = (
            f"{evidence_safety}\n\n"
            "You are Noelys, a learning knowledge assistant.\n"
            f"Summarize the evidence below in {lang_name}. Use ONLY the supplied evidence.\n\n"
            "Rules:\n"
            "- Write 3-5 clear, connected sentences.\n"
            "- Do not add outside knowledge or speculation.\n"
            "- Focus on the main ideas, not minor details.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "SUMMARY:"
        )
        try:
            summary = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.exception("LLM failed in SummaryService", extra={"owner_id": request.owner_id})
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason="Summary pipeline failed. Please retry later.",
                coverage=coverage,
            )
        from src.inference.response_parser import _fix_numbered_lists
        summary = self.response_parser.inject_citations(_fix_numbered_lists(summary), reranked)
        # Verify ONLY against contradiction (token-overlap based, false positives common).
        # Summary is inherently a paraphrase task — NOT_ENOUGH_EVIDENCE is the default state
        # of paraphrasing and should not block the response.
        verification = await self.claim_verifier.averify(
            claim=summary,
            evidence=[evidence for chunk in reranked for evidence in chunk.evidence],
        )
        if verification.verdict == ClaimVerdict.CONTRADICTED and len(verification.corrected_facts) >= 2:
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason="Summary contradicts the retrieved evidence on multiple facts.",
                coverage=coverage,
            )
        return SummaryResponse(summary=summary, citations=citations, confidence=confidence, coverage=coverage)

    async def _retrieve_summary_chunks(self, *, query: str, request: SummaryRequest, scope: RetrievalScope) -> list:
        material_ids = scope.material_ids or await self._indexed_material_ids_for_collection(request)
        if len(material_ids) <= 1:
            return await self.retriever.retrieve(query=query, scope=scope, limit=max(request.top_k or 0, self.settings.rerank_input_k))

        chunks_by_id = {}
        global_limit = max(self.settings.rerank_input_k, request.top_k or 0, len(material_ids) * 3)
        global_chunks = await self.retriever.retrieve(query=query, scope=scope, limit=global_limit)
        for chunk in global_chunks:
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

        covered_material_ids = {chunk.material_id for chunk in chunks_by_id.values()}
        missing_material_ids = [material_id for material_id in material_ids if material_id not in covered_material_ids]
        per_material_limit = 1
        for material_id in missing_material_ids:
            material_scope = RetrievalScope(
                owner_id=request.owner_id,
                collection_id=request.collection_id,
                material_ids=[material_id],
            )
            material_chunks = await self.retriever.retrieve(query=query, scope=material_scope, limit=per_material_limit)
            for chunk in material_chunks[:1]:
                chunks_by_id[chunk.chunk_id] = chunk
        return list(chunks_by_id.values())

    async def _indexed_material_ids_for_collection(self, request: SummaryRequest) -> list[str]:
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
        except Exception as exc:
            logger.debug("Could not list indexed materials for collection summary", extra={"error": str(exc)})
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

    @staticmethod
    def _request_material_ids(request: SummaryRequest) -> list[str]:
        ids = list(request.material_ids or [])
        if request.material_id:
            ids.append(request.material_id)
        return list(dict.fromkeys(mid for mid in ids if mid))

    def _summary_target_limit(self, *, request: SummaryRequest, material_ids: list[str]) -> int:
        requested = request.top_k or self.settings.final_top_k
        return max(requested, len(material_ids), 1)

    @staticmethod
    def _ensure_material_coverage(*, chunks: list, selected: list, limit: int) -> list:
        by_material = {}
        for chunk in chunks:
            by_material.setdefault(chunk.material_id, chunk)

        result = []
        seen_chunk_ids = set()
        seen_material_ids = set()
        for chunk in selected:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            if chunk.material_id in seen_material_ids:
                continue
            result.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            seen_material_ids.add(chunk.material_id)
            if len(result) >= limit:
                return result[:limit]

        for material_id, chunk in by_material.items():
            if len(result) >= limit:
                break
            if material_id in seen_material_ids or chunk.chunk_id in seen_chunk_ids:
                continue
            result.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            seen_material_ids.add(material_id)

        for chunk in selected:
            if len(result) >= limit:
                break
            if chunk.chunk_id in seen_chunk_ids:
                continue
            result.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
        return result[:limit]

    @staticmethod
    def _format_summary_evidence(chunks: list, max_chars_per_chunk: int = 700) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            pages = sorted(set(chunk.source_pages))
            page_text = f"trang {pages[0]}" if len(pages) == 1 else (f"trang {pages[0]}-{pages[-1]}" if pages else "không rõ trang")
            content = " ".join(chunk.content.split())
            if len(content) > max_chars_per_chunk:
                content = content[: max_chars_per_chunk - 3].rstrip() + "..."
            lines.append(f"[{index}] Nguồn: {chunk.document_name} ({page_text})\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extractive_collection_summary(chunks: list) -> str:
        lines = ["Tóm tắt theo từng nguồn:"]
        for index, chunk in enumerate(chunks, start=1):
            snippet = " ".join(chunk.content.split())
            if len(snippet) > 320:
                snippet = snippet[:317].rstrip() + "..."
            pages = sorted(set(chunk.source_pages))
            page_text = f", trang {pages[0]}" if pages else ""
            lines.append(f"{index}. {chunk.document_name}{page_text}: {snippet}")
        lines.append("")
        lines.append("Nhận định chung: các ý trên được rút trực tiếp từ từng nguồn trong collection; bấm citation để kiểm tra bằng chứng gốc.")
        return "\n".join(lines)

