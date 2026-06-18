from __future__ import annotations

import asyncio
import logging
import re

_MD_BOLD = re.compile(r"\*{1,2}(.+?)\*{1,2}")
_MD_CLEAN = re.compile(r"[`_~]")

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.core.model_factory import build_llm
from src.guardrails.claim_verifier import ClaimVerdict, ClaimVerifier
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.inference_engine import InferenceEngine
from src.inference.response_parser import ResponseParser
from src.rag.evidence import CitationBuilder, EvidenceBundle
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from src.schemas.query import StudyGuideRequest, StudyGuideResponse

logger = logging.getLogger(__name__)

_REFUSAL_TEXT = "Tôi không tìm thấy đủ bằng chứng trong tài liệu để tạo study guide."


class StudyGuideService:
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

    async def build(self, request: StudyGuideRequest) -> StudyGuideResponse:
        scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=[request.material_id] if request.material_id else [],
        )
        scope.ensure_scoped()
        query = f"study guide outline key concepts {request.scope}"
        try:
            retrieved = await self.retriever.retrieve(query=query, scope=scope, limit=self.settings.rerank_input_k)
        except Exception as exc:
            logger.error("Retrieval failed in StudyGuideService", exc_info=True, extra={"owner_id": request.owner_id, "error": str(exc)})
            return StudyGuideResponse(
                overview=_REFUSAL_TEXT,
                key_concepts=[],
                outline=[],
                citations=[],
                confidence=0.0,
            )
        async with self._rerank_semaphore:
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                query=query,
                chunks=retrieved,
                limit=request.top_k or self.settings.final_top_k,
            )
        confidence = self.confidence_scorer.score(reranked)
        evidence_bundle = EvidenceBundle.from_chunks(reranked)
        citations = CitationBuilder.from_evidence_bundle(
            evidence_bundle,
            owner_id=request.owner_id,
            api_v1_prefix=self.settings.api_v1_prefix,
        )
        if not reranked:
            return StudyGuideResponse(
                overview=_REFUSAL_TEXT,
                key_concepts=[],
                outline=[],
                citations=[],
                confidence=0.0,
            )
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(request.answer_language, request.answer_language)
        evidence_text = evidence_bundle.format_for_prompt()
        evidence_safety = InferenceEngine._evidence_safety_rules()
        guide_prompt = (
            f"{evidence_safety}\n\n"
            f"Bạn là Noelys, trợ lý tri thức học tập của Noelys.\n"
            f"Tạo Study Guide bằng {lang_name}, CHỈ từ BẰNG CHỨNG bên dưới. Không thêm kiến thức ngoài tài liệu.\n\n"
            f"Trả lời theo đúng cấu trúc sau (giữ nguyên các tiêu đề):\n\n"
            f"TỔNG QUAN:\n3 đến 5 câu tóm tắt nội dung chính của tài liệu.\n\n"
            f"KHÁI NIỆM CHÍNH:\n- Khái niệm 1\n- Khái niệm 2\n- ...\n\n"
            f"DÀN Ý:\n1. Mục chính 1\n2. Mục chính 2\n3. ...\n\n"
            f"BẰNG CHỨNG:\n{evidence_text}"
        )
        try:
            raw = await self.llm.generate(prompt=guide_prompt)
        except Exception as exc:
            logger.error("LLM failed in StudyGuideService", exc_info=True, extra={"owner_id": request.owner_id, "error": str(exc)})
            return StudyGuideResponse(
                overview=_REFUSAL_TEXT,
                key_concepts=[],
                outline=[],
                citations=citations,
                confidence=confidence,
            )

        overview, key_concepts, outline = self._parse_guide_output(raw)

        # Fallback: LLM-based concept extraction if structured parse yielded nothing
        if not key_concepts:
            key_concepts = await self._extract_concepts_llm(evidence_text, request.owner_id)

        from src.inference.response_parser import _fix_numbered_lists
        verification_text = "\n".join(
            item
            for item in [overview, "\n".join(key_concepts[:10]), "\n".join(outline[:12])]
            if item.strip()
        )
        verification_text = self.response_parser.inject_citations(verification_text, evidence_bundle.to_legacy_chunks())
        verification = await self.claim_verifier.averify(
            claim=verification_text,
            evidence=[evidence for chunk in reranked for evidence in chunk.evidence],
        )
        if verification.verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.NOT_ENOUGH_EVIDENCE}:
            return StudyGuideResponse(
                overview=_REFUSAL_TEXT,
                key_concepts=[],
                outline=[],
                citations=citations,
                confidence=confidence,
            )
        return StudyGuideResponse(
            overview=_fix_numbered_lists(overview or _REFUSAL_TEXT),
            key_concepts=key_concepts[:10],
            outline=[_fix_numbered_lists(item) for item in outline[:12]],
            citations=citations,
            confidence=confidence,
        )

    @staticmethod
    def _strip_md(text: str) -> str:
        text = _MD_BOLD.sub(r"\1", text)
        return _MD_CLEAN.sub("", text).strip()

    @staticmethod
    def _parse_guide_output(raw: str) -> tuple[str, list[str], list[str]]:
        overview = ""
        key_concepts: list[str] = []
        outline: list[str] = []
        current = None
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            low = stripped.lower()
            if "tổng quan" in low or "overview" in low:
                current = "overview"
            elif "khái niệm" in low or "key concept" in low:
                current = "concepts"
            elif "dàn ý" in low or "outline" in low:
                current = "outline"
            elif current == "overview":
                overview += (" " if overview else "") + StudyGuideService._strip_md(stripped)
            elif current == "concepts" and stripped.startswith("-"):
                val = StudyGuideService._strip_md(stripped.lstrip("- "))
                if val:
                    key_concepts.append(val)
            elif current == "outline" and (stripped[0].isdigit() or stripped.startswith("-")):
                val = StudyGuideService._strip_md(stripped.lstrip("0123456789.- "))
                if val:
                    outline.append(val)
        return overview.strip(), key_concepts, outline

    async def _extract_concepts_llm(self, evidence_text: str, owner_id: str) -> list[str]:
        evidence_safety = InferenceEngine._evidence_safety_rules()
        prompt = (
            f"{evidence_safety}\n\n"
            "Từ đoạn văn dưới đây, liệt kê 5-8 khái niệm quan trọng nhất.\n"
            "Yêu cầu: mỗi khái niệm 1-3 từ, mỗi dòng một khái niệm, bắt đầu bằng dấu gạch ngang (-).\n"
            "Chỉ liệt kê khái niệm, không giải thích.\n\n"
            f"Văn bản:\n{evidence_text[:2000]}"
        )
        try:
            raw = await self.llm.generate(prompt=prompt)
            return [line.strip("- •").strip() for line in raw.strip().splitlines() if line.strip()]
        except Exception as exc:
            logger.warning("Concept extraction failed", exc_info=True, extra={"owner_id": owner_id, "error": str(exc)})
            return []

