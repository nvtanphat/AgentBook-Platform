from __future__ import annotations

import logging

from src.core.base_llm import BaseLLM
from src.core.config import Settings
from src.core.model_factory import build_llm
from src.inference.confidence_scorer import ConfidenceScorer
from src.inference.response_parser import ResponseParser
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from src.schemas.query import SummaryRequest, SummaryResponse

logger = logging.getLogger(__name__)

_REFUSAL_TEXT = "Tôi không tìm thấy đủ bằng chứng trong tài liệu để tóm tắt phạm vi này."


class SummaryService:
    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm: BaseLLM | None = None,
        response_parser: ResponseParser | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings=settings, qdrant_client=get_qdrant_client_for_settings(settings))
        self.reranker = reranker or CrossEncoderReranker(settings)
        self.llm = llm or build_llm(settings)
        self.response_parser = response_parser or ResponseParser()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(settings)

    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        scope = RetrievalScope(
            owner_id=request.owner_id,
            collection_id=request.collection_id,
            material_ids=[request.material_id] if request.material_id else [],
        )
        scope.ensure_scoped()
        query = f"summarize {request.scope} key concepts"
        try:
            retrieved = await self.retriever.retrieve(query=query, scope=scope, limit=self.settings.rerank_input_k)
        except Exception as exc:
            logger.error("Retrieval failed in SummaryService", exc_info=True, extra={"owner_id": request.owner_id, "error": str(exc)})
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=[],
                confidence=0.0,
                was_refused=True,
                refusal_reason=f"Retrieval failed: {type(exc).__name__}",
            )
        reranked = self.reranker.rerank(query=query, chunks=retrieved, limit=request.top_k or self.settings.final_top_k)
        confidence = self.confidence_scorer.score(reranked)
        should_refuse, refusal_reason = self.confidence_scorer.should_refuse(chunks=reranked, confidence=confidence)
        citations = self.response_parser.citations_from_chunks(reranked)
        if should_refuse:
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=refusal_reason,
            )
        _LANG_NAMES = {"vi": "tiếng Việt", "en": "English"}
        lang_name = _LANG_NAMES.get(request.answer_language, request.answer_language)
        prompt = (
            f"Bạn là Prism — trợ lý tri thức học tập của AgentBook.\n"
            f"Tóm tắt nội dung bên dưới bằng {lang_name}, CHỈ dựa trên BẰNG CHỨNG được cung cấp.\n\n"
            f"QUY TẮC:\n"
            f"- Viết 3–5 câu đoạn văn liền mạch, dễ hiểu, nắm bắt ý chính.\n"
            f"- Không thêm kiến thức ngoài BẰNG CHỨNG, không suy diễn.\n"
            f"- Không liệt kê chi tiết vụn vặt — ưu tiên tổng hợp ý lớn.\n\n"
            f"BẰNG CHỨNG:\n{self.response_parser.format_evidence_for_prompt(reranked)}\n\n"
            f"TÓM TẮT:"
        )
        try:
            summary = await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.error("LLM failed in SummaryService", exc_info=True, extra={"owner_id": request.owner_id, "error": str(exc)})
            return SummaryResponse(
                summary=_REFUSAL_TEXT,
                citations=citations,
                confidence=confidence,
                was_refused=True,
                refusal_reason=f"LLM generation failed: {type(exc).__name__}",
            )
        from src.inference.response_parser import _fix_numbered_lists
        return SummaryResponse(summary=_fix_numbered_lists(summary), citations=citations, confidence=confidence)
