from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from src.rag.query_rewriter import LLMQueryRewriter

logger = logging.getLogger(__name__)


class ProcessedQuery(BaseModel):
    original_query: str
    query_language: str
    translated_query: str | None = None
    answer_language: str
    retrieval_queries: list[str] = Field(default_factory=list)


class QueryProcessor:
    VI_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    VI_WORDS = {"la", "là", "gi", "gì", "nhu", "như", "nao", "nào", "giup", "giúp", "giam", "giảm", "so", "sanh", "sánh"}

    # Longer phrases must come before shorter overlapping ones (sorted by length in process())
    TRANSLATIONS: dict[str, str] = {
        # Question phrases
        "như thế nào": "how",
        "như nào": "how",
        "là gì": "what is",
        "là loại gì": "what type of",
        "có nghĩa là gì": "what does it mean",
        "có ý nghĩa gì": "what is the meaning of",
        "có thể là gì": "what can be",
        "hiểu thế nào về": "how to understand",
        "thế nào là": "what is",
        "tại sao": "why",
        "vì sao": "why",
        "khi nào": "when",
        "ở đâu": "where",
        "ai": "who",
        # Comparison / analysis
        "so sánh": "compare",
        "phân biệt": "differentiate",
        "điểm khác nhau": "differences between",
        "điểm giống nhau": "similarities between",
        "tương đồng": "similar",
        "phân tích": "analyze",
        "đánh giá": "evaluate",
        "giải thích": "explain",
        "mô tả": "describe",
        "trình bày": "present",
        "liệt kê": "list",
        "nêu rõ": "clarify",
        "tóm tắt": "summarize",
        "tổng hợp": "synthesize",
        # Common academic verbs
        "giúp": "helps",
        "giảm": "reduces",
        "tăng": "increases",
        "cải thiện": "improves",
        "ảnh hưởng đến": "affects",
        "ảnh hưởng": "affects",
        "tác động": "impact",
        "gây ra": "causes",
        "dẫn đến": "leads to",
        "sử dụng": "uses",
        "áp dụng": "applies",
        "được gọi là": "called",
        "bao gồm": "includes",
        "dựa trên": "based on",
        "liên quan": "related to",
        "khác nhau": "different",
        "phụ thuộc vào": "depends on",
        # Definition / concept
        "định nghĩa": "definition of",
        "khái niệm": "concept of",
        "nguyên lý": "principle of",
        "nguyên tắc": "principle of",
        "ví dụ": "example of",
        "hạn chế": "limitation of",
        "ưu điểm": "advantage of",
        "nhược điểm": "disadvantage of",
        "mục đích": "purpose of",
        "vai trò": "role of",
        "cách hoạt động": "how it works",
        "cách sử dụng": "how to use",
        "cơ chế": "mechanism of",
        # ML / Deep Learning
        "dropout": "dropout",
        "overfitting": "overfitting",
        "underfitting": "underfitting",
        "regularization": "regularization",
        "chuẩn hóa": "normalization",
        "mô hình": "model",
        "dữ liệu": "data",
        "tập dữ liệu": "dataset",
        "huấn luyện": "training",
        "kiểm tra": "testing",
        "xác thực": "validation",
        "độ chính xác": "accuracy",
        "hàm mất mát": "loss function",
        "hàm chi phí": "cost function",
        "gradient": "gradient",
        "lan truyền ngược": "backpropagation",
        "học máy": "machine learning",
        "học sâu": "deep learning",
        "mạng nơ-ron": "neural network",
        "mạng tích chập": "convolutional neural network",
        "transformer": "transformer",
        "attention": "attention mechanism",
        "embedding": "embedding",
        "thuật toán": "algorithm",
        "tối ưu hóa": "optimization",
        "siêu tham số": "hyperparameter",
        "epoch": "epoch",
        "batch": "batch",
        "lớp": "layer",
        # General CS / Data
        "cơ sở dữ liệu": "database",
        "lập trình": "programming",
        "phần mềm": "software",
        "phần cứng": "hardware",
        "mạng máy tính": "computer network",
        "giao thức": "protocol",
        "bảo mật": "security",
        "mã hóa": "encryption",
        "xác thực danh tính": "authentication",
    }

    # Vietnamese tokens to strip from translated output
    _VI_NOISE = re.compile(
        r"\b(của|về|trong|tại|liệu|tài liệu|này|đó|và|với|cho|các|một|những|"
        r"được|bị|có|không|rất|cũng|hay|hoặc|tức là|đó là|thì|mà|nếu|vì|"
        r"theo|qua|bởi|do|khi|sau|trước|đến|từ)\b",
        re.IGNORECASE,
    )

    def process(self, query: str, *, answer_language: str | None = None) -> ProcessedQuery:
        normalized = " ".join(query.split())
        query_language = self.detect_language(normalized)
        translated_query = self.translate_to_english(normalized) if query_language == "vi" else None
        retrieval_queries = [normalized]
        if translated_query and translated_query.lower() != normalized.lower():
            retrieval_queries.append(translated_query)
        return ProcessedQuery(
            original_query=normalized,
            query_language=query_language,
            translated_query=translated_query,
            answer_language=answer_language or ("vi" if query_language == "vi" else "en"),
            retrieval_queries=retrieval_queries,
        )

    async def process_async(
        self,
        query: str,
        *,
        answer_language: str | None = None,
        rewriter: LLMQueryRewriter | None = None,
    ) -> ProcessedQuery:
        """LLM-based query rewriting (Multi-Query / RAG-Fusion).

        Falls back to the dictionary-based ``process`` if the rewriter is unavailable
        or returns invalid output. Generates 1 original + 1 translation + up to 3
        paraphrases for parallel retrieval and RRF fusion downstream.
        """
        normalized = " ".join(query.split())
        if rewriter is None:
            return self.process(normalized, answer_language=answer_language)

        result = await rewriter.rewrite(normalized)
        if result is None:
            query_language = self.detect_language(normalized)
            logger.info("Query rewriter failed; using original multilingual query only")
            return ProcessedQuery(
                original_query=normalized,
                query_language=query_language,
                translated_query=None,
                answer_language=answer_language or ("vi" if query_language == "vi" else "en"),
                retrieval_queries=[normalized],
            )

        retrieval_queries = [normalized]
        translated_query = result.translated_query.strip() if result.translated_query else None
        if translated_query and translated_query.lower() != normalized.lower():
            retrieval_queries.append(translated_query)
        for paraphrase in result.paraphrases:
            if paraphrase.lower() not in {q.lower() for q in retrieval_queries}:
                retrieval_queries.append(paraphrase)

        return ProcessedQuery(
            original_query=normalized,
            query_language=result.language,
            translated_query=translated_query,
            answer_language=answer_language or ("vi" if result.language == "vi" else "en"),
            retrieval_queries=retrieval_queries,
        )

    def detect_language(self, query: str) -> str:
        lowered = query.lower()
        if any(char in self.VI_CHARS for char in lowered):
            return "vi"
        tokens = set(re.findall(r"\w+", lowered, flags=re.UNICODE))
        return "vi" if tokens & self.VI_WORDS else "en"

    def translate_to_english(self, query: str) -> str | None:
        translated = query.lower()
        # Apply longest-match substitution first
        for vi_phrase, en_phrase in sorted(self.TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
            translated = re.sub(rf"\b{re.escape(vi_phrase)}\b", en_phrase, translated, flags=re.IGNORECASE)
        # Strip remaining Vietnamese noise words
        translated = self._VI_NOISE.sub(" ", translated)
        translated = " ".join(translated.strip(" ?.,").split())
        if not translated or translated == query.lower():
            # No useful translation produced — BGE-M3 handles VI natively, skip dual retrieval
            return None
        return translated
