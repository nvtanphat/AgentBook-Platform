from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_TRANSLATION_PROMPT = (
    "Translate the following Vietnamese search query into concise, natural English "
    "suitable for document retrieval. Keep technical terms as-is. Output ONLY the "
    "English translation on a single line — no quotes, no notes.\n\n"
    "Vietnamese: {query}\nEnglish:"
)

# HyDE: a short hypothetical English passage that *looks like* source-document
# text. Embedding it bridges the VI→EN gap better than a translated query alone,
# because it matches the documents' style and terminology, not just keywords.
_HYDE_PROMPT = (
    "Write a short, factual English passage (2-3 sentences) that could plausibly "
    "appear in an academic document and would directly answer the question below. "
    "Use precise domain terminology. Do not hedge or say you are unsure. Output "
    "only the passage.\n\nQuestion: {query}\nPassage:"
)


class ProcessedQuery(BaseModel):
    original_query: str
    query_language: str
    translated_query: str | None = None
    answer_language: str
    retrieval_queries: list[str] = Field(default_factory=list)
    # HyDE hypothetical passages — retrieval-only signals, never sent to rerank.
    hyde_passages: list[str] = Field(default_factory=list)


class QueryProcessor:
    VI_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    VI_WORDS = {
        # With diacritics
        "là", "gì", "như", "nào", "giúp", "giảm", "sánh",
        # Without diacritics — common query tokens that are unambiguously Vietnamese
        "la", "gi", "nhu", "nao", "giup", "giam", "so", "sanh",
        "khong",   # không
        "doanh",   # doanh thu / doanh nghiệp
        "thuan",   # thuần / thuận
        "nhieu",   # nhiều
        "truoc",   # trước
        "cuoi",    # cuối
        "gop",     # gộp / góp
        "phan",    # phần / phân
        "tang",    # tăng
        "biet",    # biết / biệt
        "toan",    # toàn / tổng
        "theo",    # theo
        "dung",    # đúng
        "khac",    # khác
        "nhan",    # nhân / nhận
        "tien",    # tiền / tiến
        "quan",    # quan hệ / quản
    }

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

    # Instruction verbs that should be stripped from retrieval queries so that
    # BGE-M3 focuses on the topic rather than the command.
    _INSTRUCTION_PREFIX_RE = re.compile(
        r"^(tóm tắt|tổng hợp|liệt kê|nêu|trình bày|giải thích|mô tả|hãy|cho biết|"
        r"summarize|list|describe|explain|outline|give me|tell me|what are)\s+",
        re.IGNORECASE,
    )

    # Vietnamese anaphoric pronouns that carry no retrievable content on their own.
    # Stripping them lets BGE-M3 focus on the actual predicate/comparison.
    _ANAPHORA_PRONOUN_RE = re.compile(
        r"^(nó|chúng|họ|đây|đó|này|chúng nó|chúng ta|chúng tôi|cái này|cái đó)\b\s*",
        re.IGNORECASE,
    )

    def __init__(self, llm: "BaseLLM | None" = None) -> None:
        # When an LLM is supplied, VI→EN query translation falls back to the
        # model whenever the static TRANSLATIONS dict can't produce a usable
        # English query. This is what makes cross-lingual retrieval work for
        # arbitrary English documents, not just the known ML vocabulary.
        self._llm = llm

    def _strip_anaphora(self, query: str) -> str:
        """Remove leading Vietnamese pronoun so retrieval targets the predicate."""
        return self._ANAPHORA_PRONOUN_RE.sub("", query).strip()

    def _strip_instruction(self, query: str) -> str:
        """Remove leading instruction verb so retrieval focuses on the topic."""
        return self._INSTRUCTION_PREFIX_RE.sub("", query).strip()

    def process(self, query: str, *, answer_language: str | None = None) -> ProcessedQuery:
        normalized = " ".join(query.split())
        query_language = self.detect_language(normalized)
        translated_query = self.translate_to_english(normalized) if query_language == "vi" else None

        # Build retrieval queries: always include topic-focused variant (instruction stripped).
        topic_query = self._strip_instruction(normalized)
        retrieval_queries = [normalized]
        if topic_query and topic_query.lower() != normalized.lower():
            retrieval_queries.insert(0, topic_query)  # topic first for primary embedding

        # Anaphora: if query starts with a pronoun, add a pronoun-stripped variant so
        # BGE-M3 can match the predicate even without co-reference context.
        deref_query = self._strip_anaphora(normalized)
        if deref_query and deref_query.lower() != normalized.lower() and deref_query not in retrieval_queries:
            retrieval_queries.insert(0, deref_query)

        if translated_query and translated_query.lower() != normalized.lower():
            topic_en = self._strip_instruction(translated_query)
            if topic_en not in retrieval_queries:
                retrieval_queries.append(topic_en)
            if translated_query not in retrieval_queries:
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
        hyde_enabled: bool = False,
        rewriter=None,
    ) -> ProcessedQuery:
        if rewriter is not None:
            # External rewriter takes over query expansion; bypass built-in translation.
            normalized = " ".join(query.split())
            query_language = self.detect_language(normalized)
            try:
                rewritten = await rewriter.rewrite(query)
            except Exception:
                rewritten = None
                logger.debug("Query rewriter raised — using original query only")
            if rewritten and isinstance(rewritten, list):
                retrieval_queries = [q for q in rewritten if q]
            else:
                retrieval_queries = [normalized]
            return ProcessedQuery(
                original_query=normalized,
                query_language=query_language,
                translated_query=None,
                answer_language=answer_language or ("vi" if query_language == "vi" else "en"),
                retrieval_queries=retrieval_queries,
            )
        processed = self.process(query, answer_language=answer_language)
        # Cross-lingual fallback: the static dict only covers known ML terms, so
        # a VI question over general English sources usually yields no EN query
        # and BGE-M3 alone underperforms → false refusal. When an LLM is wired,
        # translate the VI query for real and add it as a retrieval query.
        if (
            self._llm is not None
            and processed.query_language == "vi"
            and not processed.translated_query
        ):
            english = await self._translate_to_english_llm(processed.original_query)
            if english:
                processed.translated_query = english
                topic_en = self._strip_instruction(english)
                for candidate in (topic_en, english):
                    if candidate and candidate not in processed.retrieval_queries:
                        processed.retrieval_queries.append(candidate)
        # HyDE (cross-lingual recall boost): generate a hypothetical English
        # passage answering the question and add it as a retrieval-only signal.
        # Only fires for VI queries — that is the case where lexical/embedding
        # mismatch against English sources causes false refusal.
        if hyde_enabled and self._llm is not None and processed.query_language == "vi":
            seed = processed.translated_query or processed.original_query
            passage = await self._generate_hyde_en(seed)
            if passage:
                processed.hyde_passages.append(passage)
        return processed

    async def _generate_hyde_en(self, query: str) -> str | None:
        """LLM-generated hypothetical English passage for HyDE retrieval (cached)."""
        if self._llm is None:
            return None
        cached = await self._cached_translation(query, target="hyde_en")
        if cached is not None:
            return cached or None
        try:
            raw = await self._llm.generate(prompt=_HYDE_PROMPT.format(query=query))
        except Exception:
            logger.debug("HyDE generation failed", exc_info=True)
            return None
        passage = " ".join((raw or "").split()).strip()
        if len(passage) < 20:  # empty / refusal-style output is useless for retrieval
            passage = ""
        await self._store_translation(query, passage, target="hyde_en")
        return passage or None

    async def _translate_to_english_llm(self, query: str) -> str | None:
        """LLM-backed VI→EN translation with a persistent cache.

        Returns the English query, or None if translation failed / the LLM
        echoed Vietnamese back. Never raises — a failure just degrades to the
        dict-only behaviour.
        """
        if self._llm is None:
            return None
        cached = await self._cached_translation(query)
        if cached is not None:
            return cached or None  # "" = known-untranslatable, cached to skip retry
        try:
            raw = await self._llm.generate(prompt=_TRANSLATION_PROMPT.format(query=query))
        except Exception:
            logger.debug("LLM query translation failed", exc_info=True)
            return None
        english = self._clean_llm_translation(raw)
        await self._store_translation(query, english or "")
        return english

    def _clean_llm_translation(self, raw: str) -> str | None:
        if not raw or not raw.strip():
            return None
        text = raw.strip().splitlines()[0]
        text = re.sub(r"^(english|translation)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = text.strip().strip('"').strip("'").strip()
        if len(text) < 2:
            return None
        # Reject if the model echoed Vietnamese back (failed translation).
        if any(ch in self.VI_CHARS for ch in text.lower()):
            return None
        return text

    @staticmethod
    def _query_hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    async def _cached_translation(self, query: str, target: str = "en") -> str | None:
        try:
            from src.models.translation_cache import TranslationCache

            doc = await TranslationCache.find_one(
                TranslationCache.source_text_hash == self._query_hash(query),
                TranslationCache.source_language == "vi",
                TranslationCache.target_language == target,
            )
            return doc.translated_text if doc else None
        except Exception:
            return None

    async def _store_translation(self, query: str, text: str, target: str = "en") -> None:
        try:
            from src.models.translation_cache import TranslationCache

            if await self._cached_translation(query, target=target) is not None:
                return
            await TranslationCache(
                source_text_hash=self._query_hash(query),
                source_language="vi",
                target_language=target,
                translated_text=text,
                model_used=getattr(self._llm, "model_name", None) or getattr(self._llm, "model", "llm"),
            ).insert()
        except Exception:
            logger.debug("translation cache write skipped", exc_info=True)

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
        # Reject pidgin output that still contains Vietnamese diacritics or
        # untranslated Vietnamese tokens. A broken translation (e.g. "why
        # dùng wape dùng metric khác") pollutes the multi-query pool with
        # garbage chunks that crowd out genuinely-relevant evidence.
        if any(ch in self.VI_CHARS for ch in translated):
            return None
        residual_vi_tokens = set(re.findall(r"\w+", translated, flags=re.UNICODE)) & self.VI_WORDS
        if residual_vi_tokens:
            return None
        return translated
