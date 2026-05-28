from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from typing import TYPE_CHECKING

from src.core.config import project_root
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain keyword seed list — regex fast-path only
# ---------------------------------------------------------------------------

_METHOD_KEYWORDS: frozenset[str] = frozenset({
    # Optimisation
    "dropout", "regularization", "l1", "l2", "early stopping",
    "batch normalization", "layer normalization", "weight decay",
    "gradient descent", "stochastic gradient descent", "sgd", "adam",
    "adagrad", "rmsprop", "momentum",
    # Architectures
    "transformer", "attention", "self-attention", "cross-attention",
    "encoder", "decoder", "bert", "gpt", "t5", "llama", "mistral",
    "cnn", "rnn", "lstm", "gru", "autoencoder", "vae", "gan",
    "diffusion model", "unet",
    # Techniques
    "fine-tuning", "transfer learning", "few-shot", "zero-shot",
    "prompt engineering", "rag", "retrieval augmented generation",
    "knowledge distillation", "quantization", "pruning",
    # Vietnamese equivalents
    "học máy", "học sâu", "mạng neural", "mạng nơ-ron",
    "trí tuệ nhân tạo", "xử lý ngôn ngữ", "thị giác máy tính",
})

_METRIC_PATTERN = re.compile(
    r"\b(?:accuracy|precision|recall|f1[\s\-]?score|loss|error|auc|bleu|rouge|"
    r"perplexity|map|mrr|ndcg|throughput|latency)\b",
    re.IGNORECASE,
)

_CAPITALIZED_TERM_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,4}\b"
)

# ---------------------------------------------------------------------------
# Stopword sets — English + Vietnamese common words
# ---------------------------------------------------------------------------

_EN_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "we", "our", "they", "their", "he", "she", "his", "her",
    "you", "your", "i", "my", "me", "us",
    "and", "or", "but", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "by", "up", "as",
    "with", "from", "into", "onto", "upon", "over", "under",
    "about", "than", "then", "when", "where", "while",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "also", "both", "each", "all", "any", "some", "more", "most",
    "such", "not", "no", "if", "how", "why", "what", "who", "which",
    "figure", "table", "section", "chapter", "page", "appendix",
    "example", "note", "see", "cf", "ref", "eq",
    "abstract", "introduction", "conclusion", "references",
    "using", "based", "used", "shown", "given", "thus", "hence", "therefore",
    "however", "moreover", "furthermore", "additionally", "finally",
    "first", "second", "third", "last", "next", "previous",
    "following", "above", "below", "here", "there",
    "new", "high", "low", "large", "small", "good", "best", "better",
    "different", "same", "similar", "other", "another", "many", "few",
})

_VI_STOPWORDS: frozenset[str] = frozenset({
    "là", "và", "của", "trong", "với", "cho", "các", "có", "được",
    "này", "đó", "những", "một", "không", "khi", "từ", "tại", "theo",
    "bởi", "vì", "nên", "mà", "thì", "đã", "sẽ", "đang", "rằng",
    "như", "hay", "hoặc", "cũng", "còn", "đến", "lên", "xuống",
    "vào", "ra", "về", "do", "nếu", "để", "qua", "sau", "trước",
    "trên", "dưới", "giữa", "ngoài", "giữa", "bằng", "hơn",
    "nhất", "rất", "quá", "chỉ", "cần", "phải", "nên", "muốn",
    "thêm", "tất", "cả", "nhiều", "ít", "mỗi", "toàn", "bộ",
    "phần", "điều", "việc", "cách", "loại", "dạng", "kiểu",
    "trang", "bảng", "hình", "mục", "chương", "ví", "dụ",
    "thứ", "nhất", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám",
})

_ALL_STOPWORDS: frozenset[str] = _EN_STOPWORDS | _VI_STOPWORDS

# ---------------------------------------------------------------------------
# Junk-entity heuristics
# ---------------------------------------------------------------------------

_JUNK_PATTERNS = re.compile(
    r"^[\d\W]+$"           # purely digits / punctuation
    r"|^.{1}$"             # single character
    r"|[/\\<>{}\[\]|]"     # path / HTML / bracket chars
    r"|^\d+[\d\.,\s%]+$"  # numeric expressions
    r"|\bpage\s+\d+\b"    # "page 12"
    r"|\bfig(?:ure)?\s*\d+\b"  # "figure 3"
    r"|\btable\s+\d+\b",   # "table 2"
    re.IGNORECASE,
)


def _is_junk(name: str) -> bool:
    """Return True if the candidate is almost certainly not a real entity."""
    name = name.strip()
    if not name:
        return True
    # Too short (≤ 1 char) or too long (> 7 words — likely a phrase/sentence)
    if len(name) < 2 or len(name.split()) > 7:
        return True
    # Regex-detected junk patterns
    if _JUNK_PATTERNS.search(name):
        return True
    # All words are stopwords
    words = [w.lower() for w in name.split()]
    if all(w in _ALL_STOPWORDS for w in words):
        return True
    # Starts AND ends with stopword AND only 1–2 words → "The Model" etc.
    if len(words) <= 2 and words[0] in _ALL_STOPWORDS:
        return True
    return False


def _clean_name(name: str) -> str:
    """Strip leading/trailing stopwords and normalise whitespace."""
    name = re.sub(r"\s+", " ", name).strip()
    words = name.split()
    # Strip leading stopwords
    while words and words[0].lower() in _ALL_STOPWORDS:
        words.pop(0)
    # Strip trailing stopwords
    while words and words[-1].lower() in _ALL_STOPWORDS:
        words.pop()
    return " ".join(words) if words else name

# ---------------------------------------------------------------------------
# Prompt loader — loaded once from prompts/entity_extraction.txt
# ---------------------------------------------------------------------------

_MAX_CHARS_PER_BATCH = 3000  # token budget per LLM call
_PROMPT_TEMPLATE: str | None = None


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        path = project_root() / "backend" / "src" / "prompts" / "entity_extraction.txt"
        _PROMPT_TEMPLATE = path.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


_DEFAULT_TYPES = (
    "concept", "person", "organization", "location",
    "event", "artifact", "time", "quantity",
)


class EntityExtractor:
    """
    Two-path entity extractor:
    - Async (LLM-based): structured extraction with few-shot prompting + heuristic cleaning.
    - Sync  (regex-based): fast fallback when LLM is unavailable.

    Domain-agnostic by design — types/few-shots come from extraction_config.yaml,
    domain hint comes from KnowledgeCollection.subject at call time.

    Public interface is unchanged: extract(evidence_map) → list[ExtractedEntity].
    New async interface:         extract_async(evidence_map, domain_hint=…) → list[ExtractedEntity].
    """

    def __init__(
        self,
        *,
        llm: BaseLLM | None = None,
        default_entity_types: list[str] | None = None,
        few_shots: list[dict] | None = None,
        mode: str = "dynamic",
    ) -> None:
        self._llm = llm
        self._default_types: tuple[str, ...] = tuple(default_entity_types or _DEFAULT_TYPES)
        self._few_shots: list[dict] = list(few_shots or [])
        self._mode: str = (mode or "dynamic").lower()
        self._underthesea_ner = None
        self._underthesea_checked = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, evidence_map: EvidenceMap) -> list[ExtractedEntity]:
        """Sync regex-based extraction. Used when no LLM is configured."""
        return self._extract_regex(evidence_map)

    async def extract_async(
        self,
        evidence_map: EvidenceMap,
        *,
        domain_hint: str | None = None,
    ) -> list[ExtractedEntity]:
        """Async LLM-first extraction with regex fallback.

        `domain_hint` is a free-text description of the collection topic
        (typically `KnowledgeCollection.subject`). It tunes the LLM toward
        domain-specific entity types without code changes.
        """
        if self._llm is None:
            return self._extract_regex(evidence_map)
        try:
            return await self._extract_llm(evidence_map, domain_hint=domain_hint)
        except Exception as exc:
            logger.warning(
                "LLM entity extraction failed, falling back to regex",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            return self._extract_regex(evidence_map)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _extract_llm(
        self, evidence_map: EvidenceMap, *, domain_hint: str | None = None,
    ) -> list[ExtractedEntity]:
        """Batch blocks into LLM calls, parse structured JSON output."""
        batches = self._make_batches(evidence_map.blocks)
        all_raw: list[tuple[dict, EvidenceBlock]] = []

        tasks = [self._call_llm_batch(batch, domain_hint=domain_hint) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for batch, result in zip(batches, results):
            if isinstance(result, Exception):
                logger.warning(
                    "LLM batch extraction failed",
                    extra={"error": str(result), "blocks": len(batch)},
                )
                continue
            for raw_entity in result:
                # Attach first block in batch as evidence anchor
                all_raw.append((raw_entity, batch[0]))

        # Build entity objects, run heuristic cleaner
        entities: OrderedDict[str, ExtractedEntity] = OrderedDict()
        block_index = self._build_block_index(evidence_map)

        for raw, anchor_block in all_raw:
            name = str(raw.get("name", "")).strip()
            name = _clean_name(name)
            if _is_junk(name):
                continue
            entity_type = str(raw.get("type", "concept")).lower()
            confidence = float(raw.get("confidence", 0.7))
            if confidence < 0.5:
                continue
            # Strict mode: drop entities whose type isn't in the default seed list
            if self._mode == "strict" and entity_type not in self._default_types:
                continue

            key = name.lower()
            existing = entities.get(key)
            # Find evidence blocks that mention this entity name
            mention_blocks = self._find_mentions(name, block_index)
            if not mention_blocks:
                mention_blocks = [anchor_block]

            if existing is None:
                entities[key] = ExtractedEntity(
                    canonical_name=name,
                    entity_type=entity_type,
                    confidence=confidence,
                    mention_refs=mention_blocks,
                )
            else:
                # Merge: keep higher confidence, extend mentions
                merged_confidence = min(0.97, max(existing.confidence, confidence) + 0.02)
                seen_ids = {b.block_id for b in existing.mention_refs}
                new_mentions = [b for b in mention_blocks if b.block_id not in seen_ids]
                entities[key] = existing.model_copy(update={
                    "confidence": merged_confidence,
                    "mention_refs": existing.mention_refs + new_mentions,
                })

        extracted = list(entities.values())
        logger.info(
            "LLM entity extraction completed",
            extra={
                "material_id": evidence_map.material_id,
                "entities": len(extracted),
                "blocks_processed": len(evidence_map.blocks),
            },
        )
        return extracted

    async def _call_llm_batch(
        self, blocks: list[EvidenceBlock], *, domain_hint: str | None = None,
    ) -> list[dict]:
        """Call LLM for a batch of blocks, return parsed JSON list."""
        text = "\n\n".join(b.snippet_original for b in blocks if b.snippet_original.strip())
        if not text.strip():
            return []

        prompt = _load_prompt_template().format(
            text=text[:_MAX_CHARS_PER_BATCH],
            entity_types_block=self._render_entity_types_block(),
            domain_hint_block=self._render_domain_hint_block(domain_hint),
            few_shots_block=self._render_few_shots_block(),
        )
        raw = await self._llm.generate(prompt=prompt)  # type: ignore[union-attr]
        return self._parse_llm_json(raw)

    # ------------------------------------------------------------------
    # Prompt rendering helpers
    # ------------------------------------------------------------------

    def _render_entity_types_block(self) -> str:
        """Bullet list for prompt {entity_types_block}. Empty when mode=simple."""
        if self._mode == "simple":
            return "(không bắt buộc loại cụ thể — hãy chọn nhãn snake_case phù hợp nhất với ngữ cảnh)\n"
        return "".join(f"- {t}\n" for t in self._default_types)

    @staticmethod
    def _render_domain_hint_block(domain_hint: str | None) -> str:
        """Optional context line. Empty when no hint provided."""
        hint = (domain_hint or "").strip()
        if not hint:
            return ""
        return (
            f"Tài liệu thuộc lĩnh vực: {hint}.\n"
            "Khi gặp thực thể không khớp các loại bên dưới, hãy đề xuất loại mới snake_case "
            "tiếng Anh phù hợp lĩnh vực này (vd. \"regulation\", \"court_case\", \"dataset\").\n\n"
        )

    def _render_few_shots_block(self) -> str:
        """Render YAML-loaded few-shot examples into prompt-ready text.

        Falls back to one minimal abstract example if no shots configured —
        keeps the prompt grounded even with empty extraction_config.yaml.
        """
        shots = self._few_shots or [{
            "input": "Tổ chức A được B sáng lập năm 2020 tại C để nghiên cứu D.",
            "output": [
                {"name": "A", "type": "organization", "confidence": 0.9},
                {"name": "B", "type": "person", "confidence": 0.85},
                {"name": "2020", "type": "time", "confidence": 0.95},
                {"name": "C", "type": "location", "confidence": 0.9},
                {"name": "D", "type": "concept", "confidence": 0.7},
            ],
        }]
        parts: list[str] = []
        for shot in shots:
            inp = str(shot.get("input", "")).strip()
            out = shot.get("output", [])
            if not inp:
                continue
            try:
                out_json = json.dumps(out, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
            parts.append(f"Input: \"{inp}\"\nOutput:\n{out_json}\n")
        return "\n".join(parts) + "\n"

    @staticmethod
    def _parse_llm_json(raw: str) -> list[dict]:
        """Extract JSON array from LLM output — tolerant of surrounding text."""
        raw = raw.strip()
        # Try direct parse first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [r for r in parsed if isinstance(r, dict) and "name" in r]
        except json.JSONDecodeError:
            pass
        # Find the first [...] block
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, list):
                    return [r for r in parsed if isinstance(r, dict) and "name" in r]
            except json.JSONDecodeError:
                pass
        logger.debug("LLM returned unparseable entity JSON", extra={"raw_preview": raw[:200]})
        return []

    # ------------------------------------------------------------------
    # Regex path (enhanced fallback)
    # ------------------------------------------------------------------

    def _extract_regex(self, evidence_map: EvidenceMap) -> list[ExtractedEntity]:
        entities: OrderedDict[str, ExtractedEntity] = OrderedDict()

        for block in evidence_map.blocks:
            text = block.snippet_original

            # 1. Domain keyword seeds
            for keyword in _METHOD_KEYWORDS:
                if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE):
                    key = keyword.lower()
                    _upsert_entity(
                        entities, key,
                        canonical_name=keyword.title() if keyword not in {"l1", "l2", "sgd", "adam", "rag"} else keyword.upper(),
                        entity_type="algorithm",
                        confidence=0.78,
                        block=block,
                    )

            # 2. Metric terms
            for match in _METRIC_PATTERN.finditer(text):
                raw = match.group(0)
                name = _clean_name(raw)
                if _is_junk(name):
                    continue
                _upsert_entity(entities, name.lower(), canonical_name=name.lower(), entity_type="metric", confidence=0.72, block=block)

            # 3. Capitalized terms — stricter gate than before
            for match in _CAPITALIZED_TERM_PATTERN.finditer(text):
                term = _clean_name(match.group(0))
                if _is_junk(term):
                    continue
                if len(term) < 3:
                    continue
                if term.lower() in _ALL_STOPWORDS:
                    continue
                key = term.lower()
                _upsert_entity(entities, key, canonical_name=term, entity_type="concept", confidence=0.55, block=block)

            # 4. Vietnamese NER
            for name, entity_type, confidence in self._extract_vietnamese_ner(text):
                name = _clean_name(name)
                if _is_junk(name):
                    continue
                key = f"{entity_type}:{name.lower()}"
                _upsert_entity(entities, key, canonical_name=name, entity_type=entity_type, confidence=confidence, block=block)

        extracted = list(entities.values())
        logger.info(
            "Regex entity extraction completed",
            extra={"material_id": evidence_map.material_id, "entities": len(extracted)},
        )
        return extracted

    # ------------------------------------------------------------------
    # Vietnamese NER (underthesea, optional)
    # ------------------------------------------------------------------

    def _extract_vietnamese_ner(self, text: str) -> list[tuple[str, str, float]]:
        ner = self._load_underthesea_ner()
        if ner is None:
            return []
        try:
            tokens = ner(text)
        except Exception as exc:
            logger.debug("underthesea NER failed", extra={"error": str(exc)})
            return []

        entities: list[tuple[str, str, float]] = []
        current_words: list[str] = []
        current_type: str | None = None

        def flush() -> None:
            nonlocal current_words, current_type
            if current_words and current_type:
                entities.append((" ".join(current_words), current_type.lower(), 0.68))
            current_words = []
            current_type = None

        for item in tokens:
            if not item:
                continue
            word = str(item[0])
            tag = str(item[-1])
            if tag == "O" or "-" not in tag:
                flush()
                continue
            prefix, _, raw_type = tag.partition("-")
            entity_type = raw_type.lower()
            if prefix == "B" or entity_type != current_type:
                flush()
                current_words = [word]
                current_type = entity_type
            elif prefix == "I" and current_type == entity_type:
                current_words.append(word)
            else:
                flush()
        flush()
        return entities

    def _load_underthesea_ner(self):
        if self._underthesea_checked:
            return self._underthesea_ner
        self._underthesea_checked = True
        try:
            from underthesea import ner
            self._underthesea_ner = ner
        except Exception:
            self._underthesea_ner = None
        return self._underthesea_ner

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_batches(blocks: list[EvidenceBlock]) -> list[list[EvidenceBlock]]:
        """Group blocks into batches that fit within the char budget."""
        batches: list[list[EvidenceBlock]] = []
        current: list[EvidenceBlock] = []
        current_chars = 0
        for block in blocks:
            text = block.snippet_original or ""
            if current and current_chars + len(text) > _MAX_CHARS_PER_BATCH:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(block)
            current_chars += len(text)
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _build_block_index(evidence_map: EvidenceMap) -> dict[str, EvidenceBlock]:
        return {b.block_id: b for b in evidence_map.blocks}

    @staticmethod
    def _find_mentions(name: str, block_index: dict[str, EvidenceBlock]) -> list[EvidenceBlock]:
        """Return all blocks whose text contains the entity name (case-insensitive)."""
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        return [b for b in block_index.values() if pattern.search(b.snippet_original)]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _upsert_entity(
    entities: OrderedDict[str, ExtractedEntity],
    key: str,
    *,
    canonical_name: str,
    entity_type: str,
    confidence: float,
    block: EvidenceBlock,
) -> None:
    existing = entities.get(key)
    if existing is None:
        entities[key] = ExtractedEntity(
            canonical_name=canonical_name,
            entity_type=entity_type,
            confidence=confidence,
            mention_refs=[block],
        )
    else:
        if block.block_id not in {b.block_id for b in existing.mention_refs}:
            existing.mention_refs.append(block)
