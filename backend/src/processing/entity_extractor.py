from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING

from src.core.config import get_settings, project_root
from src.processing.types import EvidenceBlock, EvidenceMap, ExtractedEntity

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capitalized-term pattern (structural, not domain-specific — stays in code)
# ---------------------------------------------------------------------------

_CAPITALIZED_TERM_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,4}\b"
)

# ---------------------------------------------------------------------------
# Junk-entity heuristics (structural patterns — not domain-specific)
# ---------------------------------------------------------------------------

# Structural patterns that never change — compiled once at import time.
_JUNK_PATTERNS_STATIC = re.compile(
    r"^[\d\W]+$"           # purely digits / punctuation
    r"|^.{1}$"             # single character
    r"|[/\\<>{}\[\]|]"     # path / HTML / bracket chars
    r"|^\d+[\d\.,\s%]+$"  # numeric expressions
    r"|\bpage\s+\d+\b"    # "page 12"
    r"|\bfig(?:ure)?\s*\d+\b"  # "figure 3"
    r"|\btable\s+\d+\b",  # "table 2"
    re.IGNORECASE | re.UNICODE,
)


@lru_cache(maxsize=1)
def _junk_pattern_extra() -> re.Pattern[str] | None:
    """Compile domain-specific junk patterns from config (structural_junk_patterns).

    Returns None when the config list is empty so the hot path avoids a
    no-op regex call on every entity candidate.
    """
    patterns = get_settings().extraction_structural_junk_patterns
    if not patterns:
        return None
    return re.compile("|".join(patterns), re.IGNORECASE | re.UNICODE)

# ---------------------------------------------------------------------------
# Config-driven lazy loaders — loaded once, cached for the process lifetime
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _method_keywords() -> frozenset[str]:
    return frozenset(get_settings().extraction_method_keywords)


@lru_cache(maxsize=1)
def _metric_pattern() -> re.Pattern[str]:
    terms = get_settings().extraction_metric_terms
    if not terms:
        # Fallback if config is missing
        terms = ["accuracy", "precision", "recall", "f1", "loss", "error", "auc", "bleu", "rouge"]
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _en_stopwords() -> frozenset[str]:
    return frozenset(get_settings().extraction_en_stopwords)


@lru_cache(maxsize=1)
def _vi_stopwords() -> frozenset[str]:
    return frozenset(get_settings().extraction_vi_stopwords)


@lru_cache(maxsize=1)
def _all_stopwords() -> frozenset[str]:
    return _en_stopwords() | _vi_stopwords()


@lru_cache(maxsize=1)
def _compound_heads() -> frozenset[str]:
    return frozenset(get_settings().extraction_compound_heads)


@lru_cache(maxsize=1)
def _edge_strip_stopwords() -> frozenset[str]:
    # Vietnamese compound-head nouns must NOT be stripped from entity edges.
    # See extraction_config.yaml → compound_heads for the rationale.
    return _all_stopwords() - _compound_heads()

def _is_junk(name: str) -> bool:
    """Return True if the candidate is almost certainly not a real entity."""
    name = name.strip()
    if not name:
        return True
    cfg = get_settings()
    if len(name) < 2 or len(name.split()) > cfg.extraction_max_entity_words:
        return True
    if _JUNK_PATTERNS_STATIC.search(name):
        return True
    _extra = _junk_pattern_extra()
    if _extra and _extra.search(name):
        return True
    words = [w.lower() for w in name.split()]
    if all(w in _all_stopwords() for w in words):
        return True
    if len(words) <= cfg.extraction_stopword_only_max_words and words[0] in _all_stopwords():
        return True
    return False


def _clean_name(name: str) -> str:
    """Strip leading/trailing stopwords and normalise whitespace."""
    name = re.sub(r"\s+", " ", name).strip()
    words = name.split()
    # Strip leading/trailing stopwords — but keep compound-head nouns so phrases
    # like "điều kiện kết hôn" survive intact.
    while words and words[0].lower() in _edge_strip_stopwords():
        words.pop(0)
    while words and words[-1].lower() in _edge_strip_stopwords():
        words.pop()
    return " ".join(words) if words else name

# ---------------------------------------------------------------------------
# Prompt loader — loaded once from prompts/entity_extraction.txt
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# P3.3 — Semantic / hash-based extraction cache
# Caches GLiNER/LLM results per block content hash within a Celery worker's
# lifetime.  Same block text appearing in multiple materials is only processed
# once, giving ~30% cost reduction on collections with repeated passages.
# ---------------------------------------------------------------------------

class _ExtractionCache:
    """In-memory LRU-style cache keyed by (block_content_hash, backend)."""

    def __init__(self, max_entries: int = 2048) -> None:
        self._store: OrderedDict[str, list[dict]] = OrderedDict()
        self._max = max_entries

    def _key(self, text: str, backend: str) -> str:
        return hashlib.sha256(f"{backend}:{text}".encode()).hexdigest()

    def get(self, text: str, backend: str) -> list[dict] | None:
        k = self._key(text, backend)
        if k not in self._store:
            return None
        self._store.move_to_end(k)
        return self._store[k]

    def put(self, text: str, backend: str, entities: list[dict]) -> None:
        k = self._key(text, backend)
        self._store[k] = entities
        self._store.move_to_end(k)
        if len(self._store) > self._max:
            self._store.popitem(last=False)


_EXTRACTION_CACHE = _ExtractionCache()


# ---------------------------------------------------------------------------
# P2.1 — Smart gleaning condition
# Gleaning only fires when entity density is low OR average confidence is low.
# Avoids the brute-force +100% cost of unconditional gleaning.
# ---------------------------------------------------------------------------

def _should_glean(entities: list[ExtractedEntity], text: str) -> bool:
    """Return True when a second extraction pass is likely to find more entities."""
    density = len(entities) / max(len(text) / 1000, 1)
    avg_conf = sum(e.confidence for e in entities) / max(len(entities), 1) if entities else 0.0
    return density < 2.0 or avg_conf < 0.7


# ---------------------------------------------------------------------------
# GLiNER2 entity path — zero-LLM local span extraction (EMNLP 2025)
# Loaded lazily as a process-level singleton; ~500MB RAM, ~50ms/batch.
# Requires: pip install gliner
# ---------------------------------------------------------------------------

_gliner_instance = None
_gliner_instance_name: str | None = None


def _get_gliner_model(model_name: str):
    """Return the process-level GLiNER singleton, loading it on first call."""
    global _gliner_instance, _gliner_instance_name
    if _gliner_instance is not None and _gliner_instance_name == model_name:
        return _gliner_instance
    try:
        from gliner import GLiNER  # type: ignore[import-not-found]
        _gliner_instance = GLiNER.from_pretrained(model_name)
        _gliner_instance_name = model_name
        logger.info("GLiNER model loaded", extra={"model": model_name})
        return _gliner_instance
    except ImportError:
        logger.warning("gliner package not installed; use 'pip install gliner' to enable zero-LLM extraction")
        return None
    except Exception as exc:
        logger.warning(
            "GLiNER model load failed",
            extra={"model": model_name, "error": str(exc), "error_type": type(exc).__name__},
        )
        return None


class _GLiNEREntityPath:
    """Zero-LLM entity extraction using GLiNER2 span extraction (no hallucinations, exact spans)."""

    def __init__(self, model_name: str, threshold: float) -> None:
        self._model_name = model_name
        self._threshold = threshold

    def extract(
        self,
        evidence_map: EvidenceMap,
        entity_types: tuple[str, ...],
        *,
        enable_gleaning: bool = True,
    ) -> list[ExtractedEntity]:
        model = _get_gliner_model(self._model_name)
        if model is None:
            raise RuntimeError("GLiNER model unavailable")

        labels = list(entity_types)
        cfg = get_settings()
        block_index = {b.block_id: b for b in evidence_map.blocks}
        entities_dict: OrderedDict[str, ExtractedEntity] = OrderedDict()
        glean_threshold = max(0.25, self._threshold - 0.2)  # lower threshold for gleaning pass

        for block in evidence_map.blocks:
            text = block.snippet_original
            if not text.strip():
                continue

            # P3.3: check cache first
            cached = _EXTRACTION_CACHE.get(text, "gliner")
            if cached is not None:
                raw = cached
            else:
                try:
                    raw = model.predict_entities(text, labels, threshold=self._threshold)
                    _EXTRACTION_CACHE.put(text, "gliner", raw)
                except Exception as exc:
                    logger.debug(
                        "GLiNER prediction failed on block",
                        extra={"block_id": block.block_id, "error": str(exc)},
                    )
                    continue

            self._upsert_raw(raw, block, block_index, entities_dict, cfg)

            # P2.1: smart gleaning — second pass with lower threshold on sparse blocks
            if enable_gleaning:
                block_entities = [e for e in entities_dict.values() if block in e.mention_refs]
                if _should_glean(block_entities, text):
                    try:
                        glean_raw = model.predict_entities(text, labels, threshold=glean_threshold)
                        # Only add entities not yet found (avoid duplicates)
                        new_raw = [e for e in glean_raw if e.get("score", 0) < self._threshold]
                        self._upsert_raw(new_raw, block, block_index, entities_dict, cfg)
                    except Exception:
                        pass

        extracted = list(entities_dict.values())
        logger.info(
            "GLiNER entity extraction completed",
            extra={
                "material_id": evidence_map.material_id,
                "entities": len(extracted),
                "blocks": len(evidence_map.blocks),
            },
        )
        return extracted

    @staticmethod
    def _upsert_raw(
        raw: list[dict],
        block: EvidenceBlock,
        block_index: dict,
        entities_dict: OrderedDict,
        cfg,
    ) -> None:
        for ent in raw:
            name = _clean_name(str(ent.get("text", "")))
            if _is_junk(name):
                continue
            confidence = float(ent.get("score", 0.5))
            if confidence < cfg.extraction_min_confidence:
                continue
            entity_type = str(ent.get("label", "concept")).lower()
            key = name.lower()
            mention_blocks = _GLiNEREntityPath._find_mentions(name, block_index)
            if not mention_blocks:
                mention_blocks = [block]
            existing = entities_dict.get(key)
            if existing is None:
                entities_dict[key] = ExtractedEntity(
                    canonical_name=name,
                    entity_type=entity_type,
                    confidence=confidence,
                    mention_refs=mention_blocks,
                )
            else:
                merged_conf = min(
                    cfg.extraction_merge_confidence_max,
                    max(existing.confidence, confidence) + cfg.extraction_merge_confidence_boost,
                )
                seen_ids = {b.block_id for b in existing.mention_refs}
                new_mentions = [b for b in mention_blocks if b.block_id not in seen_ids]
                entities_dict[key] = existing.model_copy(update={
                    "confidence": merged_conf,
                    "mention_refs": existing.mention_refs + new_mentions,
                })

    @staticmethod
    def _find_mentions(name: str, block_index: dict) -> list[EvidenceBlock]:
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        return [b for b in block_index.values() if pattern.search(b.snippet_original)]


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
        """Async extraction: GLiNER2 (zero-LLM) → LLM → regex fallback chain.

        Backend is controlled by extraction_config.yaml → entity_backend:
          "gliner" — local GLiNER2 model, 0 LLM calls, no hallucinations
          "llm"    — LLM batch extraction (default)
        """
        cfg = get_settings()
        if cfg.extraction_entity_backend == "gliner":
            gliner_path = _GLiNEREntityPath(
                model_name=cfg.extraction_gliner_model,
                threshold=cfg.extraction_gliner_threshold,
            )
            try:
                return await asyncio.to_thread(
                    gliner_path.extract, evidence_map, self._default_types
                )
            except Exception as exc:
                logger.warning(
                    "GLiNER extraction failed, falling back to LLM/regex",
                    extra={"error": str(exc), "error_type": type(exc).__name__},
                )

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
            if confidence < get_settings().extraction_min_confidence:
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
                _cfg = get_settings()
                merged_confidence = min(_cfg.extraction_merge_confidence_max, max(existing.confidence, confidence) + _cfg.extraction_merge_confidence_boost)
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
            text=text[:get_settings().extraction_max_chars_per_llm_batch],
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
            return "(no specific types required — choose the most fitting snake_case label for the context)\n"
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
        cfg = get_settings()
        conf_keyword = cfg.extraction_confidence_method_keyword
        conf_metric = cfg.extraction_confidence_metric
        conf_cap = cfg.extraction_confidence_capitalized_term

        for block in evidence_map.blocks:
            text = block.snippet_original

            # 1. Domain keyword seeds
            for keyword in _method_keywords():
                if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, re.IGNORECASE):
                    key = keyword.lower()
                    _upsert_entity(
                        entities, key,
                        canonical_name=keyword.title() if keyword not in {"l1", "l2", "sgd", "adam", "rag"} else keyword.upper(),
                        entity_type="algorithm",
                        confidence=conf_keyword,
                        block=block,
                    )

            # 2. Metric terms
            for match in _metric_pattern().finditer(text):
                raw = match.group(0)
                name = _clean_name(raw)
                if _is_junk(name):
                    continue
                _upsert_entity(entities, name.lower(), canonical_name=name.lower(), entity_type="metric", confidence=conf_metric, block=block)

            # 3. Capitalized terms — stricter gate than before
            for match in _CAPITALIZED_TERM_PATTERN.finditer(text):
                term = _clean_name(match.group(0))
                if _is_junk(term):
                    continue
                if len(term) < 3:
                    continue
                if term.lower() in _all_stopwords():
                    continue
                key = term.lower()
                _upsert_entity(entities, key, canonical_name=term, entity_type="concept", confidence=conf_cap, block=block)

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

        _vi_ner_conf = get_settings().extraction_confidence_vietnamese_ner

        def flush() -> None:
            nonlocal current_words, current_type
            if current_words and current_type:
                entities.append((" ".join(current_words), current_type.lower(), _vi_ner_conf))
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
            if current and current_chars + len(text) > get_settings().extraction_max_chars_per_llm_batch:
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
