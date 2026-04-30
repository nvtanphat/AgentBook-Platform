from __future__ import annotations

import re
import logging
from collections import OrderedDict

from src.processing.types import EvidenceMap, ExtractedEntity

logger = logging.getLogger(__name__)


METHOD_KEYWORDS = {
    "dropout",
    "regularization",
    "l1",
    "l2",
    "early stopping",
    "batch normalization",
    "gradient descent",
    "transformer",
    "attention",
}

METRIC_PATTERN = re.compile(r"\b(?:accuracy|precision|recall|f1|loss|error|auc|bleu|rouge)\b", re.IGNORECASE)
CAPITALIZED_TERM_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*){0,4}\b")

# Common words that appear capitalized at sentence starts or in titles — low entity value
_STOPWORDS: frozenset[str] = frozenset({
    # Articles / determiners
    "the", "a", "an",
    # Demonstratives
    "this", "that", "these", "those",
    # Pronouns
    "it", "its", "we", "our", "they", "their", "he", "she", "his", "her",
    "you", "your", "i", "my", "me", "us",
    # Conjunctions / prepositions
    "and", "or", "but", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "by", "up", "as",
    "with", "from", "into", "onto", "upon", "over", "under",
    "about", "than", "then", "when", "where", "while",
    # Auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Common sentence starters
    "also", "both", "each", "all", "any", "some", "more", "most",
    "such", "not", "no", "if", "how", "why", "what", "who", "which",
    # Academic / document structure words
    "figure", "table", "section", "chapter", "page", "appendix",
    "example", "note", "see", "cf", "ref", "eq",
    "abstract", "introduction", "conclusion", "references",
    # Common sentence content words that are almost never named entities
    "using", "based", "used", "shown", "given", "thus", "hence", "therefore",
    "however", "moreover", "furthermore", "additionally", "finally", "first",
    "second", "third", "last", "next", "previous", "following", "above", "below",
})


class EntityExtractor:
    def __init__(self) -> None:
        self._underthesea_ner = None
        self._underthesea_checked = False

    def extract(self, evidence_map: EvidenceMap) -> list[ExtractedEntity]:
        entities: OrderedDict[str, ExtractedEntity] = OrderedDict()
        for block in evidence_map.blocks:
            text = block.snippet_original
            for keyword in METHOD_KEYWORDS:
                if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                    key = keyword.lower()
                    entity = entities.get(key)
                    if entity is None:
                        entity = ExtractedEntity(
                            canonical_name=keyword.title() if keyword not in {"l1", "l2"} else keyword.upper(),
                            entity_type="method",
                            confidence=0.72,
                        )
                        entities[key] = entity
                    entity.mention_refs.append(block)

            for match in METRIC_PATTERN.finditer(text):
                key = match.group(0).lower()
                entity = entities.get(key)
                if entity is None:
                    entity = ExtractedEntity(canonical_name=match.group(0).lower(), entity_type="metric", confidence=0.68)
                    entities[key] = entity
                entity.mention_refs.append(block)

            for match in CAPITALIZED_TERM_PATTERN.finditer(text):
                term = match.group(0).strip()
                # Require min length 4 and filter common non-entity words
                if len(term) < 4 or term.lower() in _STOPWORDS:
                    continue
                key = term.lower()
                entity = entities.get(key)
                if entity is None:
                    entity = ExtractedEntity(canonical_name=term, entity_type="concept", confidence=0.55)
                    entities[key] = entity
                entity.mention_refs.append(block)

            for name, entity_type, confidence in self._extract_vietnamese_ner(text):
                if len(name) < 2 or name.lower() in _STOPWORDS:
                    continue
                key = f"{entity_type}:{name.lower()}"
                entity = entities.get(key)
                if entity is None:
                    entity = ExtractedEntity(canonical_name=name, entity_type=entity_type, confidence=confidence)
                    entities[key] = entity
                entity.mention_refs.append(block)

        return list(entities.values())

    def _extract_vietnamese_ner(self, text: str) -> list[tuple[str, str, float]]:
        ner = self._load_underthesea_ner()
        if ner is None:
            return []
        try:
            tokens = ner(text)
        except Exception as exc:
            logger.debug("underthesea NER failed", extra={"error": str(exc), "error_type": type(exc).__name__})
            return []

        entities: list[tuple[str, str, float]] = []
        current_words: list[str] = []
        current_type: str | None = None

        def flush() -> None:
            nonlocal current_words, current_type
            if current_words and current_type:
                entities.append((" ".join(current_words), current_type.lower(), 0.65))
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
        except Exception:
            self._underthesea_ner = None
        else:
            self._underthesea_ner = ner
        return self._underthesea_ner
