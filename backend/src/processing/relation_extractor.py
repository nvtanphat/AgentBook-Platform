from __future__ import annotations

import re
import logging

from src.processing.types import EvidenceMap, ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)

# Confidence tiers by pattern specificity.
# Verb-explicit patterns (causes, uses, improves…) are more reliable than
# structural ones (is_a, part_of) because they require a specific predicate word.
_CONFIDENCE = {
    "causes":   0.75,
    "prevents": 0.75,
    "improves": 0.75,
    "uses":     0.70,
    "is_a":     0.65,
    "part_of":  0.60,
}


class RelationExtractor:
    """Extract semantic relations between entities using pattern matching.

    Supports both English and Vietnamese patterns for common relation types:
    - is_a: taxonomic/classification relations (explicit "is a type of" only)
    - part_of: compositional relations (explicit "within"/"is part of")
    - causes: causal relations
    - uses: instrumental relations
    - prevents: prevention relations
    - improves: improvement relations

    Noisy broad patterns ("X is Y", "X of Y") removed — they generated
    false positives on almost every sentence.
    """

    # English patterns — verb-specific predicates only; no broad copula/prep patterns
    EN_PATTERNS = {
        "is_a": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+is\s+a\s+type\s+of\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+is\s+an?\s+example\s+of\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:refers?\s+to|denotes?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "part_of": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+is\s+(?:a\s+)?part\s+of\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:belongs?\s+to|is\s+contained\s+in)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "causes": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<target>\b[\w][\w\s-]{1,25}?)\s+is\s+caused\s+by\s+(?P<source>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "uses": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:uses?|utilizes?|employs?|applies?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "prevents": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:prevents?|reduces?|mitigates?|avoids?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "improves": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:improves?|enhances?|increases?|boosts?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
    }

    # Vietnamese patterns — verb-specific only; "của" and "là" removed (too noisy)
    VI_PATTERNS = {
        "is_a": [
            r"(?P<source>[\w\s-]{2,30})\s+là\s+(?:một\s+loại|một\s+dạng)\s+(?P<target>[\w\s-]{2,30})",
            r"(?P<source>[\w\s-]{2,30})\s+(?:được\s+gọi\s+là|còn\s+gọi\s+là)\s+(?P<target>[\w\s-]{2,30})",
        ],
        "part_of": [
            r"(?P<source>[\w\s-]{2,30})\s+là\s+(?:một\s+)?thành\s+phần\s+(?:của\s+)?(?P<target>[\w\s-]{2,30})",
            r"(?P<source>[\w\s-]{2,30})\s+thuộc\s+(?:về\s+)?(?P<target>[\w\s-]{2,30})",
        ],
        "causes": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:gây\s+ra|dẫn\s+đến|tạo\s+ra)\s+(?P<target>[\w\s-]{2,30})",
            r"(?P<target>[\w\s-]{2,30})\s+(?:do|bởi\s+vì|vì)\s+(?P<source>[\w\s-]{2,30})\s+(?:gây|tạo)",
        ],
        "uses": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:sử\s+dụng|áp\s+dụng|dùng)\s+(?P<target>[\w\s-]{2,30})",
        ],
        "prevents": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:ngăn\s+chặn|hạn\s+chế|giảm\s+thiểu|tránh)\s+(?P<target>[\w\s-]{2,30})",
        ],
        "improves": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:cải\s+thiện|nâng\s+cao|tăng\s+cường|tối\s+ưu)\s+(?P<target>[\w\s-]{2,30})",
        ],
    }

    def __init__(self) -> None:
        self._en_compiled = {
            rel_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for rel_type, patterns in self.EN_PATTERNS.items()
        }
        self._vi_compiled = {
            rel_type: [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns]
            for rel_type, patterns in self.VI_PATTERNS.items()
        }

    def extract(self, evidence_map: EvidenceMap, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        """Extract relations from evidence blocks given known entities."""
        if not entities:
            return []

        entity_lookup = self._build_entity_lookup(entities)
        relations_dict: dict[tuple[str, str, str], ExtractedRelation] = {}

        for block in evidence_map.blocks:
            text = block.snippet_original
            language = block.source_language or "en"
            patterns = self._vi_compiled if language == "vi" else self._en_compiled

            for rel_type, compiled_patterns in patterns.items():
                base_conf = _CONFIDENCE.get(rel_type, 0.65)
                for pattern in compiled_patterns:
                    for match in pattern.finditer(text):
                        source_text = match.group("source").strip()
                        target_text = match.group("target").strip()

                        source_entity = self._find_matching_entity(source_text, entity_lookup)
                        target_entity = self._find_matching_entity(target_text, entity_lookup)

                        if source_entity and target_entity and source_entity != target_entity:
                            key = (source_entity, rel_type, target_entity)
                            if key not in relations_dict:
                                relations_dict[key] = ExtractedRelation(
                                    source_id=f"entity:{self._slug(source_entity)}",
                                    target_id=f"entity:{self._slug(target_entity)}",
                                    relation_type=rel_type,
                                    confidence=base_conf,
                                    evidence_refs=[],
                                )
                            relations_dict[key].evidence_refs.append(block)

        relations = list(relations_dict.values())
        for relation in relations:
            n = len(relation.evidence_refs)
            if n >= 3:
                relation.confidence = min(0.9, relation.confidence + 0.15)
            elif n == 2:
                relation.confidence = min(0.85, relation.confidence + 0.10)

        logger.info(
            "Relation extraction completed",
            extra={
                "total_relations": len(relations),
                "avg_confidence": sum(r.confidence for r in relations) / len(relations) if relations else 0,
            },
        )
        return relations

    def _build_entity_lookup(self, entities: list[ExtractedEntity]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for entity in entities:
            canonical_lower = entity.canonical_name.lower()
            lookup[canonical_lower] = entity.canonical_name
            for alias in entity.aliases:
                lookup[alias.lower()] = entity.canonical_name
            if " " in canonical_lower:
                lookup[canonical_lower.replace(" ", "")] = entity.canonical_name
                lookup[re.sub(r"\s+", " ", canonical_lower)] = entity.canonical_name
        return lookup

    def _find_matching_entity(self, text: str, entity_lookup: dict[str, str]) -> str | None:
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in entity_lookup:
            return entity_lookup[text_lower]

        # Normalised match (remove hyphens/extra spaces)
        text_norm = re.sub(r"[-\s]+", " ", text_lower).strip()
        for entity_text, canonical in entity_lookup.items():
            if re.sub(r"[-\s]+", " ", entity_text).strip() == text_norm:
                return canonical

        # Partial match — raised to 0.85 word-overlap to reduce false positives
        for entity_text, canonical in entity_lookup.items():
            if entity_text in text_lower or text_lower in entity_text:
                words_e = set(entity_text.split())
                words_t = set(text_lower.split())
                overlap = len(words_e & words_t)
                total = len(words_e | words_t)
                if total > 0 and overlap / total >= 0.85:
                    return canonical

        return None

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"
