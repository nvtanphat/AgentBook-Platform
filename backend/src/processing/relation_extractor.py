from __future__ import annotations

import re
import logging

from src.processing.types import EvidenceMap, ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extract semantic relations between entities using pattern matching.

    Supports both English and Vietnamese patterns for common relation types:
    - is_a: taxonomic/classification relations
    - part_of: compositional relations
    - causes: causal relations
    - uses: instrumental relations
    - related_to: general association
    """

    # English patterns (case-insensitive matching for entities)
    # Use non-greedy matching and word boundaries to avoid capturing too much text
    EN_PATTERNS = {
        "is_a": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+is\s+(?:a|an)\s+(?:type\s+of\s+)?(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:are|is)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "part_of": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:in|within|inside)\s+(?:the\s+)?(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:of|from)\s+(?:the\s+)?(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "causes": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<target>\b[\w][\w\s-]{1,25}?)\s+(?:is\s+)?(?:caused\s+by|due\s+to)\s+(?P<source>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "uses": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:uses?|utilizes?|employs?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:with|using)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "prevents": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:prevents?|reduces?|mitigates?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
        "improves": [
            r"(?P<source>\b[\w][\w\s-]{1,25}?)\s+(?:improves?|enhances?|increases?)\s+(?P<target>\b[\w][\w\s-]{1,25}?)\b",
        ],
    }

    # Vietnamese patterns
    VI_PATTERNS = {
        "is_a": [
            r"(?P<source>[\w\s-]{2,30})\s+là\s+(?:một\s+)?(?P<target>[\w\s-]{2,30})",
            r"(?P<source>[\w\s-]{2,30})\s+thuộc\s+(?P<target>[\w\s-]{2,30})",
        ],
        "part_of": [
            r"(?P<source>[\w\s-]{2,30})\s+trong\s+(?P<target>[\w\s-]{2,30})",
            r"(?P<source>[\w\s-]{2,30})\s+của\s+(?P<target>[\w\s-]{2,30})",
        ],
        "causes": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:gây\s+ra|dẫn\s+đến|tạo\s+ra)\s+(?P<target>[\w\s-]{2,30})",
            r"(?P<target>[\w\s-]{2,30})\s+(?:do|bởi\s+vì)\s+(?P<source>[\w\s-]{2,30})",
        ],
        "uses": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:sử\s+dụng|dùng)\s+(?P<target>[\w\s-]{2,30})",
        ],
        "prevents": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:ngăn\s+chặn|giảm|tránh)\s+(?P<target>[\w\s-]{2,30})",
        ],
        "improves": [
            r"(?P<source>[\w\s-]{2,30})\s+(?:cải\s+thiện|tăng|nâng\s+cao)\s+(?P<target>[\w\s-]{2,30})",
        ],
    }

    def __init__(self) -> None:
        # Compile patterns for performance
        self._en_compiled = {
            rel_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for rel_type, patterns in self.EN_PATTERNS.items()
        }
        self._vi_compiled = {
            rel_type: [re.compile(pattern, re.IGNORECASE | re.UNICODE) for pattern in patterns]
            for rel_type, patterns in self.VI_PATTERNS.items()
        }

    def extract(self, evidence_map: EvidenceMap, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        """Extract relations from evidence blocks given known entities.

        Args:
            evidence_map: Evidence blocks to extract relations from
            entities: Previously extracted entities to match against

        Returns:
            List of extracted relations with evidence references
        """
        if not entities:
            return []

        # Build entity lookup for fast matching
        entity_lookup = self._build_entity_lookup(entities)

        # Extract relations from each block
        relations_dict: dict[tuple[str, str, str], ExtractedRelation] = {}

        for block in evidence_map.blocks:
            text = block.snippet_original
            language = block.source_language or "en"

            # Choose pattern set based on language
            patterns = self._vi_compiled if language == "vi" else self._en_compiled

            for rel_type, compiled_patterns in patterns.items():
                for pattern in compiled_patterns:
                    for match in pattern.finditer(text):
                        source_text = match.group("source").strip()
                        target_text = match.group("target").strip()

                        # Match to known entities
                        source_entity = self._find_matching_entity(source_text, entity_lookup)
                        target_entity = self._find_matching_entity(target_text, entity_lookup)

                        if source_entity and target_entity and source_entity != target_entity:
                            # Create relation key
                            key = (source_entity, rel_type, target_entity)

                            if key not in relations_dict:
                                relations_dict[key] = ExtractedRelation(
                                    source_id=f"entity:{self._slug(source_entity)}",
                                    target_id=f"entity:{self._slug(target_entity)}",
                                    relation_type=rel_type,
                                    confidence=0.7,  # Pattern-based confidence
                                    evidence_refs=[],
                                )

                            # Add evidence reference
                            relations_dict[key].evidence_refs.append(block)

        # Boost confidence for relations with multiple evidence
        relations = list(relations_dict.values())
        for relation in relations:
            evidence_count = len(relation.evidence_refs)
            if evidence_count >= 3:
                relation.confidence = min(0.9, relation.confidence + 0.15)
            elif evidence_count == 2:
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
        """Build lookup map from entity text variations to canonical names."""
        lookup: dict[str, str] = {}
        for entity in entities:
            canonical_lower = entity.canonical_name.lower()

            # Add canonical name
            lookup[canonical_lower] = entity.canonical_name

            # Add aliases
            for alias in entity.aliases:
                lookup[alias.lower()] = entity.canonical_name

            # Add common variations for multi-word entities
            if ' ' in canonical_lower:
                # Remove spaces: "batch normalization" → "batchnormalization"
                lookup[canonical_lower.replace(' ', '')] = entity.canonical_name
                # Normalize multiple spaces: "batch  normalization" → "batch normalization"
                lookup[re.sub(r'\s+', ' ', canonical_lower)] = entity.canonical_name

        return lookup

    def _find_matching_entity(self, text: str, entity_lookup: dict[str, str]) -> str | None:
        """Find entity canonical name matching the given text."""
        text_lower = text.lower().strip()

        # 1. Exact match (case-insensitive)
        if text_lower in entity_lookup:
            return entity_lookup[text_lower]

        # 2. Normalized match (remove hyphens, extra spaces)
        text_normalized = re.sub(r'[-\s]+', ' ', text_lower).strip()
        for entity_text, canonical in entity_lookup.items():
            entity_normalized = re.sub(r'[-\s]+', ' ', entity_text).strip()
            if text_normalized == entity_normalized:
                return canonical

        # 3. Partial match (text contains entity or entity contains text)
        for entity_text, canonical in entity_lookup.items():
            if entity_text in text_lower or text_lower in entity_text:
                # Require at least 70% overlap to avoid false positives
                overlap = len(set(entity_text.split()) & set(text_lower.split()))
                total = len(set(entity_text.split()) | set(text_lower.split()))
                if total > 0 and overlap / total >= 0.7:
                    return canonical

        return None

    @staticmethod
    def _slug(value: str) -> str:
        """Convert entity name to slug for ID generation."""
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"
