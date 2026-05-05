from __future__ import annotations

import logging
from collections import defaultdict

from src.processing.types import ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)


class GraphQualityGate:
    """Quality control for graph entities and relations.

    Responsibilities:
    - Prune low-confidence entities/relations
    - Deduplicate similar entities (entity resolution)
    - Remove orphan entities (no relations)
    - Validate evidence references
    """

    def __init__(
        self,
        *,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.5,
        min_mention_count: int = 1,
    ) -> None:
        self.min_entity_confidence = min_entity_confidence
        self.min_relation_confidence = min_relation_confidence
        self.min_mention_count = min_mention_count

    def prune_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Remove low-quality entities based on confidence and mention count."""
        before_count = len(entities)

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= self.min_entity_confidence]

        # Filter by mention count
        entities = [e for e in entities if len(e.mention_refs) >= self.min_mention_count]

        # Remove entities with empty names
        entities = [e for e in entities if e.canonical_name.strip()]

        after_count = len(entities)
        if before_count > after_count:
            logger.info(
                "Entity pruning completed",
                extra={
                    "before": before_count,
                    "after": after_count,
                    "removed": before_count - after_count,
                },
            )

        return entities

    def prune_relations(
        self,
        relations: list[ExtractedRelation],
        valid_entity_ids: set[str],
    ) -> list[ExtractedRelation]:
        """Remove low-quality relations and orphan relations."""
        before_count = len(relations)

        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.min_relation_confidence]

        # Filter by evidence (must have at least 1 evidence ref)
        relations = [r for r in relations if r.evidence_refs]

        # Remove relations pointing to non-existent entities
        relations = [
            r for r in relations
            if r.source_id in valid_entity_ids and r.target_id in valid_entity_ids
        ]

        after_count = len(relations)
        if before_count > after_count:
            logger.info(
                "Relation pruning completed",
                extra={
                    "before": before_count,
                    "after": after_count,
                    "removed": before_count - after_count,
                },
            )

        return relations

    def resolve_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Merge similar entities (entity resolution).

        Strategy:
        1. Group entities by normalized name
        2. Within each group, merge entities with high similarity
        3. Keep the entity with highest confidence as canonical
        """
        if not entities:
            return []

        # Group by normalized name (lowercase, no punctuation)
        groups: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            normalized = self._normalize_name(entity.canonical_name)
            groups[normalized].append(entity)

        resolved: list[ExtractedEntity] = []
        merge_count = 0

        for group_entities in groups.values():
            if len(group_entities) == 1:
                resolved.append(group_entities[0])
                continue

            # Sort by confidence (highest first)
            group_entities.sort(key=lambda e: e.confidence, reverse=True)

            # Merge all into the highest-confidence entity
            canonical = group_entities[0]
            for other in group_entities[1:]:
                # Merge aliases
                for alias in other.aliases:
                    if alias not in canonical.aliases and alias != canonical.canonical_name:
                        canonical.aliases.append(alias)

                # Add other's canonical name as alias if different
                if other.canonical_name != canonical.canonical_name:
                    if other.canonical_name not in canonical.aliases:
                        canonical.aliases.append(other.canonical_name)

                # Merge mention refs
                canonical.mention_refs.extend(other.mention_refs)

                # Boost confidence slightly for multiple mentions
                canonical.confidence = min(0.95, canonical.confidence + 0.05)

                merge_count += 1

            resolved.append(canonical)

        if merge_count > 0:
            logger.info(
                "Entity resolution completed",
                extra={
                    "before": len(entities),
                    "after": len(resolved),
                    "merged": merge_count,
                },
            )

        return resolved

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize entity name for grouping."""
        import re
        import unicodedata

        # Lowercase
        name = name.lower()

        # Remove diacritics (for better matching across languages)
        name = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )

        # Remove punctuation and extra spaces
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()

        return name

    def remove_orphan_entities(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[ExtractedEntity]:
        """Remove entities that have no relations (optional quality gate)."""
        # Build set of entity IDs that appear in relations
        connected_ids: set[str] = set()
        for relation in relations:
            connected_ids.add(relation.source_id)
            connected_ids.add(relation.target_id)

        # Keep entities that are connected OR have high confidence
        kept_entities = [
            entity for entity in entities
            if f"entity:{self._slug(entity.canonical_name)}" in connected_ids
            or entity.confidence >= 0.8  # Keep high-confidence entities even if orphaned
        ]

        removed = len(entities) - len(kept_entities)
        if removed > 0:
            logger.info(
                "Orphan entity removal completed",
                extra={"removed": removed, "kept": len(kept_entities)},
            )

        return kept_entities

    @staticmethod
    def _slug(value: str) -> str:
        """Convert entity name to slug for ID generation."""
        import re
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"
