from __future__ import annotations

import re
import logging
import unicodedata
from collections import defaultdict

from src.processing.slug import slugify
from src.processing.types import ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy backend — same as entity_resolution.py
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore[import-not-found]

    def _fuzzy_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0

except ImportError:
    import difflib

    def _fuzzy_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return difflib.SequenceMatcher(None, a, b).ratio()

_FUZZY_THRESHOLD = 0.88

# ---------------------------------------------------------------------------
# Abbreviation expansion — handles "AI" ↔ "Artificial Intelligence" etc.
# This is a lightweight supplement to EntityResolver's synonym KB.
# ---------------------------------------------------------------------------

_ABBREV_EXPANSIONS: dict[str, str] = {
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "rl": "Reinforcement Learning",
    "nn": "Neural Network",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "gru": "Gated Recurrent Unit",
    "gan": "Generative Adversarial Network",
    "vae": "Variational Autoencoder",
    "llm": "Large Language Model",
    "rag": "Retrieval-Augmented Generation",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
    "pca": "Principal Component Analysis",
    "sgd": "Stochastic Gradient Descent",
    "map": "Mean Average Precision",
    "auc": "Area Under Curve",
}


class GraphQualityGate:
    """Quality control for graph entities and relations.

    Responsibilities:
    - Prune low-confidence / low-mention entities and relations
    - Remove orphan entities
    - Validate evidence references
    - resolve_entities(): multi-strategy deduplication
        1. Exact normalised match  (strip diacritics + lowercase)
        2. Abbreviation expansion  ("AI" → "Artificial Intelligence")
        3. Fuzzy string similarity (rapidfuzz or difflib, threshold 88%)
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

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        before = len(entities)
        entities = [e for e in entities if e.confidence >= self.min_entity_confidence]
        entities = [e for e in entities if len(e.mention_refs) >= self.min_mention_count]
        entities = [e for e in entities if e.canonical_name.strip()]
        after = len(entities)
        if before > after:
            logger.info("Entity pruning", extra={"before": before, "after": after, "removed": before - after})
        return entities

    def prune_relations(
        self,
        relations: list[ExtractedRelation],
        valid_entity_ids: set[str],
    ) -> list[ExtractedRelation]:
        before = len(relations)
        relations = [r for r in relations if r.confidence >= self.min_relation_confidence]
        relations = [r for r in relations if r.evidence_refs]
        relations = [r for r in relations if r.source_id in valid_entity_ids and r.target_id in valid_entity_ids]
        after = len(relations)
        if before > after:
            logger.info("Relation pruning", extra={"before": before, "after": after, "removed": before - after})
        return relations

    # ------------------------------------------------------------------
    # resolve_entities — multi-strategy dedup
    # ------------------------------------------------------------------

    def resolve_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """
        Three-pass merge:
        Pass 1 — exact normalised key grouping (strips diacritics, lower, no punct)
        Pass 2 — abbreviation expansion  ("AI" merges into "Artificial Intelligence")
        Pass 3 — fuzzy string similarity (union-find, threshold 88%)
        """
        if not entities:
            return []

        # ── Pass 1: exact normalised key ──────────────────────────────
        groups: dict[str, list[ExtractedEntity]] = defaultdict(list)
        for entity in entities:
            key = self._normalize_name(entity.canonical_name)
            groups[key].append(entity)

        after_pass1: list[ExtractedEntity] = []
        for group_entities in groups.values():
            after_pass1.append(_merge_group(group_entities))

        # ── Pass 2: abbreviation expansion ───────────────────────────
        after_pass2 = self._abbrev_pass(after_pass1)

        # ── Pass 3: fuzzy merge ───────────────────────────────────────
        after_pass3 = self._fuzzy_pass(after_pass2)

        merge_count = len(entities) - len(after_pass3)
        if merge_count > 0:
            logger.info(
                "Entity resolution completed",
                extra={"before": len(entities), "after": len(after_pass3), "merged": merge_count},
            )
        return after_pass3

    # ------------------------------------------------------------------
    # Pass 2: abbreviation expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _abbrev_pass(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Merge entities whose lower-cased name is a known abbreviation for another entity."""
        name_to_idx: dict[str, int] = {e.canonical_name.lower(): i for i, e in enumerate(entities)}
        merged_into: dict[int, int] = {}  # idx → target idx

        for i, entity in enumerate(entities):
            abbrev_key = entity.canonical_name.lower().strip()
            expanded = _ABBREV_EXPANSIONS.get(abbrev_key)
            if expanded is None:
                continue
            # Find the entity whose name matches the expansion
            target_idx = name_to_idx.get(expanded.lower())
            if target_idx is None:
                # Check partial: any entity whose canonical name starts with expansion
                for j, other in enumerate(entities):
                    if j == i:
                        continue
                    if other.canonical_name.lower().startswith(expanded.lower()[:10]):
                        target_idx = j
                        break
            if target_idx is not None and target_idx != i:
                merged_into[i] = target_idx

        # Apply merges
        result: list[ExtractedEntity] = []
        for i, entity in enumerate(entities):
            if i in merged_into:
                continue  # will be absorbed
            # Collect all that merged into i
            absorbed = [entities[j] for j, t in merged_into.items() if t == i]
            base = entity
            for other in absorbed:
                base = _merge_two(base, other, base.canonical_name)
            result.append(base)
        return result

    # ------------------------------------------------------------------
    # Pass 3: fuzzy merge (union-find)
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_pass(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        if len(entities) <= 1:
            return entities

        parent = list(range(len(entities)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                if entities[pi].confidence >= entities[pj].confidence:
                    parent[pj] = pi
                else:
                    parent[pi] = pj

        names = [e.canonical_name.lower() for e in entities]
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if find(i) == find(j):
                    continue
                if _fuzzy_ratio(names[i], names[j]) >= _FUZZY_THRESHOLD:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(len(entities)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        resolved: list[ExtractedEntity] = []
        for root, indices in groups.items():
            if len(indices) == 1:
                resolved.append(entities[indices[0]])
                continue
            indices.sort(key=lambda i: entities[i].confidence, reverse=True)
            base = entities[indices[0]]
            for i in indices[1:]:
                base = _merge_two(base, entities[i], base.canonical_name)
            resolved.append(base)

        return resolved

    # ------------------------------------------------------------------
    # Orphan removal
    # ------------------------------------------------------------------

    def remove_orphan_entities(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[ExtractedEntity]:
        connected_ids: set[str] = set()
        for relation in relations:
            connected_ids.add(relation.source_id)
            connected_ids.add(relation.target_id)

        kept = [
            e for e in entities
            if f"entity:{self._slug(e.canonical_name)}" in connected_ids
            or e.confidence >= 0.8
        ]
        removed = len(entities) - len(kept)
        if removed > 0:
            logger.info("Orphan entity removal", extra={"removed": removed, "kept": len(kept)})
        return kept

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Stable dedup key: strip diacritics, lowercase, no punctuation."""
        name = name.lower()
        # NFD decompose → remove combining marks
        name = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    @staticmethod
    def _slug(value: str) -> str:
        return slugify(value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_group(group: list[ExtractedEntity]) -> ExtractedEntity:
    """Merge a list of entities into one, keeping highest confidence as canonical."""
    if len(group) == 1:
        return group[0]
    group.sort(key=lambda e: e.confidence, reverse=True)
    base = group[0]
    for other in group[1:]:
        base = _merge_two(base, other, base.canonical_name)
    return base


def _merge_two(base: ExtractedEntity, other: ExtractedEntity, canonical_name: str) -> ExtractedEntity:
    new_aliases = sorted(set(
        base.aliases + other.aliases
        + [other.canonical_name]
        + ([base.canonical_name] if base.canonical_name != canonical_name else [])
    ))
    seen_ids = {b.block_id for b in base.mention_refs}
    new_mentions = base.mention_refs + [b for b in other.mention_refs if b.block_id not in seen_ids]
    return base.model_copy(update={
        "canonical_name": canonical_name,
        "aliases": new_aliases,
        "mention_refs": new_mentions,
        "confidence": min(0.97, max(base.confidence, other.confidence) + 0.02),
    })
