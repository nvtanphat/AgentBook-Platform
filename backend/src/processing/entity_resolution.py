from __future__ import annotations

import asyncio
import re
import logging
from collections import OrderedDict

from src.processing.types import ExtractedEntity

logger = logging.getLogger(__name__)

# BGE-M3 cosine similarity thresholds for entity dedup.
#   _EMBEDDING_MERGE_THRESHOLD       — cross-TYPE pairs (stricter; avoids merging
#                                      a concept into a person, etc.)
#   _CROSS_LINGUAL_MERGE_THRESHOLD   — same-TYPE pairs (looser; catches
#                                      cross-lingual synonyms like
#                                      "Machine Learning" ↔ "Học máy" whose
#                                      BGE-M3 cosine sits ~0.72-0.80, below the
#                                      0.82 used for safe cross-type merges).
# Overridable from config (extraction_config.yaml → resolution.*).
_EMBEDDING_MERGE_THRESHOLD = 0.82
_CROSS_LINGUAL_MERGE_THRESHOLD = 0.74


def _load_merge_thresholds() -> tuple[float, float]:
    """Return (cross_type, same_type) merge thresholds from config; fall back."""
    try:
        from src.core.config import get_settings
        s = get_settings()
        cross = float(getattr(s, "extraction_merge_threshold", 0.0) or _EMBEDDING_MERGE_THRESHOLD)
        same = float(getattr(s, "extraction_cross_lingual_merge_threshold", 0.0) or _CROSS_LINGUAL_MERGE_THRESHOLD)
        return cross, same
    except Exception:
        return _EMBEDDING_MERGE_THRESHOLD, _CROSS_LINGUAL_MERGE_THRESHOLD

# ---------------------------------------------------------------------------
# Synonym knowledge base
# Each group maps all surface forms to a single canonical name (first element).
# Lower-cased keys are used for matching.
# ---------------------------------------------------------------------------

# Synonym groups removed — fully domain-agnostic now.
# Cross-document entity dedup is handled by:
#   1. Exact normalised match (ASCII fold + lowercase) — handles "Dropout" ≡ "dropout"
#   2. Fuzzy string match (rapidfuzz/difflib ≥ 88%) — handles "Transformer" ≡ "Transformers"
#   3. BGE-M3 embedding cosine ≥ 0.82 — handles semantic+cross-lingual ("Học máy" ≡ "Machine Learning")
# Add domain-specific synonyms here ONLY if you need to override the auto-detection.
_SYNONYM_GROUPS: list[list[str]] = []

# Build lookup: lower-cased surface form → canonical name
_SYNONYM_MAP: dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    _canonical = _group[0]
    for _surface in _group:
        _SYNONYM_MAP[_surface.lower()] = _canonical

# ---------------------------------------------------------------------------
# Fuzzy matching — use rapidfuzz if available, fall back to difflib
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore[import-not-found]

    def _fuzzy_ratio(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0

    _FUZZY_BACKEND = "rapidfuzz"
except ImportError:
    import difflib

    def _fuzzy_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return difflib.SequenceMatcher(None, a, b).ratio()

    _FUZZY_BACKEND = "difflib"

_FUZZY_THRESHOLD = 0.88  # strings must be ≥ 88% similar to merge

logger.debug("EntityResolver using fuzzy backend: %s", _FUZZY_BACKEND)


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------

class EntityResolver:
    """
    Resolve and deduplicate extracted entities using three strategies:

    1. Exact normalised match  — e.g. "Transformer" vs "transformer"
    2. Synonym knowledge base  — e.g. "AI" ↔ "Trí tuệ nhân tạo"
    3. Fuzzy string matching   — e.g. "Transformers" ↔ "Transformer"
       (rapidfuzz.token_sort_ratio ≥ 88%, or difflib as fallback)

    Interface is unchanged: resolve(entities) → list[ExtractedEntity].
    """

    def resolve(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        if not entities:
            return []

        # Phase 1: map every entity to its canonical synonym form
        normalised: list[tuple[str, ExtractedEntity]] = []
        for entity in entities:
            canon_key = self._to_canonical_key(entity.canonical_name)
            normalised.append((canon_key, entity))

        # Phase 2: exact + synonym merge
        merged: OrderedDict[str, ExtractedEntity] = OrderedDict()
        for canon_key, entity in normalised:
            canonical_name = _SYNONYM_MAP.get(entity.canonical_name.lower(), entity.canonical_name)
            # Re-derive key after synonym resolution
            final_key = self.normalize(canonical_name)
            existing = merged.get(final_key)
            if existing is None:
                merged[final_key] = entity.model_copy(update={
                    "canonical_name": canonical_name,
                    "aliases": sorted(set(entity.aliases + [entity.canonical_name])),
                })
            else:
                merged[final_key] = _merge(existing, entity, canonical_name)

        # Phase 3: fuzzy merge — O(n²) over remaining entities, acceptable for <500 entities
        result_list = list(merged.values())
        result_list = self._fuzzy_merge(result_list)

        logger.info(
            "Entity resolution completed",
            extra={"before": len(entities), "after": len(result_list)},
        )
        return result_list

    # ------------------------------------------------------------------
    # Pass 4 — BGE-M3 embedding merge (async, zero extra cost)
    # ------------------------------------------------------------------

    async def resolve_async(
        self,
        entities: list[ExtractedEntity],
        *,
        embedder=None,
    ) -> list[ExtractedEntity]:
        """Sync resolve (passes 1-3) + optional BGE-M3 embedding merge (pass 4).

        Pass 4 catches cross-lingual pairs like "Học máy" ↔ "Machine Learning"
        that fuzzy string match misses. Uses the existing BGEM3Embedder already
        in memory from the indexer — zero extra model loading cost.
        """
        result = self.resolve(entities)
        if embedder is None or len(result) < 2:
            return result
        try:
            result = await self._embedding_merge(result, embedder)
        except Exception as exc:
            logger.warning(
                "BGE-M3 embedding merge failed, using fuzzy-only result",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
        return result

    @staticmethod
    async def _embedding_merge(
        entities: list[ExtractedEntity],
        embedder,
    ) -> list[ExtractedEntity]:
        """Union-find merge on BGE-M3 dense cosine similarity.

        Same entity_type pairs use a looser threshold so cross-lingual synonyms
        ("Machine Learning" ↔ "Học máy") merge; cross-type pairs use the stricter
        threshold to avoid collapsing distinct kinds of entities.
        """
        import numpy as np

        cross_type_thr, same_type_thr = _load_merge_thresholds()

        names = [e.canonical_name for e in entities]
        types = [(e.entity_type or "").lower() for e in entities]
        embedded = await asyncio.to_thread(embedder.encode, names)
        dense = [e.dense for e in embedded]

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

        vecs = [np.array(d, dtype=np.float32) for d in dense]
        norms = [float(np.linalg.norm(v)) for v in vecs]

        for i in range(len(entities)):
            if norms[i] < 1e-8:
                continue
            for j in range(i + 1, len(entities)):
                if find(i) == find(j) or norms[j] < 1e-8:
                    continue
                sim = float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j]))
                threshold = same_type_thr if types[i] == types[j] else cross_type_thr
                if sim >= threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(len(entities)):
            groups.setdefault(find(i), []).append(i)

        resolved: list[ExtractedEntity] = []
        for root, indices in groups.items():
            if len(indices) == 1:
                resolved.append(entities[indices[0]])
                continue
            indices.sort(key=lambda k: entities[k].confidence, reverse=True)
            base = entities[indices[0]]
            for k in indices[1:]:
                base = _merge(base, entities[k], base.canonical_name)
            resolved.append(base)

        merged_count = len(entities) - len(resolved)
        if merged_count > 0:
            logger.info(
                "BGE-M3 embedding merge resolved %d cross-lingual entity pairs",
                merged_count,
            )
        return resolved

    # ------------------------------------------------------------------
    # Fuzzy merge pass
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_merge(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Merge pairs whose canonical names are fuzzy-similar."""
        if len(entities) <= 1:
            return entities

        # Build a union-find structure keyed by index
        parent = list(range(len(entities)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                # Keep the higher-confidence entity as root
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

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(len(entities)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        resolved: list[ExtractedEntity] = []
        for root, indices in groups.items():
            if len(indices) == 1:
                resolved.append(entities[indices[0]])
                continue
            # Sort: highest confidence first
            indices.sort(key=lambda i: entities[i].confidence, reverse=True)
            base = entities[indices[0]]
            for i in indices[1:]:
                base = _merge(base, entities[i], base.canonical_name)
            resolved.append(base)

        return resolved

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(value: str) -> str:
        """Produce a stable dedup key: lowercase, alphanum only, no extra spaces."""
        return re.sub(r"[^a-z0-9À-ɏḀ-ỿ]+", " ", value.lower()).strip()

    @staticmethod
    def _to_canonical_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge(base: ExtractedEntity, other: ExtractedEntity, canonical_name: str) -> ExtractedEntity:
    """Merge `other` into `base`, returning a new ExtractedEntity."""
    new_aliases = sorted(set(
        base.aliases
        + other.aliases
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
