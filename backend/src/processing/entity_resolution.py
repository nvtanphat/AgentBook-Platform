from __future__ import annotations

import re
import logging
from collections import OrderedDict

from src.processing.types import ExtractedEntity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synonym knowledge base
# Each group maps all surface forms to a single canonical name (first element).
# Lower-cased keys are used for matching.
# ---------------------------------------------------------------------------

_SYNONYM_GROUPS: list[list[str]] = [
    # ── AI / ML field names ────────────────────────────────────────────────
    ["Artificial Intelligence", "AI", "Trí tuệ nhân tạo", "trí tuệ nhân tạo"],
    ["Machine Learning", "ML", "Học máy"],
    ["Deep Learning", "DL", "Học sâu", "deep neural network"],
    ["Natural Language Processing", "NLP", "Xử lý ngôn ngữ tự nhiên", "Xử lý ngôn ngữ"],
    ["Computer Vision", "CV", "Thị giác máy tính"],
    ["Reinforcement Learning", "RL", "Học tăng cường"],
    ["Transfer Learning", "Học chuyển giao"],
    ["Few-Shot Learning", "Few-shot", "Few Shot Learning"],
    ["Zero-Shot Learning", "Zero-shot", "Zero Shot Learning"],
    # ── Model families ─────────────────────────────────────────────────────
    ["Neural Network", "NN", "Mạng neural", "Mạng nơ-ron", "Artificial Neural Network", "ANN"],
    ["Convolutional Neural Network", "CNN", "Mạng tích chập"],
    ["Recurrent Neural Network", "RNN", "Mạng hồi tiếp"],
    ["Long Short-Term Memory", "LSTM"],
    ["Gated Recurrent Unit", "GRU"],
    ["Generative Adversarial Network", "GAN"],
    ["Variational Autoencoder", "VAE"],
    ["Large Language Model", "LLM"],
    ["Retrieval-Augmented Generation", "RAG"],
    # ── Algorithms / techniques ────────────────────────────────────────────
    ["Gradient Descent", "Giảm gradient"],
    ["Stochastic Gradient Descent", "SGD"],
    ["Backpropagation", "Lan truyền ngược", "back propagation", "back-propagation"],
    ["Batch Normalization", "BatchNorm", "Batch Norm"],
    ["Layer Normalization", "LayerNorm", "Layer Norm"],
    ["Dropout", "Dropout Regularization", "Dropout Layer"],
    ["Attention Mechanism", "Attention", "Self-Attention", "Cơ chế Attention"],
    ["Overfitting", "Quá khớp", "Over-fitting"],
    ["Underfitting", "Chưa khớp", "Under-fitting"],
    ["Knowledge Distillation", "Model Distillation"],
    ["Fine-tuning", "Fine tuning", "Tinh chỉnh mô hình"],
    # ── Metrics ────────────────────────────────────────────────────────────
    ["Accuracy", "Độ chính xác"],
    ["Precision", "Độ chính xác (Precision)"],
    ["Recall", "Độ nhớ", "Sensitivity"],
    ["F1-score", "F1 Score", "F1", "F-measure"],
    ["Mean Average Precision", "MAP", "mAP"],
    ["Area Under Curve", "AUC", "AUC-ROC", "ROC-AUC"],
    # ── Normalisation aliases (legacy hard-coded list) ─────────────────────
    ["L1 Regularization", "L1", "l1"],
    ["L2 Regularization", "L2", "l2", "Weight Decay"],
    ["Support Vector Machine", "SVM", "Support Vector Machines"],
    ["K-Nearest Neighbors", "KNN", "K-Nearest Neighbours", "k-NN"],
    ["Principal Component Analysis", "PCA"],
    ["Random Forest", "Random Forests"],
]

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
