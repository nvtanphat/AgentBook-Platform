from __future__ import annotations

import logging

from src.core.config import Settings
from src.rag.embedding_provider import VisualEmbeddingProvider

logger = logging.getLogger(__name__)


def build_visual_provider(settings: Settings) -> VisualEmbeddingProvider | None:
    """Return a VisualEmbeddingProvider based on config, or None if disabled.

    Backend selection is driven by settings.visual_embedding_backend:
      - "siglip"  → SigLIPProvider (google/siglip-base-patch16-224, 768d)
      - "noop"    → None (silently disabled, used in tests / CI)

    To switch from CPU PyTorch to GPU or ONNX INT8 later, change
    visual_embedding.device / visual_embedding.embedding_backend in model_config.yaml
    and restart. No code changes needed.
    """
    if not settings.visual_embedding_enabled:
        return None

    backend = settings.visual_embedding_backend.lower()

    if backend == "siglip":
        from src.rag.visual_embedder import SigLIPProvider
        return SigLIPProvider(settings)

    if backend == "noop":
        return None

    logger.warning(
        "Unknown visual_embedding.embedding_backend=%s — visual embedding disabled",
        backend,
    )
    return None
