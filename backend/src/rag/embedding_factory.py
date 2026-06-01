from __future__ import annotations

import logging

from src.core.config import Settings
from src.rag.embedding_provider import VisualEmbeddingProvider

logger = logging.getLogger(__name__)


def build_visual_provider(settings: Settings) -> VisualEmbeddingProvider | None:
    """Return a VisualEmbeddingProvider based on config, or None if disabled.

    `visual_embedding.embedding_backend` selects the RUNTIME, not the model
    family (the model itself comes from `visual_embedding.model`):
      - "pytorch"           → SigLIPProvider on PyTorch (current path)
      - "onnx"              → SigLIPProvider (ONNX INT8 path is not implemented
                              yet; falls back to the PyTorch impl)
      - "noop"/"none"/"disabled" → None (explicit off, used in tests / CI)

    To switch CPU↔GPU change visual_embedding.device in model_config.yaml; no
    code changes needed. Disabling entirely is done via `enabled: false`.
    """
    if not settings.visual_embedding_enabled:
        return None

    backend = settings.visual_embedding_backend.lower().strip()

    # Explicit opt-out values keep visual embedding off even when enabled=true.
    if backend in {"noop", "none", "disabled", ""}:
        return None

    # All active runtimes currently resolve to the SigLIP PyTorch provider.
    # ("onnx" is reserved for a future INT8 path; it uses the same impl today.)
    if backend not in {"pytorch", "onnx", "siglip"}:
        logger.warning(
            "Unknown visual_embedding.embedding_backend=%s — defaulting to PyTorch SigLIP",
            backend,
        )

    from src.rag.visual_embedder import SigLIPProvider
    return SigLIPProvider(settings)
