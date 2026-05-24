from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TextEmbeddingProvider(ABC):
    """ABC for text embedding backends (dense vector output)."""

    @abstractmethod
    def encode_dense(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised dense vector per text."""
        ...

    @property
    @abstractmethod
    def dense_dimension(self) -> int:
        """Output vector dimensionality."""
        ...


class VisualEmbeddingProvider(ABC):
    """ABC for image embedding backends with cross-modal text query support."""

    @abstractmethod
    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Return one normalised dense vector per image.

        Must return a vector for every input path (use zero-vector for failures).
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a text query into the same vector space as images (cross-modal)."""
        ...

    @property
    @abstractmethod
    def dense_dimension(self) -> int:
        """Output vector dimensionality (must match the Qdrant collection config)."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights from RAM / VRAM."""
        ...
