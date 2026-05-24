from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path

from src.core.config import Settings
from src.rag.embedding_provider import VisualEmbeddingProvider

logger = logging.getLogger(__name__)


class SigLIPProvider(VisualEmbeddingProvider):
    """SigLIP image/text embedding provider (google/siglip-base-patch16-224, 768d).

    Lazy-loads on first encode call and is thread-safe via an internal RLock.
    Call unload() after bulk indexing to free RAM before the next pipeline stage.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._processor = None
        self._lock = threading.RLock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Embed images; returns one 768d normalised vector per path.

        Skips unreadable images and fills their slot with a zero vector so the
        caller always receives len(image_paths) vectors.
        """
        if not image_paths:
            return []
        self._lazy_load()
        import torch
        from PIL import Image

        device = self._settings.visual_embedding_device
        batch_size = max(1, self._settings.visual_embedding_batch_size)
        zero_vec = [0.0] * self.dense_dimension
        all_vecs: list[list[float]] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images: list = []
            mask: list[bool] = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    mask.append(True)
                except Exception as exc:
                    logger.warning(
                        "SigLIP: cannot open image",
                        extra={"path": str(p), "error": str(exc)},
                    )
                    images.append(None)
                    mask.append(False)

            valid_images = [img for img, ok in zip(images, mask) if ok]
            if not valid_images:
                all_vecs.extend([zero_vec] * len(batch_paths))
                continue

            inputs = self._processor(images=valid_images, return_tensors="pt", padding=True)
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                batch_vecs = feats.cpu().float().tolist()

            vec_iter = iter(batch_vecs)
            for ok in mask:
                all_vecs.append(next(vec_iter) if ok else zero_vec)

        return all_vecs

    def embed_image_bytes(self, data: bytes) -> list[float]:
        """Embed an in-memory image (e.g. from an upload) into a normalised 768d vector.

        Used by the Image-as-Query endpoint so we don't need to persist the file first.
        """
        if not data:
            return [0.0] * self.dense_dimension
        self._lazy_load()
        import io
        import torch
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            logger.warning("SigLIP: cannot decode image bytes", extra={"error": str(exc)})
            return [0.0] * self.dense_dimension

        device = self._settings.visual_embedding_device
        inputs = self._processor(images=[img], return_tensors="pt", padding=True)
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].cpu().float().tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a text query into the same vector space as images (cross-modal)."""
        self._lazy_load()
        import torch

        device = self._settings.visual_embedding_device
        inputs = self._processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        )
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats[0].cpu().float().tolist()

    @property
    def dense_dimension(self) -> int:
        return self._settings.visual_embedding_dense_size

    def unload(self) -> None:
        """Release model weights from memory; safe to call multiple times."""
        with self._lock:
            if self._model is None:
                return
            try:
                import torch
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("SigLIP model unloaded from memory")
            except Exception as exc:
                logger.warning(
                    "SigLIP unload error (non-fatal)",
                    extra={"error": str(exc)},
                )
                self._model = None
                self._processor = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_name = self._settings.visual_embedding_model
            device = self._settings.visual_embedding_device
            try:
                from transformers import SiglipModel, SiglipProcessor
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for SigLIP visual embedding. "
                    "Install it with: pip install transformers"
                ) from exc
            logger.info(
                "Loading SigLIP model",
                extra={"model": model_name, "device": device},
            )
            self._processor = SiglipProcessor.from_pretrained(model_name)
            self._model = SiglipModel.from_pretrained(model_name)
            self._model.eval()
            if device != "cpu":
                self._model = self._model.to(device)
            logger.info(
                "SigLIP model ready",
                extra={"model": model_name, "device": device, "dim": self.dense_dimension},
            )
