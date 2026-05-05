from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from src.core.config import Settings
from src.processing.types import DependencyUnavailableError

logger = logging.getLogger(__name__)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")


def _patch_flag_embedding_dtype_kwarg() -> None:
    # FlagEmbedding 1.4.x bug: passes `dtype=` to AutoModel.from_pretrained() but
    # transformers requires `torch_dtype=`. This patch fixes it without modifying site-packages.
    try:
        import FlagEmbedding.finetune.embedder.encoder_only.m3.runner as _runner
        import inspect
        src = inspect.getsource(_runner.EncoderOnlyEmbedderM3Runner.get_model)
        if "dtype=torch_dtype" not in src:
            return  # already fixed (site-packages patch applied or newer FlagEmbedding)

        import os as _os
        import torch as _torch
        from transformers import AutoModel
        from huggingface_hub import snapshot_download

        @staticmethod  # type: ignore[misc]
        def _patched(model_name_or_path, trust_remote_code=False, colbert_dim=-1, cache_dir=None, torch_dtype=None):
            cache_folder = _os.getenv("HF_HUB_CACHE", None) if cache_dir is None else cache_dir
            if not _os.path.exists(model_name_or_path):
                model_name_or_path = snapshot_download(
                    repo_id=model_name_or_path,
                    cache_dir=cache_folder,
                    ignore_patterns=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
                )
            model = AutoModel.from_pretrained(
                model_name_or_path,
                cache_dir=cache_folder,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            colbert_linear = _torch.nn.Linear(
                in_features=model.config.hidden_size,
                out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim,
                dtype=torch_dtype,
            )
            sparse_linear = _torch.nn.Linear(
                in_features=model.config.hidden_size,
                out_features=1,
                dtype=torch_dtype,
            )
            colbert_path = _os.path.join(model_name_or_path, "colbert_linear.pt")
            sparse_path = _os.path.join(model_name_or_path, "sparse_linear.pt")
            if _os.path.exists(colbert_path) and _os.path.exists(sparse_path):
                colbert_linear.load_state_dict(_torch.load(colbert_path, map_location="cpu", weights_only=True))
                sparse_linear.load_state_dict(_torch.load(sparse_path, map_location="cpu", weights_only=True))
            return {"model": model, "colbert_linear": colbert_linear, "sparse_linear": sparse_linear}

        _runner.EncoderOnlyEmbedderM3Runner.get_model = _patched
        logger.warning("Applied FlagEmbedding 1.4.x dtype→torch_dtype patch. Upgrade FlagEmbedding to remove this.")
    except Exception:
        pass


_patch_flag_embedding_dtype_kwarg()


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddedText:
    dense: list[float]
    sparse: SparseEmbedding


class BGEM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self):
        return get_cached_bge_m3_model(self.settings)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            # Fallback: word-count heuristic
            return max(1, int(len(text.split()) * 1.5))
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        return len(encoded.get("input_ids", []))

    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        if not texts:
            return []
        batch_size = max(1, self.settings.embedding_batch_size)
        max_length = max(1, self.settings.embedding_max_length)
        logger.info(
            "Encoding chunks with BGE-M3",
            extra={
                "text_count": len(texts),
                "batch_size": batch_size,
                "max_length": max_length,
                "device": self.settings.embedding_device,
            },
        )
        # Retry with halved batch size on OOM to survive low-RAM environments
        for attempt in range(4):
            try:
                output = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                break
            except (MemoryError, RuntimeError) as exc:
                if batch_size <= 1:
                    logger.error(
                        "BGE-M3 encode OOM at batch_size=1 — cannot reduce further",
                        extra={"error": str(exc)},
                    )
                    raise
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "BGE-M3 encode OOM, retrying with batch_size=%d", batch_size,
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )

        dense_vectors = output.get("dense_vecs", [])
        sparse_vectors = output.get("lexical_weights", [])
        embeddings: list[EmbeddedText] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors):
            sparse_embedding = self._to_sparse_embedding(sparse)
            dense_list = dense.tolist() if hasattr(dense, "tolist") else list(dense)
            embeddings.append(
                EmbeddedText(
                    dense=[float(value) for value in dense_list],
                    sparse=sparse_embedding,
                )
            )
        return embeddings

    @staticmethod
    def _to_sparse_embedding(value) -> SparseEmbedding:
        if isinstance(value, dict):
            items = sorted((int(index), float(weight)) for index, weight in value.items() if float(weight) != 0.0)
            return SparseEmbedding(indices=[index for index, _ in items], values=[weight for _, weight in items])
        indices = getattr(value, "indices", [])
        values = getattr(value, "values", [])
        return SparseEmbedding(indices=[int(index) for index in indices], values=[float(weight) for weight in values])


_MODEL_CACHE: dict[tuple[str, str, bool], object] = {}


def get_cached_bge_m3_model(settings: Settings):
    cache_key = (settings.embedding_model, settings.embedding_device, settings.embedding_use_fp16)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise DependencyUnavailableError("FlagEmbedding is required for BGE-M3 dense+sparse embeddings") from exc
    logger.info(
        "Loading BGE-M3 model",
        extra={
            "model": settings.embedding_model,
            "device": settings.embedding_device,
            "use_fp16": settings.embedding_use_fp16,
        },
    )
    model = BGEM3FlagModel(
        settings.embedding_model,
        use_fp16=settings.embedding_use_fp16,
        device=settings.embedding_device,
    )
    _MODEL_CACHE[cache_key] = model
    return model
