from __future__ import annotations

import logging
import os
from functools import lru_cache

from src.core.config import Settings

logger = logging.getLogger(__name__)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")


@lru_cache(maxsize=2)
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for tokenizer-accurate chunking") from exc
    logger.info("Loading tokenizer for chunking", extra={"model": model_name})
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, settings: Settings) -> int:
    """Return the BGE-M3 tokenizer token count for ``text``.

    Falls back to a word-count heuristic if the tokenizer cannot be loaded
    (e.g. transformers not installed or offline). Used by the layout-aware
    chunker to keep chunks safely within ``embedding_max_length``.
    """
    if not text:
        return 0
    try:
        tokenizer = _load_tokenizer(settings.embedding_model)
    except Exception as exc:
        logger.warning("Tokenizer unavailable, using heuristic token count", extra={"error": str(exc)})
        return max(1, int(len(text.split()) * 1.5))
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    return len(encoded.get("input_ids", []))
