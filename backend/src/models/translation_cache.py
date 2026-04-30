from __future__ import annotations

from datetime import datetime

from beanie import Document
from pydantic import ConfigDict, Field
from pymongo import IndexModel

from src.models.common import utc_now


class TranslationCache(Document):
    model_config = ConfigDict(protected_namespaces=())

    source_text_hash: str
    source_language: str
    target_language: str
    translated_text: str
    model_used: str
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "translation_cache"
        indexes = [
            IndexModel(
                [("source_text_hash", 1), ("source_language", 1), ("target_language", 1)],
                unique=True,
                name="translation_cache_source_target_unique",
            )
        ]
