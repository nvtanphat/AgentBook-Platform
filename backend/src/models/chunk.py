from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import IndexModel

from src.models.common import Modality, SourceLanguage, utc_now


class Chunk(Document):
    owner_id: str
    material_id: PydanticObjectId
    collection_id: PydanticObjectId
    content: str
    language: str = SourceLanguage.UNKNOWN.value
    modality: str = Modality.TEXT.value
    source_block_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    token_count: int | None = None
    embedding_model: str
    embedding_version: str
    chunk_strategy: str
    chunker_version: str
    parser_version: str
    indexed_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "chunks"
        indexes = [
            IndexModel(
                [("owner_id", 1), ("collection_id", 1), ("material_id", 1), ("language", 1), ("modality", 1)],
                name="chunks_scope_material_language_modality",
            )
        ]
