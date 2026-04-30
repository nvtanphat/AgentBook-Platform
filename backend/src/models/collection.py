from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import IndexModel

from src.models.common import utc_now


class KnowledgeCollection(Document):
    name: str
    subject: str | None = None
    description: str | None = None
    owner_id: str
    material_ids: list[PydanticObjectId] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "collections"
        indexes = [
            IndexModel([("owner_id", 1), ("name", 1)], name="collections_owner_name"),
        ]
