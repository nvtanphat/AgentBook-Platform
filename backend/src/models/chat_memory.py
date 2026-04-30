from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import IndexModel

from src.models.common import utc_now


class ChatSummaryMemory(Document):
    owner_id: str
    collection_id: PydanticObjectId | None = None
    conversation_id: str = "default"
    summary: str = ""
    source_query_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "chat_summary_memories"
        indexes = [
            IndexModel(
                [("owner_id", 1), ("collection_id", 1), ("conversation_id", 1)],
                name="chat_summary_scope",
                unique=True,
            ),
            IndexModel([("updated_at", -1)], name="chat_summary_updated_at"),
        ]
