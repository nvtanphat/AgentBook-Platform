from __future__ import annotations

from datetime import datetime
from typing import Any

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import IndexModel

from src.models.common import utc_now
from src.models.material import BoundingBox


class QueryCitation(BaseModel):
    material_id: PydanticObjectId
    doc_name: str
    page: int | None = None
    block_id: str | None = None
    block_type: str | None = None
    content_snippet: str
    bbox: BoundingBox | None = None
    role: str = "primary"
    source_language: str
    confidence: float


class EmbeddedFeedback(BaseModel):
    rating: str
    comment: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class QueryLog(Document):
    owner_id: str
    collection_id: PydanticObjectId | None = None
    conversation_id: str = "default"
    query: str
    query_language: str
    answer: str
    citations: list[QueryCitation] = Field(default_factory=list)
    confidence: float
    was_refused: bool = False
    refusal_reason: str | None = None
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    feedback: list[EmbeddedFeedback] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "query_logs"
        indexes = [
            IndexModel([("created_at", -1)], name="query_logs_created_at"),
            IndexModel([("owner_id", 1), ("collection_id", 1), ("created_at", -1)], name="query_logs_scope_created_at"),
            IndexModel([("owner_id", 1), ("collection_id", 1), ("conversation_id", 1), ("created_at", -1)], name="query_logs_conversation_created_at"),
        ]
