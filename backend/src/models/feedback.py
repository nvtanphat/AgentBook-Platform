from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import Field
from pymongo import IndexModel

from src.models.common import utc_now


class Feedback(Document):
    owner_id: str
    query_log_id: PydanticObjectId
    rating: str
    comment: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "feedback"
        indexes = [
            IndexModel([("owner_id", 1), ("query_log_id", 1)], name="feedback_owner_query_log"),
        ]
