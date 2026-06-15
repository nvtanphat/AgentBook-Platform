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


class RequestTraceModel(BaseModel):
    """Per-request observability record (see core/trace.py::RequestTrace).

    Fields beyond query_id/latency_by_stage are filled progressively by later
    pipeline phases (validator_result, quality stage_verdicts, claim_count,
    citation_error_count); all optional so older logs and partial traces are valid.
    """

    query_id: str | None = None
    route: str | None = None
    modality: str | None = None
    difficulty: str | None = None
    table_query_type: str | None = None
    prompt_file: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    rerank_scores: list[float] = Field(default_factory=list)
    latency_by_stage: dict[str, int] = Field(default_factory=dict)
    validator_result: dict[str, Any] | None = None
    quality_stage_verdicts: dict[str, Any] | None = None
    claim_count: int | None = None
    citation_error_count: int | None = None


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
    trace: RequestTraceModel | None = None
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
