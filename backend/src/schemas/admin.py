from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class QueryStats(BaseModel):
    total_queries: int
    refused_queries: int
    average_confidence: float
    average_latency_ms: float


class RetrievalStats(BaseModel):
    average_top_k: float
    average_sources_used: float
    average_retrieval_time_ms: float


class AdminMetricsResponse(BaseModel):
    total_docs: int
    failed_jobs: int
    indexed_docs: int
    query_stats: QueryStats
    retrieval_stats: RetrievalStats
    feedback_count: int


class FeedbackRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    query_log_id: str = Field(min_length=1)
    rating: str = Field(min_length=1)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    query_log_id: str
    rating: str
