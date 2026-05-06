from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.evidence import CitationSchema


class ReasoningStep(BaseModel):
    """
    Single step in the reasoning path showing how AI traversed the graph.

    Used for transparency - shows users which entities and relations were used.
    """
    step_type: Literal["retrieve", "traverse", "synthesize"]
    entities: list[str] = Field(default_factory=list)  # Entity labels involved
    relations: list[str] = Field(default_factory=list)  # Relation types used
    confidence: float = Field(ge=0.0, le=1.0)
    description: str  # Human-readable explanation


class QueryRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)
    conversation_id: str = "default"
    query: str = Field(min_length=1)
    top_k: int | None = None
    answer_language: str | None = None


class QueryResponse(BaseModel):
    answer: str
    answer_language: str = "vi"
    query_language: str
    translated_query: str | None = None
    source_languages: list[str] = Field(default_factory=list)
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
    was_refused: bool
    refusal_reason: str | None = None

    # NEW: Reasoning path for transparency
    reasoning_path: list[ReasoningStep] = Field(default_factory=list)
    coverage: "CoverageReport | None" = None
    agent_trace: "AgentTrace | None" = None


class CoverageSource(BaseModel):
    material_id: str
    name: str
    covered: bool = False


class CoverageReport(BaseModel):
    requested_count: int = 0
    covered_count: int = 0
    sources: list[CoverageSource] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    name: str
    status: Literal["pending", "running", "completed", "skipped", "failed"] = "pending"
    query: str | None = None
    sources_requested: int | None = None
    sources_covered: int | None = None
    evidence_count: int | None = None
    warning: str | None = None


class AgentVerification(BaseModel):
    verdict: str
    confidence: float
    warning: str | None = None


class AgentTrace(BaseModel):
    plan_type: str
    steps: list[AgentTraceStep] = Field(default_factory=list)
    repair_attempted: bool = False
    verification: AgentVerification | None = None


class CompareRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)
    topic: str = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=lambda: ["definition", "intuition", "example", "limitation"])
    top_k: int | None = None
    answer_language: str = "vi"


class ComparisonCell(BaseModel):
    dimension: str
    value: str
    source: str
    citation: CitationSchema | None = None
    confidence: float


class CompareResponse(BaseModel):
    topic: str
    comparison_table: list[ComparisonCell] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    citations: list[CitationSchema] = Field(default_factory=list)
    coverage: CoverageReport | None = None


class SummaryRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)
    scope: str = "document"
    top_k: int | None = None
    answer_language: str = "vi"


class SummaryResponse(BaseModel):
    summary: str
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
    was_refused: bool = False
    refusal_reason: str | None = None
    coverage: CoverageReport | None = None


class StudyGuideRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_id: str | None = None
    scope: str = "collection"
    format: str = "outline"
    top_k: int | None = None
    answer_language: str = "vi"


class StudyGuideResponse(BaseModel):
    overview: str
    key_concepts: list[str] = Field(default_factory=list)
    outline: list[str] = Field(default_factory=list)
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
