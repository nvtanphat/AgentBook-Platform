from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.evidence import CitationSchema


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


class SummaryRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_id: str | None = None
    scope: str = "document"
    top_k: int | None = None
    answer_language: str = "vi"


class SummaryResponse(BaseModel):
    summary: str
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
    was_refused: bool = False
    refusal_reason: str | None = None


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
