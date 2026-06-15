from __future__ import annotations

from typing import Any
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
    # G3 — structured ids so the frontend can highlight the exact graph elements
    # used in this step (slug-form `entity:foo` matching GraphNode.id; Mongo
    # `_id` strings for relations).
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str  # Human-readable explanation


class QueryRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str | None = None
    # Per-request technique overrides for ablation testing.
    # Keys: reranker_enabled, agentic_rag_enabled
    rag_flags: dict[Literal["reranker_enabled", "agentic_rag_enabled"], bool] = Field(default_factory=dict)


class QueryByGraphRequest(BaseModel):
    """GraphRAG anchored query — user has selected one or more graph elements
    in the UI; backend uses them as the primary retrieval anchor."""
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    entity_ids: list[str] = Field(default_factory=list, max_length=20)   # slug-form ids
    relation_ids: list[str] = Field(default_factory=list, max_length=20) # Mongo Relation _id strings
    hops: int = Field(default=2, ge=1, le=2)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str | None = None


class QueryByImageRequest(BaseModel):
    """Multipart form fields for /query/ask-image (image-as-query)."""
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    conversation_id: str = Field(default="default", min_length=1, max_length=128)
    query_text: str | None = Field(default=None, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
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
    # Sentence-level Evidence Coverage report — populated by the SLEC gate.
    # Frontend uses `sentences` to render per-sentence support badges.
    sentence_coverage: "SentenceCoverageReport | None" = None
    # GraphRAG provenance — populated when the answer was anchored by graph
    # selection OR when reasoning_path captured graph elements. Frontend uses
    # these slug-style ids / Mongo ids to highlight nodes + edges that backed
    # the answer in GraphCanvas.
    used_entity_ids: list[str] = Field(default_factory=list)
    used_relation_ids: list[str] = Field(default_factory=list)
    # Observability (core/trace.py): per-request id + per-stage latency and routing
    # fields. Debug-visible; persisted into QueryLog.trace.
    query_id: str | None = None
    trace: dict[str, Any] | None = None


class SentenceSupport(BaseModel):
    """Per-sentence evidence support verdict — the visible artefact of SLEC.

    `status` drives UI rendering: supported → solid citation, partial → soft hedge
    marker (italic + question mark badge), unsupported → flagged or dropped server-side.
    """
    index: int                                     # position in original answer
    text: str                                      # sentence text after final processing
    status: Literal["supported", "partial", "unsupported"]
    score: float = Field(ge=0.0, le=1.0)           # best reranker score vs any evidence block
    supporting_block_ids: list[str] = Field(default_factory=list)  # top 1-2 blocks that grounded it
    citation_refs: list[int] = Field(default_factory=list)         # citation indices into QueryResponse.citations (1-based)


class SentenceCoverageReport(BaseModel):
    """Output of the Sentence-level Evidence Coverage gate."""
    enabled: bool = True
    total_sentences: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    dropped_count: int = 0                         # how many sentences were removed from final answer
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)  # supported / total (weighted: partial = 0.5)
    refused: bool = False                          # whether the gate triggered a full refusal
    sentences: list[SentenceSupport] = Field(default_factory=list)


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
    tool: str | None = None
    duration_ms: int | None = None
    sources_requested: int | None = None
    sources_covered: int | None = None
    evidence_count: int | None = None
    warning: str | None = None
    metadata: dict[str, Any] | None = None


class AgentVerification(BaseModel):
    verdict: str
    confidence: float
    warning: str | None = None
    unsupported_sentence_count: int | None = None
    invalid_citation_count: int | None = None
    repair_attempted: bool = False


class AgentTrace(BaseModel):
    plan_type: str
    steps: list[AgentTraceStep] = Field(default_factory=list)
    repair_attempted: bool = False
    verification: AgentVerification | None = None


class CompareRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    topic: str = Field(min_length=1, max_length=1000)
    dimensions: list[str] = Field(default_factory=lambda: ["definition", "intuition", "example", "limitation"], max_length=12)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str = "vi"


class ComparisonCell(BaseModel):
    dimension: str
    value: str
    source: str
    citation: CitationSchema | None = None
    confidence: float
    source_id: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False


class CompareSource(BaseModel):
    source_id: str
    name: str


class CompareMatrixCell(BaseModel):
    value: str
    confidence: float
    citation_ids: list[str] = Field(default_factory=list)
    missing_evidence: bool = False


class DimensionCoverage(BaseModel):
    dimension: str
    requested_count: int = 0
    covered_count: int = 0
    missing_source_ids: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    topic: str
    comparison_table: list[ComparisonCell] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    citations: list[CitationSchema] = Field(default_factory=list)
    coverage: CoverageReport | None = None
    sources: list[CompareSource] = Field(default_factory=list)
    matrix: dict[str, dict[str, CompareMatrixCell]] = Field(default_factory=dict)
    cell_citations: dict[str, list[str]] = Field(default_factory=dict)
    dimension_coverage: list[DimensionCoverage] = Field(default_factory=list)


class SummaryRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    scope: str = Field(default="document", min_length=1, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str = "vi"


class SummaryResponse(BaseModel):
    summary: str
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
    was_refused: bool = False
    refusal_reason: str | None = None
    coverage: CoverageReport | None = None


class StudyGuideRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_id: str | None = None
    scope: str = Field(default="collection", min_length=1, max_length=64)
    format: str = Field(default="outline", min_length=1, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    answer_language: str = "vi"


class StudyGuideResponse(BaseModel):
    overview: str
    key_concepts: list[str] = Field(default_factory=list)
    outline: list[str] = Field(default_factory=list)
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence: float
