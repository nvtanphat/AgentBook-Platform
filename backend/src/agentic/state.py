"""Blackboard shared state for the multi-agent RAG coordinating engine.

`AgentState` is the single source of truth that every specialist agent reads
from and writes to. It is the contract that turns the previously linear
pipeline into a coordinated multi-agent system — agents mutate the state in
place; the coordinator inspects state flags between turns to decide whether to
loop, refine, or finalise.

Design rules:
  - Pydantic v2 model with `arbitrary_types_allowed=True` so it can carry
    runtime objects (RouteDecision, ProcessedQuery, RetrievedChunk).
  - All evidence-trace fields ride on the chunks themselves; the state never
    flattens citations down to text-only.
  - The state never raises; agents that fail set `last_error` and the
    coordinator decides the fallback path.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.agentic.planner import AgenticSubQuestion
from src.rag.evidence import EvidenceBundle
from src.rag.types import RetrievalScope, RetrievedChunk
from src.schemas.query import AgentTraceStep, CoverageReport


class CRAGLabel(StrEnum):
    """Per-evidence CRAG triage label."""
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class CRAGEvidenceVerdict(BaseModel):
    """CRAG triage outcome for a single evidence chunk."""
    chunk_id: str
    label: CRAGLabel = CRAGLabel.AMBIGUOUS
    score: float = 0.0
    reason: str | None = None


class GuardrailReport(BaseModel):
    """Output of the GuardrailsAgent — captured per claim-verification pass."""
    verdict: str = "not_run"
    confidence: float = 0.0
    warning: str | None = None
    unsupported_sentence_count: int = 0
    invalid_citation_count: int = 0
    contradictions: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Shared blackboard for the AgenticCoordinatingEngine.

    Agents accept this state, do their work, mutate fields they own, and
    return the same instance. The coordinator orchestrates the iteration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Inputs ─────────────────────────────────────────────────────────────
    query: str
    resolved_query: str = ""
    scope: RetrievalScope
    memory_context: str | None = None
    answer_language: str | None = None
    top_k: int | None = None
    expected_material_ids: list[str] = Field(default_factory=list)

    # ── Planning ──────────────────────────────────────────────────────────
    plan_type: str = "general"
    sub_questions: list[AgenticSubQuestion] = Field(default_factory=list)
    use_graph: bool = False
    use_per_source: bool = False
    use_multi_query: bool = False
    requires_coverage: bool = False
    route: Any = None  # RouteDecision (kept untyped to avoid cycles)
    processed_query: Any = None  # ProcessedQuery
    retrieval_queries: list[str] = Field(default_factory=list)
    routed_sub_questions: list[Any] = Field(default_factory=list)  # list[RoutedSubQuestion]

    # ── Evidence layer (Blackboard) ───────────────────────────────────────
    raw_evidence: list[RetrievedChunk] = Field(default_factory=list)
    cleaned_evidence: list[RetrievedChunk] = Field(default_factory=list)
    graph_evidence: list[RetrievedChunk] = Field(default_factory=list)
    context_chunks: list[RetrievedChunk] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    crag_verdicts: list[CRAGEvidenceVerdict] = Field(default_factory=list)
    coverage: CoverageReport | None = None

    # ── Answer + verification ─────────────────────────────────────────────
    draft_answer: str = ""
    final_answer: str = ""
    citations: list[Any] = Field(default_factory=list)
    confidence_score: float = 0.0
    was_refused: bool = False
    refusal_reason: str | None = None
    claims_verified: bool = False
    critic_warnings: list[str] = Field(default_factory=list)
    guardrail_report: GuardrailReport = Field(default_factory=GuardrailReport)
    sentence_coverage_report: Any = None  # SentenceCoverageReport — see guardrails.sentence_coverage

    # ── Coordination state ────────────────────────────────────────────────
    current_iteration: int = 0
    repair_attempted: bool = False
    answer_repair_attempted: bool = False
    should_stop: bool = False
    last_error: str | None = None
    steps_history: list[AgentTraceStep] = Field(default_factory=list)

    # ── Helpers ───────────────────────────────────────────────────────────
    def record_step(self, step: AgentTraceStep) -> None:
        self.steps_history.append(step)

    def add_warning(self, message: str) -> None:
        if message and message not in self.critic_warnings:
            self.critic_warnings.append(message)

    def needs_more_evidence(self) -> bool:
        """Coordinator hook: true when CRAG flagged that evidence is weak
        AND we still have iteration budget left."""
        if not self.crag_verdicts:
            # No CRAG run yet — fall back to coverage check.
            return bool(self.requires_coverage and self.coverage and self.coverage.covered_count < self.coverage.requested_count)
        correct_count = sum(1 for v in self.crag_verdicts if v.label == CRAGLabel.CORRECT)
        ambiguous_count = sum(1 for v in self.crag_verdicts if v.label == CRAGLabel.AMBIGUOUS)
        # If we have some useful evidence (CORRECT or AMBIGUOUS) and the route is
        # FACTUAL or GENERAL, don't replan — replanning with low-quality CRAG scores
        # sends the agent into off-topic retrieval loops.
        from src.rag.query_router import RouteType
        if self.route and self.route.route_type in {RouteType.FACTUAL, RouteType.GENERAL}:
            if correct_count + ambiguous_count > 0:
                return False
        return correct_count == 0 or (len(self.crag_verdicts) > 0 and correct_count / max(1, len(self.crag_verdicts)) < 0.25)
