"""RetrieverDirectorAgent — picks the right retrieval tool per sub-question
AND (in blackboard mode) executes the routed tools to populate
`state.raw_evidence`.

Rule-based (no LLM call) to keep latency manageable. Decisions follow the
planner's hints (tool field on sub-questions) but apply safety overrides:

  - If sub-question mentions a relation/dependency keyword → trace_graph
  - If sub-question asks "what does each source say" → retrieve_per_source
  - Otherwise → retrieve_text (hybrid dense + sparse + reranker)

The director also dedupes sub-questions that would route to the same tool
with near-identical text, avoiding wasted retrieval cycles.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent
from src.agentic.planner import AgenticSubQuestion

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.agentic.tools import GraphRelationSearchTool, HybridTextSearchTool
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_PER_SOURCE_KEYWORDS = re.compile(
    r"\b(?:each source|each document|per source|từng tài liệu|từng nguồn)\b",
    re.IGNORECASE,
)
_GRAPH_KEYWORDS = re.compile(
    r"\b(?:relationship|relation|impact|depend|influence|cause|connect|liên quan|tác động|phụ thuộc|dẫn đến)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutedSubQuestion:
    text: str
    tool: str  # retrieve_text | retrieve_per_source | trace_graph
    critical: bool
    rationale: str
    depends_on: int | None = None  # index into the routed list; None = independent


class RetrieverDirectorAgent(BaseAgent):
    name = "retriever_director"

    def __init__(
        self,
        *,
        text_tool: "HybridTextSearchTool | None" = None,
        graph_tool: "GraphRelationSearchTool | None" = None,
        per_source_limit: int = 3,
        budget_phase2_enabled: bool = True,
        budget_min_strong_score: float = 0.045,
        budget_min_strong_count: int = 3,
        semantic_dedup_threshold: float = 0.85,
    ) -> None:
        super().__init__()
        self.text_tool = text_tool
        self.graph_tool = graph_tool
        self.per_source_limit = per_source_limit
        self.budget_phase2_enabled = budget_phase2_enabled
        self.budget_min_strong_score = budget_min_strong_score
        self.budget_min_strong_count = budget_min_strong_count
        self.semantic_dedup_threshold = semantic_dedup_threshold

    def run(self, sub_questions: list[AgenticSubQuestion]) -> list[RoutedSubQuestion]:
        """Decide a tool per sub-question. Returns a deduped routed list."""
        if not sub_questions:
            return []

        routed: list[RoutedSubQuestion] = []
        seen: set[tuple[str, str]] = set()

        for sq in sub_questions[:8]:
            text = sq.text.strip()
            if not text:
                continue
            tool = sq.tool or "retrieve_text"
            rationale = "planner-hint"

            # Safety overrides — apply only if planner didn't already pick the same tool.
            if tool == "retrieve_text":
                if _GRAPH_KEYWORDS.search(text):
                    tool = "trace_graph"
                    rationale = "graph-keyword"
                elif _PER_SOURCE_KEYWORDS.search(text):
                    tool = "retrieve_per_source"
                    rationale = "per-source-keyword"

            key = (text.lower()[:80], tool)
            if key in seen:
                continue
            seen.add(key)

            routed.append(
                RoutedSubQuestion(text=text, tool=tool, critical=sq.critical, rationale=rationale, depends_on=sq.depends_on)
            )

        logger.info(
            "RetrieverDirector: routed sub-questions",
            extra={
                "total": len(routed),
                "tools": {r.tool: sum(1 for x in routed if x.tool == r.tool) for r in routed},
            },
        )
        return routed

    def _evidence_sufficient(self, chunks: "list[RetrievedChunk]") -> bool:
        """True when phase-1 chunks are strong enough to skip optional sub-questions.

        Uses fused_score (pre-rerank RRF scale: ~0.02–0.5) because the reranker
        has not run yet at director time. The thresholds are conservative to
        avoid skipping phase-2 on genuinely weak retrievals.
        """
        if not chunks:
            return False
        strong = sum(
            1 for c in chunks
            if (c.fused_score or 0.0) >= self.budget_min_strong_score
        )
        return strong >= self.budget_min_strong_count

    async def act(self, state: "AgentState", *, limit: int) -> "AgentState":
        """Blackboard entry: route sub-questions and execute the picked tools.

        Three-phase execution:
          Phase 1  — main queries + critical level-0 sub-questions (no deps) + graph + per-source.
          Phase 2a — optional level-0 sub-questions, skipped when phase-1 evidence is strong.
          Phase 2b — level-1 dependent sub-questions (always run), queries augmented with
                     context snippets from their prerequisite's phase-1 chunks.

        Falls back to a single hybrid_text_search over the resolved query when no sub-questions exist.
        """
        from src.rag.retriever import dedupe_retrieved_chunks, semantic_dedupe_chunks  # avoid cycle
        from src.rag.types import RetrievalScope

        owner_id = state.scope.owner_id
        collection_id = state.scope.collection_id

        queries: list[str] = list(state.retrieval_queries) if state.retrieval_queries else [state.resolved_query or state.query]
        routed = self.run(state.sub_questions)
        state.routed_sub_questions = list(routed)

        scheduled_texts: set[str] = {q.lower().strip() for q in queries}

        # ── Build per-phase task lists ─────────────────────────────────────────
        main_tasks: list = []
        sq_crit_tasks: list = []     # level-0 critical (phase 1)
        sq_crit_routing: list[int] = []  # routed index for each sq_crit_tasks entry
        # Store (text, sub_limit) specs — NOT coroutines — so we only create coroutines
        # for optional tasks if we actually decide to run them (avoids unawaited-coroutine warnings).
        sq_opt_specs: list[tuple[str, int]] = []
        sq_dep_specs: list[tuple[int, "RoutedSubQuestion"]] = []  # (routed_idx, r) level-1 (phase 2b)
        per_source_tasks: list = []
        graph_tasks: list = []

        if self.text_tool:
            for q in queries:
                main_tasks.append(self.text_tool.run(query=q, scope=state.scope, limit=limit))

            for ridx, r in enumerate(routed):
                if r.tool == "retrieve_text":
                    norm = r.text.lower().strip()
                    if norm in scheduled_texts:
                        continue
                    scheduled_texts.add(norm)
                    sub_limit = max(1, min(self.per_source_limit, limit))
                    if r.depends_on is not None:
                        # Defer: needs result of prerequisite first
                        sq_dep_specs.append((ridx, r))
                    elif r.critical:
                        sq_crit_tasks.append(self.text_tool.run(query=r.text, scope=state.scope, limit=sub_limit))
                        sq_crit_routing.append(ridx)
                    else:
                        sq_opt_specs.append((r.text, sub_limit))

            if state.use_per_source and state.expected_material_ids:
                for mid in state.expected_material_ids:
                    per_scope = RetrievalScope(owner_id=owner_id, collection_id=collection_id, material_ids=[mid])
                    per_source_tasks.append(
                        self.text_tool.run(query=state.resolved_query or state.query, scope=per_scope, limit=max(1, min(self.per_source_limit, limit)))
                    )

        if self.graph_tool and (state.use_graph or any(r.tool == "trace_graph" for r in routed)):
            graph_tasks.append(self.graph_tool.run(query=state.resolved_query or state.query, scope=state.scope, priority=False))
            for r in routed:
                if r.tool == "trace_graph":
                    graph_tasks.append(self.graph_tool.run(query=r.text, scope=state.scope, priority=False))

        # ── Phase 1 ───────────────────────────────────────────────────────────
        phase1_results = await asyncio.gather(
            *main_tasks, *sq_crit_tasks, *per_source_tasks, *graph_tasks,
            return_exceptions=True,
        )

        n_main = len(main_tasks)
        n_sq_crit = len(sq_crit_tasks)
        n_per_src = len(per_source_tasks)

        text_chunks: list["RetrievedChunk"] = []
        per_source_chunks: list["RetrievedChunk"] = []
        graph_chunks: list["RetrievedChunk"] = []
        # Track per-sub-question chunks for dependency injection
        sq_chunks_by_ridx: dict[int, list["RetrievedChunk"]] = {}

        for idx, res in enumerate(phase1_results):
            if isinstance(res, Exception):
                logger.info("Director phase-1 tool error: %s", res, extra={"owner_id": owner_id})
                continue
            if not res.success:
                continue
            chunks = res.data or []
            if idx < n_main:
                text_chunks.extend(chunks)
            elif idx < n_main + n_sq_crit:
                ridx = sq_crit_routing[idx - n_main]
                sq_chunks_by_ridx[ridx] = chunks
                text_chunks.extend(chunks)
            elif idx < n_main + n_sq_crit + n_per_src:
                per_source_chunks.extend(chunks)
            else:
                graph_chunks.extend(chunks)

        # ── Phase 2a: optional independent sub-questions (skippable) ─────────
        if sq_opt_specs:
            if self.budget_phase2_enabled and self._evidence_sufficient(text_chunks):
                logger.info(
                    "RetrieverDirector: phase-1 sufficient — skipping %d optional sub-questions",
                    len(sq_opt_specs),
                    extra={"owner_id": owner_id, "strong_chunks": sum(1 for c in text_chunks if (c.fused_score or 0.0) >= self.budget_min_strong_score)},
                )
            else:
                # Create coroutines only now — avoids unawaited-coroutine warnings on skip path.
                logger.info("RetrieverDirector: running %d optional sub-questions (phase 2a)", len(sq_opt_specs), extra={"owner_id": owner_id})
                sq_opt_tasks = [self.text_tool.run(query=t, scope=state.scope, limit=lim) for t, lim in sq_opt_specs]
                phase2a_results = await asyncio.gather(*sq_opt_tasks, return_exceptions=True)
                for res in phase2a_results:
                    if isinstance(res, Exception) or not getattr(res, "success", False):
                        continue
                    text_chunks.extend(res.data or [])

        # ── Phase 2b: dependent sub-questions (always run, augmented) ─────────
        if sq_dep_specs and self.text_tool:
            dep_tasks = []
            for ridx, r in sq_dep_specs:
                prereq_idx = r.depends_on  # index into routed list
                prereq_chunks = sq_chunks_by_ridx.get(prereq_idx or -1, text_chunks[:2])
                context_snippet = prereq_chunks[0].content[:150].strip() if prereq_chunks else ""
                augmented = (
                    f"{r.text} [dựa trên: {context_snippet}]" if context_snippet else r.text
                )
                dep_tasks.append(
                    self.text_tool.run(query=augmented, scope=state.scope, limit=max(1, min(self.per_source_limit, limit)))
                )
            dep_results = await asyncio.gather(*dep_tasks, return_exceptions=True)
            for res in dep_results:
                if isinstance(res, Exception) or not getattr(res, "success", False):
                    continue
                text_chunks.extend(res.data or [])
            logger.info("RetrieverDirector: ran %d dependent (multi-hop) sub-questions", len(dep_tasks), extra={"owner_id": owner_id})

        # ── Merge + dedup ──────────────────────────────────────────────────────
        all_chunks = [*text_chunks, *per_source_chunks, *graph_chunks]
        merged = dedupe_retrieved_chunks(all_chunks)
        if self.semantic_dedup_threshold < 1.0 and not state.requires_coverage:
            merged = semantic_dedupe_chunks(merged, threshold=self.semantic_dedup_threshold)

        state.raw_evidence = merged
        state.graph_evidence = graph_chunks
        logger.info(
            "RetrieverDirector.act: retrieved chunks",
            extra={
                "owner_id": owner_id,
                "collection_id": collection_id,
                "text": len(text_chunks),
                "per_source": len(per_source_chunks),
                "graph": len(graph_chunks),
                "merged": len(merged),
            },
        )
        return state
