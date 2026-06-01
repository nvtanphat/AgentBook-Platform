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


class RetrieverDirectorAgent(BaseAgent):
    name = "retriever_director"

    def __init__(
        self,
        *,
        text_tool: "HybridTextSearchTool | None" = None,
        graph_tool: "GraphRelationSearchTool | None" = None,
        per_source_limit: int = 3,
    ) -> None:
        super().__init__()
        self.text_tool = text_tool
        self.graph_tool = graph_tool
        self.per_source_limit = per_source_limit

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
                RoutedSubQuestion(text=text, tool=tool, critical=sq.critical, rationale=rationale)
            )

        logger.info(
            "RetrieverDirector: routed sub-questions",
            extra={
                "total": len(routed),
                "tools": {r.tool: sum(1 for x in routed if x.tool == r.tool) for r in routed},
            },
        )
        return routed

    async def act(self, state: "AgentState", *, limit: int) -> "AgentState":
        """Blackboard entry: route sub-questions and execute the picked tools.

        Falls back to a single hybrid_text_search over the resolved query when
        no sub-questions exist.
        """
        from src.rag.retriever import dedupe_retrieved_chunks  # local import — avoid cycle
        from src.rag.types import RetrievalScope

        owner_id = state.scope.owner_id
        collection_id = state.scope.collection_id

        # Always include the resolved query path (multi-query is handled by
        # the engine's QueryProcessor; here we keep things explicit).
        queries: list[str] = list(state.retrieval_queries) if state.retrieval_queries else [state.resolved_query or state.query]
        routed = self.run(state.sub_questions)
        state.routed_sub_questions = list(routed)

        text_tasks: list = []
        graph_tasks: list = []
        per_source_tasks: list = []

        # Track all query texts already scheduled so sub-questions that duplicate
        # a main query don't trigger a redundant embedding + retrieval pass.
        scheduled_texts: set[str] = {q.lower().strip() for q in queries}

        if self.text_tool:
            for q in queries:
                text_tasks.append(self.text_tool.run(query=q, scope=state.scope, limit=limit))
            for r in routed:
                if r.tool == "retrieve_text":
                    if r.text.lower().strip() in scheduled_texts:
                        continue  # already covered by a main query
                    scheduled_texts.add(r.text.lower().strip())
                    text_tasks.append(self.text_tool.run(query=r.text, scope=state.scope, limit=max(1, min(self.per_source_limit, limit))))
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

        results = await asyncio.gather(*text_tasks, *per_source_tasks, *graph_tasks, return_exceptions=True)

        text_count = len(text_tasks)
        per_source_count = len(per_source_tasks)

        text_chunks: list[RetrievedChunk] = []
        per_source_chunks: list[RetrievedChunk] = []
        graph_chunks: list[RetrievedChunk] = []

        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.info("Director tool error: %s", res, extra={"owner_id": owner_id})
                continue
            if not res.success:
                continue
            chunks = res.data or []
            if idx < text_count:
                text_chunks.extend(chunks)
            elif idx < text_count + per_source_count:
                per_source_chunks.extend(chunks)
            else:
                graph_chunks.extend(chunks)

        merged = dedupe_retrieved_chunks([*text_chunks, *per_source_chunks, *graph_chunks])
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

