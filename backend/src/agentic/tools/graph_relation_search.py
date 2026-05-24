"""Graph relation search tool — wraps `GraphRetriever.retrieve_paths` and
projects the paths back into evidence-bearing chunks via the inference
engine's `_chunks_from_graph_paths`. The conversion preserves owner +
collection scope and inherits evidence trace from the path."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from src.inference.inference_engine import InferenceEngine
    from src.rag.graph_retriever import GraphRetriever
    from src.rag.types import RetrievalScope, RetrievedChunk

logger = logging.getLogger(__name__)


class GraphRelationSearchTool(BaseTool):
    name = "graph_relation_search"
    description = (
        "Traverse the knowledge graph to surface relationship/causal/dependency "
        "evidence. Scope-isolated. Returns chunks decorated with graph_score."
    )

    def __init__(self, *, graph_retriever: "GraphRetriever", engine: "InferenceEngine") -> None:
        self.graph_retriever = graph_retriever
        self.engine = engine

    async def _run(
        self,
        *,
        query: str,
        scope: "RetrievalScope",
        priority: bool = False,
    ) -> list["RetrievedChunk"]:
        if not query or not query.strip():
            return []
        scope.ensure_scoped()
        paths = await self.graph_retriever.retrieve_paths(query=query, scope=scope)
        chunks = self.engine._chunks_from_graph_paths(paths, scope=scope, priority=priority)
        logger.debug(
            "GraphRelationSearchTool returned %d chunks (%d paths)",
            len(chunks),
            len(paths),
            extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id},
        )
        return chunks
