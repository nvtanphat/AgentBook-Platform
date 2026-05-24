"""Hybrid text search tool — wraps `HybridRetriever.retrieve`.

Dense + sparse + RRF + reranker remain in the retriever; this tool only
guarantees scoped, evidence-trace-preserving access from agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever
    from src.rag.types import RetrievalScope, RetrievedChunk

logger = logging.getLogger(__name__)


class HybridTextSearchTool(BaseTool):
    name = "hybrid_text_search"
    description = (
        "Hybrid dense+sparse retrieval over indexed chunks. Always scoped by "
        "owner_id + collection_id. Returns evidence-bearing RetrievedChunks."
    )

    def __init__(self, *, retriever: "HybridRetriever") -> None:
        self.retriever = retriever

    async def _run(
        self,
        *,
        query: str,
        scope: "RetrievalScope",
        limit: int | None = None,
    ) -> list["RetrievedChunk"]:
        if not query or not query.strip():
            return []
        scope.ensure_scoped()
        chunks = await self.retriever.retrieve(query=query, scope=scope, limit=limit)
        logger.debug(
            "HybridTextSearchTool returned %d chunks",
            len(chunks),
            extra={"owner_id": scope.owner_id, "collection_id": scope.collection_id},
        )
        return chunks
