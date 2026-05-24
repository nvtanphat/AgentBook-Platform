"""Visual image search tool — cross-modal SigLIP retrieval over figures.

Wraps `HybridRetriever.retrieve_visual` if the visual provider is configured.
Returns an empty list when visual retrieval is disabled, so agents can call
unconditionally and the coordinator decides whether to use the result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from src.rag.embedding_provider import VisualEmbeddingProvider
    from src.rag.retriever import HybridRetriever
    from src.rag.types import RetrievalScope, RetrievedVisualChunk

logger = logging.getLogger(__name__)


class VisualImageSearchTool(BaseTool):
    name = "visual_image_search"
    description = (
        "Cross-modal visual retrieval (text → figure) over the indexed figure "
        "collection. Scoped per owner+collection. Returns RetrievedVisualChunks."
    )

    def __init__(
        self,
        *,
        retriever: "HybridRetriever",
        visual_provider: "VisualEmbeddingProvider | None",
    ) -> None:
        self.retriever = retriever
        self.visual_provider = visual_provider

    @property
    def enabled(self) -> bool:
        return self.visual_provider is not None

    async def _run(
        self,
        *,
        query: str,
        scope: "RetrievalScope",
        limit: int | None = None,
    ) -> list["RetrievedVisualChunk"]:
        if not self.enabled or not query or not query.strip():
            return []
        scope.ensure_scoped()
        return await self.retriever.retrieve_visual(
            query=query, scope=scope, visual_provider=self.visual_provider, limit=limit,
        )
