"""SUMMARIZATION route — material-coverage + relaxed refusal.

Behaviour vs. baseline:
  - Force `_ensure_material_coverage` so every input document contributes
    at least one chunk to the context (preserves outline breadth).
  - Relax refusal: any retrieved evidence is good enough; reranker scores
    look low because the query is an instruction, not a content match.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.inference.route_pipelines.base import BaseRoutePipeline
from src.rag.query_router import RouteType

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class SummarizationPipeline(BaseRoutePipeline):
    name = "summarization"

    DEFAULT_FORCE_MATERIAL_COVERAGE = True
    DEFAULT_RELAX_REFUSAL = True
    DEFAULT_SKIP_LLM_RETRY = False
    DEFAULT_ENABLE_SELF_RAG = False
    DEFAULT_ENABLE_CLAIM_VERIFIER = False

    def post_retrieval(
        self,
        *,
        reranked: list["RetrievedChunk"],
        candidates: list["RetrievedChunk"],
        final_limit: int,
        ensure_material_coverage_fn,
    ) -> list["RetrievedChunk"]:
        """Inject the cross-document coverage helper from InferenceEngine."""
        return ensure_material_coverage_fn(
            reranked=reranked,
            candidates=candidates,
            final_limit=final_limit,
            route=RouteType.SUMMARIZATION,
        )
