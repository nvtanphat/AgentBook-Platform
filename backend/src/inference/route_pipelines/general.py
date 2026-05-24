"""GENERAL / COMPARISON / GRAPH_RELATION fallback.

These routes share the relaxed-refusal behaviour (partial evidence is enough)
and skip the LLM retry-with-stricter-prompt step to save 80-100s when the
first generation already emitted a (false) "no evidence" prefix. The
relax/skip flags are toggled via constructor args by the dispatcher.
"""

from __future__ import annotations

import logging

from src.inference.route_pipelines.base import BaseRoutePipeline

logger = logging.getLogger(__name__)


class GeneralPipeline(BaseRoutePipeline):
    name = "general"

    DEFAULT_FORCE_MATERIAL_COVERAGE = False
    DEFAULT_RELAX_REFUSAL = False        # plain GENERAL keeps the hard refusal
    DEFAULT_SKIP_LLM_RETRY = False
    DEFAULT_ENABLE_SELF_RAG = False
    DEFAULT_ENABLE_CLAIM_VERIFIER = False
