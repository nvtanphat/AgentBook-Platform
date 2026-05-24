"""FACTUAL route — tight grounded QA with Self-RAG hedging.

Behaviour vs. baseline:
  - No material-coverage forcing (one strong chunk beats five mediocre).
  - Standard refusal policy (don't relax — we want hard refusals on no-evidence).
  - Self-RAG reflection enabled to hedge sentences not fully supported.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.inference.route_pipelines.base import BaseRoutePipeline

if TYPE_CHECKING:
    from src.guardrails.claim_verifier import ClaimVerifier
    from src.guardrails.refusal_policy import RefusalPolicy
    from src.inference.response_parser import ResponseParser
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class FactualPipeline(BaseRoutePipeline):
    name = "factual"

    DEFAULT_FORCE_MATERIAL_COVERAGE = False
    DEFAULT_RELAX_REFUSAL = False
    DEFAULT_SKIP_LLM_RETRY = False
    DEFAULT_ENABLE_SELF_RAG = True
    DEFAULT_ENABLE_CLAIM_VERIFIER = False

    # The Self-RAG reflection step lives in InferenceEngine._self_reflect_claims;
    # the pipeline just signals via `hooks.enable_self_rag` and the orchestrator
    # invokes the existing implementation. Keeping the heavy LLM call there
    # avoids duplicating prompt + retry logic across pipelines.
