"""FACTUAL route — tight grounded QA.

Behaviour vs. baseline:
  - No material-coverage forcing (one strong chunk beats five mediocre).
  - Standard refusal policy (don't relax — we want hard refusals on no-evidence).
  - No LLM-based Self-RAG reflection; SLEC + quality gate handle support checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.inference.route_pipelines.base import BaseRoutePipeline

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FactualPipeline(BaseRoutePipeline):
    name = "factual"

    DEFAULT_FORCE_MATERIAL_COVERAGE = False
    DEFAULT_RELAX_REFUSAL = False
    DEFAULT_SKIP_LLM_RETRY = False
    DEFAULT_ENABLE_SELF_RAG = False
    DEFAULT_ENABLE_CLAIM_VERIFIER = False

    # Self-RAG used to live behind this hook, but it adds a second LLM call on the
    # hot path. Leave it disabled now that SLEC + quality gate cover support.
