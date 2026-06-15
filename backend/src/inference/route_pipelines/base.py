"""Base contract for per-route execution strategies.

Each pipeline exposes a small set of *hooks* the orchestrator
(`inference_engine.answer`) calls at well-known decision points:

  1. `post_retrieval(reranked, candidates, final_limit)` — modify the chunk
     ordering / material coverage AFTER reranking. SUMMARIZATION uses this to
     force per-document coverage; others pass through.

  2. `override_evidence_refusal(should_refuse, reason, reranked)` — relax the
     RefusalPolicy verdict when the route allows partial-evidence answers
     (SUMMARIZATION / COMPARISON / CLAIM_CHECK / GRAPH_RELATION). Returns a
     potentially-flipped `(should_refuse, reason)` tuple.

  3. `skip_llm_retry_on_refusal` — when the LLM emits a "no evidence" refusal,
     should we skip the retry-with-stricter-prompt step? Override-relaxed
     routes set this True to save 80-100s on retry latency.

  4. `post_generation(answer, context_chunks, response_parser, claim_verifier,
     refusal_policy)` — return `(maybe_modified_answer, should_refuse, reason)`.
     CLAIM_CHECK runs the NLI verifier here; factual support checks now happen
     in SLEC + quality gate.

Pipelines are stateless w.r.t. requests; their constructors take only the
shared collaborators (settings, claim_verifier, …). Hot-path hooks accept the
per-request state as arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.guardrails.claim_verifier import ClaimVerifier
    from src.guardrails.refusal_policy import RefusalPolicy
    from src.inference.response_parser import ResponseParser
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RouteHooks:
    """Static behaviour switches a pipeline exposes to the orchestrator."""
    force_material_coverage: bool = False
    relax_refusal: bool = False
    skip_llm_retry_on_refusal: bool = False
    enable_self_rag: bool = False
    enable_claim_verifier: bool = False


class BaseRoutePipeline:
    """Default no-op pipeline. Subclass and override hooks per route."""

    name: str = "general"
    hooks: RouteHooks

    def __init__(
        self,
        *,
        name: str | None = None,
        relax_refusal: bool | None = None,
        skip_llm_retry_on_refusal: bool | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        # Start from class default, allow overrides for instances that share class
        # but with different hook tweaks (e.g., COMPARISON vs GRAPH_RELATION).
        self.hooks = RouteHooks(
            force_material_coverage=getattr(self.__class__, "DEFAULT_FORCE_MATERIAL_COVERAGE", False),
            relax_refusal=relax_refusal if relax_refusal is not None else getattr(self.__class__, "DEFAULT_RELAX_REFUSAL", False),
            skip_llm_retry_on_refusal=(
                skip_llm_retry_on_refusal
                if skip_llm_retry_on_refusal is not None
                else getattr(self.__class__, "DEFAULT_SKIP_LLM_RETRY", False)
            ),
            enable_self_rag=getattr(self.__class__, "DEFAULT_ENABLE_SELF_RAG", False),
            enable_claim_verifier=getattr(self.__class__, "DEFAULT_ENABLE_CLAIM_VERIFIER", False),
        )

    # ── Retrieval-side hook ────────────────────────────────────────────────
    def post_retrieval(
        self,
        *,
        reranked: list["RetrievedChunk"],
        candidates: list["RetrievedChunk"],
        final_limit: int,
        ensure_material_coverage_fn,
    ) -> list["RetrievedChunk"]:
        """Default: no transformation. Override per route if needed."""
        return reranked

    # ── Refusal-policy hook ────────────────────────────────────────────────
    def override_evidence_refusal(
        self,
        *,
        should_refuse: bool,
        reason: str | None,
        reranked: list["RetrievedChunk"],
        rule_was_no_evidence: bool,
    ) -> tuple[bool, str | None]:
        """Default: pass through."""
        return should_refuse, reason

    # ── Post-generation hook ───────────────────────────────────────────────
    async def post_generation(
        self,
        *,
        answer: str,
        context_chunks: list["RetrievedChunk"],
        response_parser: "ResponseParser",
        claim_verifier: "ClaimVerifier",
        refusal_policy: "RefusalPolicy",
    ) -> tuple[str, bool, str | None]:
        """Default: leave answer untouched, no refusal.

        Subclasses may rewrite `answer`, set `should_refuse=True`, or attach a
        `refusal_reason`.
        """
        return answer, False, None
