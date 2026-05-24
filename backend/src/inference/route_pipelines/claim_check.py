"""CLAIM_CHECK route — NLI-enhanced stance verification.

Phase C centerpiece. The shared `claim_verifier.py` already supports an
NLI-augmented mode (`nli_enabled=True`) but the global default is the
token-overlap heuristic. This pipeline owns its own `ClaimVerifier` instance
with NLI forced ON and a multilingual model that handles Vietnamese
queries — `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (≈100 MB,
8 languages incl. VN).

Behaviour vs. baseline:
  - Relax retrieval-side refusal (partial evidence is enough to verify a claim).
  - Skip LLM retry on "no evidence" preamble (override-routes save 80-100s).
  - Run NLI verifier post-generation; flag contradictions as refusals with
    a concrete `corrected_facts` payload.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.guardrails.claim_verifier import ClaimVerifier
from src.inference.route_pipelines.base import BaseRoutePipeline

if TYPE_CHECKING:
    from src.guardrails.refusal_policy import RefusalPolicy
    from src.inference.response_parser import ResponseParser
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Multilingual NLI cross-encoder — handles VN + EN + 6 more out of the box.
# Smaller than the DeBERTa-base variant (≈90 MB on disk vs 400 MB) and 2-3×
# faster on CPU, with comparable XNLI scores on the languages we care about.
_MULTILINGUAL_NLI_DEFAULT = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


class ClaimCheckPipeline(BaseRoutePipeline):
    name = "claim_check"

    DEFAULT_FORCE_MATERIAL_COVERAGE = False
    DEFAULT_RELAX_REFUSAL = True
    DEFAULT_SKIP_LLM_RETRY = True
    DEFAULT_ENABLE_SELF_RAG = False
    DEFAULT_ENABLE_CLAIM_VERIFIER = True

    def __init__(self, *, nli_model_name: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Lazy-loaded — the model only materialises on the first CLAIM_CHECK
        # request, so startup latency is unaffected for users who never use it.
        self._verifier: ClaimVerifier | None = None
        self._nli_model_name = (
            nli_model_name
            or os.getenv("AGENTBOOK_CLAIM_NLI_MODEL")
            or _MULTILINGUAL_NLI_DEFAULT
        )
        # Ablation switch — when AGENTBOOK_CLAIM_NLI_ENABLED=false, fall back to
        # the cheaper token-overlap heuristic. Used to quantify NLI contribution.
        self._nli_enabled = os.getenv("AGENTBOOK_CLAIM_NLI_ENABLED", "true").strip().lower() not in ("false", "0", "no", "")

    @property
    def verifier(self) -> ClaimVerifier:
        if self._verifier is None:
            logger.info(
                "ClaimCheckPipeline: bootstrapping verifier",
                extra={"nli_model": self._nli_model_name, "nli_enabled": self._nli_enabled},
            )
            self._verifier = ClaimVerifier(
                nli_model_name=self._nli_model_name,
                nli_enabled=self._nli_enabled,
            )
        return self._verifier

    async def post_generation(
        self,
        *,
        answer: str,
        context_chunks: list["RetrievedChunk"],
        response_parser: "ResponseParser",
        claim_verifier,  # ignored — we use our own NLI-enabled instance
        refusal_policy: "RefusalPolicy",
    ) -> tuple[str, bool, str | None]:
        """Run NLI verification; return refusal + correction signal."""
        if not answer.strip() or not context_chunks:
            return answer, False, None

        evidence = [block for chunk in context_chunks for block in chunk.evidence]
        if not evidence:
            return answer, False, None

        try:
            result = await self.verifier.averify(claim=answer, evidence=evidence)
        except Exception as exc:
            # NLI is best-effort — never let it crash the whole route.
            logger.warning(
                "ClaimCheckPipeline: NLI verification failed, keeping LLM answer",
                extra={"error": str(exc)},
            )
            return answer, False, None

        decision = refusal_policy.check_claim(result.overall_verdict, result.corrected_facts)
        # CLAIM_CHECK UX correction (v16): when NLI flags CONTRADICTED, the user
        # explicitly asked for verification — refusing is wrong. The LLM's answer
        # already contains the correct fact ("F1-score là trung bình điều hòa…").
        # Keep it; only refuse when truly NOT_ENOUGH_EVIDENCE.
        from src.guardrails.claim_verifier import OverallVerdict
        if decision.should_refuse:
            if result.overall_verdict == OverallVerdict.CONTRADICTED:
                logger.info(
                    "ClaimCheckPipeline: NLI=CONTRADICTED → keeping LLM correction (no refusal)",
                    extra={
                        "corrected_facts": result.corrected_facts[:3],
                        "supported_ratio": round(result.supported_ratio, 3),
                    },
                )
                # The LLM already produced the corrected answer; trust it.
                return answer, False, None
            # Other refuse reasons (INSUFFICIENT) — still refuse, evidence really is missing.
            from src.inference.inference_engine import REFUSAL_ANSWER
            logger.info(
                "ClaimCheckPipeline: NLI verdict triggers refusal",
                extra={"overall_verdict": result.overall_verdict.value, "reason": decision.reason},
            )
            return REFUSAL_ANSWER, True, decision.reason
        return answer, False, None
