"""NLI verifier tool — wraps the ClaimVerifier so any agent can request a
fully-typed claim verdict without touching the verifier internals."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from src.agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from src.guardrails.claim_verifier import ClaimVerificationResult, ClaimVerifier
    from src.processing.types import EvidenceBlock

logger = logging.getLogger(__name__)


class NLIVerifierTool(BaseTool):
    name = "nli_verifier"
    description = (
        "Claim verification via NLI. Compares a draft answer sentence by "
        "sentence against the cited evidence and emits SUPPORTED / "
        "CONTRADICTED / NOT_ENOUGH_EVIDENCE."
    )

    def __init__(self, *, verifier: "ClaimVerifier") -> None:
        self.verifier = verifier

    async def _run(
        self,
        *,
        claim: str,
        evidence: list["EvidenceBlock"],
    ) -> "ClaimVerificationResult":
        if hasattr(self.verifier, "averify"):
            return await self.verifier.averify(claim=claim, evidence=evidence)
        result = self.verifier.verify(claim=claim, evidence=evidence)
        if inspect.isawaitable(result):
            return await result
        return result
