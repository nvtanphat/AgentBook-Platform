"""GuardrailsAgent — claim verification and self-repair gatekeeper.

Runs NLI-style cross-checking over the draft answer and flags
contradictions or unsupported sentences. When the verdict is
CONTRADICTED or NOT_ENOUGH_EVIDENCE, it records a warning so the
coordinator can trigger an answer-repair pass on the SynthesizerAgent.

Failure semantics: any internal error degrades to verdict='not_run', the
coordinator still finalises the answer based on the draft.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.agentic.agents.base import BaseAgent
from src.agentic.state import GuardrailReport
from src.guardrails.claim_verifier import ClaimVerdict

if TYPE_CHECKING:
    from src.agentic.state import AgentState
    from src.agentic.tools import NLIVerifierTool

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


class GuardrailsAgent(BaseAgent):
    name = "guardrails"

    def __init__(self, *, verifier_tool: "NLIVerifierTool") -> None:
        super().__init__()
        self.verifier_tool = verifier_tool

    async def act(self, state: "AgentState") -> "AgentState":
        answer = (state.final_answer or state.draft_answer or "").strip()
        chunks = state.context_chunks or state.cleaned_evidence
        if not answer or not chunks:
            state.guardrail_report = GuardrailReport(verdict="not_run", warning="no_answer_or_evidence")
            state.claims_verified = False
            return state

        # Skip token-overlap claim verification when answer language differs
        # from the dominant chunk language. The verifier intersects answer
        # tokens with evidence tokens — meaningless across languages because
        # only acronyms and numbers survive (every claim degrades to
        # NOT_ENOUGH_EVIDENCE). Use chunk.language as the source of truth
        # for evidence language, not query_language (which describes input).
        processed = state.processed_query
        answer_lang = (
            getattr(processed, "answer_language", None)
            or state.answer_language
            or ""
        ).lower()
        if answer_lang:
            chunk_langs = [(c.language or "").lower() for c in chunks if c.language]
            if chunk_langs:
                from collections import Counter
                dominant_lang, count = Counter(chunk_langs).most_common(1)[0]
                if dominant_lang and dominant_lang != answer_lang and count >= len(chunk_langs) / 2:
                    state.guardrail_report = GuardrailReport(
                        verdict="not_run", warning="cross_lingual_skip",
                    )
                    state.claims_verified = False
                    return state

        evidence_blocks = [block for chunk in chunks for block in chunk.evidence]
        try:
            result = await self.verifier_tool.run(claim=answer, evidence=evidence_blocks)
        except Exception as exc:
            logger.warning("GuardrailsAgent: verifier call failed", extra={"error": str(exc)})
            state.guardrail_report = GuardrailReport(verdict="not_run", warning="verifier_error")
            state.claims_verified = False
            return state

        if not result.success or result.data is None:
            state.guardrail_report = GuardrailReport(verdict="not_run", warning=result.error or "verifier_failed")
            state.claims_verified = False
            return state

        verification = result.data
        unsupported, invalid = self._grounding_report(answer=answer, citation_count=len(state.citations))

        warning: str | None = None
        if verification.verdict == ClaimVerdict.CONTRADICTED:
            warning = "Answer appears to conflict with retrieved evidence."
            state.add_warning("Answer contradicts cited evidence.")
        elif verification.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE:
            warning = "Evidence may not directly support every claim."
            state.add_warning("Not enough direct evidence to back the answer.")

        state.guardrail_report = GuardrailReport(
            verdict=verification.verdict.value,
            confidence=float(getattr(verification, "confidence", 0.0)),
            warning=warning,
            unsupported_sentence_count=unsupported,
            invalid_citation_count=invalid,
            contradictions=list(getattr(verification, "corrected_facts", []) or []),
        )
        state.claims_verified = verification.verdict == ClaimVerdict.SUPPORTED and not unsupported and not invalid
        logger.info(
            "GuardrailsAgent verdict",
            extra={
                "owner_id": state.scope.owner_id,
                "collection_id": state.scope.collection_id,
                "verdict": verification.verdict.value,
                "unsupported": unsupported,
                "invalid_citations": invalid,
            },
        )
        return state

    @staticmethod
    def _grounding_report(*, answer: str, citation_count: int) -> tuple[int, int]:
        if citation_count <= 0 or not answer.strip():
            return 0, 0
        markers = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
        invalid = sum(1 for m in markers if m < 1 or m > citation_count)
        paragraphs = re.split(r"\n\s*\n", answer)
        unsupported = 0
        for para in paragraphs:
            if _CITATION_RE.search(para):
                continue
            sentences = [
                s.strip() for s in _SENTENCE_RE.findall(para)
                if len(s.strip()) >= 12 and not s.strip().startswith(">")
            ]
            if len(sentences) >= 2:
                unsupported += 1
        return unsupported, invalid
