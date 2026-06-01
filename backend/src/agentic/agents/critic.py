"""CriticAgent — reviews the draft answer + triggers re-retrieve loop.

This is the agent that turns the orchestrator from "single-pass" to "multi-
agent collaborative" — it inspects the synthesizer's draft and decides:

  - ACCEPT: answer is well-grounded and complete → return as-is
  - REFINE: answer is missing context → propose `follow_up_queries` for the
           retriever to fetch, then the synthesizer runs again with the
           augmented context.

To keep latency manageable, the critic only fires when confidence < threshold
(see CRITIC_ACTIVATION_CONFIDENCE). On high-confidence answers it's skipped.

Output contract (strict JSON):
  {"verdict": "accept|refine", "reason": "...", "follow_up_queries": ["..."]}

Failure modes (all degrade gracefully to "accept"):
  - LLM error → accept
  - JSON parse error → accept
  - LLM emits gibberish → accept
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.agentic.agents.base import BaseAgent

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Critic only fires below this confidence threshold. Above it, the answer is
# trusted and we save 60-100s of LLM time per query.
CRITIC_ACTIVATION_CONFIDENCE = 0.65

_CRITIC_PROMPT = """\
You are a strict answer critic. Review the DRAFT ANSWER below for grounding,
completeness, and citation hygiene against the EVIDENCE.

QUESTION: {query}

EVIDENCE:
{evidence}

DRAFT ANSWER:
{answer}

Decide one of:
  - "accept" if the answer is well-grounded and complete.
  - "refine" if a specific piece of evidence is missing that would materially
    improve the answer. Provide 1–2 targeted follow-up queries.

Output STRICTLY this JSON (no prose, no markdown fences):
{{"verdict": "accept", "reason": "...", "follow_up_queries": []}}
or
{{"verdict": "refine", "reason": "...", "follow_up_queries": ["query 1", "query 2"]}}

JSON:\
"""


@dataclass(frozen=True)
class CriticVerdict:
    verdict: Literal["accept", "refine"]
    reason: str
    follow_up_queries: list[str]


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(self, *, llm: "BaseLLM") -> None:
        super().__init__(llm=llm)

    def should_fire(self, *, confidence: float, route_type: str | None = None) -> bool:
        """Critic only fires below activation threshold AND not on chitchat /
        off-topic / claim_check routes (those have their own verification).

        FACTUAL and GENERAL routes are excluded: direct lookups don't benefit
        from critic refinement — it only adds 2 extra embedding batches (~54s)
        without improving answer quality for single-hop legal queries.
        """
        if confidence >= CRITIC_ACTIVATION_CONFIDENCE:
            return False
        if route_type in ("claim_check", "chitchat", "off_topic", "factual", "general"):
            return False
        return True

    async def run(
        self,
        *,
        query: str,
        answer: str,
        context_chunks: list["RetrievedChunk"],
    ) -> CriticVerdict:
        """Review the draft. Returns ACCEPT on any failure (graceful)."""
        if not answer.strip() or not context_chunks:
            return CriticVerdict(verdict="accept", reason="no draft or evidence", follow_up_queries=[])

        evidence_text = "\n".join(
            f"[{i+1}] {(c.content or '').strip()[:250]}"
            for i, c in enumerate(context_chunks[:6])
        )
        prompt = _CRITIC_PROMPT.format(
            query=query[:400],
            evidence=evidence_text[:2500],
            answer=answer[:1500],
        )

        raw = await self._safe_generate(prompt, label="critic-review")
        if not raw.strip():
            return CriticVerdict(verdict="accept", reason="critic_no_output", follow_up_queries=[])

        verdict = self._parse(raw)
        if verdict is None:
            return CriticVerdict(verdict="accept", reason="critic_parse_failed", follow_up_queries=[])

        logger.info(
            "CriticAgent verdict",
            extra={
                "verdict": verdict.verdict,
                "follow_up_count": len(verdict.follow_up_queries),
                "reason_preview": verdict.reason[:80],
            },
        )
        return verdict

    @staticmethod
    def _parse(raw: str) -> CriticVerdict | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        verdict_raw = str(data.get("verdict", "accept")).lower().strip()
        if verdict_raw not in ("accept", "refine"):
            verdict_raw = "accept"

        follow_ups_raw = data.get("follow_up_queries", []) or []
        follow_ups: list[str] = []
        if isinstance(follow_ups_raw, list):
            for q in follow_ups_raw[:2]:
                qs = str(q).strip()
                if qs and 5 <= len(qs) <= 300:
                    follow_ups.append(qs)
        if verdict_raw == "refine" and not follow_ups:
            verdict_raw = "accept"  # cannot refine without targets

        return CriticVerdict(
            verdict=verdict_raw,  # type: ignore[arg-type]
            reason=str(data.get("reason", ""))[:240],
            follow_up_queries=follow_ups,
        )
