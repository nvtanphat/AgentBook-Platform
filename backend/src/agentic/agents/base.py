"""Base contract for specialised agents.

An agent wraps a single LLM call (or rule-based logic) with:
  - A focused prompt template / persona
  - A typed input/output contract
  - Telemetry hook (records to AgentTrace via the orchestrator)

Subclasses override:
  - `name` (str): unique persona name used in logs + AgentTrace
  - `async def run(...)`: the actual call

Failure semantics: agents must NEVER raise. On error, return a best-effort
default that the orchestrator can interpret as "skip" or "fall back to rule".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentInvocation:
    """Telemetry payload — one row per agent run, attached to AgentTrace."""
    agent: str
    duration_ms: int
    success: bool
    detail: str | None = None
    metadata: dict[str, Any] | None = None


class BaseAgent:
    """All specialised agents share this contract.

    Concrete `run()` signatures differ by role — declared in subclasses.
    """

    name: str = "base"

    def __init__(self, *, llm: "BaseLLM | None" = None) -> None:
        self.llm = llm

    async def _safe_generate(self, prompt: str, *, label: str | None = None) -> str:
        """Wrap llm.generate with a guard that swallows errors. Agents must
        never propagate exceptions to the orchestrator — they degrade.
        """
        if self.llm is None:
            return ""
        try:
            return await self.llm.generate(prompt=prompt)
        except Exception as exc:
            logger.warning(
                "Agent LLM call failed",
                extra={"agent": self.name, "label": label or "", "error": str(exc)},
            )
            return ""
