"""Specialised agent personas for multi-agentic RAG.

Each agent is a single-responsibility persona that wraps an LLM call (or rule-
based logic) with a focused prompt template + structured I/O. The
`AgenticCoordinatingEngine` in `src.agentic.service` composes them via a
shared `AgentState` blackboard with bounded iteration.

Roles:
  PlannerAgent           — decomposes the query into sub-questions + plan
  RetrieverDirectorAgent — picks the right retrieval tool per sub-question
  CRAGCriticAgent        — Corrective RAG triage of retrieved evidence
  SynthesizerAgent       — composes a grounded answer from collected evidence
  GuardrailsAgent        — claim verification + self-repair gate
  CriticAgent            — (legacy) refine-loop critic over the draft answer
"""

from src.agentic.agents.base import AgentInvocation, BaseAgent
from src.agentic.agents.crag_critic import CRAGCriticAgent
from src.agentic.agents.critic import CriticAgent, CriticVerdict
from src.agentic.agents.guardrails_agent import GuardrailsAgent
from src.agentic.agents.planner_agent import PlannerAgent
from src.agentic.agents.retriever_director import RetrieverDirectorAgent
from src.agentic.agents.synthesizer import SynthesizerAgent

__all__ = [
    "AgentInvocation",
    "BaseAgent",
    "CRAGCriticAgent",
    "CriticAgent",
    "CriticVerdict",
    "GuardrailsAgent",
    "PlannerAgent",
    "RetrieverDirectorAgent",
    "SynthesizerAgent",
]
