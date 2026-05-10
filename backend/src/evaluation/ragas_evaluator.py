from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")

_FAITHFULNESS_PROMPT = """\
Given the following evidence passages and an answer, count how many sentences in the answer make factual claims NOT supported by the evidence.

Evidence:
{evidence}

Answer:
{answer}

Output ONLY a JSON object: {{"supported": <int>, "unsupported": <int>}}
JSON:\
"""

_RELEVANCY_PROMPT = """\
Given the following question and answer, rate how well the answer addresses the question on a scale of 0.0 to 1.0.
- 1.0 = directly and completely answers the question
- 0.5 = partially answers the question
- 0.0 = does not address the question

Question: {query}
Answer: {answer}

Output ONLY a JSON object: {{"score": 0.85}}
JSON:\
"""


@dataclass
class RAGASMetrics:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    citation_coverage: float = 0.0
    refusal_rate: float = 0.0
    sample_count: int = 0
    details: dict = field(default_factory=dict)


class RAGASEvaluator:
    """Lightweight RAGAS-style evaluator for answer quality.

    Metrics:
    - faithfulness: % of answer sentences that have citation or evidence support
    - answer_relevancy: LLM-judged score (0-1) of how well answer addresses query
    - context_precision: ratio of high-quality chunks among retrieved (CRAG proxy)
    - citation_coverage: % of answer paragraphs that contain at least one citation
    """

    def __init__(self, llm: "BaseLLM | None" = None) -> None:
        self.llm = llm
        self._results: list[dict] = []

    def evaluate_faithfulness(self, *, answer: str) -> float:
        """Score-based faithfulness: ratio of sentences with citations."""
        sentences = [s.strip() for s in _SENTENCE_RE.findall(answer) if len(s.strip()) >= 10]
        if not sentences:
            return 1.0
        supported = sum(1 for s in sentences if _CITATION_RE.search(s))
        return supported / len(sentences)

    def evaluate_citation_coverage(self, *, answer: str) -> float:
        """Ratio of non-empty paragraphs that contain at least one citation."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
        if not paragraphs:
            return 1.0
        covered = sum(1 for p in paragraphs if _CITATION_RE.search(p))
        return covered / len(paragraphs)

    def evaluate_context_precision(self, *, chunks: list["RetrievedChunk"], threshold: float = 0.4) -> float:
        """Ratio of chunks with score above threshold — proxy for context precision."""
        if not chunks:
            return 0.0
        high_quality = sum(
            1 for c in chunks
            if (getattr(c, "reranker_score", None) or c.fused_score or 0.0) >= threshold
        )
        return high_quality / len(chunks)

    async def evaluate_answer_relevancy(self, *, query: str, answer: str) -> float:
        """LLM-judged relevancy score (0-1)."""
        if self.llm is None:
            return -1.0
        prompt = _RELEVANCY_PROMPT.format(query=query[:500], answer=answer[:1000])
        try:
            import json
            raw = await self.llm.generate(prompt=prompt)
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                score = float(data.get("score", 0.0))
                return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning("Answer relevancy LLM call failed", extra={"error": str(exc)})
        return -1.0

    async def evaluate_faithfulness_llm(self, *, answer: str, chunks: list["RetrievedChunk"]) -> float:
        """LLM-judged faithfulness (fallback to citation-based if LLM unavailable)."""
        if self.llm is None:
            return self.evaluate_faithfulness(answer=answer)
        evidence = "\n\n".join(f"[{i+1}] {c.content[:400]}" for i, c in enumerate(chunks))
        prompt = _FAITHFULNESS_PROMPT.format(evidence=evidence[:3000], answer=answer[:1500])
        try:
            import json
            raw = await self.llm.generate(prompt=prompt)
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                supported = int(data.get("supported", 0))
                unsupported = int(data.get("unsupported", 0))
                total = supported + unsupported
                return supported / total if total > 0 else 1.0
        except Exception as exc:
            logger.warning("Faithfulness LLM call failed", extra={"error": str(exc)})
        return self.evaluate_faithfulness(answer=answer)

    def record(
        self,
        *,
        query: str,
        answer: str,
        chunks: list["RetrievedChunk"],
        was_refused: bool,
        faithfulness: float,
        answer_relevancy: float,
        context_precision: float,
        citation_coverage: float,
    ) -> None:
        self._results.append({
            "query_preview": query[:80],
            "was_refused": was_refused,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "citation_coverage": citation_coverage,
            "chunk_count": len(chunks),
        })

    def aggregate(self) -> RAGASMetrics:
        """Compute aggregate metrics over all recorded evaluations."""
        if not self._results:
            return RAGASMetrics()

        def _mean(key: str) -> float:
            vals = [r[key] for r in self._results if r[key] >= 0]
            return sum(vals) / len(vals) if vals else 0.0

        return RAGASMetrics(
            faithfulness=_mean("faithfulness"),
            answer_relevancy=_mean("answer_relevancy"),
            context_precision=_mean("context_precision"),
            citation_coverage=_mean("citation_coverage"),
            refusal_rate=sum(1 for r in self._results if r["was_refused"]) / len(self._results),
            sample_count=len(self._results),
            details={"samples": self._results[-10:]},
        )

    def reset(self) -> None:
        self._results.clear()
