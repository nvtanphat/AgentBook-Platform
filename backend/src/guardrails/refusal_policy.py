from __future__ import annotations

import logging
import math
import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_refusal_cfg() -> dict:
    config_path = Path(__file__).parents[3] / "config" / "guardrails_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as fh:
            return (yaml.safe_load(fh) or {}).get("refusal", {})
    except Exception:
        return {}


# ── Sigmoid (same formula as ConfidenceScorer) ────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


# ── Topic-coverage helper ─────────────────────────────────────────────────────

_VI_STOPWORDS = {
    "là", "gì", "có", "và", "hay", "hoặc", "của", "trong", "trên", "về",
    "cho", "với", "tại", "bởi", "từ", "đến", "khi", "nếu", "thì", "mà",
    "cũng", "đã", "sẽ", "đang", "được", "bị", "các", "một", "những",
    "này", "đó", "thế", "nào", "sao", "như", "vậy",
    "the", "and", "for", "are", "was", "how", "what", "why", "when",
    "does", "can", "not", "this", "that", "with", "from", "into",
    "is", "in", "of", "to", "be", "it", "an", "at", "by", "do",
}
_WORD_RE = re.compile(r"[\w]{2,}", re.UNICODE)


# ── Public types ──────────────────────────────────────────────────────────────

class RefusalRule(StrEnum):
    NO_EVIDENCE = "no_evidence"
    LOW_CONFIDENCE = "low_confidence"
    SCOPE_MISMATCH = "scope_mismatch"
    CLAIM_CONTRADICTED = "claim_contradicted"
    GRAPH_NO_PATH = "graph_no_path"
    CHITCHAT = "chitchat"


class RefusalDecision(BaseModel):
    should_refuse: bool
    reason: str | None = None
    confidence: float = 0.0
    rule: RefusalRule | None = None


# ── Policy ────────────────────────────────────────────────────────────────────

class RefusalPolicy:
    """Centralises all refusal decisions with thresholds from guardrails_config.yaml.

    Threshold semantics (all in sigmoid-normalised [0, 1] space):
      min_rerank_score          – NO_EVIDENCE gate: ALL chunks must exceed this
      min_confidence_threshold  – LOW_CONFIDENCE gate: the BEST chunk must exceed this
    """

    def __init__(self, cfg: dict | None = None) -> None:
        if cfg is None:
            cfg = _load_refusal_cfg()
        self.min_rerank_score: float = float(cfg.get("min_rerank_score", 0.30))
        self.min_confidence_threshold: float = float(cfg.get("min_confidence_threshold", 0.40))
        self.claim_overlap_threshold: float = float(cfg.get("claim_overlap_threshold", 0.50))
        self.max_false_accept_rate: float = float(cfg.get("max_false_accept_rate", 0.05))

    # ── Rule a + b: evidence quality ─────────────────────────────────────────

    def check_evidence(self, chunks: list, query: str = "") -> RefusalDecision:
        """Rules a (NO_EVIDENCE) and b (LOW_CONFIDENCE).

        Parameters
        ----------
        chunks:
            RetrievedChunk objects after reranking.
        query:
            Original query string used for topic-coverage validation.
        """
        if not chunks:
            return RefusalDecision(
                should_refuse=True,
                reason="no relevant evidence was found in the scoped materials",
                confidence=0.0,
                rule=RefusalRule.NO_EVIDENCE,
            )

        # Normalise rerank logits → [0, 1]; fall back to fused scores when unavailable.
        rerank_logits = [c.rerank_score for c in chunks if c.rerank_score is not None]
        if rerank_logits:
            normalized = [_sigmoid(s) for s in rerank_logits]
        else:
            fused = [c.fused_score for c in chunks]
            mx = max(fused) or 1.0
            normalized = [s / mx for s in fused]

        best = max(normalized)

        # Rule a: NO_EVIDENCE — every chunk is below the minimum relevance floor.
        if all(s < self.min_rerank_score for s in normalized):
            logger.info(
                "RefusalPolicy: NO_EVIDENCE — best normalized score %.4f < min_rerank_score %.4f",
                best,
                self.min_rerank_score,
            )
            return RefusalDecision(
                should_refuse=True,
                reason="all retrieved chunks scored below the minimum relevance threshold",
                confidence=best,
                rule=RefusalRule.NO_EVIDENCE,
            )

        # Rule b: LOW_CONFIDENCE — best score is between floors → partial confidence.
        if best < self.min_confidence_threshold:
            logger.info(
                "RefusalPolicy: LOW_CONFIDENCE partial — best %.4f in [%.4f, %.4f)",
                best,
                self.min_rerank_score,
                self.min_confidence_threshold,
            )
            return RefusalDecision(
                should_refuse=False,
                reason="partial_confidence",
                confidence=best,
                rule=RefusalRule.LOW_CONFIDENCE,
            )

        # Adequate score — but validate topic coverage before declaring success.
        if query and not self._topic_coverage(query, chunks):
            logger.info(
                "RefusalPolicy: LOW_CONFIDENCE topic coverage failed for query: %.60s", query
            )
            return RefusalDecision(
                should_refuse=True,
                reason="retrieved evidence does not cover the query topic",
                confidence=best,
                rule=RefusalRule.LOW_CONFIDENCE,
            )

        return RefusalDecision(should_refuse=False, confidence=best)

    # ── Rule c: scope mismatch (static) ──────────────────────────────────────

    @staticmethod
    def for_scope_mismatch() -> RefusalDecision:
        """Rule c: called when IntentClassifier returns OFF_TOPIC."""
        return RefusalDecision(
            should_refuse=True,
            reason="off_topic",
            confidence=0.0,
            rule=RefusalRule.SCOPE_MISMATCH,
        )

    # ── Rule d: claim contradicted ────────────────────────────────────────────

    @staticmethod
    def check_claim(overall_verdict, corrected_facts: list[str] | None = None) -> RefusalDecision:
        """Rule d: CLAIM_CONTRADICTED.

        Parameters
        ----------
        overall_verdict:
            ``OverallVerdict`` from ``ClaimVerificationResult``.
        corrected_facts:
            Optional list of corrected fact strings from the verifier.
        """
        from src.guardrails.claim_verifier import OverallVerdict  # local import avoids circular dep

        # A contradiction from the verifier is a hard refusal. The verifier is
        # responsible for avoiding false positives; policy should not downgrade
        # an explicit CONTRADICTED verdict just because only one fact failed.
        if overall_verdict == OverallVerdict.CONTRADICTED:
            detail = ""
            if corrected_facts:
                snippets = "; ".join(f[:80] for f in corrected_facts[:2])
                detail = f": {snippets}"
            return RefusalDecision(
                should_refuse=True,
                reason=f"claim_verification_contradicted{detail}",
                confidence=0.0,
                rule=RefusalRule.CLAIM_CONTRADICTED,
            )
        return RefusalDecision(should_refuse=False)

    # ── Rule e: graph no path (soft) ──────────────────────────────────────────

    @staticmethod
    def check_graph(paths: list, use_graph: bool) -> RefusalDecision:
        """Rule e: GRAPH_NO_PATH — soft signal; not a hard refusal.

        Returns should_refuse=False but flags the rule so the engine can log
        the fallback to text retrieval.
        """
        if use_graph and not paths:
            logger.info("RefusalPolicy: GRAPH_NO_PATH — falling back to text retrieval")
            return RefusalDecision(
                should_refuse=False,
                reason=None,
                confidence=0.0,
                rule=RefusalRule.GRAPH_NO_PATH,
            )
        return RefusalDecision(should_refuse=False)

    # ── Rule f: chitchat (static) ─────────────────────────────────────────────

    @staticmethod
    def for_chitchat() -> RefusalDecision:
        """Rule f: CHITCHAT — not a refusal; response follows a different path."""
        return RefusalDecision(
            should_refuse=False,
            reason=None,
            confidence=1.0,
            rule=RefusalRule.CHITCHAT,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _topic_coverage(query: str, chunks: list, min_overlap: int = 1) -> bool:
        tokens = {t for t in _WORD_RE.findall(query.lower()) if t not in _VI_STOPWORDS}
        if len(tokens) < 2:
            return True  # query too short to judge
        for chunk in chunks:
            text = chunk.content.lower()
            if sum(1 for t in tokens if t in text) >= min_overlap:
                return True
        return False
