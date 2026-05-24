from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from src.guardrails.refusal_policy import RefusalPolicy, RefusalRule


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _policy(**overrides) -> RefusalPolicy:
    """Build a RefusalPolicy with test thresholds."""
    cfg = {
        "min_rerank_score": 0.30,
        "min_confidence_threshold": 0.40,
        "claim_overlap_threshold": 0.50,
        "max_false_accept_rate": 0.05,
        **overrides,
    }
    return RefusalPolicy(cfg=cfg)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _chunk(*, rerank_score: float | None = None, fused_score: float = 0.5, content: str = "neural network regularisation dropout") -> SimpleNamespace:
    return SimpleNamespace(
        rerank_score=rerank_score,
        fused_score=fused_score,
        content=content,
    )


# ── Rule a: NO_EVIDENCE ───────────────────────────────────────────────────────

def test_no_evidence_empty_chunks():
    """Empty chunk list → NO_EVIDENCE refusal."""
    policy = _policy()
    decision = policy.check_evidence([])
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.NO_EVIDENCE


def test_no_evidence_all_below_min_rerank_score():
    """All chunks have normalized score < 0.30 → NO_EVIDENCE refusal."""
    policy = _policy()
    # logit whose sigmoid is ~0.25 (well below 0.30)
    low_logit = math.log(0.25 / 0.75)  # logit(0.25) ≈ -1.099
    chunks = [_chunk(rerank_score=low_logit), _chunk(rerank_score=low_logit - 0.5)]
    decision = policy.check_evidence(chunks)
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.NO_EVIDENCE
    assert decision.confidence < 0.30


def test_no_evidence_one_chunk_just_at_floor():
    """Single chunk exactly at min_rerank_score is counted as passing the floor."""
    policy = _policy()
    # sigmoid(logit) == 0.30 exactly → at threshold, should NOT trigger NO_EVIDENCE
    # logit for 0.30 = log(0.30/0.70)
    logit_at_floor = math.log(0.30 / 0.70)
    chunk = _chunk(rerank_score=logit_at_floor)
    decision = policy.check_evidence([chunk])
    # normalized score == 0.30 which is NOT < 0.30, so NO_EVIDENCE doesn't fire.
    # But it IS < min_confidence_threshold (0.40) → LOW_CONFIDENCE / partial.
    assert decision.rule != RefusalRule.NO_EVIDENCE


# ── Rule b: LOW_CONFIDENCE ────────────────────────────────────────────────────

def test_low_confidence_partial_between_thresholds():
    """Best score in [0.30, 0.40) → not refused, reason='partial_confidence'."""
    policy = _policy()
    # logit for sigmoid ≈ 0.35
    logit_35 = math.log(0.35 / 0.65)
    chunk = _chunk(rerank_score=logit_35)
    decision = policy.check_evidence([chunk])
    assert decision.should_refuse is False
    assert decision.reason == "partial_confidence"
    assert decision.rule == RefusalRule.LOW_CONFIDENCE


def test_low_confidence_no_refuse_above_threshold():
    """Best score >= 0.40 with topic coverage → not refused."""
    policy = _policy()
    logit_60 = math.log(0.60 / 0.40)
    chunk = _chunk(rerank_score=logit_60)
    decision = policy.check_evidence([chunk], query="dropout neural")
    assert decision.should_refuse is False
    assert decision.reason != "partial_confidence"


def test_low_confidence_fused_score_fallback():
    """When rerank_score is None, fused_score path is used.

    A fused_score of 0.0 → max(fused)=0 → mx falls back to 1.0 → normalized=0.0
    which is below min_rerank_score (0.30) → NO_EVIDENCE refusal.
    """
    policy = _policy()
    chunks = [_chunk(rerank_score=None, fused_score=0.0)]
    decision = policy.check_evidence(chunks)
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.NO_EVIDENCE


# ── Rule b edge: topic coverage ───────────────────────────────────────────────

def test_topic_coverage_fails_for_off_topic_chunks():
    """High rerank score but chunk content unrelated to query → refuse."""
    policy = _policy()
    logit_high = math.log(0.80 / 0.20)
    chunk = _chunk(rerank_score=logit_high, content="pandas DataFrame merge join groupby")
    decision = policy.check_evidence([chunk], query="quantum entanglement photon")
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.LOW_CONFIDENCE
    assert "topic" in (decision.reason or "")


def test_topic_coverage_skipped_for_short_query():
    """Query with < 2 content tokens skips topic-coverage check."""
    policy = _policy()
    logit_high = math.log(0.80 / 0.20)
    chunk = _chunk(rerank_score=logit_high, content="unrelated content here entirely different")
    # very short query — stopwords stripped leaves < 2 tokens
    decision = policy.check_evidence([chunk], query="the")
    assert decision.should_refuse is False


# ── Rule c: SCOPE_MISMATCH ────────────────────────────────────────────────────

def test_scope_mismatch_always_refuses():
    decision = RefusalPolicy.for_scope_mismatch()
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.SCOPE_MISMATCH
    assert decision.reason == "off_topic"
    assert decision.confidence == 0.0


# ── Rule d: CLAIM_CONTRADICTED ────────────────────────────────────────────────

def test_claim_contradicted_refuses():
    from src.guardrails.claim_verifier import OverallVerdict

    decision = RefusalPolicy.check_claim(OverallVerdict.CONTRADICTED)
    assert decision.should_refuse is True
    assert decision.rule == RefusalRule.CLAIM_CONTRADICTED
    assert "contradicted" in decision.reason


def test_claim_contradicted_includes_corrected_facts():
    from src.guardrails.claim_verifier import OverallVerdict

    facts = ["Dropout rate should be 0.5 not 0.9", "Accuracy ≠ Precision"]
    decision = RefusalPolicy.check_claim(OverallVerdict.CONTRADICTED, corrected_facts=facts)
    assert decision.should_refuse is True
    assert "Dropout rate" in decision.reason


def test_claim_supported_does_not_refuse():
    from src.guardrails.claim_verifier import OverallVerdict

    for verdict in (OverallVerdict.SUPPORTED, OverallVerdict.PARTIAL, OverallVerdict.INSUFFICIENT):
        decision = RefusalPolicy.check_claim(verdict)
        assert decision.should_refuse is False, f"Expected no refusal for {verdict}"


# ── Rule e: GRAPH_NO_PATH ─────────────────────────────────────────────────────

def test_graph_no_path_is_soft_not_refusal():
    """Empty graph paths when use_graph=True → soft signal, not a refusal."""
    decision = RefusalPolicy.check_graph([], use_graph=True)
    assert decision.should_refuse is False
    assert decision.rule == RefusalRule.GRAPH_NO_PATH


def test_graph_no_path_skipped_when_graph_disabled():
    """When use_graph=False, GRAPH_NO_PATH rule doesn't fire."""
    decision = RefusalPolicy.check_graph([], use_graph=False)
    assert decision.rule != RefusalRule.GRAPH_NO_PATH


def test_graph_has_paths_returns_clean_decision():
    """Non-empty paths → no rule fires."""
    fake_path = SimpleNamespace(path=["entity:a", "relation:r", "entity:b"], confidence=0.8)
    decision = RefusalPolicy.check_graph([fake_path], use_graph=True)
    assert decision.should_refuse is False
    assert decision.rule is None


# ── Rule f: CHITCHAT ──────────────────────────────────────────────────────────

def test_chitchat_is_not_a_refusal():
    decision = RefusalPolicy.for_chitchat()
    assert decision.should_refuse is False
    assert decision.rule == RefusalRule.CHITCHAT
    assert decision.confidence == 1.0
    assert decision.reason is None


# ── RefusalDecision schema ────────────────────────────────────────────────────

def test_refusal_decision_serialises():
    """RefusalDecision is a Pydantic model and serialises cleanly."""
    decision = RefusalPolicy.for_scope_mismatch()
    dumped = decision.model_dump()
    assert dumped["should_refuse"] is True
    assert dumped["rule"] == "scope_mismatch"
    assert "reason" in dumped


# ── Config loading ────────────────────────────────────────────────────────────

def test_custom_thresholds_are_respected():
    """Thresholds passed via cfg dict override defaults."""
    policy = _policy(min_rerank_score=0.10, min_confidence_threshold=0.20)
    # logit for sigmoid ≈ 0.15 → above the custom floor (0.10)
    logit_15 = math.log(0.15 / 0.85)
    chunk = _chunk(rerank_score=logit_15)
    decision = policy.check_evidence([chunk])
    # Normalized ≈ 0.15 ≥ custom min_rerank_score (0.10) so NO_EVIDENCE doesn't fire.
    assert decision.rule != RefusalRule.NO_EVIDENCE
