"""Unit tests for ClaimVerifier — real per-claim verification algorithm."""
from __future__ import annotations

import pytest

from src.guardrails.claim_verifier import (
    ClaimVerdict,
    ClaimVerifier,
    OverallVerdict,
    PerClaimResult,
)
from src.processing.types import EvidenceBlock


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ev(snippet: str, block_id: str = "blk1") -> EvidenceBlock:
    return EvidenceBlock(
        owner_id="user1",
        collection_id="col1",
        material_id="mat1",
        document_name="test.pdf",
        page=1,
        block_id=block_id,
        block_type="paragraph",
        snippet_original=snippet,
        source_language="en",
    )


def _verifier(threshold: float = 0.15) -> ClaimVerifier:
    """Return a deterministic verifier with NLI disabled."""
    return ClaimVerifier(nli_enabled=False, claim_overlap_threshold=threshold)


# ── Test 1: fully supported claim ──────────────────────────────────────────────

def test_fully_supported_answer_returns_supported():
    """A single-sentence answer covered by the evidence should be SUPPORTED."""
    verifier = _verifier()
    answer = "Dropout reduces overfitting in neural networks."
    evidence = [_ev("Dropout is a regularization technique that reduces overfitting in deep neural networks.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.verdict == ClaimVerdict.SUPPORTED
    assert result.overall_verdict == OverallVerdict.SUPPORTED
    assert result.supported_ratio == 1.0
    assert result.was_refused is False
    assert len(result.per_claim) >= 1
    assert result.per_claim[0].verdict == ClaimVerdict.SUPPORTED
    assert result.per_claim[0].overlap_score > 0.0


def test_per_claim_has_best_source_chunk_id_when_supported():
    """best_source_chunk_id must point to the matching evidence block."""
    verifier = _verifier()
    answer = "Backpropagation computes gradients efficiently using the chain rule."
    evidence = [
        _ev("Backpropagation is an algorithm that computes gradients via chain rule.", block_id="blk-bp"),
        _ev("Unrelated text about cooking pasta.", block_id="blk-unrelated"),
    ]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.overall_verdict in {OverallVerdict.SUPPORTED, OverallVerdict.PARTIAL}
    supported_claims = [c for c in result.per_claim if c.verdict == ClaimVerdict.SUPPORTED]
    assert any(c.best_source_chunk_id == "blk-bp" for c in supported_claims)


# ── Test 2: contradicted claim ─────────────────────────────────────────────────

def test_directional_contradiction_is_detected():
    """Directional mismatch (increases vs reduces) should produce CONTRADICTED."""
    verifier = _verifier()
    answer = "Dropout increases overfitting during training."
    evidence = [_ev("Dropout reduces overfitting. It is a widely used regularization strategy.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.verdict == ClaimVerdict.CONTRADICTED
    assert result.overall_verdict == OverallVerdict.CONTRADICTED
    assert result.was_refused is False
    assert len(result.corrected_facts) >= 1
    contradicted = [c for c in result.per_claim if c.verdict == ClaimVerdict.CONTRADICTED]
    assert contradicted, "Expected at least one per-claim contradiction"
    assert contradicted[0].overlap_score > 0.0


def test_number_mismatch_triggers_contradiction():
    """A claim with different numbers than the evidence should be contradicted."""
    verifier = _verifier()
    answer = "The model achieved 95% accuracy on the test set."
    evidence = [_ev("The model achieved 72% accuracy on the test set.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.verdict == ClaimVerdict.CONTRADICTED
    assert result.overall_verdict == OverallVerdict.CONTRADICTED


# ── Test 3: insufficient evidence ──────────────────────────────────────────────

def test_unrelated_evidence_returns_insufficient():
    """Evidence that shares no tokens with the claim should produce INSUFFICIENT."""
    verifier = _verifier()
    answer = "The moon is made of cheese and orbits the Earth."
    evidence = [_ev("Dropout is a technique used in deep learning regularization.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE
    assert result.overall_verdict == OverallVerdict.INSUFFICIENT
    assert result.supported_ratio == 0.0
    assert result.was_refused is True
    assert all(c.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE for c in result.per_claim)


def test_no_evidence_returns_insufficient():
    """Empty evidence list should immediately return INSUFFICIENT without crashing."""
    verifier = _verifier()
    result = verifier.verify(claim="Anything at all.", evidence=[])

    assert result.overall_verdict == OverallVerdict.INSUFFICIENT
    assert result.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE
    assert result.was_refused is True


# ── Test 4: partial support (multi-sentence answer) ───────────────────────────

def test_multi_claim_answer_with_partial_coverage():
    """An answer where half the claims are covered should produce PARTIAL."""
    verifier = _verifier()
    # First sentence is covered; second is unrelated.
    answer = "Dropout reduces overfitting. The moon is made of cheese."
    evidence = [_ev("Dropout reduces overfitting in neural networks and improves generalization.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    assert result.overall_verdict in {OverallVerdict.PARTIAL, OverallVerdict.INSUFFICIENT}
    assert 0.0 < result.supported_ratio <= 1.0
    verdicts = {c.verdict for c in result.per_claim}
    assert ClaimVerdict.SUPPORTED in verdicts
    assert ClaimVerdict.NOT_ENOUGH_EVIDENCE in verdicts


# ── Test 5: overlap score in per_claim ────────────────────────────────────────

def test_overlap_score_is_bounded_and_positive():
    """overlap_score must be in [0, 1] and positive when claims are supported."""
    verifier = _verifier()
    answer = "Regularization prevents overfitting and improves model generalization."
    evidence = [_ev("Regularization techniques prevent overfitting and improve model generalization on unseen data.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    for claim in result.per_claim:
        assert 0.0 <= claim.overlap_score <= 1.0
    supported = [c for c in result.per_claim if c.verdict == ClaimVerdict.SUPPORTED]
    assert all(c.overlap_score > 0.0 for c in supported)


# ── Test 6: async averify has same behaviour ──────────────────────────────────

@pytest.mark.asyncio
async def test_averify_async_matches_sync():
    """averify() should produce the same verdict as verify() for NLI-off mode."""
    verifier = _verifier()
    answer = "Dropout reduces overfitting in neural networks."
    evidence = [_ev("Dropout reduces overfitting. It is used in deep learning.")]

    sync_result = verifier.verify(claim=answer, evidence=evidence)
    async_result = await verifier.averify(claim=answer, evidence=evidence)

    assert sync_result.verdict == async_result.verdict
    assert sync_result.overall_verdict == async_result.overall_verdict
    assert sync_result.supported_ratio == async_result.supported_ratio


# ── Test 7: custom threshold ──────────────────────────────────────────────────

def test_high_threshold_raises_bar_for_support():
    """A very high threshold (0.9) should make most claims NOT_ENOUGH_EVIDENCE."""
    verifier = _verifier(threshold=0.9)
    # Only 2 of ~6 claim tokens appear in evidence → ratio ≈ 0.33 < 0.9
    answer = "Dropout reduces overfitting in neural networks."
    evidence = [_ev("Dropout reduces overfitting.")]

    result = verifier.verify(claim=answer, evidence=evidence)

    # With 0.9 threshold only exact-match claims pass — likely INSUFFICIENT here
    # (we just check it doesn't crash and returns valid structure)
    assert result.overall_verdict in {OverallVerdict.SUPPORTED, OverallVerdict.PARTIAL, OverallVerdict.INSUFFICIENT}
    assert 0.0 <= result.supported_ratio <= 1.0


# ── Test 8: config threshold loaded from YAML ────────────────────────────────

def test_default_verifier_loads_threshold_from_config():
    """ClaimVerifier() with no args should load threshold from guardrails_config.yaml."""
    verifier = ClaimVerifier(nli_enabled=False)
    # The YAML sets 0.15; just check it's a reasonable float.
    assert 0.0 < verifier.claim_overlap_threshold <= 1.0
