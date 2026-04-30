from __future__ import annotations

from src.guardrails.claim_verifier import ClaimVerdict, ClaimVerifier
from src.processing.types import EvidenceBlock


def test_claim_verifier_detects_numeric_false_premise() -> None:
    evidence = [
        EvidenceBlock(
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="lecture.pdf",
            page=14,
            block_id="blk-002",
            block_type="paragraph",
            snippet_original="Dropout reduced validation error to 9.8%.",
            source_language="en",
            confidence=0.92,
        )
    ]

    result = ClaimVerifier().verify(claim="Dropout increases validation error to 92%.", evidence=evidence)

    assert result.verdict == ClaimVerdict.CONTRADICTED
    assert result.was_refused is False
    assert result.citations[0].block_id == "blk-002"


class FakeNLIConfig:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class FakeNLIInner:
    config = FakeNLIConfig()


class FakeNLIModel:
    model = FakeNLIInner()

    def predict(self, pairs):
        return [[0.91, 0.04, 0.05] for _ in pairs]


def test_claim_verifier_uses_optional_nli_model_for_semantic_contradiction() -> None:
    evidence = [
        EvidenceBlock(
            owner_id="user_demo",
            collection_id="c",
            material_id="m",
            document_name="lecture.pdf",
            page=14,
            block_id="blk-003",
            block_type="paragraph",
            snippet_original="Dropout reduces overfitting by randomly disabling activations.",
            source_language="en",
            confidence=0.92,
        )
    ]
    verifier = ClaimVerifier(nli_enabled=True)
    verifier._nli_model = FakeNLIModel()

    result = verifier.verify(claim="Dropout increases overfitting.", evidence=evidence)

    assert result.verdict == ClaimVerdict.CONTRADICTED
    assert result.confidence >= 0.9
