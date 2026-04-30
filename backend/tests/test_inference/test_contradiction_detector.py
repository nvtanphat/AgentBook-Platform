from __future__ import annotations

from src.guardrails.contradiction_detector import ContradictionDetector
from src.processing.types import EvidenceBlock


def make_evidence(text: str, *, doc: str, page: int, block: str) -> EvidenceBlock:
    return EvidenceBlock(
        owner_id="user_demo",
        collection_id="c",
        material_id=doc,
        document_name=f"{doc}.pdf",
        page=page,
        block_id=block,
        block_type="paragraph",
        snippet_original=text,
        source_language="en",
        confidence=0.9,
    )


def test_contradiction_detector_ignores_numeric_differences_in_different_contexts() -> None:
    evidence = [
        make_evidence("Experiment A reports model accuracy 95%.", doc="a", page=1, block="a1"),
        make_evidence("Experiment B reports model accuracy 92%.", doc="b", page=1, block="b1"),
    ]

    contradictions = ContradictionDetector().detect(evidence)

    assert contradictions == []


def test_contradiction_detector_flags_same_context_numeric_conflict() -> None:
    evidence = [
        make_evidence("Dropout experiment reports validation accuracy 95%.", doc="a", page=1, block="a1"),
        make_evidence("Dropout experiment reports validation accuracy 92%.", doc="b", page=1, block="b1"),
    ]

    contradictions = ContradictionDetector().detect(evidence)

    assert len(contradictions) == 1
    assert "accuracy" in contradictions[0].description


def test_contradiction_detector_flags_semantic_negation_conflict() -> None:
    evidence = [
        make_evidence("Dropout reduces overfitting in neural networks.", doc="a", page=1, block="a1"),
        make_evidence("Dropout does not reduce overfitting in neural networks.", doc="b", page=1, block="b1"),
    ]

    contradictions = ContradictionDetector().detect(evidence)

    assert len(contradictions) == 1
    assert contradictions[0].confidence >= 0.6
