# -*- coding: utf-8 -*-
"""Tests for the unified EvidenceValidator (Phase 4)."""
from src.guardrails.evidence_validator import EvidenceValidator
from src.guardrails.refusal_policy import RefusalDecision, RefusalRule
from src.rag.types import RetrievedChunk


class _FakeRefusal:
    def __init__(self, decision: RefusalDecision) -> None:
        self._d = decision

    def check_evidence(self, chunks, query, aux_query=""):
        return self._d


def _chunk(cid: str, modality: str = "text", metadata: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, owner_id="o", collection_id="c", material_id="m",
        document_name="d.pdf", content="x", language="vi", modality=modality,
        metadata=metadata or {}, rerank_score=0.9,
    )


def _validator(decision: RefusalDecision) -> EvidenceValidator:
    return EvidenceValidator(_FakeRefusal(decision))  # type: ignore[arg-type]


def test_sufficient_when_not_refused():
    v = _validator(RefusalDecision(should_refuse=False, confidence=0.8))
    r = v.validate(query="q", chunks=[_chunk("c1")])
    assert r.sufficient is True
    assert r.should_refuse is False
    assert r.risk == "low"
    assert r.selected_evidence_ids == ["c1"]


def test_refusal_passthrough_high_risk():
    v = _validator(RefusalDecision(should_refuse=True, reason="no_evidence", rule=RefusalRule.NO_EVIDENCE))
    r = v.validate(query="q", chunks=[_chunk("c1")])
    assert r.should_refuse is True
    assert r.sufficient is False
    assert r.risk == "high"
    assert r.rule == RefusalRule.NO_EVIDENCE


def test_low_confidence_is_medium_risk():
    v = _validator(RefusalDecision(should_refuse=False, reason="partial_confidence", rule=RefusalRule.LOW_CONFIDENCE))
    r = v.validate(query="q", chunks=[_chunk("c1")])
    assert r.risk == "medium"
    assert r.sufficient is True  # not refused


def test_table_query_without_table_chunk_flags_modality():
    v = _validator(RefusalDecision(should_refuse=False, confidence=0.8))
    r = v.validate(query="tổng giá?", chunks=[_chunk("c1", modality="text")], preferred_modality="table")
    assert r.modality_ok is False
    assert "table_evidence" in r.missing
    assert r.sufficient is False     # modality gap makes it insufficient
    assert r.risk == "medium"
    assert r.should_refuse is False  # but refusal stays authoritative (unchanged)


def test_table_query_with_table_chunk_is_ok():
    tbl = _chunk("c1", modality="table", metadata={"sheet_names": ["S"]})
    v = _validator(RefusalDecision(should_refuse=False, confidence=0.8))
    r = v.validate(query="tổng giá?", chunks=[tbl], preferred_modality="table")
    assert r.modality_ok is True
    assert r.sufficient is True


def test_result_is_json_serializable_for_trace():
    v = _validator(RefusalDecision(should_refuse=False, confidence=0.8))
    r = v.validate(query="q", chunks=[_chunk("c1")])
    dumped = r.model_dump(mode="json")
    assert dumped["sufficient"] is True and dumped["risk"] == "low"
