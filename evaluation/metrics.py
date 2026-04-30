from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvidenceKey:
    doc_id: str
    page: int
    block_id: str

    @classmethod
    def from_mapping(cls, value: dict) -> "EvidenceKey":
        return cls(doc_id=str(value.get("doc_id") or value.get("material_id")), page=int(value["page"]), block_id=str(value["block_id"]))


def recall_at_k(expected: Iterable[EvidenceKey], retrieved: list[EvidenceKey], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return len(expected_set & set(retrieved[:k])) / len(expected_set)


def precision_at_k(expected: Iterable[EvidenceKey], retrieved: list[EvidenceKey], k: int) -> float:
    if k <= 0:
        return 0.0
    expected_set = set(expected)
    return len(expected_set & set(retrieved[:k])) / k


def mrr_at_k(expected: Iterable[EvidenceKey], retrieved: list[EvidenceKey], k: int) -> float:
    expected_set = set(expected)
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(expected: Iterable[EvidenceKey], retrieved: list[EvidenceKey], k: int) -> float:
    expected_set = set(expected)
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in expected_set:
            dcg += 1.0 / _log2(rank + 1)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def citation_accuracy(expected: Iterable[EvidenceKey], citations: list[EvidenceKey]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return 1.0 if expected_set & set(citations) else 0.0


def ragas_stub() -> dict[str, str]:
    return {
        "status": "not_configured",
        "message": "RAGAS integration hook is present; install and configure ragas datasets before live scoring.",
    }


def _log2(value: int) -> float:
    import math

    return math.log(value, 2)
