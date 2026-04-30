from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.processing.types import EvidenceBlock


class Contradiction(BaseModel):
    description: str
    evidence_refs: list[EvidenceBlock] = Field(default_factory=list)
    confidence: float


@dataclass(frozen=True)
class _NumericSignal:
    metric: str
    value: str
    context_terms: frozenset[str]
    evidence: EvidenceBlock


@dataclass(frozen=True)
class _SemanticSignal:
    polarity: str
    relation: str
    context_terms: frozenset[str]
    negated: bool
    evidence: EvidenceBlock


class ContradictionDetector:
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    METRIC_PATTERN = re.compile(r"\b(?:accuracy|precision|recall|f1|loss|error|auc|bleu|rouge)\b", re.IGNORECASE)
    NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|khong|không|does not|do not|did not)\b", re.IGNORECASE)
    POSITIVE_DIRECTION = re.compile(
        r"\b(?:increase|increases|increased|improve|improves|improved|cause|causes|lead|leads|tăng|cải thiện|gây|dẫn)\b",
        re.IGNORECASE,
    )
    NEGATIVE_DIRECTION = re.compile(
        r"\b(?:decrease|decreases|decreased|reduce|reduces|reduced|prevent|prevents|avoid|avoids|giảm|ngăn|tránh)\b",
        re.IGNORECASE,
    )
    STOPWORDS = {
        "the", "and", "or", "with", "from", "this", "that", "there", "their", "document",
        "experiment", "experiments", "report", "reports", "reported", "model", "models",
        "accuracy", "precision", "recall", "loss", "error", "auc", "bleu", "rouge",
        "khong", "không", "trong", "của", "cho", "các", "một", "những", "được", "với",
    }

    def detect(self, evidence: list[EvidenceBlock]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []
        numeric_signals = [signal for item in evidence for signal in self._numeric_signals(item)]
        semantic_signals = [signal for item in evidence for signal in self._semantic_signals(item)]

        for left_index, left in enumerate(numeric_signals):
            for right in numeric_signals[left_index + 1 :]:
                if left.metric != right.metric or left.value == right.value:
                    continue
                if not self._same_claim_context(left.context_terms, right.context_terms, left.evidence, right.evidence):
                    continue
                contradictions.append(
                    Contradiction(
                        description=f"Conflicting {left.metric} values: {left.value} vs {right.value}",
                        evidence_refs=[left.evidence, right.evidence],
                        confidence=0.72,
                    )
                )

        for left_index, left in enumerate(semantic_signals):
            for right in semantic_signals[left_index + 1 :]:
                if not self._same_claim_context(left.context_terms, right.context_terms, left.evidence, right.evidence):
                    continue
                opposite_direction = left.polarity != right.polarity and left.relation == right.relation
                negation_flip = left.polarity == right.polarity and left.negated != right.negated
                if not (opposite_direction or negation_flip):
                    continue
                contradictions.append(
                    Contradiction(
                        description="Semantic evidence appears to conflict on direction or negation",
                        evidence_refs=[left.evidence, right.evidence],
                        confidence=0.66,
                    )
                )
        return self._dedupe(contradictions)

    def _numeric_signals(self, item: EvidenceBlock) -> list[_NumericSignal]:
        text = item.snippet_original
        numbers = list(self.NUMBER_PATTERN.finditer(text))
        if not numbers:
            return []
        metrics = list(self.METRIC_PATTERN.finditer(text))
        if not metrics:
            return []
        context_terms = self._context_terms(text)
        signals: list[_NumericSignal] = []
        for number in numbers:
            nearest_metric = min(metrics, key=lambda metric: abs(metric.start() - number.start()))
            if abs(nearest_metric.start() - number.start()) > 80:
                continue
            signals.append(
                _NumericSignal(
                    metric=nearest_metric.group(0).lower(),
                    value=number.group(0),
                    context_terms=context_terms,
                    evidence=item,
                )
            )
        return signals

    def _semantic_signals(self, item: EvidenceBlock) -> list[_SemanticSignal]:
        text = item.snippet_original
        context_terms = self._context_terms(text)
        if len(context_terms) < 2:
            return []
        signals: list[_SemanticSignal] = []
        negated = bool(self.NEGATION_PATTERN.search(text))
        if self.POSITIVE_DIRECTION.search(text):
            signals.append(_SemanticSignal("positive", "direction", context_terms, negated, item))
        if self.NEGATIVE_DIRECTION.search(text):
            signals.append(_SemanticSignal("negative", "direction", context_terms, negated, item))
        return signals

    @classmethod
    def _context_terms(cls, text: str) -> frozenset[str]:
        terms = {
            token.lower()
            for token in re.findall(r"[\w\-]{4,}", text, flags=re.UNICODE)
            if token.lower() not in cls.STOPWORDS and not cls.NUMBER_PATTERN.fullmatch(token)
        }
        return frozenset(terms)

    @staticmethod
    def _same_claim_context(
        left_terms: frozenset[str],
        right_terms: frozenset[str],
        left: EvidenceBlock,
        right: EvidenceBlock,
    ) -> bool:
        shared = left_terms & right_terms
        if len(shared) >= 2:
            return True
        if left.material_id == right.material_id and left.page == right.page and len(shared) >= 1:
            return True
        return False

    @staticmethod
    def _dedupe(contradictions: list[Contradiction]) -> list[Contradiction]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[Contradiction] = []
        for contradiction in contradictions:
            ref_key = tuple(sorted(f"{ref.material_id}:{ref.page}:{ref.block_id}" for ref in contradiction.evidence_refs))
            key = (contradiction.description, ref_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(contradiction)
        return deduped
