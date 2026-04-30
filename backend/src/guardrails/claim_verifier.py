from __future__ import annotations

import logging
import os
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from src.processing.types import EvidenceBlock

logger = logging.getLogger(__name__)


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class ClaimVerificationResult(BaseModel):
    verdict: ClaimVerdict
    corrected_facts: list[str] = Field(default_factory=list)
    citations: list[EvidenceBlock] = Field(default_factory=list)
    confidence: float
    was_refused: bool
    refusal_reason: str | None = None


class ClaimVerifier:
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|khong|không|khong phai|không phải|does not|do not|did not)\b", re.IGNORECASE)
    DIRECTIONAL_PAIRS = [
        (
            re.compile(r"\b(?:increase|increases|increased|raise|raises|raised|higher|tang|tăng)\b", re.IGNORECASE),
            re.compile(r"\b(?:decrease|decreases|decreased|reduce|reduces|reduced|lower|giam|giảm)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(?:improve|improves|improved|better|cai thien|cải thiện)\b", re.IGNORECASE),
            re.compile(r"\b(?:worsen|worsens|worsened|worse|lam xau|làm xấu)\b", re.IGNORECASE),
        ),
        (
            re.compile(r"\b(?:cause|causes|caused|lead to|leads to|dan den|dẫn đến|gay ra|gây ra)\b", re.IGNORECASE),
            re.compile(r"\b(?:prevent|prevents|prevented|avoid|avoids|avoided|ngan|ngăn|tranh|tránh)\b", re.IGNORECASE),
        ),
    ]

    def __init__(self, *, nli_model_name: str | None = None, nli_enabled: bool | None = None) -> None:
        self.nli_model_name = nli_model_name or os.getenv("AGENTBOOK_CLAIM_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
        if nli_enabled is None:
            nli_enabled = os.getenv("AGENTBOOK_CLAIM_NLI_ENABLED", "").strip().lower() in {"1", "true", "yes"}
        self.nli_enabled = nli_enabled
        self._nli_model = None

    def verify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        if not evidence:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.0,
                was_refused=True,
                refusal_reason="no evidence available to verify the claim",
            )
        nli_result = self._verify_nli(claim=claim, evidence=evidence)
        if nli_result is not None:
            return nli_result

        claim_numbers = set(self.NUMBER_PATTERN.findall(claim))
        evidence_text = "\n".join(item.snippet_original for item in evidence)
        evidence_numbers = set(self.NUMBER_PATTERN.findall(evidence_text))
        if claim_numbers and claim_numbers.isdisjoint(evidence_numbers):
            return ClaimVerificationResult(
                verdict=ClaimVerdict.CONTRADICTED,
                corrected_facts=[f"Evidence numbers {sorted(evidence_numbers)} differ from claim numbers {sorted(claim_numbers)}"],
                citations=evidence,
                confidence=0.72,
                was_refused=False,
            )
        if self._semantic_contradiction(claim=claim, evidence_text=evidence_text):
            return ClaimVerificationResult(
                verdict=ClaimVerdict.CONTRADICTED,
                corrected_facts=["Claim wording appears to invert negation, direction, or causality relative to the evidence."],
                citations=evidence,
                confidence=0.62,
                was_refused=False,
            )
        claim_terms = self._important_terms(claim)
        evidence_terms = self._important_terms(evidence_text)
        if claim_terms and len(claim_terms & evidence_terms) >= max(1, min(3, len(claim_terms))):
            return ClaimVerificationResult(
                verdict=ClaimVerdict.SUPPORTED,
                citations=evidence,
                confidence=0.68,
                was_refused=False,
            )
        return ClaimVerificationResult(
            verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
            citations=evidence,
            confidence=0.35,
            was_refused=True,
            refusal_reason="evidence does not directly support or contradict the claim",
        )

    def _verify_nli(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult | None:
        if not self.nli_enabled:
            return None
        model = self._load_nli_model()
        if model is None:
            return None
        evidence_ranked = sorted(
            evidence,
            key=lambda item: len(self._important_terms(claim) & self._important_terms(item.snippet_original)),
            reverse=True,
        )[:5]
        if not evidence_ranked:
            return None
        pairs = [(item.snippet_original, claim) for item in evidence_ranked]
        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("NLI claim verification failed", extra={"error": str(exc), "error_type": type(exc).__name__})
            return None

        rows = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if rows and isinstance(rows[0], (float, int)):
            rows = [rows]
        label_order = self._nli_label_order(model)
        best_label = "neutral"
        best_score = 0.0
        best_index = 0
        for index, row in enumerate(rows):
            if not row:
                continue
            max_position = max(range(len(row)), key=lambda pos: float(row[pos]))
            label = label_order[max_position] if max_position < len(label_order) else "neutral"
            score = float(row[max_position])
            if label in {"contradiction", "entailment"} and score > best_score:
                best_label = label
                best_score = score
                best_index = index
        if best_label == "contradiction" and best_score >= 0.45:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.CONTRADICTED,
                corrected_facts=["NLI model classified the claim as contradicted by the evidence."],
                citations=[evidence_ranked[best_index]],
                confidence=min(0.95, max(0.65, best_score)),
                was_refused=False,
            )
        if best_label == "entailment" and best_score >= 0.45:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.SUPPORTED,
                citations=[evidence_ranked[best_index]],
                confidence=min(0.95, max(0.65, best_score)),
                was_refused=False,
            )
        return None

    def _load_nli_model(self):
        if self._nli_model is not None:
            return self._nli_model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.info("NLI verifier unavailable because sentence-transformers is not installed", extra={"error": str(exc)})
            return None
        try:
            self._nli_model = CrossEncoder(self.nli_model_name)
        except Exception as exc:
            logger.warning("NLI verifier model could not be loaded", extra={"model": self.nli_model_name, "error": str(exc)})
            return None
        return self._nli_model

    @staticmethod
    def _nli_label_order(model) -> list[str]:
        id2label = getattr(getattr(getattr(model, "model", None), "config", None), "id2label", None)
        if id2label:
            return [str(id2label[index]).lower() for index in sorted(id2label)]
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _important_terms(text: str) -> set[str]:
        stopwords = {"the", "and", "or", "is", "are", "la", "là", "va", "và", "cua", "của", "trong", "khong", "không"}
        return {token.lower() for token in re.findall(r"[\w\-]{4,}", text, flags=re.UNICODE) if token.lower() not in stopwords}

    @classmethod
    def _semantic_contradiction(cls, *, claim: str, evidence_text: str) -> bool:
        claim_terms = cls._important_terms(claim)
        evidence_terms = cls._important_terms(evidence_text)
        if len(claim_terms & evidence_terms) < 2:
            return False
        claim_negated = bool(cls.NEGATION_PATTERN.search(claim))
        evidence_negated = bool(cls.NEGATION_PATTERN.search(evidence_text))
        if claim_negated != evidence_negated and len(claim_terms & evidence_terms) >= 3:
            return True
        for positive, negative in cls.DIRECTIONAL_PAIRS:
            claim_positive = bool(positive.search(claim))
            claim_negative = bool(negative.search(claim))
            evidence_positive = bool(positive.search(evidence_text))
            evidence_negative = bool(negative.search(evidence_text))
            if (claim_positive and evidence_negative) or (claim_negative and evidence_positive):
                return True
        return False
