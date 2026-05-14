from __future__ import annotations

import asyncio
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
    # Only match numbers >= 10 or decimals/percentages to avoid false positives
    # from list ordinals (1. 2. 3.) and citation markers ([1]).
    NUMBER_PATTERN = re.compile(r"\b(?:[1-9]\d+(?:\.\d+)?%?|\d+\.\d+%?)\b")
    CITATION_PATTERN = re.compile(r"\[\d+\]")
    SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?")
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
        self._nli_model_load_lock = asyncio.Lock()
        self._nli_predict_semaphore = asyncio.Semaphore(1)

    def verify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        if not evidence:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.0,
                was_refused=True,
                refusal_reason="no evidence available to verify the claim",
            )
        if self.nli_enabled:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._verify_atomic_claims_async(claim=claim, evidence=evidence))
        return self._verify_atomic_claims(claim=claim, evidence=evidence)

    async def averify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        if not evidence:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                confidence=0.0,
                was_refused=True,
                refusal_reason="no evidence available to verify the claim",
            )
        return await self._verify_atomic_claims_async(claim=claim, evidence=evidence)

    def _verify_atomic_claims(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        atomic_claims = self._split_atomic_claims(claim)
        if not atomic_claims:
            stripped = self.CITATION_PATTERN.sub("", claim).strip()
            atomic_claims = [stripped] if stripped else []
        atomic_claims = self._restore_citation_signal(source=claim, claims=atomic_claims)
        results = [self._verify_single_claim(claim=item, evidence=evidence) for item in atomic_claims if item]
        return self._aggregate_atomic_results(results=results, evidence=evidence)

    async def _verify_atomic_claims_async(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        atomic_claims = self._split_atomic_claims(claim)
        if not atomic_claims:
            stripped = self.CITATION_PATTERN.sub("", claim).strip()
            atomic_claims = [stripped] if stripped else []
        atomic_claims = self._restore_citation_signal(source=claim, claims=atomic_claims)
        results = [await self._verify_single_claim_async(claim=item, evidence=evidence) for item in atomic_claims if item]
        return self._aggregate_atomic_results(results=results, evidence=evidence)

    def _aggregate_atomic_results(
        self,
        *,
        results: list[ClaimVerificationResult],
        evidence: list[EvidenceBlock],
    ) -> ClaimVerificationResult:
        if not results:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                citations=evidence,
                confidence=0.0,
                was_refused=True,
                refusal_reason="no verifiable atomic claims found",
            )

        contradicted = [item for item in results if item.verdict == ClaimVerdict.CONTRADICTED]
        if contradicted:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.CONTRADICTED,
                corrected_facts=[fact for item in contradicted for fact in item.corrected_facts],
                citations=self._dedupe_citations([citation for item in contradicted for citation in item.citations]),
                confidence=max(item.confidence for item in contradicted),
                was_refused=False,
            )

        unsupported = [item for item in results if item.verdict == ClaimVerdict.NOT_ENOUGH_EVIDENCE]
        if unsupported:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                corrected_facts=[fact for item in unsupported for fact in item.corrected_facts],
                citations=self._dedupe_citations([citation for item in results for citation in item.citations]),
                confidence=min(item.confidence for item in unsupported),
                was_refused=True,
                refusal_reason="one or more atomic claims are not directly supported by the evidence",
            )

        return ClaimVerificationResult(
            verdict=ClaimVerdict.SUPPORTED,
            citations=self._dedupe_citations([citation for item in results for citation in item.citations]),
            confidence=sum(item.confidence for item in results) / len(results),
            was_refused=False,
        )

    def _verify_single_claim(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        nli_result = self._verify_nli(claim=claim, evidence=evidence)
        if nli_result is not None:
            return nli_result
        return self._verify_single_claim_without_nli(claim=claim, evidence=evidence)

    def _verify_single_claim_without_nli(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
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
        claim_terms = self._important_terms(claim)
        evidence_terms = self._important_terms(evidence_text)
        shared_terms = claim_terms & evidence_terms
        if self.CITATION_PATTERN.search(claim) and len(shared_terms) >= 2:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.SUPPORTED,
                citations=evidence,
                confidence=0.58,
                was_refused=False,
            )
        if claim_terms and len(shared_terms) >= max(1, min(3, len(claim_terms))):
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

    async def _verify_single_claim_async(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        nli_result = await self._verify_nli_async(claim=claim, evidence=evidence)
        if nli_result is not None:
            return nli_result
        return self._verify_single_claim_without_nli(claim=claim, evidence=evidence)

    @classmethod
    def _split_atomic_claims(cls, text: str) -> list[str]:
        cleaned = cls.CITATION_PATTERN.sub("", text or "")
        cleaned = re.sub(r"\[[^\]]*(?:source|nguồn|nguá»“n)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        claims: list[str] = []
        for sentence in cls.SENTENCE_PATTERN.findall(cleaned):
            item = sentence.strip().strip("-*• ")
            if len(item) < 12 or item.startswith(">"):
                continue
            claims.append(item)
        return claims

    @classmethod
    def _restore_citation_signal(cls, *, source: str, claims: list[str]) -> list[str]:
        match = cls.CITATION_PATTERN.search(source or "")
        if match is None:
            return claims
        citation = match.group(0)
        return [claim if cls.CITATION_PATTERN.search(claim) else f"{claim} {citation}" for claim in claims]

    @staticmethod
    def _dedupe_citations(citations: list[EvidenceBlock]) -> list[EvidenceBlock]:
        deduped: list[EvidenceBlock] = []
        seen: set[tuple[str, int | None, str | None]] = set()
        for citation in citations:
            key = (citation.material_id, citation.page, citation.block_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(citation)
        return deduped

    def _verify_nli(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult | None:
        if self.nli_enabled:
            logger.debug("NLI verifier skipped in sync verify(); use averify() for semaphore-guarded NLI execution.")
        return None

    def _nli_result_from_scores(
        self,
        *,
        model,
        raw_scores,
        evidence_ranked: list[EvidenceBlock],
    ) -> ClaimVerificationResult | None:
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

    async def _verify_nli_async(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult | None:
        if not self.nli_enabled:
            return None
        model = await self._aload_nli_model()
        if model is None:
            return None
        evidence_ranked = self._rank_evidence_for_claim(claim=claim, evidence=evidence)
        if not evidence_ranked:
            return None
        pairs = [(item.snippet_original, claim) for item in evidence_ranked]
        try:
            async with self._nli_predict_semaphore:
                raw_scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:
            logger.warning("NLI claim verification failed", extra={"error": str(exc), "error_type": type(exc).__name__})
            return None
        return self._nli_result_from_scores(model=model, raw_scores=raw_scores, evidence_ranked=evidence_ranked)

    async def _aload_nli_model(self):
        if self._nli_model is not None:
            return self._nli_model
        async with self._nli_model_load_lock:
            if self._nli_model is None:
                await asyncio.to_thread(self._load_nli_model)
            return self._nli_model

    def _rank_evidence_for_claim(self, *, claim: str, evidence: list[EvidenceBlock]) -> list[EvidenceBlock]:
        claim_terms = self._important_terms(claim)
        return sorted(
            evidence,
            key=lambda item: len(claim_terms & self._important_terms(item.snippet_original)),
            reverse=True,
        )[:5]

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
        terms: set[str] = set()
        for token in re.findall(r"[\w\-]{4,}", text, flags=re.UNICODE):
            normalized = token.lower()
            if len(normalized) > 4 and normalized.endswith("s"):
                normalized = normalized[:-1]
            if normalized not in stopwords:
                terms.add(normalized)
        return terms

    @classmethod
    def _semantic_contradiction(cls, *, claim: str, evidence_text: str) -> bool:
        claim_terms = cls._important_terms(claim)
        evidence_terms = cls._important_terms(evidence_text)
        shared = claim_terms & evidence_terms
        if len(shared) < 3:
            return False
        # Negation mismatch: require strong term overlap and that the negation appears
        # in the same sentence as a shared term (not just anywhere in the text).
        claim_negated = bool(cls.NEGATION_PATTERN.search(claim))
        evidence_negated = bool(cls.NEGATION_PATTERN.search(evidence_text))
        if claim_negated != evidence_negated and len(shared) >= 5:
            return True
        # Directional mismatch: only flag when a directional word appears in the same
        # sentence as a shared key term in BOTH claim and evidence — avoids false positives
        # from documents that mention both positive and negative directions in different contexts.
        for positive, negative in cls.DIRECTIONAL_PAIRS:
            claim_has_pos = cls._directional_near_shared(claim, positive, shared)
            claim_has_neg = cls._directional_near_shared(claim, negative, shared)
            evidence_has_pos = cls._directional_near_shared(evidence_text, positive, shared)
            evidence_has_neg = cls._directional_near_shared(evidence_text, negative, shared)
            if (claim_has_pos and evidence_has_neg) or (claim_has_neg and evidence_has_pos):
                return True
        return False

    @staticmethod
    def _directional_near_shared(text: str, direction_re: re.Pattern, shared_terms: set[str]) -> bool:
        """Return True if direction_re matches in a sentence that also contains a shared term."""
        for sentence in re.split(r"[.!?\n]", text):
            if direction_re.search(sentence):
                sentence_terms = {t.lower() for t in re.findall(r"[\w\-]{4,}", sentence, flags=re.UNICODE)}
                if sentence_terms & shared_terms:
                    return True
        return False
