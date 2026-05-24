from __future__ import annotations

import asyncio
import logging
import os
import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.processing.types import EvidenceBlock

logger = logging.getLogger(__name__)

# ── Config loader ──────────────────────────────────────────────────────────────

def _load_guardrails_config() -> dict:
    config_path = Path(__file__).parents[3] / "config" / "guardrails_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ── Enums ──────────────────────────────────────────────────────────────────────

class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class OverallVerdict(StrEnum):
    SUPPORTED = "supported"      # ≥ 90 % of claims supported, none contradicted
    PARTIAL = "partial"          # ≥ 50 % supported, none contradicted
    CONTRADICTED = "contradicted"  # at least one claim contradicted
    INSUFFICIENT = "insufficient"  # < 50 % supported


# ── Result models ──────────────────────────────────────────────────────────────

class PerClaimResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    claim_text: str
    verdict: ClaimVerdict
    best_source_chunk_id: str | None = None
    overlap_score: float = 0.0
    # Model-level confidence (NLI score when available, else same as overlap_score).
    model_confidence: float | None = None


class ClaimVerificationResult(BaseModel):
    # Legacy field — mirrors overall_verdict for backward-compat callers.
    verdict: ClaimVerdict
    corrected_facts: list[str] = Field(default_factory=list)
    citations: list[EvidenceBlock] = Field(default_factory=list)
    confidence: float
    was_refused: bool
    refusal_reason: str | None = None
    # New structured fields
    per_claim: list[PerClaimResult] = Field(default_factory=list)
    overall_verdict: OverallVerdict = OverallVerdict.INSUFFICIENT
    supported_ratio: float = 0.0


# ── Verifier ───────────────────────────────────────────────────────────────────

class ClaimVerifier:
    NUMBER_PATTERN = re.compile(r"\b(?:[1-9]\d+(?:\.\d+)?%?|\d+\.\d+%?)\b")
    CITATION_PATTERN = re.compile(r"\[\d+\]")
    SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]?")
    NEGATION_PATTERN = re.compile(
        r"\b(?:not|no|never|khong|không|khong phai|không phải|does not|do not|did not)\b",
        re.IGNORECASE,
    )
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

    def __init__(
        self,
        *,
        nli_model_name: str | None = None,
        nli_enabled: bool | None = None,
        claim_overlap_threshold: float | None = None,
        min_shared_tokens_contradiction: int | None = None,
    ) -> None:
        self.nli_model_name = nli_model_name or os.getenv(
            "AGENTBOOK_CLAIM_NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
        )
        if nli_enabled is None:
            nli_enabled = os.getenv("AGENTBOOK_CLAIM_NLI_ENABLED", "").strip().lower() in {"1", "true", "yes"}
        self.nli_enabled = nli_enabled
        self._nli_model = None
        self._nli_model_load_lock = asyncio.Lock()
        self._nli_predict_semaphore = asyncio.Semaphore(1)

        cfg = _load_guardrails_config().get("claim_verification", {})
        self.claim_overlap_threshold: float = (
            claim_overlap_threshold
            if claim_overlap_threshold is not None
            else float(cfg.get("claim_overlap_threshold", 0.15))
        )
        self.min_shared_tokens_contradiction: int = (
            min_shared_tokens_contradiction
            if min_shared_tokens_contradiction is not None
            else int(cfg.get("min_shared_tokens_contradiction", 3))
        )
        self.contradicted_majority_fraction: float = float(
            cfg.get("contradicted_majority_fraction", 0.5)
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def verify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        if not evidence:
            return self._empty_result()
        if self.nli_enabled:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.averify(claim=claim, evidence=evidence))
        return self._run_per_claim_pipeline(claim=claim, evidence=evidence)

    async def averify(self, *, claim: str, evidence: list[EvidenceBlock]) -> ClaimVerificationResult:
        if not evidence:
            return self._empty_result()
        # NLI path: attempt to enhance per-claim verdicts with cross-encoder scores.
        if self.nli_enabled:
            nli_model = await self._aload_nli_model()
            if nli_model is not None:
                return await self._run_per_claim_nli(claim=claim, evidence=evidence, model=nli_model)
        return self._run_per_claim_pipeline(claim=claim, evidence=evidence)

    # ── Core per-claim pipeline ─────────────────────────────────────────────────

    def _run_per_claim_pipeline(
        self, *, claim: str, evidence: list[EvidenceBlock]
    ) -> ClaimVerificationResult:
        sentences = self._split_atomic_claims(claim)
        if not sentences:
            stripped = self.CITATION_PATTERN.sub("", claim).strip()
            sentences = [stripped] if stripped else []
        sentences = self._restore_citation_signal(source=claim, claims=sentences)

        per_claim: list[PerClaimResult] = []
        for sentence in sentences:
            if not sentence:
                continue
            pc = self._classify_claim(claim_text=sentence, evidence=evidence)
            per_claim.append(pc)

        return self._aggregate(per_claim=per_claim, evidence=evidence)

    def _classify_claim(
        self,
        *,
        claim_text: str,
        evidence: list[EvidenceBlock],
    ) -> PerClaimResult:
        """Score one sentence against all evidence blocks; return a PerClaimResult."""
        claim_tokens = self._important_terms(claim_text)
        if not claim_tokens:
            return PerClaimResult(
                claim_text=claim_text,
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                overlap_score=0.0,
            )

        # Find best-matching evidence block by token-overlap ratio.
        best_score = 0.0
        best_ev: EvidenceBlock | None = None
        for ev in evidence:
            ev_tokens = self._important_terms(ev.snippet_original)
            shared = claim_tokens & ev_tokens
            score = len(shared) / len(claim_tokens)
            if score > best_score:
                best_score = score
                best_ev = ev

        best_chunk_id = best_ev.block_id if best_ev else None

        # Threshold gate: only declare support/contradiction when the evidence is
        # about the same topic as the claim (overlap_score ≥ threshold).
        if best_score >= self.claim_overlap_threshold and best_ev is not None:
            ev_text = best_ev.snippet_original
            shared_count = len(claim_tokens & self._important_terms(ev_text))

            # Contradiction: number mismatch against the best block.
            claim_numbers = set(self.NUMBER_PATTERN.findall(claim_text))
            if claim_numbers:
                ev_numbers = set(self.NUMBER_PATTERN.findall(ev_text))
                if ev_numbers and claim_numbers.isdisjoint(ev_numbers):
                    return PerClaimResult(
                        claim_text=claim_text,
                        verdict=ClaimVerdict.CONTRADICTED,
                        best_source_chunk_id=best_chunk_id,
                        overlap_score=best_score,
                    )

            # Contradiction: directional / negation mismatch.
            # Require min shared tokens so we don't flag unrelated pairs.
            if shared_count >= self.min_shared_tokens_contradiction:
                if self._semantic_contradiction(
                    claim=claim_text,
                    evidence_text=ev_text,
                    min_shared=self.min_shared_tokens_contradiction,
                ):
                    return PerClaimResult(
                        claim_text=claim_text,
                        verdict=ClaimVerdict.CONTRADICTED,
                        best_source_chunk_id=best_chunk_id,
                        overlap_score=best_score,
                    )

            return PerClaimResult(
                claim_text=claim_text,
                verdict=ClaimVerdict.SUPPORTED,
                best_source_chunk_id=best_chunk_id,
                overlap_score=best_score,
            )

        return PerClaimResult(
            claim_text=claim_text,
            verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
            best_source_chunk_id=best_chunk_id,
            overlap_score=best_score,
        )

    def _aggregate(
        self,
        *,
        per_claim: list[PerClaimResult],
        evidence: list[EvidenceBlock],
    ) -> ClaimVerificationResult:
        if not per_claim:
            return ClaimVerificationResult(
                verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
                citations=evidence[:5],
                confidence=0.0,
                was_refused=True,
                refusal_reason="no verifiable atomic claims found",
                per_claim=[],
                overall_verdict=OverallVerdict.INSUFFICIENT,
                supported_ratio=0.0,
            )

        total = len(per_claim)
        supported = [c for c in per_claim if c.verdict == ClaimVerdict.SUPPORTED]
        contradicted = [c for c in per_claim if c.verdict == ClaimVerdict.CONTRADICTED]
        supported_ratio = len(supported) / total

        def _eff_confidence(c: PerClaimResult) -> float:
            return c.model_confidence if c.model_confidence is not None else c.overlap_score

        # Only declare overall CONTRADICTED when contradictions actually
        # dominate. NLI cross-encoders give frequent false positives on
        # Vietnamese text; require a configurable majority before erasing
        # the answer. See guardrails_config.yaml → claim_verification →
        # contradicted_majority_fraction.
        if contradicted and len(contradicted) > total * self.contradicted_majority_fraction:
            overall = OverallVerdict.CONTRADICTED
            verdict = ClaimVerdict.CONTRADICTED
            confidence = max(_eff_confidence(c) for c in contradicted)
            corrected = [f"Claim may conflict with evidence: «{c.claim_text[:100]}»" for c in contradicted]
            return ClaimVerificationResult(
                verdict=verdict,
                corrected_facts=corrected,
                citations=self._citations_for(contradicted, evidence),
                confidence=min(1.0, confidence),
                was_refused=False,
                per_claim=per_claim,
                overall_verdict=overall,
                supported_ratio=supported_ratio,
            )

        if supported_ratio >= 0.5:
            overall = OverallVerdict.SUPPORTED if supported_ratio >= 0.9 else OverallVerdict.PARTIAL
            verdict = ClaimVerdict.SUPPORTED
            confidence = sum(_eff_confidence(c) for c in supported) / len(supported)
            return ClaimVerificationResult(
                verdict=verdict,
                citations=self._citations_for(supported, evidence),
                confidence=min(1.0, confidence),
                was_refused=False,
                per_claim=per_claim,
                overall_verdict=overall,
                supported_ratio=supported_ratio,
            )

        return ClaimVerificationResult(
            verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
            citations=evidence[:3],
            confidence=supported_ratio,
            was_refused=True,
            refusal_reason="insufficient evidence coverage for most claims",
            per_claim=per_claim,
            overall_verdict=OverallVerdict.INSUFFICIENT,
            supported_ratio=supported_ratio,
        )

    # ── NLI-enhanced path ───────────────────────────────────────────────────────

    async def _run_per_claim_nli(
        self,
        *,
        claim: str,
        evidence: list[EvidenceBlock],
        model,
    ) -> ClaimVerificationResult:
        sentences = self._split_atomic_claims(claim)
        if not sentences:
            stripped = self.CITATION_PATTERN.sub("", claim).strip()
            sentences = [stripped] if stripped else []
        sentences = self._restore_citation_signal(source=claim, claims=sentences)

        per_claim: list[PerClaimResult] = []
        for sentence in sentences:
            if not sentence:
                continue
            # First try NLI against top evidence blocks.
            ranked_ev = self._rank_evidence_for_claim(claim=sentence, evidence=evidence)
            pairs = [(ev.snippet_original, sentence) for ev in ranked_ev]
            try:
                async with self._nli_predict_semaphore:
                    raw_scores = await asyncio.to_thread(model.predict, pairs)
                label_order = self._nli_label_order(model)
                verdict, best_idx, nli_score = self._best_nli_verdict(raw_scores, label_order)
                overlap = self._compute_overlap_score(sentence, ranked_ev[best_idx].snippet_original) if ranked_ev else 0.0
                chunk_id = ranked_ev[best_idx].block_id if ranked_ev else None
                per_claim.append(PerClaimResult(
                    claim_text=sentence,
                    verdict=verdict,
                    best_source_chunk_id=chunk_id,
                    overlap_score=overlap,
                    model_confidence=nli_score,
                ))
                continue
            except Exception as exc:
                logger.warning("NLI inference failed for claim; falling back to token overlap",
                               extra={"error": str(exc), "claim": sentence[:60]})
            # Fallback to token overlap for this sentence.
            per_claim.append(self._classify_claim(claim_text=sentence, evidence=evidence))

        return self._aggregate(per_claim=per_claim, evidence=evidence)

    @staticmethod
    def _best_nli_verdict(
        raw_scores, label_order: list[str]
    ) -> tuple[ClaimVerdict, int, float]:
        """Return (verdict, best_evidence_index, nli_confidence_score)."""
        rows = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if rows and isinstance(rows[0], (float, int)):
            rows = [rows]
        best_label = "neutral"
        best_score = 0.0
        best_idx = 0
        for idx, row in enumerate(rows):
            if not row:
                continue
            top_pos = max(range(len(row)), key=lambda p: float(row[p]))
            label = label_order[top_pos] if top_pos < len(label_order) else "neutral"
            score = float(row[top_pos])
            if label in {"contradiction", "entailment"} and score > best_score:
                best_label = label
                best_score = score
                best_idx = idx
        if best_label == "contradiction" and best_score >= 0.45:
            return ClaimVerdict.CONTRADICTED, best_idx, best_score
        if best_label == "entailment" and best_score >= 0.45:
            return ClaimVerdict.SUPPORTED, best_idx, best_score
        return ClaimVerdict.NOT_ENOUGH_EVIDENCE, best_idx, best_score

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _compute_overlap_score(self, claim: str, evidence_text: str) -> float:
        claim_tokens = self._important_terms(claim)
        if not claim_tokens:
            return 0.0
        ev_tokens = self._important_terms(evidence_text)
        return len(claim_tokens & ev_tokens) / len(claim_tokens)

    @staticmethod
    def _citations_for(
        results: list[PerClaimResult], all_evidence: list[EvidenceBlock]
    ) -> list[EvidenceBlock]:
        ids = {r.best_source_chunk_id for r in results if r.best_source_chunk_id}
        prioritised = [ev for ev in all_evidence if ev.block_id in ids]
        remainder = [ev for ev in all_evidence if ev.block_id not in ids]
        return (prioritised + remainder)[:5]

    def _empty_result(self) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            verdict=ClaimVerdict.NOT_ENOUGH_EVIDENCE,
            confidence=0.0,
            was_refused=True,
            refusal_reason="no evidence available to verify the claim",
            overall_verdict=OverallVerdict.INSUFFICIENT,
            supported_ratio=0.0,
        )

    @classmethod
    def _split_atomic_claims(cls, text: str) -> list[str]:
        cleaned = cls.CITATION_PATTERN.sub("", text or "")
        cleaned = re.sub(r"\[[^\]]*(?:source|ngu[^\]]*n)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
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
        return [
            claim if cls.CITATION_PATTERN.search(claim) else f"{claim} {citation}"
            for claim in claims
        ]

    def _rank_evidence_for_claim(
        self, *, claim: str, evidence: list[EvidenceBlock]
    ) -> list[EvidenceBlock]:
        claim_tokens = self._important_terms(claim)
        return sorted(
            evidence,
            key=lambda ev: len(claim_tokens & self._important_terms(ev.snippet_original)),
            reverse=True,
        )[:5]

    @classmethod
    def _semantic_contradiction(
        cls, *, claim: str, evidence_text: str, min_shared: int = 3
    ) -> bool:
        claim_tokens = cls._important_terms(claim)
        evidence_tokens = cls._important_terms(evidence_text)
        shared = claim_tokens & evidence_tokens
        if len(shared) < min_shared:
            return False
        claim_negated = bool(cls.NEGATION_PATTERN.search(claim))
        evidence_negated = bool(cls.NEGATION_PATTERN.search(evidence_text))
        if claim_negated != evidence_negated and len(shared) >= 5:
            return True
        for positive, negative in cls.DIRECTIONAL_PAIRS:
            claim_has_pos = cls._directional_near_shared(claim, positive, shared)
            claim_has_neg = cls._directional_near_shared(claim, negative, shared)
            evidence_has_pos = cls._directional_near_shared(evidence_text, positive, shared)
            evidence_has_neg = cls._directional_near_shared(evidence_text, negative, shared)
            if (claim_has_pos and evidence_has_neg) or (claim_has_neg and evidence_has_pos):
                return True
        return False

    @staticmethod
    def _directional_near_shared(
        text: str, direction_re: re.Pattern, shared_terms: set[str]
    ) -> bool:
        for sentence in re.split(r"[.!?\n]", text):
            if direction_re.search(sentence):
                sentence_terms = {
                    t.lower() for t in re.findall(r"[\w\-]{4,}", sentence, flags=re.UNICODE)
                }
                if sentence_terms & shared_terms:
                    return True
        return False

    @staticmethod
    def _important_terms(text: str) -> set[str]:
        stopwords = {
            "the", "and", "or", "is", "are",
            "la", "là", "va", "và", "cua", "của",
            "trong", "khong", "không",
        }
        terms: set[str] = set()
        for token in re.findall(r"[\w\-]{4,}", text, flags=re.UNICODE):
            normalized = token.lower()
            if len(normalized) > 4 and normalized.endswith("s"):
                normalized = normalized[:-1]
            if normalized not in stopwords:
                terms.add(normalized)
        return terms

    # ── NLI model management ───────────────────────────────────────────────────

    async def _aload_nli_model(self):
        if self._nli_model is not None:
            return self._nli_model
        async with self._nli_model_load_lock:
            if self._nli_model is None:
                await asyncio.to_thread(self._load_nli_model)
            return self._nli_model

    def _load_nli_model(self):
        if self._nli_model is not None:
            return self._nli_model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            logger.info("NLI verifier unavailable — sentence-transformers not installed",
                        extra={"error": str(exc)})
            return None
        try:
            self._nli_model = CrossEncoder(self.nli_model_name)
        except Exception as exc:
            logger.warning("NLI verifier model could not be loaded",
                           extra={"model": self.nli_model_name, "error": str(exc)})
            return None
        return self._nli_model

    @staticmethod
    def _nli_label_order(model) -> list[str]:
        id2label = getattr(getattr(getattr(model, "model", None), "config", None), "id2label", None)
        if id2label:
            return [str(id2label[i]).lower() for i in sorted(id2label)]
        return ["contradiction", "entailment", "neutral"]

    # ── Backward-compat stubs (kept so existing call sites need no change) ─────

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
