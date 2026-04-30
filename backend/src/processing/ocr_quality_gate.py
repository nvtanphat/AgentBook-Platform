"""OCR text quality gate for printed scan images.

Scores the OCR output of a ParsedDocument using four lightweight signals:
  - valid_char_ratio    : fraction of printable, non-noise characters
  - meaningful_word_ratio: fraction of tokens with ≥2 alphabetic chars
  - repetition_penalty  : fraction of text in repetitive char runs (aaaa, ----)
  - symbol_density      : fraction of chars that are unusual symbols (OCR artefacts)

The composite score is in [0, 1]. Scores below ``min_ocr_text_quality`` from
config cause the pipeline to set ``failed_stage="ocr_quality"`` instead of
proceeding to indexing. Scores in (min, warn_ocr_text_quality) emit a warning
but allow indexing to continue.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.processing.types import ParsedDocument

# ── Constants ────────────────────────────────────────────────────────────────

_REPETITION_RE = re.compile(r"(.)\1{3,}")  # 4+ consecutive identical chars

# Punctuation that should NOT count as "unusual symbols" in normal text.
_ALLOWED_PUNCTUATION = frozenset(".,;:!?-–—\"'`()[]{}…/\\|@#$%^&*+=<>~")

# ── Public API ────────────────────────────────────────────────────────────────


@dataclass
class OCRQualityReport:
    score: float                       # composite in [0, 1]
    valid_char_ratio: float
    meaningful_word_ratio: float
    repetition_ratio: float
    symbol_density: float
    total_chars: int
    warnings: list[str] = field(default_factory=list)

    def is_acceptable(self, min_score: float) -> bool:
        return self.score >= min_score

    def flag_summary(self) -> str:
        return "; ".join(self.warnings) if self.warnings else "ok"


def score_ocr_document(
    parsed: ParsedDocument,
    *,
    min_score: float = 0.35,
    warn_score: float = 0.55,
) -> OCRQualityReport:
    """Return an OCRQualityReport for a parsed document.

    Only meaningful for documents produced by an OCR engine (PNG/JPG/JPEG path).
    Calling this on Docling-parsed PDFs is harmless but unnecessary.
    """
    text = "\n".join(block.content for block in parsed.blocks)
    total_chars = len(text)

    if total_chars == 0:
        return OCRQualityReport(
            score=0.0,
            valid_char_ratio=0.0,
            meaningful_word_ratio=0.0,
            repetition_ratio=0.0,
            symbol_density=0.0,
            total_chars=0,
            warnings=["empty OCR output — no text extracted from image"],
        )

    # 1. Valid char ratio — printable, non-replacement, non-control chars.
    valid_chars = sum(
        1 for c in text
        if c.isprintable() and c != "�" and unicodedata.category(c) not in ("Cc", "Cf")
    )
    valid_char_ratio = valid_chars / total_chars

    # 2. Meaningful word ratio — tokens with ≥2 alphabetic chars.
    tokens = text.split()
    if tokens:
        meaningful = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 2)
        meaningful_word_ratio = meaningful / len(tokens)
    else:
        meaningful_word_ratio = 0.0

    # 3. Repetition ratio — chars inside repetitive runs of length ≥4.
    repetitive_chars = sum(len(m.group(0)) for m in _REPETITION_RE.finditer(text))
    repetition_ratio = repetitive_chars / total_chars

    # 4. Symbol density — unusual non-alpha, non-space, non-allowed-punct chars.
    symbol_count = sum(
        1 for c in text
        if not c.isalnum() and not c.isspace() and c not in _ALLOWED_PUNCTUATION
    )
    symbol_density = symbol_count / total_chars

    # Composite score (weights tuned empirically for typical scan artefacts).
    score = (
        0.45 * valid_char_ratio
        + 0.35 * meaningful_word_ratio
        - 0.15 * min(repetition_ratio * 2, 1.0)
        - 0.05 * min(symbol_density * 3, 1.0)
    )
    score = max(0.0, min(1.0, round(score, 4)))

    warnings: list[str] = []
    if valid_char_ratio < 0.80:
        warnings.append(
            f"high noise char ratio: {1 - valid_char_ratio:.1%} chars are invalid/replacement"
        )
    if meaningful_word_ratio < 0.50:
        warnings.append(f"low meaningful word ratio: {meaningful_word_ratio:.1%}")
    if repetition_ratio > 0.10:
        warnings.append(f"high char repetition: {repetition_ratio:.1%} of text in repetitive runs")
    if symbol_density > 0.12:
        warnings.append(f"high symbol density: {symbol_density:.1%}")
    if total_chars < 30:
        warnings.append(f"very short OCR output: only {total_chars} chars extracted")
    if score < min_score:
        warnings.append(
            f"overall OCR quality score {score:.2f} is below fail threshold {min_score:.2f}"
        )
    elif score < warn_score:
        warnings.append(
            f"OCR quality score {score:.2f} is below warning threshold {warn_score:.2f} — indexing continues with reduced confidence"
        )

    return OCRQualityReport(
        score=score,
        valid_char_ratio=valid_char_ratio,
        meaningful_word_ratio=meaningful_word_ratio,
        repetition_ratio=repetition_ratio,
        symbol_density=symbol_density,
        total_chars=total_chars,
        warnings=warnings,
    )
