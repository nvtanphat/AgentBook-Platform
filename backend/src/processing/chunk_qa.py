"""Chunk quality assurance metrics for post-chunking validation.

Checks each TextChunk for common chunking pathologies and emits structured
warnings per material. Does not raise — callers decide whether to block or log.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.processing.types import TextChunk

logger = logging.getLogger(__name__)

_HEADING_ONLY_RE = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)
_NOISE_CHARS_RE = re.compile(r"[^\x20-\x7EàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀ-ɏḀ-ỿ]")


@dataclass
class ChunkQAReport:
    material_id: str
    total_chunks: int
    too_short: list[int] = field(default_factory=list)       # chunk indices
    too_long: list[int] = field(default_factory=list)
    heading_only: list[int] = field(default_factory=list)
    duplicate: list[int] = field(default_factory=list)
    noisy_ocr: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.too_short or self.too_long or self.heading_only or self.duplicate or self.noisy_ocr)

    def summary(self) -> str:
        parts = []
        if self.too_short:
            parts.append(f"{len(self.too_short)} too-short")
        if self.too_long:
            parts.append(f"{len(self.too_long)} too-long")
        if self.heading_only:
            parts.append(f"{len(self.heading_only)} heading-only")
        if self.duplicate:
            parts.append(f"{len(self.duplicate)} duplicate")
        if self.noisy_ocr:
            parts.append(f"{len(self.noisy_ocr)} noisy-ocr")
        return ", ".join(parts) if parts else "ok"


def run_chunk_qa(
    chunks: list[TextChunk],
    *,
    material_id: str,
    min_tokens: int = 10,
    max_tokens: int = 600,
    noise_char_ratio_threshold: float = 0.15,
) -> ChunkQAReport:
    """Run quality checks on a list of chunks and return a structured report.

    Warnings are emitted via structured logging — callers receive the report
    to decide whether to block or continue.
    """
    report = ChunkQAReport(material_id=material_id, total_chunks=len(chunks))
    if not chunks:
        report.warnings.append("no chunks produced — possible parsing failure")
        logger.warning(
            "Chunk QA: no chunks",
            extra={"material_id": material_id, "stage": "chunk_qa"},
        )
        return report

    seen_fingerprints: dict[str, int] = {}

    for index, chunk in enumerate(chunks):
        content = chunk.content.strip()
        token_count = chunk.token_count

        # Too short
        if token_count < min_tokens:
            report.too_short.append(index)

        # Too long
        if token_count > max_tokens:
            report.too_long.append(index)

        # Heading-only: all non-empty lines look like markdown headings
        non_empty_lines = [ln for ln in content.splitlines() if ln.strip()]
        if non_empty_lines and all(_HEADING_ONLY_RE.match(ln.strip()) for ln in non_empty_lines):
            report.heading_only.append(index)

        # Duplicate: compare by full content to avoid false positives from shared
        # preamble text (e.g. spreadsheet chunks that share the same sheet summary header).
        fingerprint = content
        if fingerprint in seen_fingerprints:
            report.duplicate.append(index)
        else:
            seen_fingerprints[fingerprint] = index

        # Noisy OCR: high ratio of unusual characters
        if content:
            noise_chars = len(_NOISE_CHARS_RE.findall(content))
            if noise_chars / len(content) > noise_char_ratio_threshold:
                report.noisy_ocr.append(index)

    if report.has_issues:
        report.warnings = _build_warnings(report, chunks)
        logger.warning(
            "Chunk QA issues detected",
            extra={
                "material_id": material_id,
                "stage": "chunk_qa",
                "summary": report.summary(),
                "too_short": len(report.too_short),
                "too_long": len(report.too_long),
                "heading_only": len(report.heading_only),
                "duplicate": len(report.duplicate),
                "noisy_ocr": len(report.noisy_ocr),
                "total_chunks": report.total_chunks,
            },
        )
    else:
        logger.info(
            "Chunk QA passed",
            extra={"material_id": material_id, "stage": "chunk_qa", "total_chunks": report.total_chunks},
        )

    return report


def _build_warnings(report: ChunkQAReport, chunks: list[TextChunk]) -> list[str]:
    warnings: list[str] = []
    if report.too_short:
        examples = [chunks[i].token_count for i in report.too_short[:3]]
        warnings.append(f"{len(report.too_short)} chunks below min token threshold (e.g. {examples} tokens)")
    if report.too_long:
        examples = [chunks[i].token_count for i in report.too_long[:3]]
        warnings.append(f"{len(report.too_long)} chunks exceed max token threshold (e.g. {examples} tokens)")
    if report.heading_only:
        warnings.append(f"{len(report.heading_only)} heading-only chunks (no body content)")
    if report.duplicate:
        warnings.append(f"{len(report.duplicate)} duplicate chunks by content fingerprint")
    if report.noisy_ocr:
        warnings.append(f"{len(report.noisy_ocr)} chunks with high noise-char ratio (likely OCR artefacts)")
    return warnings
