from __future__ import annotations

import math
import re

from src.rag.types import RetrievedChunk
from src.schemas.evidence import BoundingBoxSchema, CitationSchema, EvidenceBlockSchema

# Vietnamese + Latin word pattern covering all Vietnamese diacritics
_WORD_RE = re.compile(r"[\wÀ-ɏḀ-ỿ]{3,}", re.UNICODE)


def _citation_confidence(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        score = chunk.rerank_score
        if score >= 0:
            value = 1.0 / (1.0 + math.exp(-score))
        else:
            exp_score = math.exp(score)
            value = exp_score / (1.0 + exp_score)
    elif chunk.graph_score is not None:
        value = chunk.graph_score
    else:
        value = chunk.fused_score if chunk.fused_score else 0.5
    return min(1.0, max(0.0, float(value)))


_ORDERED_LIST_RE = re.compile(r"^(\s*)(\d+)\.\s", re.MULTILINE)


def _fix_numbered_lists(text: str) -> str:
    """Renumber ordered list items when the LLM repeats '1.' for every item."""
    lines = text.split("\n")
    result: list[str] = []
    counter: dict[str, int] = {}  # indent -> current counter
    prev_indent: str | None = None

    for line in lines:
        m = _ORDERED_LIST_RE.match(line)
        if m:
            indent = m.group(1)
            # Reset child counters when indent changes up
            if prev_indent is not None and len(indent) < len(prev_indent):
                keys_to_remove = [k for k in counter if len(k) > len(indent)]
                for k in keys_to_remove:
                    del counter[k]
            counter[indent] = counter.get(indent, 0) + 1
            line = indent + str(counter[indent]) + ". " + line[m.end():]
            prev_indent = indent
        else:
            if not line.strip():
                # blank line resets list context
                counter.clear()
                prev_indent = None
        result.append(line)
    return "\n".join(result)


class ResponseParser:
    _MIN_CITATION_OVERLAP = 3  # require ≥3 content-word overlaps to suppress spurious citations
    _REFUSAL_RE = re.compile(
        r"(kh[oô]ng\s+t[iì]m\s+th[aấ]y|kh[oô]ng\s+[dđ][uủ]|not\s+enough\s+evidence|cannot\s+answer|can't\s+answer)",
        re.IGNORECASE,
    )

    def format_evidence_for_prompt(self, chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            doc_name = chunk.evidence[0].document_name if chunk.evidence else chunk.document_name
            pages = sorted({e.page for e in chunk.evidence}) if chunk.evidence else chunk.source_pages
            page_str = f"trang {pages[0]}" if len(pages) == 1 else f"trang {pages[0]}-{pages[-1]}"
            lines.append(
                f"[{index}] Nguồn: {doc_name} ({page_str})\n"
                f"{chunk.content}"
            )
        return "\n\n".join(lines)

    def citations_from_chunks(self, chunks: list[RetrievedChunk]) -> list[CitationSchema]:
        """One citation per chunk — index N in this list matches [N] in the LLM answer.

        Chunk order must match the order used in format_evidence_for_prompt so that
        [1]…[N] markers in the LLM answer map to citations[0]…[N-1].
        """
        citations: list[CitationSchema] = []
        for i, chunk in enumerate(chunks):
            evs = chunk.evidence
            pages = sorted({e.page for e in evs}) if evs else sorted(set(chunk.source_pages))
            page = pages[0] if pages else None
            primary_ev = evs[0] if evs else None

            # All contributing blocks exposed for downstream spatial rendering
            evidence_blocks = [
                EvidenceBlockSchema(
                    block_id=e.block_id,
                    block_type=e.block_type,
                    page=e.page,
                    snippet_original=e.snippet_original,
                    source_language=e.source_language,
                    bbox=BoundingBoxSchema.model_validate(e.bbox.model_dump()) if e.bbox else None,
                    confidence=e.confidence,
                    material_id=e.material_id,
                    doc_name=e.document_name,
                )
                for e in evs
            ]

            citations.append(
                CitationSchema(
                    doc_id=primary_ev.material_id if primary_ev else chunk.material_id,
                    doc_name=primary_ev.document_name if primary_ev else chunk.document_name,
                    page=page,
                    pages=pages,
                    block_id=primary_ev.block_id if primary_ev else None,
                    block_type=primary_ev.block_type if primary_ev else None,
                    # Block-level snippet is more precise than full chunk content (500-token cap as fallback)
                    snippet_original=primary_ev.snippet_original if primary_ev else chunk.content[:500],
                    bbox=BoundingBoxSchema.model_validate(primary_ev.bbox.model_dump()) if primary_ev and primary_ev.bbox else None,
                    role="primary" if i == 0 else "supporting",
                    source_language=chunk.language,
                    confidence=_citation_confidence(chunk),
                    evidence_blocks=evidence_blocks,
                )
            )
        return citations

    def inject_citations(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Append best-matching [N] citation to each sentence if LLM didn't add any."""
        answer = _fix_numbered_lists(answer)
        if not chunks:
            return answer
        if self._REFUSAL_RE.search(answer):
            return answer
        # If model already placed at least one [N] marker, trust it
        if re.search(r"\[\d+\]", answer):
            return answer

        # Token sets per chunk — use evidence block snippets when available because they are
        # block-level text and far more specific than the full multi-block chunk content.
        chunk_tokens: list[set[str]] = []
        for chunk in chunks:
            if chunk.evidence:
                combined = " ".join(e.snippet_original for e in chunk.evidence)
            else:
                combined = chunk.content
            chunk_tokens.append(set(_WORD_RE.findall(combined.lower())))

        # Split answer into paragraphs first (preserve paragraph breaks)
        paragraphs = answer.split("\n\n")
        result_paragraphs: list[str] = []

        for para in paragraphs:
            if not para.strip():
                result_paragraphs.append(para)
                continue
            # Split paragraph into sentences on . ! ? followed by space or end
            parts = re.split(r"(?<=[.!?])(\s+)", para)
            # parts alternates: [sentence, separator, sentence, separator, ...]
            out_parts: list[str] = []
            i = 0
            while i < len(parts):
                chunk_text = parts[i]
                sep = parts[i + 1] if i + 1 < len(parts) else ""
                stripped = chunk_text.strip()
                if stripped:
                    sent_tokens = set(_WORD_RE.findall(stripped.lower()))
                    best_idx, best_score = 0, 0
                    for idx, ctokens in enumerate(chunk_tokens):
                        score = len(sent_tokens & ctokens)
                        if score > best_score:
                            best_score, best_idx = score, idx
                    if best_score >= self._MIN_CITATION_OVERLAP:
                        tag = f"[{best_idx + 1}]"
                        # Insert tag before terminal punctuation or at end
                        if stripped and stripped[-1] in ".!?":
                            chunk_text = chunk_text.rstrip()[:-1] + tag + chunk_text.rstrip()[-1]
                        else:
                            chunk_text = chunk_text.rstrip() + tag
                out_parts.append(chunk_text + sep)
                i += 2

            result_paragraphs.append("".join(out_parts))

        return "\n\n".join(result_paragraphs)
