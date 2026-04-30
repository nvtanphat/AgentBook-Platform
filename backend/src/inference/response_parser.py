from __future__ import annotations

import math
import re

from src.rag.types import RetrievedChunk
from src.schemas.evidence import BoundingBoxSchema, CitationSchema

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


class ResponseParser:
    _MIN_CITATION_OVERLAP = 1
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
        """One citation per chunk — index N in this list matches [N] in the LLM answer."""
        citations: list[CitationSchema] = []
        for i, chunk in enumerate(chunks):
            first_ev = chunk.evidence[0] if chunk.evidence else None
            pages = sorted({e.page for e in chunk.evidence}) if chunk.evidence else sorted(set(chunk.source_pages))
            page = pages[0] if pages else None
            citations.append(
                CitationSchema(
                    doc_id=first_ev.material_id if first_ev else chunk.material_id,
                    doc_name=first_ev.document_name if first_ev else chunk.document_name,
                    page=page,
                    pages=pages,
                    block_id=first_ev.block_id if first_ev else None,
                    block_type=first_ev.block_type if first_ev else None,
                    snippet_original=chunk.content,
                    bbox=BoundingBoxSchema.model_validate(first_ev.bbox.model_dump()) if first_ev and first_ev.bbox else None,
                    role="primary" if i == 0 else "supporting",
                    source_language=chunk.language,
                    confidence=_citation_confidence(chunk),
                )
            )
        return citations

    def inject_citations(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Append best-matching [N] citation to each sentence if LLM didn't add any."""
        if not chunks:
            return answer
        if self._REFUSAL_RE.search(answer):
            return answer
        # If model already placed at least one [N] marker, trust it
        if re.search(r"\[\d+\]", answer):
            return answer

        # Token sets per chunk for overlap scoring
        chunk_tokens: list[set[str]] = [
            set(_WORD_RE.findall(chunk.content.lower())) for chunk in chunks
        ]

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
