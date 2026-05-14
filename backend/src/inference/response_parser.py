from __future__ import annotations

import html
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
_SAME_CITATION_SENTENCE_RE = re.compile(r"\[(\d+)\]([.!?])(\s+)(?=[^.!?]*\[\1\][.!?])")
_TOC_ENTRY_RE = re.compile(r"^trang\s+\d+\s*[-–]", re.IGNORECASE)


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


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


def _collapse_repeated_sentence_citations(text: str) -> str:
    """Avoid noisy repeated markers like '[1]. ... [1].' within one paragraph."""
    paragraphs = text.split("\n\n")
    return "\n\n".join(_SAME_CITATION_SENTENCE_RE.sub(r"\2\3", paragraph) for paragraph in paragraphs)


class ResponseParser:
    _MIN_CITATION_OVERLAP = 2  # require content-word overlap to suppress spurious citations
    _REFUSAL_RE = re.compile(
        r"(kh[oô]ng\s+t[iì]m\s+th[aấ]y|kh[oô]ng\s+[dđ][uủ]|not\s+enough\s+evidence|cannot\s+answer|can't\s+answer)",
        re.IGNORECASE,
    )

    def format_evidence_for_prompt(self, chunks: list[RetrievedChunk]) -> str:
        return self._format_evidence_xml(chunks)

    @staticmethod
    def _format_evidence_xml(chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            doc_name = chunk.evidence[0].document_name if chunk.evidence else chunk.document_name
            pages = sorted({e.page for e in chunk.evidence}) if chunk.evidence else chunk.source_pages
            page_str = f"trang {pages[0]}" if len(pages) == 1 else f"trang {pages[0]}-{pages[-1]}"
            safe_doc_name = html.escape(doc_name or "", quote=True)
            safe_page_str = html.escape(page_str, quote=True)
            safe_content = html.escape(chunk.content or "")
            lines.append(
                f'<EVIDENCE id="{index}" citation="[{index}]" '
                f'source="{safe_doc_name}" pages="{safe_page_str}">\n'
                f"{safe_content}\n"
                f"</EVIDENCE>"
            )
        return "\n\n".join(lines)

    def citations_from_chunks(self, chunks: list[RetrievedChunk], *, focus_text: str | None = None) -> list[CitationSchema]:
        """One citation per chunk — index N in this list matches [N] in the LLM answer.

        Chunk order must match the order used in format_evidence_for_prompt so that
        [1]…[N] markers in the LLM answer map to citations[0]…[N-1].
        """
        citations: list[CitationSchema] = []
        for i, chunk in enumerate(chunks):
            evs = chunk.evidence
            pages = sorted({e.page for e in evs}) if evs else sorted(set(chunk.source_pages))
            page = pages[0] if pages else None
            # Prefer the most substantive block — filter TOC entries ("Trang N - ..."),
            # slide headers, and very short blocks; fall back to first block if needed.
            substantive_evs = [
                e for e in evs
                if len(e.snippet_original.strip()) >= 40
                and not _TOC_ENTRY_RE.match(e.snippet_original.strip())
            ]
            primary_ev = self._select_primary_evidence(
                substantive_evs if substantive_evs else evs,
                focus_text=focus_text,
            )

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

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {_normalize_token(token) for token in _WORD_RE.findall((text or "").lower())}

    def _select_primary_evidence(self, evidence: list, *, focus_text: str | None) -> object | None:
        if not evidence:
            return None
        focus_tokens = self._token_set(focus_text or "")
        if not focus_tokens:
            return evidence[0]

        def score(ev) -> tuple[int, float, int]:
            ev_tokens = self._token_set(ev.snippet_original)
            overlap = len(focus_tokens & ev_tokens)
            confidence = float(ev.confidence or 0.0)
            return (overlap, confidence, len(ev.snippet_original or ""))

        best = max(evidence, key=score)
        return best if score(best)[0] > 0 else evidence[0]

    def inject_citations(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        """Ensure every sentence has a valid [N] citation.

        - If LLM added no [N] at all: inject a best-matching citation only
          when the sentence has enough lexical overlap with retrieved evidence.
        - If LLM added some [N] but missed sentences: fill only grounded gaps.
        - Preserve invalid citation markers so downstream grounding checks can
          detect and repair them instead of silently laundering them to [1].
        """
        answer = _fix_numbered_lists(answer)
        if not chunks:
            return answer
        if self._REFUSAL_RE.search(answer):
            return answer

        # Token sets per chunk for best-match injection.
        chunk_tokens: list[set[str]] = []
        for chunk in chunks:
            if chunk.evidence:
                combined = " ".join(e.snippet_original for e in chunk.evidence)
            else:
                combined = chunk.content
            chunk_tokens.append(self._token_set(combined))

        _citation_num_re = re.compile(r"\[(\d+)\]")

        def _best_tag(sentence: str) -> str | None:
            sent_tokens = self._token_set(sentence)
            best_idx, best_score = 0, 0
            for idx, ctokens in enumerate(chunk_tokens):
                score = len(sent_tokens & ctokens)
                if score > best_score:
                    best_score, best_idx = score, idx
            if best_score < self._MIN_CITATION_OVERLAP:
                return None
            return f"[{best_idx + 1}]"

        def _process_sentence(text: str) -> str:
            stripped = text.strip()
            if not stripped:
                return text
            has_citation = bool(_citation_num_re.search(stripped))
            if not has_citation:
                tag = _best_tag(stripped)
                if tag is None:
                    return text
                if stripped[-1] in ".!?":
                    text = text.rstrip()[:-1] + tag + text.rstrip()[-1]
                else:
                    text = text.rstrip() + tag
            return text

        # Split answer into paragraphs, then sentences.
        paragraphs = answer.split("\n\n")
        result_paragraphs: list[str] = []

        for para in paragraphs:
            if not para.strip():
                result_paragraphs.append(para)
                continue
            parts = re.split(r"(?<=[.!?])(\s+)", para)
            out_parts: list[str] = []
            i = 0
            while i < len(parts):
                chunk_text = parts[i]
                sep = parts[i + 1] if i + 1 < len(parts) else ""
                out_parts.append(_process_sentence(chunk_text) + sep)
                i += 2
            result_paragraphs.append("".join(out_parts))

        result = "\n\n".join(result_paragraphs)
        return _collapse_repeated_sentence_citations(result)

    @staticmethod
    def invalid_citation_numbers(answer: str, citation_count: int) -> list[int]:
        markers = [int(match.group(1)) for match in re.finditer(r"\[(\d+)\]", answer or "")]
        return sorted({marker for marker in markers if marker < 1 or marker > citation_count})
