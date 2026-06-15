"""Rule-based sentence decomposer for better relation extraction (CoDe-KG pattern, ACL 2025).

Splits complex compound/relative-clause sentences into simple clauses, improving
pairwise entity relation extraction from dense academic text.

Zero LLM cost — pure regex rules. Works for EN and VI.
"""
from __future__ import annotations

import re


# ── EN: coordinating / subordinating conjunctions that split independent clauses ──
_EN_SPLIT = re.compile(
    r",?\s+(?:and|but|while|whereas|although|however|therefore|thus|hence|moreover|furthermore|additionally)\s+",
    re.IGNORECASE,
)
# EN relative/appositive clauses
_EN_RELATIVE = re.compile(r",\s+(?:which|who|that)\s+", re.IGNORECASE)

# ── VI: coordinating conjunctions ──
_VI_SPLIT = re.compile(
    r",?\s+(?:và|nhưng|hoặc|hay|trong khi|mặc dù|tuy nhiên|do đó|vì vậy|vì thế|ngoài ra|hơn nữa|bên cạnh đó)\s+",
    re.IGNORECASE | re.UNICODE,
)
_VI_RELATIVE = re.compile(r",\s+(?:mà|trong đó|điều này|điều đó)\s+", re.IGNORECASE | re.UNICODE)

_MIN_CLAUSE_CHARS = 20  # discard fragments shorter than this


def decompose(text: str, language: str = "en") -> list[str]:
    """Split a text into simple clauses on coordinating conjunctions.

    Returns `[text]` unchanged when no split points qualify.
    """
    text = text.strip()
    if not text:
        return []

    patterns = (_VI_SPLIT, _VI_RELATIVE) if language == "vi" else (_EN_SPLIT, _EN_RELATIVE)

    split_pos: list[int] = sorted({m.start() for p in patterns for m in p.finditer(text)})
    if not split_pos:
        return [text]

    clauses: list[str] = []
    prev = 0
    for pos in split_pos:
        clause = text[prev:pos].strip().rstrip(",").strip()
        if len(clause) >= _MIN_CLAUSE_CHARS:
            clauses.append(clause)
        prev = pos
    last = text[prev:].strip()
    if len(last) >= _MIN_CLAUSE_CHARS:
        clauses.append(last)

    return clauses if len(clauses) > 1 else [text]


def decompose_blocks(blocks: list, max_clauses_per_block: int = 4) -> list:
    """Expand a list of EvidenceBlocks by decomposing each into clause-level snippets.

    Each original block may expand to 1-N lightweight copies with shorter
    `snippet_original`. Downstream code sees more, smaller passages which
    surface pairwise entity co-mentions more reliably.
    """
    result = []
    for block in blocks:
        text = block.snippet_original or ""
        lang = (getattr(block, "source_language", None) or "en").lower()[:2]
        clauses = decompose(text, language=lang)[:max_clauses_per_block]
        if len(clauses) <= 1:
            result.append(block)
        else:
            for clause in clauses:
                try:
                    result.append(block.model_copy(update={"snippet_original": clause}))
                except Exception:
                    result.append(block)
    return result
