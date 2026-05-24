"""Text cleaner tool — strips obviously irrelevant or boilerplate sentences
from retrieved chunks while keeping the evidence-trace fields intact.

Implementation is deterministic (rule-based) — fast, safe, and never
hallucinates. We only edit the `content` field; `evidence` blocks and ids
are preserved verbatim.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Heuristics: lines that are pure page headers / footers / page numbers /
# copyright noise. Matching is conservative — false negatives are safer than
# losing genuine evidence.
_BOILERPLATE_RE = re.compile(
    r"^(?:\s*(?:page\s+\d+(?:\s+of\s+\d+)?|trang\s+\d+|\d+\s*/\s*\d+|©.+|copyright.+|all rights reserved\.?))\s*$",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class TextCleanerTool(BaseTool):
    name = "text_cleaner"
    description = (
        "Drops obvious header/footer/boilerplate lines from chunk content "
        "without mutating evidence-trace fields. Returns chunk copies."
    )

    async def _run(
        self,
        *,
        chunks: list["RetrievedChunk"],
        query: str | None = None,
    ) -> list["RetrievedChunk"]:
        if not chunks:
            return []
        cleaned: list[RetrievedChunk] = []
        for chunk in chunks:
            cleaned.append(self._clean_one(chunk))
        return cleaned

    @staticmethod
    def _clean_one(chunk: "RetrievedChunk") -> "RetrievedChunk":
        original = chunk.content or ""
        if not original.strip():
            return chunk
        lines = [line for line in original.splitlines() if not _BOILERPLATE_RE.match(line.strip())]
        new_content = "\n".join(lines).strip() or original
        if new_content == original:
            return chunk
        return chunk.model_copy(update={"content": new_content})
