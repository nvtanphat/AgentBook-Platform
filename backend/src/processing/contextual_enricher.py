"""
Contextual Retrieval enrichment (Anthropic, 2024).

For each chunk, an LLM generates a short situating context that is prepended to
the chunk text before embedding. This gives embeddings awareness of where in the
document a chunk comes from without changing the stored citation content.

Reference: https://www.anthropic.com/news/contextual-retrieval
Reported improvement: ~49% reduction in retrieval failures.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.processing.types import EvidenceMap, TextChunk

if TYPE_CHECKING:
    from src.core.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_CONTEXT_PROMPT = """\
<document>
{doc_context}
</document>

Here is a chunk from that document:
<chunk>
{chunk_content}
</chunk>

Write a short (1-3 sentence) context that situates this chunk within the document \
for the purpose of improving search retrieval. Only output the context sentences, \
nothing else."""

_DOC_CONTEXT_MAX_CHARS = 2000
_MIN_CHUNK_CHARS = 60
_DEFAULT_CONCURRENCY = 4


class ContextualEnricher:
    """
    Enriches TextChunks with LLM-generated situating context.

    enriched_chunk.contextualized_content = "<context>\n\n<original content>"
    enriched_chunk.content stays unchanged (used for citations / display).
    """

    def __init__(self, llm: "BaseLLM", *, concurrency: int = _DEFAULT_CONCURRENCY) -> None:
        self.llm = llm
        self.concurrency = concurrency

    async def enrich(
        self,
        chunks: list[TextChunk],
        evidence_map: EvidenceMap,
    ) -> list[TextChunk]:
        if not chunks:
            return chunks

        doc_context = self._build_doc_context(evidence_map)
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [self._enrich_one(chunk, doc_context, sem) for chunk in chunks]
        return list(await asyncio.gather(*tasks))

    async def _enrich_one(
        self,
        chunk: TextChunk,
        doc_context: str,
        sem: asyncio.Semaphore,
    ) -> TextChunk:
        if len(chunk.content) < _MIN_CHUNK_CHARS:
            return chunk
        async with sem:
            try:
                prompt = _CONTEXT_PROMPT.format(
                    doc_context=doc_context,
                    chunk_content=chunk.content[:3000],
                )
                context_text = (await self.llm.generate(prompt=prompt)).strip()
                if context_text:
                    return chunk.model_copy(
                        update={"contextualized_content": f"{context_text}\n\n{chunk.content}"}
                    )
            except Exception:
                logger.warning(
                    "Contextual enrichment failed for chunk in %s — using raw content",
                    chunk.document_name,
                    exc_info=True,
                )
        return chunk

    @staticmethod
    def _build_doc_context(evidence_map: EvidenceMap) -> str:
        """
        Build a compact document summary from the first blocks of the evidence map.
        Headings are included in full; other blocks are truncated.
        """
        parts: list[str] = [f"Document: {evidence_map.document_name}"]
        budget = _DOC_CONTEXT_MAX_CHARS - len(parts[0])

        for block in evidence_map.blocks:
            if budget <= 0:
                break
            text = block.snippet_original.strip()
            if not text:
                continue
            if block.block_type == "heading":
                snippet = text[:200]
            else:
                snippet = text[:100]
            parts.append(snippet)
            budget -= len(snippet)

        return "\n".join(parts)
