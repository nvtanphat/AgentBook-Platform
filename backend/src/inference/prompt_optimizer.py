from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimize prompts to reduce token usage while maintaining quality.

    Techniques:
    1. Remove redundant instructions
    2. Use concise formatting
    3. Truncate long evidence
    4. Smart context selection
    """

    @staticmethod
    def optimize_evidence(chunks: List[dict], max_chunks: int = 5, max_tokens_per_chunk: int = 400) -> str:
        """
        Format evidence concisely.

        Original format (verbose):
        Document: lecture.pdf, Page: 5, Block: blk-123
        Content: [long text...]

        Optimized format (concise):
        [1] lecture.pdf p5: [text...]
        """
        evidence_parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            content = chunk.get("content", "")

            # Truncate if too long
            if len(content.split()) > max_tokens_per_chunk:
                words = content.split()[:max_tokens_per_chunk]
                content = " ".join(words) + "..."

            # Concise format
            doc_name = chunk.get("document_name", "unknown")
            page = chunk.get("page", "?")
            evidence_parts.append(f"[{i}] {doc_name} p{page}: {content}")

        return "\n\n".join(evidence_parts)

    @staticmethod
    def build_concise_prompt(query: str, evidence: str, answer_language: str = "vi") -> str:
        """
        Build concise prompt (saves ~30% tokens vs verbose version).

        Original: ~150 tokens template
        Optimized: ~100 tokens template
        """
        # Concise template (no fluff)
        template = f"""Evidence:
{evidence}

Question: {query}

Answer in {answer_language}. Cite sources as [N].
"""
        return template

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Quick token estimation (words × 1.3 for multilingual)."""
        return int(len(text.split()) * 1.3)

    @staticmethod
    def truncate_to_budget(text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        estimated = PromptOptimizer.estimate_tokens(text)
        if estimated <= max_tokens:
            return text

        # Truncate proportionally
        ratio = max_tokens / estimated
        words = text.split()
        keep_words = int(len(words) * ratio * 0.95)  # 5% safety margin
        return " ".join(words[:keep_words]) + "..."


class ContextWindowManager:
    """
    Adaptive context window management.

    Dynamically adjust number of chunks based on:
    1. Available context window
    2. Chunk sizes
    3. Query complexity
    """

    def __init__(self, max_context_tokens: int = 6000):
        self.max_context_tokens = max_context_tokens
        self.template_tokens = 150  # Estimated prompt template size

    def select_chunks(self, query: str, chunks: List[dict], max_chunks: int = 5) -> List[dict]:
        """
        Select chunks that fit in context window.

        Priority: rerank_score (descending)
        """
        query_tokens = PromptOptimizer.estimate_tokens(query)
        available_tokens = self.max_context_tokens - self.template_tokens - query_tokens

        selected = []
        used_tokens = 0

        # Sort by score
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("rerank_score") or c.get("fused_score") or 0.0,
            reverse=True
        )

        for chunk in sorted_chunks[:max_chunks]:
            chunk_tokens = chunk.get("token_count") or PromptOptimizer.estimate_tokens(chunk.get("content", ""))

            if used_tokens + chunk_tokens > available_tokens:
                logger.warning(
                    "Context window full, stopping at %d chunks",
                    len(selected),
                    extra={"used_tokens": used_tokens, "available": available_tokens}
                )
                break

            selected.append(chunk)
            used_tokens += chunk_tokens

        logger.info(
            "Selected %d chunks (%d tokens)",
            len(selected),
            used_tokens,
            extra={"available": available_tokens, "utilization": used_tokens / available_tokens}
        )

        return selected
