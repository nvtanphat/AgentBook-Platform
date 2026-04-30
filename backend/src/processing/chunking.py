from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.core.config import Settings
from src.core.tokenizer import count_tokens as _tokenizer_count_tokens
from src.processing.types import BlockType, EvidenceBlock, EvidenceMap, TextChunk

if TYPE_CHECKING:
    from src.rag.embedder import BGEM3Embedder

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")
_TABLE_BLOCK_TYPE = BlockType.TABLE.value


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class LayoutAwareChunker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _count_tokens(self, text: str) -> int:
        if self.settings.testing:
            return len(text.split()) if text else 0
        return _tokenizer_count_tokens(text, self.settings)

    def build_chunks(self, evidence_map: EvidenceMap) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current: list[EvidenceBlock] = []
        current_tokens = 0

        for source_block in evidence_map.blocks:
            for block in self._split_oversized_block(source_block):
                current, current_tokens = self._append_block(
                    evidence_map=evidence_map,
                    chunks=chunks,
                    current=current,
                    current_tokens=current_tokens,
                    block=block,
                )

        if current:
            chunks.append(self._make_chunk(evidence_map, current))

        return chunks

    def _append_block(
        self,
        *,
        evidence_map: EvidenceMap,
        chunks: list[TextChunk],
        current: list[EvidenceBlock],
        current_tokens: int,
        block: EvidenceBlock,
    ) -> tuple[list[EvidenceBlock], int]:
        block_tokens = self._count_tokens(block.snippet_original)
        starts_new_section = self._starts_new_section(block, current)
        table_boundary = self._is_table_boundary(block, current)
        exceeds_token_budget = current and current_tokens + block_tokens > self.settings.chunk_target_token_count
        exceeds_block_budget = self._exceeds_block_budget(current, current_tokens)

        if starts_new_section or exceeds_token_budget or exceeds_block_budget or table_boundary:
            chunks.append(self._make_chunk(evidence_map, current))
            reset = starts_new_section or exceeds_block_budget or table_boundary or self._is_heading_only(current)
            current = [] if reset else self._overlap_blocks(current)
            current_tokens = sum(self._count_tokens(item.snippet_original) for item in current)
            if current_tokens + block_tokens > self.settings.chunk_target_token_count:
                current = []
                current_tokens = 0

        current.append(block)
        current_tokens += block_tokens
        return current, current_tokens

    @staticmethod
    def _is_table_boundary(block: EvidenceBlock, current: list[EvidenceBlock]) -> bool:
        if not current:
            return False
        last_is_table = current[-1].block_type == _TABLE_BLOCK_TYPE
        incoming_is_table = block.block_type == _TABLE_BLOCK_TYPE
        return last_is_table != incoming_is_table

    def _make_chunk(self, evidence_map: EvidenceMap, blocks: list[EvidenceBlock]) -> TextChunk:
        content = "\n".join(block.snippet_original for block in blocks).strip()
        pages = sorted({block.page for block in blocks})
        block_ids = [block.block_id for block in blocks]
        bboxes = [block.bbox for block in blocks if block.bbox is not None]
        languages = sorted({block.source_language for block in blocks if block.source_language != "unknown"})
        modalities = {block.block_type for block in blocks}
        modality = "mixed" if len(modalities) > 1 else next(iter(modalities), "text")
        return TextChunk(
            owner_id=evidence_map.owner_id,
            collection_id=evidence_map.collection_id,
            material_id=evidence_map.material_id,
            document_name=evidence_map.document_name,
            content=content,
            language=languages[0] if len(languages) == 1 else ("mixed" if languages else "unknown"),
            modality=modality,
            source_block_ids=block_ids,
            source_pages=pages,
            bboxes=bboxes,
            token_count=self._count_tokens(content),
            chunk_strategy="layout_heading_parent_child",
            chunker_version=self.settings.chunk_version,
            parser_version=self.settings.parse_version,
            embedding_model=self.settings.embedding_model,
            embedding_version=self.settings.embedding_version,
            index_version=self.settings.index_version,
            evidence=blocks,
        )

    def _overlap_blocks(self, blocks: list[EvidenceBlock]) -> list[EvidenceBlock]:
        if self.settings.chunk_overlap_token_count <= 0:
            return []
        overlap: list[EvidenceBlock] = []
        token_count = 0
        for block in reversed(blocks):
            block_tokens = self._count_tokens(block.snippet_original)
            if token_count + block_tokens > self.settings.chunk_overlap_token_count:
                break
            overlap.insert(0, block)
            token_count += block_tokens
        return overlap

    @staticmethod
    def _starts_new_section(block: EvidenceBlock, current: list[EvidenceBlock]) -> bool:
        if block.block_type != "heading" or not current:
            return False
        return any(item.block_type != "heading" for item in current)

    def _exceeds_block_budget(self, current: list[EvidenceBlock], current_tokens: int) -> bool:
        if len(current) < self.settings.chunk_max_blocks_per_chunk:
            return False
        if self._is_heading_only(current):
            return False
        min_reasonable_tokens = max(1, self.settings.chunk_target_token_count // 2)
        return current_tokens >= min_reasonable_tokens

    @staticmethod
    def _is_heading_only(blocks: list[EvidenceBlock]) -> bool:
        return bool(blocks) and all(block.block_type == "heading" for block in blocks)

    def _split_oversized_block(self, block: EvidenceBlock) -> list[EvidenceBlock]:
        if self._count_tokens(block.snippet_original) <= self.settings.chunk_target_token_count:
            return [block]

        if block.block_type == _TABLE_BLOCK_TYPE:
            table_parts = self._split_markdown_table(block.snippet_original)
            if table_parts:
                return self._wrap_split_parts(block, table_parts)

        sentences = _SENTENCE_SPLIT.split(block.snippet_original)
        if len(sentences) <= 1:
            sentences = block.snippet_original.split()

        part_texts: list[str] = []
        current_units: list[str] = []
        current_tokens = 0

        for unit in sentences:
            unit_tokens = self._count_tokens(unit)
            if current_tokens + unit_tokens > self.settings.chunk_target_token_count and current_units:
                part_texts.append(" ".join(current_units))
                current_units = []
                current_tokens = 0
            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            part_texts.append(" ".join(current_units))

        if not part_texts:
            return [block]

        return self._wrap_split_parts(block, part_texts)

    def _wrap_split_parts(self, block: EvidenceBlock, part_texts: list[str]) -> list[EvidenceBlock]:
        part_count = len(part_texts)
        return [
            block.model_copy(
                update={
                    "snippet_original": part_text,
                    "metadata": {
                        **block.metadata,
                        "split_part_index": idx,
                        "split_part_count": part_count,
                    },
                }
            )
            for idx, part_text in enumerate(part_texts)
        ]

    def _split_markdown_table(self, content: str) -> list[str]:
        lines = content.splitlines()
        if len(lines) < 3:
            return []
        header_line = lines[0]
        separator_line = lines[1]
        if not header_line.lstrip().startswith("|") or "---" not in separator_line:
            return []
        body_lines = [line for line in lines[2:] if line.strip()]
        if not body_lines:
            return []
        header_block = f"{header_line}\n{separator_line}"
        header_tokens = self._count_tokens(header_block)
        budget = max(1, self.settings.chunk_target_token_count - header_tokens)
        parts: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0
        for row_line in body_lines:
            row_tokens = self._count_tokens(row_line)
            if current_lines and current_tokens + row_tokens > budget:
                parts.append("\n".join([header_line, separator_line, *current_lines]))
                current_lines = []
                current_tokens = 0
            current_lines.append(row_line)
            current_tokens += row_tokens
        if current_lines:
            parts.append("\n".join([header_line, separator_line, *current_lines]))
        return parts if len(parts) > 1 else []


class SemanticChunker:
    """
    Semantic chunking via cosine-distance breakpoint detection (Kamradt 2023).

    Algorithm:
    1. Embed every block with BGE-M3.
    2. Compute cosine distance between each adjacent pair of block embeddings.
    3. Split wherever distance exceeds the adaptive per-document percentile threshold.
    4. Always honour layout hard-breaks: heading after body → mandatory split;
       table ↔ non-table transitions → mandatory split.
    5. Token-budget cap identical to LayoutAwareChunker; overlap carried between
       non-mandatory splits.

    Falls back to LayoutAwareChunker when no embedder is supplied (e.g. testing).
    """

    _STRATEGY = "semantic_breakpoint_v1"

    def __init__(self, settings: Settings, embedder: "BGEM3Embedder | None" = None) -> None:
        self.settings = settings
        self.embedder = embedder
        self._layout = LayoutAwareChunker(settings)

    def build_chunks(self, evidence_map: EvidenceMap) -> list[TextChunk]:
        blocks = list(evidence_map.blocks)
        if not blocks:
            return []
        if self.embedder is None or len(blocks) < 2:
            logger.info("SemanticChunker: no embedder or too few blocks — falling back to layout chunker")
            return self._layout.build_chunks(evidence_map)

        # 1. Pre-split oversized blocks (reuse layout logic)
        expanded: list[EvidenceBlock] = []
        for block in blocks:
            expanded.extend(self._layout._split_oversized_block(block))
        blocks = expanded

        # 2. Embed all blocks in one batch
        texts = [b.snippet_original for b in blocks]
        try:
            embeddings = self.embedder.encode(texts)
        except Exception:
            logger.exception("SemanticChunker: embedding failed — falling back to layout chunker")
            return self._layout.build_chunks(evidence_map)

        dense_vecs = [e.dense for e in embeddings]

        # 3. Compute cosine distances between adjacent blocks
        distances: list[float] = [
            1.0 - _cosine_similarity(dense_vecs[i], dense_vecs[i + 1])
            for i in range(len(dense_vecs) - 1)
        ]

        # 4. Adaptive threshold: p-th percentile of this document's distances
        threshold = _percentile(distances, self.settings.semantic_chunk_breakpoint_percentile)

        # 5. Determine split points (0-indexed: split *before* block at that index)
        split_before: set[int] = set()
        for i, dist in enumerate(distances):
            if dist >= threshold:
                split_before.add(i + 1)

        # Layout hard-breaks always override
        for i in range(1, len(blocks)):
            prev, curr = blocks[i - 1], blocks[i]
            # heading after body content → new section
            if curr.block_type == "heading" and prev.block_type != "heading":
                split_before.add(i)
            # table ↔ non-table
            if (curr.block_type == _TABLE_BLOCK_TYPE) != (prev.block_type == _TABLE_BLOCK_TYPE):
                split_before.add(i)

        logger.info(
            "SemanticChunker: %d blocks → %d semantic splits + %d layout hard-breaks (threshold=%.4f)",
            len(blocks),
            sum(1 for i, d in enumerate(distances) if d >= threshold),
            len(split_before),
            threshold,
        )

        # 6. Build chunks honouring token budget
        chunks: list[TextChunk] = []
        current: list[EvidenceBlock] = []
        current_tokens = 0

        for i, block in enumerate(blocks):
            block_tokens = self._layout._count_tokens(block.snippet_original)
            hard_break = i in split_before
            token_overflow = bool(current) and current_tokens + block_tokens > self.settings.chunk_target_token_count

            if (hard_break or token_overflow) and current:
                chunks.append(self._make_chunk(evidence_map, current))
                if hard_break:
                    current = []
                    current_tokens = 0
                else:
                    current = self._layout._overlap_blocks(current)
                    current_tokens = sum(self._layout._count_tokens(b.snippet_original) for b in current)
                    if current_tokens + block_tokens > self.settings.chunk_target_token_count:
                        current = []
                        current_tokens = 0

            current.append(block)
            current_tokens += block_tokens

        if current:
            chunks.append(self._make_chunk(evidence_map, current))

        return chunks

    def _make_chunk(self, evidence_map: EvidenceMap, blocks: list[EvidenceBlock]) -> TextChunk:
        chunk = self._layout._make_chunk(evidence_map, blocks)
        return chunk.model_copy(update={"chunk_strategy": self._STRATEGY})


def build_chunker(settings: Settings, embedder: "BGEM3Embedder | None" = None) -> LayoutAwareChunker | SemanticChunker:
    """Return the configured chunker, wiring in the embedder for semantic mode."""
    if settings.chunk_strategy == "semantic":
        return SemanticChunker(settings, embedder=embedder)
    return LayoutAwareChunker(settings)
