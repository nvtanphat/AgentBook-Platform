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
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_LIST_ITEM = re.compile(r"^[\s]*(?:[\u25a1\u2022\u25cf\u25aa\u25ab\u2013\u2014\-\*\u25c6\u25c7\u25cb]|\d+\.|\w\))\s+", re.MULTILINE)
_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
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


def _merge_two_chunks(a: TextChunk, b: TextChunk) -> TextChunk:
    content = f"{a.content}\n{b.content}".strip()
    modalities = {a.modality, b.modality} - {"mixed"}
    modality = "mixed" if len(modalities) > 1 else (modalities.pop() if modalities else "mixed")
    pages = sorted(set(a.source_pages + b.source_pages))
    seen_ids = set(a.source_block_ids)
    block_ids = a.source_block_ids + [bid for bid in b.source_block_ids if bid not in seen_ids]
    bboxes = a.bboxes + b.bboxes
    langs = {a.language, b.language} - {"unknown", "mixed"}
    language = next(iter(langs)) if len(langs) == 1 else ("mixed" if langs else a.language)
    return a.model_copy(update={
        "content": content,
        "modality": modality,
        "source_pages": pages,
        "source_block_ids": block_ids,
        "bboxes": bboxes,
        "language": language,
        "token_count": a.token_count + b.token_count,
        "evidence": a.evidence + b.evidence,
    })


def _same_single_page(a: TextChunk, b: TextChunk) -> bool:
    return len(a.source_pages) == 1 and a.source_pages == b.source_pages


def _can_merge(
    a: TextChunk,
    b: TextChunk,
    target_tokens: int,
    max_blocks: int = 0,
    *,
    allow_same_page_block_overflow: bool = False,
) -> bool:
    if a.modality == "table" or b.modality == "table":
        return False
    if a.token_count + b.token_count > target_tokens:
        return False
    exceeds_block_budget = max_blocks > 0 and len(a.source_block_ids) + len(b.source_block_ids) > max_blocks
    if exceeds_block_budget and not (allow_same_page_block_overflow and _same_single_page(a, b)):
        return False
    return True


def _merge_tiny_chunks(
    chunks: list[TextChunk],
    min_tokens: int,
    target_tokens: int,
    max_blocks: int = 0,
    *,
    allow_same_page_block_overflow: bool = False,
) -> list[TextChunk]:
    """
    Bidirectional iterative merge to eliminate sub-minimum chunks.

    Each iteration runs:
      • Forward pass  — absorb tiny tail into the incoming chunk.
      • Backward pass — absorb tiny trailing chunk into its predecessor.
      • Isolated pass — absorb a tiny chunk sandwiched between two
                        non-tiny chunks into whichever neighbour gives
                        the smaller combined size.

    Loops until stable (no merge occurred in the last iteration).
    Table-modality chunks are never merged with adjacent chunks.
    """
    if not chunks:
        return chunks

    for _ in range(len(chunks)):  # at most O(n) iterations
        changed = False

        # --- forward pass ---
        result: list[TextChunk] = []
        for chunk in chunks:
            if result and result[-1].token_count < min_tokens and _can_merge(
                result[-1],
                chunk,
                target_tokens,
                max_blocks,
                allow_same_page_block_overflow=allow_same_page_block_overflow,
            ):
                result[-1] = _merge_two_chunks(result[-1], chunk)
                changed = True
            else:
                result.append(chunk)
        chunks = result

        # --- backward pass: trailing tiny chunk → predecessor ---
        if (
            len(chunks) >= 2
            and chunks[-1].token_count < min_tokens
            and _can_merge(
                chunks[-2],
                chunks[-1],
                target_tokens,
                max_blocks,
                allow_same_page_block_overflow=allow_same_page_block_overflow,
            )
        ):
            chunks[-2] = _merge_two_chunks(chunks[-2], chunks[-1])
            chunks.pop()
            changed = True

        # --- isolated tiny chunk → smaller neighbour ---
        result = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if c.token_count < min_tokens and c.modality != "table":
                prev_ok = (
                    i > 0
                    and _can_merge(
                        result[-1],
                        c,
                        target_tokens,
                        max_blocks,
                        allow_same_page_block_overflow=allow_same_page_block_overflow,
                    )
                    if result
                    else False
                )
                next_ok = i + 1 < len(chunks) and _can_merge(
                    c,
                    chunks[i + 1],
                    target_tokens,
                    max_blocks,
                    allow_same_page_block_overflow=allow_same_page_block_overflow,
                )
                if prev_ok and next_ok:
                    # merge into whichever gives the smaller combined chunk
                    if result[-1].token_count <= chunks[i + 1].token_count:
                        result[-1] = _merge_two_chunks(result[-1], c)
                    else:
                        chunks[i + 1] = _merge_two_chunks(c, chunks[i + 1])
                    changed = True
                elif prev_ok:
                    result[-1] = _merge_two_chunks(result[-1], c)
                    changed = True
                elif next_ok:
                    chunks[i + 1] = _merge_two_chunks(c, chunks[i + 1])
                    changed = True
                else:
                    result.append(c)
            else:
                result.append(c)
            i += 1
        chunks = result

        if not changed:
            break

    return chunks


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
            if not source_block.snippet_original.strip():
                continue
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

        return _merge_tiny_chunks(
            [c for c in chunks if c.content.strip() and c.token_count > 0],
            self.settings.chunk_min_token_count,
            self.settings.chunk_target_token_count,
            self.settings.chunk_max_blocks_per_chunk,
            allow_same_page_block_overflow=True,
        )

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
        # Suppress heading-triggered split when the current buffer is too small to
        # stand alone; let it accumulate until it reaches a meaningful size.
        if starts_new_section and current_tokens < self.settings.chunk_min_token_count:
            starts_new_section = False
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
        # Always enforce block count — short-block sources (PPTX) must be split even
        # when the token budget hasn't been hit yet.
        return True

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

        content = block.snippet_original

        # Preserve code blocks: extract them, split around them, then reinsert
        code_blocks: list[tuple[int, int, str]] = []
        for match in _CODE_BLOCK.finditer(content):
            code_blocks.append((match.start(), match.end(), match.group(0)))

        # If content has code blocks, split non-code parts only
        if code_blocks:
            parts: list[str] = []
            last_end = 0
            for start, end, code in code_blocks:
                # Split text before code block
                before = content[last_end:start]
                if before.strip():
                    parts.extend(self._split_text_content(before))
                # Keep code block intact if it fits, otherwise split by lines
                if self._count_tokens(code) <= self.settings.chunk_target_token_count:
                    parts.append(code)
                else:
                    # Split large code blocks by lines
                    parts.extend(self._split_by_lines(code))
                last_end = end
            # Split remaining text after last code block
            after = content[last_end:]
            if after.strip():
                parts.extend(self._split_text_content(after))
            if parts:
                return self._wrap_split_parts(block, parts)

        # No code blocks: use paragraph-aware splitting
        return self._wrap_split_parts(block, self._split_text_content(content))

    def _split_text_content(self, text: str) -> list[str]:
        """Split text content by paragraphs, then sentences, respecting list items."""
        # Try paragraph split first
        paragraphs = _PARAGRAPH_SPLIT.split(text)
        if len(paragraphs) > 1:
            # Check if splitting by paragraphs gives reasonable chunks
            para_parts: list[str] = []
            current_paras: list[str] = []
            current_tokens = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_tokens = self._count_tokens(para)

                # If single paragraph exceeds target, split it further
                if para_tokens > self.settings.chunk_target_token_count:
                    if current_paras:
                        para_parts.append("\n\n".join(current_paras))
                        current_paras = []
                        current_tokens = 0
                    # Recursively split oversized paragraph
                    para_parts.extend(self._split_by_sentences(para))
                    continue

                # Accumulate paragraphs
                if current_tokens + para_tokens > self.settings.chunk_target_token_count and current_paras:
                    para_parts.append("\n\n".join(current_paras))
                    current_paras = []
                    current_tokens = 0

                current_paras.append(para)
                current_tokens += para_tokens

            if current_paras:
                para_parts.append("\n\n".join(current_paras))

            if para_parts:
                return para_parts

        # Fallback to sentence splitting
        return self._split_by_sentences(text)

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentences, preserving list items."""
        sentences = _SENTENCE_SPLIT.split(text)
        if len(sentences) <= 1:
            return self._split_by_lines(text)

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

        return part_texts if part_texts else [text]

    def _split_by_lines(self, text: str) -> list[str]:
        """Fallback: split by lines when sentences are too long."""
        lines = text.split('\n')
        if len(lines) <= 1:
            # Ultimate fallback: split by words
            words = text.split()
            part_texts: list[str] = []
            current_words: list[str] = []
            current_tokens = 0

            for word in words:
                word_tokens = self._count_tokens(word)
                if current_tokens + word_tokens > self.settings.chunk_target_token_count and current_words:
                    part_texts.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0
                current_words.append(word)
                current_tokens += word_tokens

            if current_words:
                part_texts.append(" ".join(current_words))
            return part_texts if part_texts else [text]

        part_texts: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._count_tokens(line)
            if current_tokens + line_tokens > self.settings.chunk_target_token_count and current_lines:
                part_texts.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            part_texts.append("\n".join(current_lines))

        return part_texts if part_texts else [text]

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
        blocks = [b for b in evidence_map.blocks if b.snippet_original.strip()]
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

        # Pre-compute cumulative token counts to suppress tiny-buffer heading splits
        _tok = [self._layout._count_tokens(b.snippet_original) for b in blocks]
        _cum = [0] * (len(blocks) + 1)
        for _i, _t in enumerate(_tok):
            _cum[_i + 1] = _cum[_i] + _t

        # Layout hard-breaks always override, except when the buffer that would
        # be emitted is too small to stand alone.
        min_tok = self.settings.chunk_min_token_count
        for i in range(1, len(blocks)):
            prev, curr = blocks[i - 1], blocks[i]
            # heading after body content → new section
            if curr.block_type == "heading" and prev.block_type != "heading":
                # Find the start of the current segment (last split point before i)
                seg_start = max((s for s in split_before if s < i), default=0)
                seg_tokens = _cum[i] - _cum[seg_start]
                if seg_tokens >= min_tok:
                    split_before.add(i)
                # else: suppress — buffer too small, keep accumulating
            # table ↔ non-table (always hard-break; table isolation is non-negotiable)
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
            block_overflow = bool(current) and len(current) >= self.settings.chunk_max_blocks_per_chunk

            if (hard_break or token_overflow or block_overflow) and current:
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

        return _merge_tiny_chunks(
            [c for c in chunks if c.content.strip() and c.token_count > 0],
            self.settings.chunk_min_token_count,
            self.settings.chunk_target_token_count,
            self.settings.chunk_max_blocks_per_chunk,
            allow_same_page_block_overflow=True,
        )

    def _make_chunk(self, evidence_map: EvidenceMap, blocks: list[EvidenceBlock]) -> TextChunk:
        chunk = self._layout._make_chunk(evidence_map, blocks)
        return chunk.model_copy(update={"chunk_strategy": self._STRATEGY})


def build_chunker(settings: Settings, embedder: "BGEM3Embedder | None" = None) -> LayoutAwareChunker | SemanticChunker:
    """Return the configured chunker, wiring in the embedder for semantic mode."""
    if settings.chunk_strategy == "semantic":
        return SemanticChunker(settings, embedder=embedder)
    return LayoutAwareChunker(settings)
