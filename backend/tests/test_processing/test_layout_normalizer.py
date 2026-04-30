from __future__ import annotations

import pytest

from src.processing.layout_normalizer import LayoutNormalizer
from src.processing.types import BBox, BlockType, ParsedBlock, ParsedDocument, ParsedPage


def _ocr_block(block_id: str, content: str, *, y1: float, y2: float, x1: float = 0.0, x2: float = 100.0) -> ParsedBlock:
    return ParsedBlock(
        block_id=block_id,
        block_index=0,
        block_type=BlockType.OCR_TEXT.value,
        content=content,
        page_number=1,
        language="vi",
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        ocr_confidence=0.95,
        reading_order=0,
        source="paddleocr",
    )


def _parsed_doc(blocks: list[ParsedBlock]) -> ParsedDocument:
    return ParsedDocument(
        source_path="test_slide.png",
        file_type="png",
        language="vi",
        pages=[ParsedPage(page_number=1, blocks=blocks)],
    )


# --- _merge_ocr_lines ---

class TestMergeOcrLines:
    def test_merges_nearby_lines_into_one_block(self) -> None:
        normalizer = LayoutNormalizer()
        blocks = [
            _ocr_block("b1", "Phân tích hai biến", y1=10, y2=30),
            _ocr_block("b2", "tìm ra mối quan hệ", y1=32, y2=52),  # gap=2, threshold ~16
            _ocr_block("b3", "giữa hai biến", y1=54, y2=74),       # gap=2, merges
        ]
        merged = normalizer._merge_ocr_lines(blocks)
        assert len(merged) == 1
        assert "Phân tích hai biến" in merged[0].content
        assert "giữa hai biến" in merged[0].content

    def test_splits_on_large_vertical_gap(self) -> None:
        normalizer = LayoutNormalizer()
        blocks = [
            _ocr_block("b1", "Heading line", y1=10, y2=30),
            # large gap of 60px (threshold ~16) → new group
            _ocr_block("b2", "Bullet point one", y1=90, y2=110),
            _ocr_block("b3", "continuation of bullet", y1=112, y2=132),  # nearby
        ]
        merged = normalizer._merge_ocr_lines(blocks)
        assert len(merged) == 2
        assert merged[0].content == "Heading line"
        assert "Bullet point one" in merged[1].content
        assert "continuation of bullet" in merged[1].content

    def test_merged_bbox_covers_all_lines(self) -> None:
        normalizer = LayoutNormalizer()
        # Three blocks to meet ≥3 threshold; all close together vertically
        blocks = [
            _ocr_block("b1", "line one", y1=10, y2=30, x1=50, x2=300),
            _ocr_block("b2", "line two", y1=32, y2=52, x1=45, x2=310),
            _ocr_block("b3", "line three", y1=54, y2=74, x1=55, x2=290),
        ]
        merged = normalizer._merge_ocr_lines(blocks)
        assert len(merged) == 1
        bbox = merged[0].bbox
        assert bbox is not None
        assert bbox.x1 == 45.0
        assert bbox.x2 == 310.0
        assert bbox.y1 == 10.0
        assert bbox.y2 == 74.0

    def test_skips_merge_when_not_enough_ocr_blocks(self) -> None:
        normalizer = LayoutNormalizer()
        blocks = [
            _ocr_block("b1", "lone line", y1=10, y2=30),
            _ocr_block("b2", "another", y1=32, y2=52),
        ]
        # Only 2 ocr blocks — below the ≥3 threshold
        result = normalizer._merge_ocr_lines(blocks)
        assert len(result) == 2

    def test_skips_merge_when_mixed_block_types(self) -> None:
        normalizer = LayoutNormalizer()
        heading_block = ParsedBlock(
            block_id="h1",
            block_index=0,
            block_type=BlockType.HEADING.value,
            content="Slide Title",
            page_number=1,
            language="vi",
            bbox=BBox(x1=0, y1=0, x2=100, y2=20),
            reading_order=0,
            source="docling",
        )
        ocr_blocks = [_ocr_block(f"b{i}", f"ocr line {i}", y1=i * 22, y2=i * 22 + 20) for i in range(3)]
        blocks = [heading_block] + ocr_blocks
        # Only 3/4 = 75% ocr_text — above 60% threshold, so merge still runs on ocr lines
        result = normalizer._merge_ocr_lines(blocks)
        # heading block is not ocr_text so it should stay separate
        assert any(b.block_type == BlockType.HEADING.value for b in result)

    def test_averaged_confidence_preserved(self) -> None:
        normalizer = LayoutNormalizer()
        b1 = _ocr_block("b1", "line one", y1=10, y2=30)
        b2 = _ocr_block("b2", "line two", y1=32, y2=52)
        b1 = b1.model_copy(update={"ocr_confidence": 0.90})
        b2 = b2.model_copy(update={"ocr_confidence": 0.80})
        # Need a third block to meet ≥3 threshold
        b3 = _ocr_block("b3", "line three", y1=54, y2=74)
        b3 = b3.model_copy(update={"ocr_confidence": 0.70})
        merged = normalizer._merge_ocr_lines([b1, b2, b3])
        assert len(merged) == 1
        assert merged[0].ocr_confidence == pytest.approx((0.90 + 0.80 + 0.70) / 3)


# --- _normalize_block_type ---

class TestNormalizeBlockType:
    def test_ocr_text_short_becomes_heading(self) -> None:
        block = _ocr_block("b1", "Phân tích hai biến", y1=10, y2=30)
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.HEADING.value

    def test_ocr_text_bullet_becomes_list(self) -> None:
        block = _ocr_block("b1", "□ Phân tích hai biến tìm ra mối quan hệ", y1=50, y2=70)
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.LIST.value

    def test_ocr_text_dash_bullet_becomes_list(self) -> None:
        block = _ocr_block("b1", "- Some bullet point item here", y1=50, y2=70)
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.LIST.value

    def test_ocr_text_long_sentence_becomes_paragraph(self) -> None:
        long_text = "Phân tích hai biến tìm ra mối quan hệ giữa hai biến tìm kiếm sự liên kết association và không liên kết disassociation giữa các biến ở mức ý nghĩa được xác định trước."
        block = _ocr_block("b1", long_text, y1=50, y2=70)
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.PARAGRAPH.value

    def test_structured_parser_heading_not_reclassified(self) -> None:
        block = ParsedBlock(
            block_id="h1",
            block_index=0,
            block_type=BlockType.HEADING.value,
            content="Section Title",
            page_number=1,
            language="en",
            reading_order=0,
            source="docling",
        )
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.HEADING.value

    def test_structured_parser_paragraph_preserved(self) -> None:
        block = ParsedBlock(
            block_id="p1",
            block_index=0,
            block_type=BlockType.PARAGRAPH.value,
            content="Short text",  # would be heading if it were ocr_text
            page_number=1,
            language="en",
            reading_order=0,
            source="docling",
        )
        result = LayoutNormalizer._normalize_block_type(block)
        assert result == BlockType.PARAGRAPH.value


# --- Integration: normalize produces heading block for OCR slide ---

class TestNormalizeIntegration:
    def test_slide_ocr_heading_detected_after_normalize(self) -> None:
        # Simulate 4 OCR blocks from a simple slide: title + 3 bullet lines
        blocks = [
            _ocr_block("b0", "Phân tích hai biến", y1=10, y2=35),
            _ocr_block("b1", "□ Phân tích hai biến tìm ra mối quan hệ giữa hai biến tìm kiếm", y1=80, y2=105),
            _ocr_block("b2", "□ Chúng ta có thể thực hiện phân tích hai biến cho bất kỳ", y1=140, y2=165),
            _ocr_block("b3", "□ Các phương pháp khác nhau được sử dụng", y1=200, y2=225),
        ]
        doc = _parsed_doc(blocks)
        normalizer = LayoutNormalizer()
        normalized = normalizer.normalize(doc)
        page_blocks = normalized.pages[0].blocks
        types = [b.block_type for b in page_blocks]
        assert BlockType.HEADING.value in types
        assert types.count(BlockType.LIST.value) >= 1

    def test_ocr_lines_merged_then_heading_detected(self) -> None:
        # Title wraps across 2 OCR lines — should merge into 1 heading
        blocks = [
            _ocr_block("b0", "Phân tích hai", y1=10, y2=35),
            _ocr_block("b1", "biến", y1=37, y2=60),  # gap=2, merges with b0
            _ocr_block("b2", "□ First bullet point text here is longer", y1=100, y2=125),
            _ocr_block("b3", "□ Second bullet point text here is longer", y1=140, y2=165),
            _ocr_block("b4", "□ Third bullet point text here is longer", y1=180, y2=205),
        ]
        doc = _parsed_doc(blocks)
        normalizer = LayoutNormalizer()
        normalized = normalizer.normalize(doc)
        page_blocks = normalized.pages[0].blocks
        heading_blocks = [b for b in page_blocks if b.block_type == BlockType.HEADING.value]
        assert len(heading_blocks) >= 1
        assert "Phân tích hai biến" in heading_blocks[0].content
