from __future__ import annotations

from pathlib import Path

from src.processing.docling_parser import DoclingParser
from src.services.parse_index_pipeline import ParseIndexPipeline


class FakeDoclingDocument:
    def export_to_dict(self):
        return {
            "texts": [
                {
                    "self_ref": "#/texts/0",
                    "label": "section_header",
                    "text": "Regularization",
                    "prov": [{"page_no": 2, "bbox": {"l": 10, "t": 20, "r": 110, "b": 40}}],
                },
                {
                    "self_ref": "#/texts/1",
                    "label": "text",
                    "text": "Dropout reduces co-adaptation.",
                    "prov": [{"page_no": 2, "bbox": {"l": 10, "t": 50, "r": 210, "b": 90}}],
                },
            ]
        }


class FakeConversionResult:
    document = FakeDoclingDocument()


def test_docling_export_is_mapped_to_structured_blocks() -> None:
    parser = DoclingParser()

    pages = parser._pages_from_export(
        FakeDoclingDocument(),
        file_path=Path("lecture.pdf"),
        extension="pdf",
        language="en",
    )

    assert len(pages) == 1
    assert pages[0].page_number == 2
    assert [block.block_id for block in pages[0].blocks] == ["#/texts/0", "#/texts/1"]
    assert pages[0].blocks[0].block_type == "heading"
    assert pages[0].blocks[1].bbox.x2 == 210


def test_pdf_parse_uses_docling_before_text_fallback(monkeypatch) -> None:
    class FakeConverter:
        def convert(self, path: str):
            return FakeConversionResult()

    parser = DoclingParser()
    monkeypatch.setattr(parser, "_ensure_docling_available", lambda: None)
    monkeypatch.setattr(parser, "_converter", lambda extension: FakeConverter())
    monkeypatch.setattr(parser, "_add_pdf_text_fallback_pages", lambda *args, **kwargs: None)
    monkeypatch.setattr(parser, "_add_easyocr_pages", lambda *args, **kwargs: None)

    parsed = parser.parse(Path("lecture.pdf"), language="en")

    assert parsed.extra["parser"] == "docling"
    assert parsed.extra["pdf_strategy"] == "docling_layout_first_text_ocr_missing_pages"
    assert parsed.pages[0].blocks[0].source == "docling"


def test_pdf_parse_falls_back_when_docling_conversion_fails(monkeypatch) -> None:
    class BrokenConverter:
        def convert(self, path: str):
            raise RuntimeError("layout model failed")

    parser = DoclingParser()
    monkeypatch.setattr(parser, "_ensure_docling_available", lambda: None)
    monkeypatch.setattr(parser, "_converter", lambda extension: BrokenConverter())
    monkeypatch.setattr(parser, "_add_easyocr_pages", lambda *args, **kwargs: None)

    parsed = parser.parse(Path("lecture.pdf"), language="en")

    assert parsed.extra["parser"] == "pypdf_fallback"
    assert "layout model failed" in parsed.extra["docling_error"]


def test_pdf_text_fallback_splits_page_text_into_logical_blocks() -> None:
    parser = DoclingParser()
    blocks = parser._blocks_from_plain_text_page(
        "1. Overview\nDropout reduces overfitting.\n- Useful for neural networks.",
        file_path=Path("lecture.pdf"),
        page_number=1,
        language="en",
        source="pypdf_text_fallback",
        fallback_reason="test",
    )

    assert [block.block_type for block in blocks] == ["heading", "paragraph", "list"]
    assert blocks[0].content == "1. Overview"


def test_docx_embedded_images_are_preserved_as_figure_blocks() -> None:
    import pytest
    docx_path = Path(__file__).resolve().parents[3] / "data" / "test data" / "multimodal_rag_test_day_du.docx"
    if not docx_path.exists():
        pytest.skip(f"Test fixture not found: {docx_path}")
    parser = DoclingParser()
    parsed = parser.parse(docx_path, language="vi")

    figure_blocks = [block for block in parsed.blocks if block.block_type == "figure"]
    assert len(figure_blocks) >= 6
    assert all(block.extra.get("needs_captioning") for block in figure_blocks)
    assert all(
        isinstance(block.extra.get("embedded_image_uri"), str)
        and block.extra["embedded_image_uri"].startswith("data:image/")
        for block in figure_blocks
    )

    class FakeCaptioner:
        def caption_image_path(self, image_path: Path) -> str:
            assert image_path.exists()
            return "DOCX FIGURE CAPTION"

    caption = ParseIndexPipeline._caption_one_figure(
        figure_blocks[0],
        FakeCaptioner(),
        Path("D:/GenAI/DoAn01/data/cache/pdf_page_images"),
        parsed,
    )
    assert caption == "DOCX FIGURE CAPTION"
