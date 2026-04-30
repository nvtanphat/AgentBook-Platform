from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.core.config import Settings
from src.processing.handwriting_reader import HandwritingReader
from src.processing.image_quality_checker import ImageQualityReport
from src.processing.ocr_engine import PaddleOCREngine
from src.processing.types import DependencyUnavailableError, ParsedBlock, ParsedDocument, ParsedPage


def test_paddle_ocr_result_maps_text_bbox_and_confidence() -> None:
    engine = PaddleOCREngine(lang="vi")
    raw = [
        [
            [[[1, 2], [11, 2], [11, 8], [1, 8]], ("Xin chao", 0.93)],
        ]
    ]

    blocks = engine._parse_result(raw, image_path=Path("scan.png"), language="vi")

    assert len(blocks) == 1
    assert blocks[0].content == "Xin chao"
    assert blocks[0].bbox.x1 == 1
    assert blocks[0].bbox.x2 == 11
    assert blocks[0].ocr_confidence == 0.93


def test_paddle_ocr_metadata_uses_configured_model_names() -> None:
    class FakePaddle:
        def predict(self, path: str):
            return [
                {
                    "rec_texts": ["Xin chao"],
                    "rec_scores": [0.93],
                    "rec_polys": [[[1, 2], [11, 2], [11, 8], [1, 8]]],
                }
            ]

    settings = Settings(
        testing=True,
        ocr_text_detection_model_name="PP-OCRv5_mobile_det",
        ocr_text_recognition_model_name="PP-OCRv5_server_rec",
    )
    engine = PaddleOCREngine(lang="vi", settings=settings)
    engine._ocr = FakePaddle()

    parsed = engine.parse_image(Path("scan.png"), language="vi")

    assert parsed.extra["det_model"] == "PP-OCRv5_mobile_det"
    assert parsed.extra["rec_model"] == "PP-OCRv5_server_rec"


def test_paddle_ocr_merges_better_vietnamese_variant_text() -> None:
    engine = PaddleOCREngine(lang="vi")
    base = engine._parse_result(
        [[[[[0, 0], [100, 0], [100, 20], [0, 20]], ("trien khai du an", 0.98)]]],
        image_path=Path("scan.png"),
        language="vi",
    )
    variant = engine._parse_result(
        [[[[[0, 0], [100, 0], [100, 20], [0, 20]], ("tri\u1ec3n khai d\u1ef1 \u00e1n", 0.94)]]],
        image_path=Path("scan.png"),
        language="vi",
    )

    merged = engine._merge_variant_blocks(
        base,
        variant,
        image_path=Path("scan.png"),
        language="vi",
        variant_name="grayscale",
    )

    assert merged[0].content == "tri\u1ec3n khai d\u1ef1 \u00e1n"
    assert merged[0].extra["ocr_source_variant"] == "grayscale"
    assert merged[0].extra["ocr_raw_content"] == "trien khai du an"


def test_printed_image_ocr_requires_paddleocr() -> None:
    class BrokenPaddle:
        def ocr(self, *args, **kwargs):
            raise RuntimeError("paddle runtime unavailable")

    engine = PaddleOCREngine(lang="vi")
    engine._ocr = BrokenPaddle()

    with pytest.raises(DependencyUnavailableError, match="PaddleOCR is required"):
        engine.parse_image(Path("diagram.png"), language="en")


@dataclass
class FakeQualityChecker:
    report: ImageQualityReport

    def check(self, image_path: Path) -> ImageQualityReport:
        return self.report


class FakeOCREngine:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def parse_image(self, image_path: Path, *, language: str = "vi") -> ParsedDocument:
        return ParsedDocument(
            source_path=str(image_path),
            file_type="png",
            language=language,
            pages=[
                ParsedPage(
                    page_number=1,
                    blocks=[
                        ParsedBlock(
                            block_id="blk-hw",
                            block_index=0,
                            block_type="ocr_text",
                            content="loi giai mau",
                            page_number=1,
                            language=language,
                            ocr_confidence=self.confidence,
                            reading_order=0,
                            source="fake_ocr",
                        )
                    ],
                )
            ],
        )


def test_handwriting_reader_refuses_low_quality_image() -> None:
    reader = HandwritingReader(
        settings=Settings(testing=True),
        quality_checker=FakeQualityChecker(
            ImageQualityReport(
                score=0.3,
                is_acceptable=False,
                blur_variance=10,
                brightness=20,
                contrast=5,
                skew_degrees=0,
                warnings=["image is too blurry"],
            )
        ),
        ocr_engine=FakeOCREngine(confidence=0.99),
    )

    parsed = reader.parse_image(Path("handwriting.png"))

    assert parsed.pages == []
    assert parsed.extra["accepted_as_evidence"] is False
    assert parsed.warnings == ["image is too blurry"]


def test_handwriting_reader_accepts_clear_high_confidence_image() -> None:
    reader = HandwritingReader(
        settings=Settings(testing=True),
        quality_checker=FakeQualityChecker(
            ImageQualityReport(
                score=0.95,
                is_acceptable=True,
                blur_variance=200,
                brightness=120,
                contrast=45,
                skew_degrees=0,
                warnings=[],
            )
        ),
        ocr_engine=FakeOCREngine(confidence=0.97),
    )

    parsed = reader.parse_image(Path("handwriting.png"))

    assert parsed.extra["accepted_as_evidence"] is True
    assert parsed.blocks[0].block_type == "handwriting"
    assert parsed.blocks[0].extra["image_quality_score"] == 0.95
