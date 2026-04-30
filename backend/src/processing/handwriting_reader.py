from __future__ import annotations

from pathlib import Path

from src.core.config import Settings
from src.processing.image_quality_checker import ImageQualityChecker
from src.processing.ocr_engine import PaddleOCREngine
from src.processing.types import BlockType, ParsedDocument


class HandwritingReader:
    def __init__(
        self,
        *,
        settings: Settings,
        quality_checker: ImageQualityChecker | None = None,
        ocr_engine: PaddleOCREngine | None = None,
    ) -> None:
        self.settings = settings
        self.quality_checker = quality_checker or ImageQualityChecker(settings)
        self.ocr_engine = ocr_engine or PaddleOCREngine(lang="vi", settings=settings)

    def parse_image(self, image_path: Path, *, language: str = "vi") -> ParsedDocument:
        quality = self.quality_checker.check(image_path)
        if not quality.is_acceptable:
            return ParsedDocument(
                source_path=str(image_path),
                file_type=image_path.suffix.lower().lstrip("."),
                language=language,
                warnings=quality.warnings,
                extra={"parser": "handwriting_reader", "image_quality_score": quality.score, "accepted_as_evidence": False},
            )

        parsed = self.ocr_engine.parse_image(image_path, language=language)
        confidences = [block.ocr_confidence for block in parsed.blocks if block.ocr_confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if avg_confidence < self.settings.min_handwriting_confidence:
            parsed.warnings.append("handwriting OCR confidence is below the evidence threshold")
            parsed.pages = []
            parsed.extra.update(
                {
                    "parser": "handwriting_reader",
                    "image_quality_score": quality.score,
                    "handwriting_confidence": avg_confidence,
                    "accepted_as_evidence": False,
                }
            )
            return parsed

        for block in parsed.blocks:
            block.block_type = BlockType.HANDWRITING.value
            block.source = "handwriting_reader"
            block.extra["image_quality_score"] = quality.score
        parsed.extra.update(
            {
                "parser": "handwriting_reader",
                "image_quality_score": quality.score,
                "handwriting_confidence": avg_confidence,
                "accepted_as_evidence": True,
            }
        )
        return parsed
