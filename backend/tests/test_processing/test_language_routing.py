from __future__ import annotations

from src.core.config import Settings
from src.processing.chunking import LayoutAwareChunker
from src.processing.language_detector import detect_block_language, detect_document_language
from src.processing.types import EvidenceBlock, EvidenceMap, ParsedBlock, ParsedDocument, ParsedPage
from src.services.parse_index_pipeline import ParseIndexPipeline


def test_language_detector_handles_vietnamese_and_english_without_langdetect() -> None:
    assert detect_block_language("Cach xu ly du lieu thieu trong bai hoc nay") == "vi"
    assert detect_block_language("Dropout reduces overfitting in neural networks.") == "en"


def test_document_language_detector_marks_balanced_mixed_documents() -> None:
    detected = detect_document_language(
        [
            "Dropout reduces overfitting in neural networks.",
            "Cach xu ly du lieu thieu trong bai hoc nay",
        ]
    )

    assert detected == "mixed"


def test_parse_pipeline_assigns_block_languages_when_upload_language_unknown() -> None:
    parsed = ParsedDocument(
        source_path="mixed.pdf",
        file_type="pdf",
        language="unknown",
        pages=[
            ParsedPage(
                page_number=1,
                blocks=[
                    ParsedBlock(
                        block_id="en",
                        block_index=0,
                        block_type="paragraph",
                        content="Dropout reduces overfitting in neural networks.",
                        page_number=1,
                        language="unknown",
                        reading_order=0,
                        source="test",
                    ),
                    ParsedBlock(
                        block_id="vi",
                        block_index=1,
                        block_type="paragraph",
                        content="Cach xu ly du lieu thieu trong bai hoc nay",
                        page_number=1,
                        language="unknown",
                        reading_order=1,
                        source="test",
                    ),
                ],
            )
        ],
    )

    document_language, counts = ParseIndexPipeline._apply_language_detection(parsed, declared_language="unknown")

    assert document_language == "mixed"
    assert counts == {"en": 1, "vi": 1}
    assert [block.language for block in parsed.blocks] == ["en", "vi"]


def test_chunker_marks_chunk_language_mixed_when_evidence_is_multilingual() -> None:
    settings = Settings(testing=True, chunk_target_token_count=128, chunk_overlap_token_count=0)
    evidence_map = EvidenceMap(
        owner_id="u",
        collection_id="c",
        material_id="m",
        document_name="mixed.pdf",
        blocks=[
            EvidenceBlock(
                owner_id="u",
                collection_id="c",
                material_id="m",
                document_name="mixed.pdf",
                page=1,
                block_id="en",
                block_type="paragraph",
                snippet_original="Dropout reduces overfitting in neural networks.",
                source_language="en",
            ),
            EvidenceBlock(
                owner_id="u",
                collection_id="c",
                material_id="m",
                document_name="mixed.pdf",
                page=1,
                block_id="vi",
                block_type="paragraph",
                snippet_original="Cach xu ly du lieu thieu trong bai hoc nay",
                source_language="vi",
            ),
        ],
    )

    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    assert chunks[0].language == "mixed"


class FakeOCREngine:
    def __init__(self, *, lang: str, content: str, confidence: float) -> None:
        self.lang = lang
        self.content = content
        self.confidence = confidence

    def parse_image(self, path, *, language: str = "unknown") -> ParsedDocument:
        return ParsedDocument(
            source_path=str(path),
            file_type="png",
            language=language,
            pages=[
                ParsedPage(
                    page_number=1,
                    blocks=[
                        ParsedBlock(
                            block_id=f"blk-{self.lang}",
                            block_index=0,
                            block_type="ocr_text",
                            content=self.content,
                            page_number=1,
                            language=language,
                            ocr_confidence=self.confidence,
                            reading_order=0,
                            source="fake_ocr",
                        )
                    ],
                )
            ],
            extra={"parser": "paddleocr", "ocr_lang": self.lang},
        )


def test_unknown_image_ocr_routes_to_vietnamese_when_initial_english_output_is_vi() -> None:
    pipeline = ParseIndexPipeline(settings=Settings(testing=True))
    pipeline._ocr_engines = {
        "en": FakeOCREngine(
            lang="en",
            content="Quy trinh xu ly du lieu dur\u03c3c thuc hien trong bai hoc nay",
            confidence=0.92,
        ),
        "vi": FakeOCREngine(
            lang="vi",
            content="Quy trinh xu ly du lieu duoc thuc hien trong bai hoc nay",
            confidence=0.94,
        ),
    }

    parsed = pipeline._parse_printed_image("scan.png", declared_language="unknown")

    assert parsed.extra["ocr_language_routing"]["selected"] == "vi"
    assert parsed.blocks[0].content == "Quy trinh xu ly du lieu duoc thuc hien trong bai hoc nay"
