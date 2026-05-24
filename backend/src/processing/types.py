from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProcessingError(RuntimeError):
    """Base error for document processing failures."""


class DependencyUnavailableError(ProcessingError):
    """Raised when an optional parser/OCR/model dependency is unavailable."""


class OCRQualityError(ProcessingError):
    """Raised when OCR output quality is below the configured fail threshold."""

    failed_stage: str = "ocr_quality"

    def __init__(self, message: str, score: float, threshold: float) -> None:
        super().__init__(message)
        self.score = score
        self.threshold = threshold


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    EQUATION = "equation"
    HANDWRITING = "handwriting"
    OCR_TEXT = "ocr_text"


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ParsedBlock(BaseModel):
    block_id: str
    block_index: int
    block_type: str
    content: str
    page_number: int
    language: str = "unknown"
    bbox: BBox | None = None
    ocr_confidence: float | None = None
    reading_order: int
    source: str
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedPage(BaseModel):
    page_number: int
    image_path: str | None = None
    width: int | None = None
    height: int | None = None
    ocr_confidence: float | None = None
    blocks: list[ParsedBlock] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    source_path: str
    file_type: str
    language: str = "unknown"
    pages: list[ParsedPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def blocks(self) -> list[ParsedBlock]:
        return [block for page in self.pages for block in page.blocks]


class EvidenceBlock(BaseModel):
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    page: int
    block_id: str
    block_type: str
    snippet_original: str
    source_language: str
    bbox: BBox | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceMap(BaseModel):
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    blocks: list[EvidenceBlock] = Field(default_factory=list)


class TextChunk(BaseModel):
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    content: str
    # LLM-generated context prefix prepended before embedding (Contextual Retrieval).
    # None means contextual enrichment was skipped; content is embedded as-is.
    contextualized_content: str | None = None
    language: str
    modality: str = "text"
    source_block_ids: list[str]
    source_pages: list[int]
    bboxes: list[BBox] = Field(default_factory=list)
    token_count: int
    chunk_strategy: str
    chunker_version: str
    parser_version: str
    embedding_model: str
    embedding_version: str
    index_version: str
    evidence: list[EvidenceBlock] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str
    mention_refs: list[EvidenceBlock] = Field(default_factory=list)
    normalized_value: str | None = None
    confidence: float


class ExtractedEvent(BaseModel):
    event_name: str
    event_time: str | None = None
    participants: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceBlock] = Field(default_factory=list)
    temporal_status: str = "unknown"
    confidence: float = 0.5


class ExtractedRelation(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    evidence_refs: list[EvidenceBlock] = Field(default_factory=list)
    evidence_text_chunk: str | None = None
    confidence: float = 0.5
    is_conflicting: bool = False
