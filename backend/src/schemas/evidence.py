from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class EvidenceBlockSchema(BaseModel):
    block_id: str
    block_type: str
    page: int
    snippet_original: str
    source_language: str
    bbox: BoundingBoxSchema | None = None
    confidence: float | None = None
    material_id: str | None = None
    doc_name: str | None = None
    # Audio-only fields (populated when block came from audio transcription).
    # Allows frontend to seek to exact timestamp and render [file.mp3 03:24-04:18].
    audio_start_seconds: float | None = None
    audio_end_seconds: float | None = None
    audio_file: str | None = None
    # Figure-only: API URL to the cropped/embedded figure image on disk.
    # Set when block_type="figure" and FigureCaptioner saved an image crop.
    figure_image_url: str | None = None


class CitationSchema(BaseModel):
    doc_id: str
    doc_name: str
    page: int | None = None
    pages: list[int] = Field(default_factory=list)
    block_id: str | None = None
    block_type: str | None = None
    snippet_original: str
    snippet_translated: str | None = None
    # Span-level citation: the specific sentence within snippet_original that directly
    # proves the answer sentence. Extracted by token-overlap scoring against focus_text.
    # Frontend highlights this span inside the full blockquote.
    cited_span: str | None = None
    bbox: BoundingBoxSchema | None = None
    role: str = "primary"
    source_language: str
    confidence: float = Field(ge=0.0, le=1.0)
    # All contributing evidence blocks — exposes spatial/page data for every block in this chunk
    evidence_blocks: list[EvidenceBlockSchema] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


class EvidencePageResponse(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    blocks: list[EvidenceBlockSchema] = Field(default_factory=list)
    source_filename: str
    # Populated for image-type documents (png/jpg/jpeg) so the UI can show the source image
    raw_image_url: str | None = None
    file_type: str | None = None
