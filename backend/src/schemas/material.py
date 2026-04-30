from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MaterialUploadMetadata(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    collection_name: str | None = None
    collection_description: str | None = None
    subject: str | None = None
    topic: str | None = None
    language: str = "unknown"
    modality: str = "mixed"
    source_type: str | None = None
    version: str = "v1.0"
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class MaterialUploadResponse(BaseModel):
    material_id: str
    doc_id: str
    collection_id: str
    job_id: str
    status: str
    stage: str
    filename: str
    original_name: str
    checksum_sha256: str
    file_size_bytes: int
    storage_path: str


class MaterialBatchUploadItem(BaseModel):
    filename: str
    success: bool
    data: MaterialUploadResponse | None = None
    error: str | None = None


class MaterialBatchUploadResponse(BaseModel):
    results: list[MaterialBatchUploadItem] = Field(default_factory=list)


class MaterialResponse(BaseModel):
    material_id: str
    collection_id: str
    owner_id: str
    filename: str
    original_name: str
    file_type: str
    status: str
    subject: str | None = None
    topic: str | None = None
    page_count: int | None = None
    version: str


class MaterialStatusResponse(BaseModel):
    material_id: str
    collection_id: str
    status: str
    stage: str
    progress_pct: int
    failed_stage: str | None = None
    error_message: str | None = None


class DebugBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DebugBlock(BaseModel):
    block_id: str
    block_index: int
    block_type: str
    content: str
    language: str
    bbox: DebugBBox | None = None
    ocr_confidence: float | None = None
    reading_order: int


class DebugPage(BaseModel):
    page_number: int
    width: int | None = None
    height: int | None = None
    ocr_confidence: float | None = None
    blocks: list[DebugBlock] = Field(default_factory=list)


class DebugChunk(BaseModel):
    chunk_id: str
    content: str
    language: str
    modality: str
    token_count: int | None = None
    source_block_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    chunk_strategy: str
    embedding_model: str


class MaterialDebugResponse(BaseModel):
    material_id: str
    collection_id: str
    owner_id: str
    original_name: str
    file_type: str
    status: str
    modality: str
    language: str
    page_count: int
    pages: list[DebugPage] = Field(default_factory=list)
    chunks: list[DebugChunk] = Field(default_factory=list)
    qdrant_vector_count: int = 0
    raw_image_url: str | None = None
