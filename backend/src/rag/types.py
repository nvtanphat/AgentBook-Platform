from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any

from src.processing.types import BBox, EvidenceBlock


class RetrievalScope(BaseModel):
    owner_id: str
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)

    def ensure_scoped(self) -> None:
        if not self.owner_id:
            raise ValueError("owner_id is required for retrieval")
        if not self.collection_id and not self.material_ids:
            raise ValueError("collection_id or material_ids is required for scoped retrieval")


class RetrievedChunk(BaseModel):
    chunk_id: str
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    content: str
    language: str
    modality: str
    source_block_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    bboxes: list[BBox] = Field(default_factory=list)
    evidence: list[EvidenceBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    graph_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class GraphPath(BaseModel):
    path: list[str]
    confidence: float
    evidence_refs: list[EvidenceBlock] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class FigureIndexItem(BaseModel):
    """A figure block ready to be embedded and upserted into the visual collection.

    All evidence-trace fields are mandatory; image_path may be None for figures
    whose pixel data could not be preserved (e.g. deleted DOCX temp files) — the
    indexer will skip those with a debug log rather than raise.
    """

    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    page: int
    block_id: str
    block_type: str
    caption: str
    source_language: str
    bbox: BBox | None = None
    image_path: str | None = None


class RetrievedVisualChunk(BaseModel):
    """A figure retrieved from the visual Qdrant collection."""

    point_id: str
    owner_id: str
    collection_id: str
    material_id: str
    document_name: str
    page: int
    block_id: str
    block_type: str
    caption: str
    source_language: str
    bbox: BBox | None = None
    image_path: str | None = None
    score: float = 0.0
