from __future__ import annotations

from pydantic import BaseModel, Field

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
    dense_score: float | None = None
    sparse_score: float | None = None
    graph_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class GraphPath(BaseModel):
    path: list[str]
    confidence: float
    evidence_refs: list[EvidenceBlock] = Field(default_factory=list)
