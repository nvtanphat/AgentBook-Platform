from __future__ import annotations

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    confidence: float | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float | None = None
    evidence_refs: list[dict[str, str | int]] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class MindmapRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)
    root_topic: str | None = None
