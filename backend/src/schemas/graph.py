from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    confidence: float | None = None
    mention_count: int = 0
    source_docs: list[str] = Field(default_factory=list)
    evidence_refs: list[dict[str, str | int]] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    source_label: str | None = None
    target_label: str | None = None
    confidence: float | None = None
    evidence_count: int = 0
    evidence_refs: list[dict[str, str | int]] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class MindmapRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list)
    root_topic: str | None = None
    detail_level: Literal["overview", "detailed"] = "overview"
    use_llm: bool = False
