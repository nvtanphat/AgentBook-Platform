from __future__ import annotations

from pydantic import BaseModel, Field


class MindmapNode(BaseModel):
    id: str
    label: str
    entity_type: str = "concept"  # NEW: Explicit entity type for better clustering
    summary: str | None = None
    children: list["MindmapNode"] = Field(default_factory=list)
    citations: list[dict[str, str | int]] = Field(default_factory=list)
    collapsed: bool = False  # NEW: For collapsible branches


class MindmapResponse(BaseModel):
    root_topic: str
    nodes: list[MindmapNode] = Field(default_factory=list)
