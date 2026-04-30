from __future__ import annotations

from pydantic import BaseModel, Field


class MindmapNode(BaseModel):
    id: str
    label: str
    summary: str | None = None
    children: list["MindmapNode"] = Field(default_factory=list)
    citations: list[dict[str, str | int]] = Field(default_factory=list)


class MindmapResponse(BaseModel):
    root_topic: str
    nodes: list[MindmapNode] = Field(default_factory=list)
