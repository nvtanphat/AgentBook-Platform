from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    confidence: float | None = None
    mention_count: int = 0
    degree: int = 0
    # Centrality + clustering signals (added Phase 2 — community detection + PageRank)
    importance: float = 0.0           # normalized [0, 1] PageRank score
    community: int = 0                # community id from Louvain
    community_label: str | None = None  # optional human label for the community
    is_hub: bool = False              # top 10% by importance
    is_focused: bool = False          # In focus mode: directly matches citation evidence (primary node)
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
    evidence_text_chunk: str | None = None


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class MindmapRequest(BaseModel):
    owner_id: str = Field(min_length=1, max_length=128)
    collection_id: str | None = None
    material_ids: list[str] = Field(default_factory=list, max_length=50)
    root_topic: str | None = Field(default=None, max_length=1000)
    detail_level: Literal["brief", "overview", "detailed"] = "overview"
    use_llm: bool = False
    # When provided, graph endpoint filters/focuses on entities whose mention_refs
    # match these block_ids (from citations of the last answer). Used by the
    # "Kiểm chứng bằng Graph" button to show only entities backing the answer.
    focus_block_ids: list[str] = Field(default_factory=list, max_length=200)
    focus_material_ids: list[str] = Field(default_factory=list, max_length=50)
    # Optional: page numbers per material to widen focus to entities on cited pages.
    # Format: ["material_id:page_number"], e.g. ["65f0:1", "65f0:2"]
    focus_pages: list[str] = Field(default_factory=list, max_length=200)
    # Primary verification signal: entities whose canonical_name or aliases appear
    # in the query OR answer text get marked as focused. Far more precise than
    # block_id matching when chunk_ids don't align with entity mention_refs.
    focus_query_text: str | None = Field(default=None, max_length=4000)
    focus_answer_text: str | None = Field(default=None, max_length=8000)
