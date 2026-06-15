from __future__ import annotations

from datetime import datetime

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from pymongo import IndexModel

from src.models.common import utc_now


class EvidenceRef(BaseModel):
    material_id: PydanticObjectId
    page: int | None = None
    block_id: str | None = None
    span: list[int] | None = None


class Entity(Document):
    owner_id: str
    collection_id: PydanticObjectId
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str
    mention_refs: list[EvidenceRef] = Field(default_factory=list)
    normalized_value: str | None = None
    description: str | None = None  # short context snippet describing the entity
    confidence: float
    chunk_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "entities"
        indexes = [
            IndexModel([("owner_id", 1), ("collection_id", 1), ("canonical_name", 1)], name="entities_scope_name"),
            IndexModel([("aliases", 1)], name="entities_aliases"),
            IndexModel([("mention_refs.material_id", 1)], name="entities_mention_material_id"),
            IndexModel([("canonical_name", "text"), ("aliases", "text")], name="entities_text_search"),
        ]


class Event(Document):
    owner_id: str
    collection_id: PydanticObjectId
    event_name: str
    event_time: datetime | None = None
    participants: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    temporal_status: str = "unknown"
    chunk_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "events"
        indexes = [
            IndexModel([("owner_id", 1), ("collection_id", 1), ("event_name", 1)], name="events_scope_name"),
            IndexModel([("evidence_refs.material_id", 1)], name="events_evidence_material_id"),
        ]


class Relation(Document):
    owner_id: str
    collection_id: PydanticObjectId
    source_id: str
    target_id: str
    relation_type: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    evidence_text_chunk: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float
    is_conflicting: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    class Settings:
        name = "relations"
        indexes = [
            IndexModel([("owner_id", 1), ("collection_id", 1), ("source_id", 1), ("target_id", 1)], name="relations_scope_edge"),
            IndexModel([("owner_id", 1), ("collection_id", 1), ("source_id", 1)], name="relations_scope_source"),
            IndexModel([("owner_id", 1), ("collection_id", 1), ("target_id", 1)], name="relations_scope_target"),
            IndexModel([("confidence", -1)], name="relations_confidence"),
            IndexModel([("evidence_refs.material_id", 1)], name="relations_evidence_material_id"),
        ]
