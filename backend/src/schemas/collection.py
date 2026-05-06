from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=160)
    subject: str | None = None
    description: str | None = None


class CollectionUpdateRequest(BaseModel):
    owner_id: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1, max_length=160)
    subject: str | None = None
    description: str | None = None


class CollectionSummary(BaseModel):
    collection_id: str
    name: str
    owner_id: str
    subject: str | None = None
    description: str | None = None
    material_count: int = 0
    indexed_material_count: int = 0
    retrievable_chunk_count: int = 0
    latest_material_name: str | None = None
    created_at: datetime
    updated_at: datetime


class CollectionDashboard(BaseModel):
    collection_id: str
    name: str
    owner_id: str
    subject: str | None = None
    description: str | None = None
    material_count: int = 0
    indexed_material_count: int = 0
    retrievable_chunk_count: int = 0
    entity_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    language_counts: dict[str, int] = Field(default_factory=dict)
    latest_material_name: str | None = None
    created_at: datetime
    updated_at: datetime
