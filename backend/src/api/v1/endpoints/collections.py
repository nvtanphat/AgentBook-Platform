from __future__ import annotations

from collections import defaultdict

from beanie.operators import In
from fastapi import APIRouter, Depends, Query, Request, status

from src.dependencies import get_material_service, verify_owner_access
from src.models.common import utc_now
from src.models.chunk import Chunk
from src.models.collection import KnowledgeCollection
from src.models.material import Material
from src.schemas.collection import CollectionCreateRequest, CollectionSummary
from src.schemas.common import APIResponse
from src.services.material_service import MaterialService

router = APIRouter(prefix="/collections", tags=["collections"])


def _empty_summary(collection: KnowledgeCollection) -> CollectionSummary:
    return CollectionSummary(
        collection_id=str(collection.id),
        name=collection.name,
        owner_id=collection.owner_id,
        subject=collection.subject,
        description=collection.description,
        material_count=0,
        indexed_material_count=0,
        retrievable_chunk_count=0,
        latest_material_name=None,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )


@router.post("", response_model=APIResponse[CollectionSummary], status_code=status.HTTP_201_CREATED)
async def create_collection(request: Request, body: CollectionCreateRequest) -> APIResponse[CollectionSummary]:
    verify_owner_access(request, body.owner_id)
    name = " ".join(body.name.split())
    collection = KnowledgeCollection(
        name=name,
        owner_id=body.owner_id,
        subject=body.subject,
        description=body.description,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    await collection.insert()
    return APIResponse(
        success=True,
        message="Collection created successfully",
        data=_empty_summary(collection),
        error=None,
    )


@router.get("", response_model=APIResponse[list[CollectionSummary]])
async def list_collections(
    request: Request,
    owner_id: str = Query(..., min_length=1),
    limit: int = Query(default=100, ge=1, le=500),
) -> APIResponse[list[CollectionSummary]]:
    verify_owner_access(request, owner_id)
    collections = await KnowledgeCollection.find(KnowledgeCollection.owner_id == owner_id).sort("-updated_at").limit(limit).to_list()
    if not collections:
        return APIResponse(success=True, message="Collections loaded successfully", data=[], error=None)

    collection_ids = [c.id for c in collections]

    all_materials = await Material.aggregate([
        {"$match": {"owner_id": owner_id, "collection_id": {"$in": collection_ids}}},
        {"$sort": {"updated_at": -1}},
        {"$project": {"collection_id": 1, "original_name": 1, "updated_at": 1}},
    ]).to_list()

    # Aggregate chunk counts and indexed material ids without loading full Chunk documents
    chunk_agg = await Chunk.aggregate([
        {"$match": {"owner_id": owner_id, "collection_id": {"$in": collection_ids}}},
        {"$group": {
            "_id": "$collection_id",
            "chunk_count": {"$sum": 1},
            "indexed_material_ids": {"$addToSet": "$material_id"},
        }},
    ]).to_list()

    materials_by_collection: dict = defaultdict(list)
    for material in all_materials:
        materials_by_collection[material["collection_id"]].append(material)

    chunk_count_by_collection: dict = defaultdict(int)
    indexed_material_ids_by_collection: dict = defaultdict(set)
    for row in chunk_agg:
        cid = row["_id"]
        chunk_count_by_collection[cid] = row["chunk_count"]
        indexed_material_ids_by_collection[cid] = set(row["indexed_material_ids"])

    summaries: list[CollectionSummary] = []
    for collection in collections:
        col_materials = materials_by_collection.get(collection.id, [])
        summaries.append(
            CollectionSummary(
                collection_id=str(collection.id),
                name=collection.name,
                owner_id=collection.owner_id,
                subject=collection.subject,
                description=collection.description,
                material_count=len(col_materials),
                indexed_material_count=len(indexed_material_ids_by_collection.get(collection.id, set())),
                retrievable_chunk_count=chunk_count_by_collection.get(collection.id, 0),
                latest_material_name=col_materials[0]["original_name"] if col_materials else None,
                created_at=collection.created_at,
                updated_at=collection.updated_at,
            )
        )

    return APIResponse(success=True, message="Collections loaded successfully", data=summaries, error=None)


@router.delete("/{collection_id}", response_model=APIResponse[dict])
async def delete_collection(
    request: Request,
    collection_id: str,
    owner_id: str = Query(..., min_length=1),
    service: MaterialService = Depends(get_material_service),
) -> APIResponse[dict]:
    verify_owner_access(request, owner_id)
    try:
        counts = await service.delete_collection(collection_id=collection_id, owner_id=owner_id)
    except LookupError as exc:
        return APIResponse(success=False, message=str(exc), data=None, error=str(exc))
    except ValueError as exc:
        return APIResponse(success=False, message=str(exc), data=None, error=str(exc))
    return APIResponse(success=True, message="Collection deleted successfully", data=counts, error=None)
