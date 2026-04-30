from __future__ import annotations

import re

from beanie import PydanticObjectId
from fastapi import APIRouter, HTTPException, Request, status

from src.dependencies import verify_owner_access
from src.models.knowledge_graph import Entity, Event, Relation
from src.models.material import Material, get_material_pages
from src.schemas.common import APIResponse
from src.schemas.graph import GraphEdge, GraphNode, GraphResponse, MindmapRequest
from src.schemas.mindmap import MindmapNode, MindmapResponse

router = APIRouter(prefix="/graph", tags=["graph"])


def _entity_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
    return f"entity:{slug}"


def _event_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"
    return f"event:{slug}"


def _block_node_id(block_id: str | None) -> str:
    return f"block:{block_id or 'unknown'}"


def _short_label(text: str, *, limit: int = 90) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1].rstrip()}…"


def _scope_query(request: MindmapRequest) -> dict:
    if not request.collection_id and not request.material_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="collection_id or material_ids is required for scoped graph retrieval",
        )
    query = {"owner_id": request.owner_id}
    if request.collection_id:
        query["collection_id"] = PydanticObjectId(request.collection_id)
    return query


def _evidence_refs(refs) -> list[dict[str, str | int]]:
    return [
        {"material_id": str(ref.material_id), "page": ref.page or 0, "block_id": ref.block_id or ""}
        for ref in refs[:3]
    ]


async def _block_nodes_from_relations(relations: list[Relation]) -> list[GraphNode]:
    needed_block_ids: set[str] = set()
    material_ids: set[PydanticObjectId] = set()
    for relation in relations:
        for node_id in (relation.source_id, relation.target_id):
            if node_id.startswith("block:"):
                needed_block_ids.add(node_id.removeprefix("block:"))
        for ref in relation.evidence_refs:
            material_ids.add(ref.material_id)
            if ref.block_id:
                needed_block_ids.add(ref.block_id)

    if not needed_block_ids or not material_ids:
        return []

    materials = await Material.find({"_id": {"$in": list(material_ids)}}).to_list()
    nodes: dict[str, GraphNode] = {}
    for material in materials:
        for page in await get_material_pages(material):
            for block in page.blocks:
                if block.block_id not in needed_block_ids:
                    continue
                label = _short_label(block.content) or f"{material.original_name} p.{page.page_number}"
                nodes[_block_node_id(block.block_id)] = GraphNode(
                    id=_block_node_id(block.block_id),
                    label=f"p.{page.page_number}: {label}",
                    type=block.block_type,
                    confidence=block.ocr_confidence,
                )
    return list(nodes.values())


def _fallback_node(node_id: str) -> GraphNode:
    prefix, _, value = node_id.partition(":")
    node_type = prefix or "unknown"
    label = value.replace("-", " ") if value else node_id
    return GraphNode(id=node_id, label=_short_label(label), type=node_type, confidence=None)


def _entity_cooccurrence_edges(entities: list[Entity], *, limit: int = 120) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for index, source_entity in enumerate(entities):
        source_refs = source_entity.mention_refs
        for target_entity in entities[index + 1 :]:
            target_refs = target_entity.mention_refs
            shared_block = _shared_ref(source_refs, target_refs, require_same_block=True)
            shared_page = shared_block or _shared_ref(source_refs, target_refs, require_same_block=False)
            if shared_page is None:
                continue
            source_id = _entity_slug(source_entity.canonical_name)
            target_id = _entity_slug(target_entity.canonical_name)
            key = tuple(sorted((source_id, target_id)))
            if key in seen:
                continue
            seen.add(key)
            relation_type = "co_occurs_in_block" if shared_block is not None else "co_occurs_on_page"
            confidence = min(source_entity.confidence, target_entity.confidence, 0.58 if shared_block is not None else 0.52)
            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    relation_type=relation_type,
                    confidence=confidence,
                    evidence_refs=_evidence_refs([shared_page]),
                )
            )
            if len(edges) >= limit:
                return edges
    return edges


def _shared_ref(source_refs, target_refs, *, require_same_block: bool):
    target_keys = {
        (
            str(ref.material_id),
            ref.page,
            ref.block_id if require_same_block else None,
        )
        for ref in target_refs
    }
    for ref in source_refs:
        key = (str(ref.material_id), ref.page, ref.block_id if require_same_block else None)
        if key in target_keys:
            return ref
    return None


@router.post("", response_model=APIResponse[GraphResponse])
async def graph(request: Request, body: MindmapRequest) -> APIResponse[GraphResponse]:
    verify_owner_access(request, body.owner_id)
    query = _scope_query(body)
    entity_query = dict(query)
    relation_query = dict(query)
    if body.material_ids:
        material_ids = [PydanticObjectId(material_id) for material_id in body.material_ids]
        entity_query["mention_refs.material_id"] = {"$in": material_ids}
        relation_query["evidence_refs.material_id"] = {"$in": material_ids}

    entities = await Entity.find(entity_query).limit(100).to_list()
    events = await Event.find(relation_query).limit(100).to_list()
    relations = await Relation.find(relation_query).limit(200).to_list()

    nodes_by_id = {
        _entity_slug(entity.canonical_name): GraphNode(
            id=_entity_slug(entity.canonical_name),
            label=entity.canonical_name,
            type=entity.entity_type,
            confidence=entity.confidence,
        )
        for entity in entities
    }
    nodes_by_id.update(
        {
            _event_slug(event.event_name): GraphNode(
                id=_event_slug(event.event_name),
                label=_short_label(event.event_name),
                type="event",
                confidence=None,
            )
            for event in events
        }
    )
    nodes_by_id.update({node.id: node for node in await _block_nodes_from_relations(relations)})

    for relation in relations:
        nodes_by_id.setdefault(relation.source_id, _fallback_node(relation.source_id))
        nodes_by_id.setdefault(relation.target_id, _fallback_node(relation.target_id))

    nodes = list(nodes_by_id.values())
    node_ids = {node.id for node in nodes}
    edges = [
        GraphEdge(
            source=relation.source_id,
            target=relation.target_id,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence_refs=_evidence_refs(relation.evidence_refs),
        )
        for relation in relations
        if relation.source_id in node_ids and relation.target_id in node_ids
    ]
    if not edges:
        edges = _entity_cooccurrence_edges(entities)
    result = GraphResponse(nodes=nodes, edges=edges)
    return APIResponse(success=True, message="Graph loaded successfully", data=result, error=None)


@router.post("/mindmap", response_model=APIResponse[MindmapResponse])
async def mindmap(request: Request, body: MindmapRequest) -> APIResponse[MindmapResponse]:
    verify_owner_access(request, body.owner_id)
    query = _scope_query(body)
    if body.material_ids:
        query["mention_refs.material_id"] = {"$in": [PydanticObjectId(material_id) for material_id in body.material_ids]}
    entities = await Entity.find(query).limit(50).to_list()
    nodes = [
        MindmapNode(
            id=str(entity.id),
            label=entity.canonical_name,
            summary=f"{entity.entity_type} confidence={entity.confidence:.2f}",
            citations=_evidence_refs(entity.mention_refs),
        )
        for entity in entities
    ]
    result = MindmapResponse(root_topic=body.root_topic or "Prism Knowledge Map", nodes=nodes)
    return APIResponse(success=True, message="Mindmap generated successfully", data=result, error=None)
