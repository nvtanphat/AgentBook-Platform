"""
Helper để build reasoning path cho transparency.

Entities surfaced in the synthesis step are pulled from the **canonical
Entity DB** (built by the LLM-driven EntityResolver during indexing) and
filtered to RAG-relevant types (model/algorithm/concept/dataset/...).

Earlier versions relied on a regex heuristic over chunk text which over-picked
Vietnamese sentence-starts and truncated names (Hưng., Trọng, Mạng) and
required hardcoded VN/EN stoplists that didn't scale across domains. The
DB-driven path uses the same entities that already power the mindmap, so the
reasoning step's tags now match what users see in the knowledge graph.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from beanie import PydanticObjectId

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk
    from src.schemas.query import ReasoningStep

logger = logging.getLogger(__name__)

# Document-layout structural types — never surfaced in the reasoning trace
# because they name where content sits (table, figure) not what it means.
# All other entity types (concept, organization, person, event, artifact,
# quantity, time, metric, model, …) are domain-content and should appear.
_LAYOUT_ENTITY_TYPES = frozenset({
    "table", "figure", "equation", "section", "page", "image", "caption",
})


async def _resolve_canonical_entities(
    chunks: list["RetrievedChunk"],
    *,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Pull canonical entity labels for the given chunks from MongoDB.

    Joins via Entity.mention_refs[].block_id ∈ {chunk.source_block_ids} AND
    Entity.mention_refs[].material_id ∈ {chunk.material_id} to prevent entities
    from other materials in the same collection from leaking into the trace.
    Returns (display_labels, slug_ids) — deduplicated, capped at `limit`,
    ranked by confidence × mention_count. Empty tuple on any failure.
    """
    if not chunks:
        return [], []

    # Collect scope + block ids + material ids — all chunks share owner/collection.
    owner_id = chunks[0].owner_id
    collection_id = chunks[0].collection_id

    # Only use text-modality chunks for entity lookup. Image/figure chunks produce
    # VLM-captioned entities that may be hallucinated and unrelated to the
    # document's domain content. If no text chunks exist, return empty rather
    # than surfacing noisy image-derived entities in the reasoning trace.
    text_chunks = [c for c in chunks if getattr(c, "modality", "text") == "text"]
    if not text_chunks:
        return [], []
    lookup_chunks = text_chunks

    block_ids: set[str] = set()
    material_id_strs: set[str] = set()
    for chunk in lookup_chunks:
        block_ids.update(chunk.source_block_ids or [])
        if chunk.material_id:
            material_id_strs.add(chunk.material_id)
    if not block_ids:
        return [], []

    try:
        from src.models.knowledge_graph import Entity

        coll_oid: PydanticObjectId | str = collection_id
        try:
            coll_oid = PydanticObjectId(collection_id)
        except Exception:
            pass

        # Convert material_ids to PydanticObjectId; keep str fallback on failure.
        mat_oids: list = []
        for mid in material_id_strs:
            try:
                mat_oids.append(PydanticObjectId(mid))
            except Exception:
                mat_oids.append(mid)

        # Use $elemMatch so block_id AND material_id must match the *same*
        # mention_ref element — prevents entities from other materials leaking
        # in via a shared material_id with a mismatched block_id.
        elem_filter: dict = {"block_id": {"$in": list(block_ids)}}
        if mat_oids:
            elem_filter["material_id"] = {"$in": mat_oids}
        query: dict = {
            "owner_id": owner_id,
            "collection_id": coll_oid,
            "mention_refs": {"$elemMatch": elem_filter},
            "entity_type": {"$nin": list(_LAYOUT_ENTITY_TYPES)},
            # Require reasonably high confidence to keep OCR/VLM noise out of
            # the reasoning trace — junk entities tend to score 0.50-0.60.
            "confidence": {"$gte": 0.70},
        }

        entities = await Entity.find(query).to_list()
    except Exception as exc:
        logger.debug(
            "Entity lookup for reasoning path failed",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        return [], []

    if not entities:
        return [], []

    # Drop single-mention entities — they typically come from one-off OCR/VLM
    # noise in image captions rather than meaningful domain content. Entities
    # mentioned ≥ 2 times are far more likely to be real named entities.
    entities = [e for e in entities if len(getattr(e, "mention_refs", []) or []) >= 2]

    # Rank: confidence first, then mention count, drop duplicates by canonical_name.
    ranked = sorted(
        entities,
        key=lambda e: (
            float(getattr(e, "confidence", 0.0) or 0.0),
            len(getattr(e, "mention_refs", []) or []),
        ),
        reverse=True,
    )
    seen: set[str] = set()
    labels: list[str] = []
    slug_ids: list[str] = []
    for ent in ranked:
        name = (getattr(ent, "canonical_name", "") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        labels.append(name)
        # Match the slug convention used by graph endpoint (`entity:foo-bar`).
        slug = "entity:" + name.lower().replace(" ", "-").replace("/", "-")
        slug_ids.append(slug)
        if len(labels) >= limit:
            break
    return labels, slug_ids


async def build_reasoning_path(
    *,
    query: str,
    answer: str = "",
    retrieved_chunks: list["RetrievedChunk"],
    graph_chunks: list["RetrievedChunk"],
    reranked_chunks: list["RetrievedChunk"],
    use_graph: bool,
) -> list["ReasoningStep"]:
    """
    Build reasoning path showing how AI found the answer.

    Returns list of steps with entities, relations, and confidence.
    Async because the synthesis step pulls canonical entities from MongoDB.
    `answer` is used to validate synthesize-step entities so only names that
    actually appear in the response are surfaced (filters VLM/OCR noise).
    """
    from src.schemas.query import ReasoningStep

    del query  # reserved for future query-aware entity ranking; callers must keep passing it
    answer_lower = answer.lower()
    steps: list[ReasoningStep] = []

    # Step 1: Retrieval
    if retrieved_chunks:
        top_docs = list({c.document_name for c in retrieved_chunks[:5]})
        avg_score = sum(c.fused_score or 0.0 for c in retrieved_chunks[:5]) / min(5, len(retrieved_chunks))

        steps.append(ReasoningStep(
            step_type="retrieve",
            entities=[],
            relations=[],
            confidence=min(1.0, max(0.0, avg_score)),
            description=f"Retrieved {len(retrieved_chunks)} relevant chunks from {len(top_docs)} documents: {', '.join(top_docs[:3])}"
        ))

    # Step 2: Graph traversal — prefer canonical entity names from the DB
    # over chunk metadata, which often carries ascii-folded slugs/heading
    # fragments unreadable in the trace UI.
    if use_graph and graph_chunks:
        canonical_labels, canonical_ids = await _resolve_canonical_entities(
            graph_chunks[:5], limit=5,
        )
        relations: list[str] = []
        for chunk in graph_chunks[:5]:
            metadata = getattr(chunk, "metadata", {}) or {}
            relations.extend(str(item) for item in metadata.get("relation_types", []) if item)

        # Fall back to metadata entity_labels only when DB lookup returned nothing.
        if not canonical_labels:
            for chunk in graph_chunks[:5]:
                metadata = getattr(chunk, "metadata", {}) or {}
                canonical_labels.extend(str(item) for item in metadata.get("entity_labels", []) if item)
                canonical_ids.extend(str(item) for item in metadata.get("entity_ids", []) if item)
            canonical_labels = list(dict.fromkeys(canonical_labels))[:5]
            canonical_ids = list(dict.fromkeys(canonical_ids))[:8]
        relations = list(dict.fromkeys(relations))[:3]

        # Same answer-validation as synthesize step — filter cross-collection noise.
        if answer_lower and canonical_labels:
            filtered_t = [
                (n, s) for n, s in zip(canonical_labels, canonical_ids)
                if n.lower() in answer_lower
            ]
            canonical_labels = [p[0] for p in filtered_t]
            canonical_ids = [p[1] for p in filtered_t]

        avg_graph_score = sum(c.fused_score or 0.0 for c in graph_chunks[:5]) / min(5, len(graph_chunks))
        entity_str = ", ".join(canonical_labels) if canonical_labels else f"{len(graph_chunks)} graph evidence chunks"
        relation_str = f" via {', '.join(relations)}" if relations else ""
        steps.append(ReasoningStep(
            step_type="traverse",
            entities=canonical_labels,
            relations=relations,
            entity_ids=canonical_ids,
            confidence=min(1.0, max(0.0, avg_graph_score)),
            description=f"Traversed knowledge graph: {entity_str}{relation_str}"
        ))

    # Step 3: Reranking & synthesis — canonical entities from the DB.
    if reranked_chunks:
        top_rerank_score = reranked_chunks[0].rerank_score if reranked_chunks[0].rerank_score else 0.0
        key_concepts, key_concept_ids = await _resolve_canonical_entities(
            reranked_chunks[:5], limit=5,
        )

        # Validate against answer: only surface entities whose name actually
        # appears in the response. Filters VLM/OCR noise (e.g. "otter", "Vase")
        # that may be stored in the DB but are irrelevant to the answer.
        if answer_lower:
            filtered = [
                (name, sid) for name, sid in zip(key_concepts, key_concept_ids)
                if name.lower() in answer_lower
            ]
            key_concepts = [p[0] for p in filtered]
            key_concept_ids = [p[1] for p in filtered]

        steps.append(ReasoningStep(
            step_type="synthesize",
            entities=key_concepts,
            entity_ids=key_concept_ids,
            relations=[],
            confidence=min(1.0, max(0.0, top_rerank_score)),
            description=f"Synthesized answer from top {len(reranked_chunks)} most relevant chunks",
        ))

    return steps
