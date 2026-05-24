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
from beanie.operators import In

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk
    from src.schemas.query import ReasoningStep

logger = logging.getLogger(__name__)

# Entity types that belong in the mindmap / reasoning trace. Drops authors,
# people, organisations, locations, plus the docling "wrapper" types (table,
# figure, equation) that mention the document layout, not domain content.
_RAG_RELEVANT_ENTITY_TYPES = frozenset({
    "model", "algorithm", "concept", "dataset",
    "framework", "metric", "field", "method", "technology",
})


async def _resolve_canonical_entities(
    chunks: list["RetrievedChunk"],
    *,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Pull canonical entity labels for the given chunks from MongoDB.

    Joins via Entity.mention_refs[].block_id ∈ {chunk.source_block_ids}.
    Returns (display_labels, slug_ids) — both deduplicated, capped at `limit`,
    ranked by confidence × mention_count. Empty tuple on any failure (caller
    falls back to the document-derived heuristic).
    """
    if not chunks:
        return [], []

    # Collect scope + block ids — all chunks share owner/collection in practice.
    owner_id = chunks[0].owner_id
    collection_id = chunks[0].collection_id
    block_ids: set[str] = set()
    for chunk in chunks:
        block_ids.update(chunk.source_block_ids or [])
    if not block_ids:
        return [], []

    try:
        from src.models.knowledge_graph import Entity

        coll_oid: PydanticObjectId | str = collection_id
        try:
            coll_oid = PydanticObjectId(collection_id)
        except Exception:
            pass

        entities = await Entity.find({
            "owner_id": owner_id,
            "collection_id": coll_oid,
            "mention_refs.block_id": {"$in": list(block_ids)},
            "entity_type": {"$in": list(_RAG_RELEVANT_ENTITY_TYPES)},
        }).to_list()
    except Exception as exc:
        logger.debug(
            "Entity lookup for reasoning path failed",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        return [], []

    if not entities:
        return [], []

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
    retrieved_chunks: list["RetrievedChunk"],
    graph_chunks: list["RetrievedChunk"],
    reranked_chunks: list["RetrievedChunk"],
    use_graph: bool,
) -> list["ReasoningStep"]:
    """
    Build reasoning path showing how AI found the answer.

    Returns list of steps with entities, relations, and confidence.
    Async because the synthesis step pulls canonical entities from MongoDB.
    """
    from src.schemas.query import ReasoningStep

    steps: list[ReasoningStep] = []

    # Step 1: Retrieval
    if retrieved_chunks:
        top_docs = list({c.document_name for c in retrieved_chunks[:5]})
        avg_score = sum(c.fused_score or 0.0 for c in retrieved_chunks[:5]) / min(5, len(retrieved_chunks))

        steps.append(ReasoningStep(
            step_type="retrieve",
            entities=[],
            relations=[],
            confidence=avg_score,
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

        avg_graph_score = sum(c.fused_score or 0.0 for c in graph_chunks[:5]) / min(5, len(graph_chunks))
        entity_str = ", ".join(canonical_labels) if canonical_labels else f"{len(graph_chunks)} graph evidence chunks"
        relation_str = f" via {', '.join(relations)}" if relations else ""
        steps.append(ReasoningStep(
            step_type="traverse",
            entities=canonical_labels,
            relations=relations,
            entity_ids=canonical_ids,
            confidence=avg_graph_score,
            description=f"Traversed knowledge graph: {entity_str}{relation_str}"
        ))

    # Step 3: Reranking & synthesis — canonical entities from the DB.
    if reranked_chunks:
        top_rerank_score = reranked_chunks[0].rerank_score if reranked_chunks[0].rerank_score else 0.0
        key_concepts, key_concept_ids = await _resolve_canonical_entities(
            reranked_chunks[:5], limit=5,
        )

        steps.append(ReasoningStep(
            step_type="synthesize",
            entities=key_concepts,
            entity_ids=key_concept_ids,
            relations=[],
            confidence=top_rerank_score,
            description=f"Synthesized answer from top {len(reranked_chunks)} most relevant chunks",
        ))

    return steps
