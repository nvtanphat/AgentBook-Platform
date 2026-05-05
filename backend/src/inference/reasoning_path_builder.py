"""
Helper để build reasoning path cho transparency.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.types import RetrievedChunk
    from src.schemas.query import ReasoningStep

logger = logging.getLogger(__name__)


def build_reasoning_path(
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

    # Step 2: Graph traversal
    if use_graph and graph_chunks:
        # Extract entities from graph chunks
        entities = []
        relations = []

        for chunk in graph_chunks[:5]:
            # Try to extract entity names from chunk metadata
            if hasattr(chunk, 'metadata') and chunk.metadata:
                if 'entity_label' in chunk.metadata:
                    entities.append(chunk.metadata['entity_label'])
                if 'relation_type' in chunk.metadata:
                    relations.append(chunk.metadata['relation_type'])

        # Deduplicate
        entities = list(dict.fromkeys(entities))[:5]
        relations = list(dict.fromkeys(relations))[:3]

        avg_graph_score = sum(c.fused_score or 0.0 for c in graph_chunks[:5]) / min(5, len(graph_chunks))

        if entities:
            entity_str = ", ".join(entities)
            relation_str = f" via {', '.join(relations)}" if relations else ""
            steps.append(ReasoningStep(
                step_type="traverse",
                entities=entities,
                relations=relations,
                confidence=avg_graph_score,
                description=f"Traversed knowledge graph: {entity_str}{relation_str}"
            ))

    # Step 3: Reranking & synthesis
    if reranked_chunks:
        top_rerank_score = reranked_chunks[0].rerank_score if reranked_chunks[0].rerank_score else 0.0

        # Extract key concepts from top chunks
        key_concepts = []
        for chunk in reranked_chunks[:3]:
            # Simple heuristic: extract capitalized phrases
            words = chunk.content.split()[:50]  # First 50 words
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 3:
                    key_concepts.append(word)

        key_concepts = list(dict.fromkeys(key_concepts))[:5]

        steps.append(ReasoningStep(
            step_type="synthesize",
            entities=key_concepts,
            relations=[],
            confidence=top_rerank_score,
            description=f"Synthesized answer from top {len(reranked_chunks)} most relevant chunks"
        ))

    return steps
