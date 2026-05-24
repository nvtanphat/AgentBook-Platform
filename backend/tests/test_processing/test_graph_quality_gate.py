from __future__ import annotations

import pytest

from src.processing.graph_quality_gate import GraphQualityGate
from src.processing.types import EvidenceBlock, ExtractedEntity, ExtractedRelation


def test_prune_entities_removes_low_confidence():
    """Test that low-confidence entities are removed."""
    gate = GraphQualityGate(min_entity_confidence=0.6)

    entities = [
        ExtractedEntity(
            canonical_name="HighConf",
            entity_type="concept",
            confidence=0.8,
            mention_refs=[_dummy_block()],
        ),
        ExtractedEntity(
            canonical_name="LowConf",
            entity_type="concept",
            confidence=0.4,
            mention_refs=[_dummy_block()],
        ),
    ]

    pruned = gate.prune_entities(entities)

    assert len(pruned) == 1
    assert pruned[0].canonical_name == "HighConf"


def test_prune_entities_requires_minimum_mentions():
    """Test that entities need minimum mention count."""
    gate = GraphQualityGate(min_mention_count=2)

    entities = [
        ExtractedEntity(
            canonical_name="ManyMentions",
            entity_type="concept",
            confidence=0.8,
            mention_refs=[_dummy_block(), _dummy_block()],
        ),
        ExtractedEntity(
            canonical_name="OneMention",
            entity_type="concept",
            confidence=0.8,
            mention_refs=[_dummy_block()],
        ),
    ]

    pruned = gate.prune_entities(entities)

    assert len(pruned) == 1
    assert pruned[0].canonical_name == "ManyMentions"


def test_resolve_entities_merges_similar_names():
    """Test that similar entities are merged."""
    gate = GraphQualityGate()

    entities = [
        ExtractedEntity(
            canonical_name="Dropout",
            entity_type="method",
            confidence=0.8,
            mention_refs=[_dummy_block()],
        ),
        ExtractedEntity(
            canonical_name="dropout",
            entity_type="method",
            confidence=0.7,
            mention_refs=[_dummy_block()],
        ),
        ExtractedEntity(
            canonical_name="Drop-out",
            entity_type="method",
            confidence=0.6,
            mention_refs=[_dummy_block()],
        ),
    ]

    resolved = gate.resolve_entities(entities)

    # Should merge "Dropout" and "dropout" (same normalized form)
    # "Drop-out" normalizes differently due to hyphen
    assert len(resolved) <= 2  # At most 2 groups

    # Find the highest confidence entity
    dropout_entity = next((e for e in resolved if e.canonical_name == "Dropout"), None)
    assert dropout_entity is not None

    # Should have merged at least one other entity
    assert len(dropout_entity.mention_refs) >= 2


def test_prune_relations_removes_orphans():
    """Test that relations to non-existent entities are removed."""
    gate = GraphQualityGate()

    valid_entity_ids = {"entity:dropout", "entity:regularization"}

    relations = [
        ExtractedRelation(
            source_id="entity:dropout",
            target_id="entity:regularization",
            relation_type="is_a",
            confidence=0.8,
            evidence_refs=[_dummy_block()],
        ),
        ExtractedRelation(
            source_id="entity:dropout",
            target_id="entity:nonexistent",
            relation_type="uses",
            confidence=0.8,
            evidence_refs=[_dummy_block()],
        ),
    ]

    pruned = gate.prune_relations(relations, valid_entity_ids)

    assert len(pruned) == 1
    assert pruned[0].target_id == "entity:regularization"


def test_prune_relations_removes_low_confidence():
    """Test that low-confidence relations are removed."""
    gate = GraphQualityGate(min_relation_confidence=0.6)

    valid_entity_ids = {"entity:a", "entity:b"}

    relations = [
        ExtractedRelation(
            source_id="entity:a",
            target_id="entity:b",
            relation_type="related",
            confidence=0.8,
            evidence_refs=[_dummy_block()],
        ),
        ExtractedRelation(
            source_id="entity:a",
            target_id="entity:b",
            relation_type="weak",
            confidence=0.4,
            evidence_refs=[_dummy_block()],
        ),
    ]

    pruned = gate.prune_relations(relations, valid_entity_ids)

    assert len(pruned) == 1
    assert pruned[0].relation_type == "related"


_dummy_block_counter = 0


def _dummy_block() -> EvidenceBlock:
    """Create a dummy evidence block for testing with unique block_id."""
    global _dummy_block_counter
    _dummy_block_counter += 1
    return EvidenceBlock(
        owner_id="user1",
        collection_id="col1",
        material_id="mat1",
        document_name="test.pdf",
        page=1,
        block_id=f"blk{_dummy_block_counter}",
        block_type="paragraph",
        snippet_original="Test content",
        source_language="en",
    )
