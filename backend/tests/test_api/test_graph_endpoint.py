from __future__ import annotations

from types import SimpleNamespace

import pytest
from beanie import PydanticObjectId

from src.api.v1.endpoints import graph as graph_endpoint
from src.models.knowledge_graph import EvidenceRef


@pytest.mark.asyncio
async def test_graph_hydrates_block_nodes_from_relation_evidence(monkeypatch) -> None:
    material_id = PydanticObjectId()
    relation = SimpleNamespace(
        source_id="entity:dropout",
        target_id="block:blk-table",
        evidence_refs=[EvidenceRef(material_id=material_id, page=4, block_id="blk-table")],
    )
    material = SimpleNamespace(
        pages=[
            SimpleNamespace(
                page_number=4,
                blocks=[
                    SimpleNamespace(
                        block_id="blk-table",
                        block_type="table",
                        content="| model | accuracy |\n| --- | --- |\n| Dropout | 0.92 |",
                        ocr_confidence=None,
                    )
                ],
            )
        ],
        original_name="paper.pdf",
    )

    class FakeQuery:
        async def to_list(self):
            return [material]

    monkeypatch.setattr(graph_endpoint.Material, "find", lambda *args, **kwargs: FakeQuery())

    nodes = await graph_endpoint._block_nodes_from_relations([relation])

    assert len(nodes) == 1
    assert nodes[0].id == "block:blk-table"
    assert nodes[0].type == "table"
    assert nodes[0].label.startswith("p.4: | model | accuracy |")


def test_graph_fallback_node_keeps_unhydrated_relation_visible() -> None:
    node = graph_endpoint._fallback_node("event:dropout-improved-accuracy")

    assert node.id == "event:dropout-improved-accuracy"
    assert node.type == "event"
    assert node.label == "dropout improved accuracy"


def test_graph_builds_entity_cooccurrence_edges_when_relations_are_missing() -> None:
    material_id = PydanticObjectId()
    collection_id = PydanticObjectId()
    refs = [EvidenceRef(material_id=material_id, page=2, block_id="blk-1")]
    entities = [
        SimpleNamespace(
            canonical_name="Dropout",
            entity_type="method",
            mention_refs=refs,
            confidence=0.72,
        ),
        SimpleNamespace(
            canonical_name="accuracy",
            entity_type="metric",
            mention_refs=refs,
            confidence=0.68,
        ),
    ]

    edges = graph_endpoint._entity_cooccurrence_edges(entities)

    assert len(edges) == 1
    assert edges[0].source == "entity:dropout"
    assert edges[0].target == "entity:accuracy"
    assert edges[0].relation_type == "co_occurs_in_block"
