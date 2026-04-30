from __future__ import annotations

import asyncio
from types import SimpleNamespace

from beanie import PydanticObjectId

from src.core.config import Settings
from src.models.knowledge_graph import EvidenceRef
from src.rag.graph_retriever import GraphRetriever
from src.rag.types import RetrievalScope


def test_graph_retriever_scope_query_uses_collection_or_material_filter() -> None:
    retriever = GraphRetriever(Settings(testing=True))
    collection_scope = RetrievalScope(owner_id="user_demo", collection_id="65f000000000000000000002")
    material_scope = RetrievalScope(owner_id="user_demo", material_ids=["65f000000000000000000001"])

    collection_query = retriever._scope_query(collection_scope, {})
    material_query = retriever._scope_query(material_scope, {})

    assert collection_query["owner_id"] == "user_demo"
    assert collection_query["collection_id"] == PydanticObjectId("65f000000000000000000002")
    assert material_query["owner_id"] == "user_demo"
    assert material_query["evidence_refs.material_id"]["$in"] == [PydanticObjectId("65f000000000000000000001")]


def test_graph_retriever_hydrates_evidence_refs_with_batched_pages(monkeypatch) -> None:
    material_id = PydanticObjectId()
    collection_id = PydanticObjectId()
    material = SimpleNamespace(
        id=material_id,
        owner_id="user_demo",
        collection_id=collection_id,
        original_name="graph.pdf",
    )
    page = SimpleNamespace(
        page_number=1,
        blocks=[
            SimpleNamespace(
                block_id=f"blk-{index}",
                block_type="paragraph",
                content=f"Evidence {index}",
                language="en",
                bbox=None,
                ocr_confidence=0.9,
                extra={},
            )
            for index in range(50)
        ],
    )
    calls = {"pages": 0}

    class FakeMaterialQuery:
        async def to_list(self):
            return [material]

    async def fake_get_pages(materials):
        calls["pages"] += 1
        assert len(materials) == 1
        return {str(material_id): [page]}

    monkeypatch.setattr("src.rag.graph_retriever.Material.find", lambda *args, **kwargs: FakeMaterialQuery())
    monkeypatch.setattr("src.rag.graph_retriever.get_material_pages_by_material_ids", fake_get_pages)

    refs = [EvidenceRef(material_id=material_id, page=1, block_id=f"blk-{index}") for index in range(50)]
    evidence = asyncio.run(GraphRetriever(Settings(testing=True))._hydrate_evidence_refs(refs))

    assert len(evidence) == 50
    assert calls["pages"] == 1
