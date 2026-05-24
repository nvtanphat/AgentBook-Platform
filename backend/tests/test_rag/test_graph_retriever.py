from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


# ── retrieve_subgraph integration tests ───────────────────────────────────────

def _make_entity(canonical_name: str, chunk_ids: list[str], col_id: PydanticObjectId) -> SimpleNamespace:
    slug = GraphRetriever._slug(canonical_name)
    return SimpleNamespace(
        id=PydanticObjectId(),
        canonical_name=canonical_name,
        aliases=[],
        chunk_ids=chunk_ids,
        entity_type="concept",
        confidence=0.9,
        mention_refs=[],
        _slug=slug,
    )


def _make_relation(source_slug: str, target_slug: str, rel_type: str, evidence_chunk_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        id=PydanticObjectId(),
        source_id=f"entity:{source_slug}",
        target_id=f"entity:{target_slug}",
        relation_type=rel_type,
        confidence=0.88,
        evidence_refs=[],
        evidence_chunk_ids=evidence_chunk_ids,
    )


class _FakeRelationFind:
    def __init__(self, relations):
        self._relations = relations

    def sort(self, *args):
        return self

    def limit(self, *args):
        return self

    async def to_list(self):
        return self._relations


@pytest.mark.asyncio
async def test_retrieve_subgraph_returns_correct_path(monkeypatch):
    """Index 2 entities + 1 relation, then verify retrieve_subgraph returns the connecting path."""
    col_id = PydanticObjectId()
    entity_a = _make_entity("Dropout", ["chunk-1", "chunk-2"], col_id)
    entity_b = _make_entity("Overfitting", ["chunk-3"], col_id)
    slug_a = GraphRetriever._slug("Dropout")
    slug_b = GraphRetriever._slug("Overfitting")
    relation = _make_relation(slug_a, slug_b, "reduces", ["chunk-rel-1"])

    # Patch keyword matching to return our two entities
    async def _fake_keyword_match(self_inner, *, query, scope):
        return [entity_a, entity_b]

    monkeypatch.setattr(GraphRetriever, "_keyword_matching_entities", _fake_keyword_match)
    monkeypatch.setattr(
        "src.rag.graph_retriever.Relation.find",
        lambda *a, **kw: _FakeRelationFind([relation]),
    )
    # Skip evidence hydration (no real DB)
    monkeypatch.setattr(
        GraphRetriever,
        "_hydrate_evidence_refs",
        AsyncMock(return_value=[]),
    )

    retriever = GraphRetriever(Settings(testing=True))
    scope = RetrievalScope(owner_id="u1", collection_id=str(col_id))
    paths = await retriever.retrieve_subgraph("Dropout overfitting", scope, top_k=5)

    assert len(paths) == 1
    path = paths[0]
    assert path.path[0] == f"entity:{slug_a}"
    assert path.path[1] == "relation:reduces"
    assert path.path[2] == f"entity:{slug_b}"
    assert path.confidence == pytest.approx(0.88)
    # chunk_ids from entity_a (matched source) + relation evidence
    assert "chunk-1" in path.source_chunk_ids
    assert "chunk-rel-1" in path.source_chunk_ids


@pytest.mark.asyncio
async def test_retrieve_subgraph_returns_empty_when_no_entities(monkeypatch):
    """When no entities match the query, retrieve_subgraph must return []."""
    col_id = PydanticObjectId()

    async def _no_entities(self_inner, *, query, scope):
        return []

    monkeypatch.setattr(GraphRetriever, "_keyword_matching_entities", _no_entities)

    retriever = GraphRetriever(Settings(testing=True))
    scope = RetrievalScope(owner_id="u1", collection_id=str(col_id))
    paths = await retriever.retrieve_subgraph("unknown term xyz", scope)

    assert paths == []


@pytest.mark.asyncio
async def test_retrieve_subgraph_scopes_by_owner_and_collection(monkeypatch):
    """Verify that the Relation.find call always carries owner_id + collection_id."""
    col_id = PydanticObjectId()
    entity_a = _make_entity("Attention", ["chunk-10"], col_id)
    captured_filter: dict = {}

    async def _fake_keyword_match(self_inner, *, query, scope):
        return [entity_a]

    class _CapturingRelationFind:
        def __init__(self, filt):
            captured_filter.update(filt)

        def sort(self, *args):
            return self

        def limit(self, *args):
            return self

        async def to_list(self):
            return []

    monkeypatch.setattr(GraphRetriever, "_keyword_matching_entities", _fake_keyword_match)
    monkeypatch.setattr("src.rag.graph_retriever.Relation.find", _CapturingRelationFind)

    retriever = GraphRetriever(Settings(testing=True))
    scope = RetrievalScope(owner_id="owner-xyz", collection_id=str(col_id))
    await retriever.retrieve_subgraph("attention mechanism", scope)

    assert captured_filter.get("owner_id") == "owner-xyz"
    assert captured_filter.get("collection_id") == PydanticObjectId(str(col_id))


# ── _wire_chunk_graph_links unit tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_wire_chunk_graph_links_populates_entity_chunk_ids(monkeypatch):
    """After wire-up, entity.chunk_ids must contain the ID of the matching chunk."""
    from src.rag.indexer import QdrantMongoIndexer

    mat_id = PydanticObjectId()
    entity_id = PydanticObjectId()

    entity_doc = SimpleNamespace(
        id=entity_id,
        chunk_ids=[],
        mention_refs=[SimpleNamespace(material_id=mat_id, block_id="blk-42")],
    )

    chunk_id = PydanticObjectId()
    chunk_doc = SimpleNamespace(
        id=chunk_id,
        material_id=mat_id,
        source_block_ids=["blk-42", "blk-99"],
    )

    update_calls: list[dict] = []

    class _FakeChunkFindList:
        def __init__(self, _filt):
            pass
        async def to_list(self):
            return [chunk_doc]

    class _FakeChunkFindUpdate:
        def __init__(self, _filt):
            pass
        async def update_many(self, upd):
            update_calls.append(upd)

    class _FakeEntityFindUpdate:
        def __init__(self, _filt):
            pass
        async def update_many(self, upd):
            entity_doc.chunk_ids = upd.get("$set", {}).get("chunk_ids", entity_doc.chunk_ids)

    def _chunk_find(filt):
        if "_id" in filt:
            return _FakeChunkFindUpdate(filt)
        return _FakeChunkFindList(filt)

    monkeypatch.setattr("src.rag.indexer.Chunk.find", _chunk_find)
    monkeypatch.setattr("src.rag.indexer.Entity.find", lambda filt: _FakeEntityFindUpdate(filt))
    monkeypatch.setattr("src.rag.indexer.Relation.find", lambda filt: _FakeEntityFindUpdate(filt))

    # _wire_chunk_graph_links doesn't use self — pass a minimal stub
    indexer = SimpleNamespace()
    await QdrantMongoIndexer._wire_chunk_graph_links(
        indexer,
        entity_docs=[entity_doc],
        relation_docs=[],
    )

    assert str(chunk_id) in entity_doc.chunk_ids, "entity.chunk_ids must be populated after wire-up"
    assert any("entity_ids" in str(call) for call in update_calls), "Chunk.entity_ids must be updated via $addToSet"
