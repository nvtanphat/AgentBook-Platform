"""Unit tests for the structural KG linker (belongs_to section hierarchy)."""
from __future__ import annotations

from src.processing.slug import slugify
from src.processing.structure_linker import StructureLinker
from src.processing.types import EvidenceBlock, EvidenceMap


def _block(
    *,
    block_id: str,
    block_type: str,
    text: str,
    reading_order: int,
    page: int = 1,
    label: str = "",
) -> EvidenceBlock:
    return EvidenceBlock(
        owner_id="user1",
        collection_id="col1",
        material_id="mat1",
        document_name="paper.pdf",
        page=page,
        block_id=block_id,
        block_type=block_type,
        snippet_original=text,
        source_language="en",
        metadata={"reading_order": reading_order, "label": label},
    )


def _map(blocks: list[EvidenceBlock]) -> EvidenceMap:
    return EvidenceMap(
        owner_id="user1",
        collection_id="col1",
        material_id="mat1",
        document_name="paper.pdf",
        blocks=blocks,
    )


def test_content_block_belongs_to_enclosing_section() -> None:
    blocks = [
        _block(block_id="h1", block_type="heading", text="Introduction", reading_order=0),
        _block(block_id="p1", block_type="paragraph", text="Some text.", reading_order=1),
        _block(block_id="p2", block_type="paragraph", text="More text.", reading_order=2),
    ]
    entities, relations = StructureLinker().link(_map(blocks))

    assert len(entities) == 1
    assert entities[0].entity_type == "section"
    sec_id = f"entity:{slugify('Introduction')}"
    belongs = {(r.source_id, r.target_id) for r in relations if r.relation_type == "belongs_to"}
    assert ("block:p1", sec_id) in belongs
    assert ("block:p2", sec_id) in belongs


def test_subsection_nests_under_parent_title() -> None:
    blocks = [
        _block(block_id="t1", block_type="heading", text="Methods", reading_order=0, label="title"),
        _block(block_id="h2", block_type="heading", text="Datasets", reading_order=1, label="section_header"),
        _block(block_id="p1", block_type="paragraph", text="We use X.", reading_order=2),
    ]
    _, relations = StructureLinker().link(_map(blocks))

    child = f"entity:{slugify('Datasets')}"
    parent = f"entity:{slugify('Methods')}"
    nesting = {(r.source_id, r.target_id) for r in relations if r.relation_type == "belongs_to"}
    # sub-section nested under the title, and the paragraph under the sub-section
    assert (child, parent) in nesting
    assert ("block:p1", child) in nesting


def test_content_before_any_heading_has_no_section() -> None:
    blocks = [
        _block(block_id="p0", block_type="paragraph", text="Orphan.", reading_order=0),
        _block(block_id="h1", block_type="heading", text="Body", reading_order=1),
        _block(block_id="p1", block_type="paragraph", text="Inside.", reading_order=2),
    ]
    _, relations = StructureLinker().link(_map(blocks))
    sources = {r.source_id for r in relations}
    assert "block:p0" not in sources
    assert "block:p1" in sources


def test_confidence_comes_from_config_not_hardcoded() -> None:
    blocks = [
        _block(block_id="h1", block_type="heading", text="Sec", reading_order=0),
        _block(block_id="p1", block_type="paragraph", text="x", reading_order=1),
    ]
    entities, relations = StructureLinker(
        section_confidence=0.42, belongs_to_confidence=0.37
    ).link(_map(blocks))
    assert entities[0].confidence == 0.42
    assert all(r.confidence == 0.37 for r in relations)


def test_disabled_linker_returns_nothing() -> None:
    blocks = [_block(block_id="h1", block_type="heading", text="Sec", reading_order=0)]
    entities, relations = StructureLinker(enabled=False).link(_map(blocks))
    assert entities == []
    assert relations == []
