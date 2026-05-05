from __future__ import annotations

from src.core.config import Settings
from src.processing.chunking import LayoutAwareChunker
from src.processing.entity_extractor import EntityExtractor
from src.processing.event_extractor import EventExtractor
from src.processing.types import BBox, EvidenceBlock, EvidenceMap


def build_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="lecture.pdf",
        blocks=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=3,
                block_id="blk-001",
                block_type="heading",
                snippet_original="Regularization",
                source_language="en",
                bbox=BBox(x1=1, y1=2, x2=3, y2=4),
                confidence=0.98,
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="lecture.pdf",
                page=3,
                block_id="blk-002",
                block_type="paragraph",
                snippet_original="Dropout reduced validation error in 2014.",
                source_language="en",
                bbox=BBox(x1=5, y1=6, x2=7, y2=8),
                confidence=0.95,
            ),
        ],
    )


def test_chunker_preserves_evidence_trace_fields() -> None:
    settings = Settings(
        testing=True,
        chunk_target_token_count=128,
        chunk_overlap_token_count=0,
        chunk_max_blocks_per_chunk=4,
    )
    chunks = LayoutAwareChunker(settings).build_chunks(build_evidence_map())

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.owner_id == "user_demo"
    assert chunk.collection_id == "65f000000000000000000002"
    assert chunk.material_id == "65f000000000000000000001"
    assert chunk.source_pages == [3]
    assert chunk.source_block_ids == ["blk-001", "blk-002"]
    assert chunk.bboxes[0].x1 == 1
    assert chunk.evidence[1].snippet_original == "Dropout reduced validation error in 2014."


def test_chunker_groups_heading_runs_without_cumulative_overlap() -> None:
    """Test that heading-only chunks are grouped efficiently without unnecessary splits."""
    settings = Settings(
        testing=True,
        chunk_target_token_count=50,
        chunk_min_token_count=5,
        chunk_overlap_token_count=3,
        chunk_max_blocks_per_chunk=5,
    )
    evidence_map = EvidenceMap(
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="table-heavy.pdf",
        blocks=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="table-heavy.pdf",
                page=1,
                block_id=f"blk-{index:03d}",
                block_type="heading",
                snippet_original=f"H{index}",
                source_language="en",
            )
            for index in range(12)
        ],
    )

    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    # Heading-only chunks with small token count (12 tokens total) should be grouped into one chunk
    # because _is_heading_only() prevents premature splits
    assert len(chunks) == 1
    assert chunks[0].modality == "heading"
    # All 12 blocks should be preserved
    assert len(chunks[0].source_block_ids) == 12
    assert set(chunks[0].source_block_ids) == {f"blk-{i:03d}" for i in range(12)}


def test_chunker_splits_oversized_blocks_to_target_budget() -> None:
    settings = Settings(
        testing=True,
        chunk_target_token_count=10,
        chunk_min_token_count=3,
        chunk_overlap_token_count=2,
        chunk_max_blocks_per_chunk=4,
    )
    evidence_map = EvidenceMap(
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="long-block.docx",
        blocks=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="long-block.docx",
                page=1,
                block_id="blk-long",
                block_type="paragraph",
                snippet_original=" ".join(f"word{index}" for index in range(25)),
                source_language="en",
            )
        ],
    )

    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    assert [chunk.token_count for chunk in chunks] == [10, 10, 5]
    assert all(chunk.source_block_ids == ["blk-long"] for chunk in chunks)
    assert chunks[0].evidence[0].metadata["split_part_count"] == 3


def test_chunker_preserves_single_block_modality() -> None:
    settings = Settings(
        testing=True,
        chunk_target_token_count=128,
        chunk_overlap_token_count=0,
        chunk_max_blocks_per_chunk=4,
    )
    evidence_map = EvidenceMap(
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="table-heavy.pdf",
        blocks=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="table-heavy.pdf",
                page=1,
                block_id="blk-table",
                block_type="table",
                snippet_original="| model | accuracy |\n| --- | --- |\n| A | 0.92 |",
                source_language="en",
            )
        ],
    )

    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    assert chunks[0].modality == "table"


def test_merge_tiny_chunks_combines_small_sections() -> None:
    """Consecutive under-minimum chunks should be merged while respecting the token budget."""
    settings = Settings(
        testing=True,
        chunk_target_token_count=20,
        chunk_min_token_count=8,
        chunk_overlap_token_count=0,
        chunk_max_blocks_per_chunk=4,
    )
    # heading "A" (1 token) + body "x x x" (3 tokens) → 4-token chunk; too small
    # heading "B" (1 token) + body "y y y" (3 tokens) → 4-token chunk; too small
    # heading "C" + long paragraph filling budget → stand-alone chunk
    blocks = []
    for idx, (h, body) in enumerate([("A", "x x x"), ("B", "y y y"), ("C", " ".join(["z"] * 15))]):
        blocks.append(
            EvidenceBlock(
                owner_id="u", collection_id="c", material_id="m",
                document_name="test.pdf", page=idx + 1,
                block_id=f"h{idx}", block_type="heading",
                snippet_original=h, source_language="en",
            )
        )
        blocks.append(
            EvidenceBlock(
                owner_id="u", collection_id="c", material_id="m",
                document_name="test.pdf", page=idx + 1,
                block_id=f"p{idx}", block_type="paragraph",
                snippet_original=body, source_language="en",
            )
        )
    evidence_map = EvidenceMap(owner_id="u", collection_id="c", material_id="m", document_name="test.pdf", blocks=blocks)

    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    # Sections A and B (each 4 tokens) should merge into one chunk (8 tokens ≥ min, ≤ target)
    # Section C (16 tokens) is its own chunk
    assert len(chunks) == 2
    assert chunks[0].token_count == 8
    assert "h0" in chunks[0].source_block_ids and "h1" in chunks[0].source_block_ids
    assert chunks[1].token_count == 16


def test_baseline_extractors_link_graph_facts_to_evidence_refs() -> None:
    evidence_map = build_evidence_map()

    entities = EntityExtractor().extract(evidence_map)
    events, relations = EventExtractor().extract(evidence_map, entities)

    assert any(entity.canonical_name == "Dropout" for entity in entities)
    assert events
    assert events[0].evidence_refs[0].block_id == "blk-002"
    assert relations[0].evidence_refs[0].page == 3


def test_event_extractor_builds_cross_modal_block_relations() -> None:
    evidence_map = EvidenceMap(
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id="65f000000000000000000001",
        document_name="paper.pdf",
        blocks=[
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=1,
                block_id="blk-heading",
                block_type="heading",
                snippet_original="Dropout Results",
                source_language="en",
                metadata={"reading_order": 0},
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=1,
                block_id="blk-paragraph",
                block_type="paragraph",
                snippet_original="Dropout improves accuracy.",
                source_language="en",
                metadata={"reading_order": 1},
            ),
            EvidenceBlock(
                owner_id="user_demo",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name="paper.pdf",
                page=1,
                block_id="blk-table",
                block_type="table",
                snippet_original="| method | accuracy |\n| --- | --- |\n| Dropout | 0.92 |",
                source_language="en",
                metadata={"reading_order": 2},
            ),
        ],
    )

    entities = EntityExtractor().extract(evidence_map)
    _, relations = EventExtractor().extract(evidence_map, entities)
    relation_keys = {(relation.source_id, relation.relation_type, relation.target_id) for relation in relations}

    assert ("entity:dropout", "mentioned_in_block", "block:blk-paragraph") in relation_keys
    assert ("block:blk-heading", "section_contains", "block:blk-table") in relation_keys
    assert ("block:blk-paragraph", "adjacent_context", "block:blk-table") in relation_keys
