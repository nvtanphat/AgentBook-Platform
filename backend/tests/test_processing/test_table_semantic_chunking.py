from __future__ import annotations

from src.core.config import Settings
from src.processing.chunking import LayoutAwareChunker
from src.processing.table_serializer import to_html
from src.processing.types import EvidenceBlock, EvidenceMap


def test_table_html_is_rendered_as_semantic_chunk_text() -> None:
    table_html = to_html(
        ["model", "WMT EN-DE BLEU"],
        [["Transformer big", "41.8"], ["Transformer base", "27.3"]],
    )
    block = EvidenceBlock(
        owner_id="owner",
        collection_id="collection",
        material_id="material",
        document_name="attention.pdf",
        page=8,
        block_id="table-1",
        block_type="table",
        snippet_original=table_html,
        source_language="en",
        metadata={"block_kind": "table_block", "sheet_name": "Results", "row_start": 2},
    )
    evidence_map = EvidenceMap(
        owner_id="owner",
        collection_id="collection",
        material_id="material",
        document_name="attention.pdf",
        blocks=[block],
    )

    settings = Settings(testing=True, chunk_min_token_count=1, chunk_overlap_token_count=0)
    chunks = LayoutAwareChunker(settings).build_chunks(evidence_map)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.modality == "table"
    assert "| model | WMT EN-DE BLEU |" in chunk.content
    assert "Transformer big" in chunk.content
    assert "41.8" in chunk.content
    assert "Row 2 of table 'Results'. model: Transformer big. WMT EN-DE BLEU: 41.8." in chunk.content
    assert chunk.evidence[0].snippet_original == table_html
