from __future__ import annotations

from pathlib import Path

from backend.tools import debug_chunking as dc
from src.core.config import get_settings
from src.processing.evidence_mapper import EvidenceMapper


def test_debug_chunking_preserves_table_block_and_row_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "student,score\n"
        "An,9.5\n"
        "Binh,8.0\n",
        encoding="utf-8",
    )
    settings = get_settings()
    settings.testing = True

    parsed = dc._parse_file(csv_path, language="vi")
    evidence_map = EvidenceMapper().build(
        parsed=parsed,
        owner_id="owner",
        collection_id="collection",
        material_id="material",
        document_name=csv_path.name,
    )
    chunks = dc._chunker_for("csv", settings).build_chunks(evidence_map)
    block_to_chunks = dc._build_block_to_chunks(chunks)
    checks = dc._checks(parsed, chunks, block_to_chunks)

    assert checks["missing_non_empty_blocks"] == []
    assert checks["table"]["parsed_table_block_count"] == 1
    assert checks["table"]["parsed_table_row_count"] == 2
    assert checks["table"]["table_chunk_count"] == 1
    assert checks["table"]["table_blocks_isolated"] is True
    assert checks["table"]["table_rows_with_metadata"] == 2

    table_chunk = next(chunk for chunk in chunks if chunk.modality == "table")
    assert table_chunk.source_pages == [1]
    assert table_chunk.evidence[0].metadata["block_kind"] == "table_block"
    assert table_chunk.evidence[0].metadata["columns"] == ["student", "score"]

    row_evidence = [
        ev
        for chunk in chunks
        for ev in chunk.evidence
        if ev.metadata.get("block_kind") == "table_row"
    ]
    assert [ev.metadata["row_index"] for ev in row_evidence] == [2, 3]
