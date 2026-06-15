from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.tools import debug_retrieval as dr


def test_debug_retrieval_reports_top_chunks_and_table_extra_pass(tmp_path: Path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "Ten san pham,Gia VND,So luong\n"
        "Laptop Dell XPS 13,28500000,32\n"
        "MacBook Air M3,30900000,24\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        file=csv_path,
        query=["Bảng products cột Gia VND sản phẩm nào cao nhất?"],
        language="vi",
        owner_id="debug_owner",
        collection_id="65f000000000000000000040",
        material_id="65f000000000000000000041",
        document_name=None,
        gold_block_id=[],
        gold_material_id=[],
        preferred_modality=None,
        top_k=10,
        retrieval_limit=20,
        dense_size=8,
        batch_size=16,
        collection_name="debug_retrieval_test",
        max_content_chars=120,
        disable_modality_extra_pass=False,
        disable_modality_routing=False,
    )

    report = asyncio.run(dr._run(args))
    query_report = report["queries"][0]

    assert report["indexed_chunk_count"] > 0
    assert query_report["preferred_modality"] == "table"
    assert query_report["modality_extra_pass_ran"] is True
    assert query_report["table_chunk_in_top_k"] is True
    assert query_report["top_chunks"]
    first = query_report["top_chunks"][0]
    assert {"doc", "page", "modality", "block_id", "score", "rerank_score", "content_preview"} <= set(first)
