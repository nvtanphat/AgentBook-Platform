from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.tools import debug_reranking as dr


def test_debug_reranking_reports_before_after_gold_rank_and_latency(tmp_path: Path) -> None:
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
        query_variant=["gia cao nhat"],
        include_query_as_variant=True,
        language="vi",
        owner_id="debug_owner",
        collection_id="65f000000000000000000060",
        material_id="65f000000000000000000061",
        document_name=None,
        gold_block_id=[],
        gold_material_id=["65f000000000000000000061"],
        preferred_modality="table",
        top_k=10,
        retrieval_limit=20,
        dense_size=8,
        batch_size=16,
        collection_name="debug_reranking_test",
        max_content_chars=120,
        disable_modality_extra_pass=False,
        disable_modality_routing=False,
        use_mmr=False,
        production_reranker=False,
    )

    report = asyncio.run(dr._run(args))
    query_report = report["queries"][0]

    assert report["query_count"] == 1
    assert query_report["before_rerank_top_k"]
    assert query_report["after_rerank_top_k"]
    assert query_report["rerank_latency_ms"] >= 0
    assert query_report["gold_rank_before"] is not None
    assert query_report["gold_rank_after"] is not None
    assert query_report["gold_rerank_score"] is not None
    assert query_report["recommendation"]["action"] in {
        "ok",
        "increase_final_top_k_or_raise_rerank_input_k",
        "inspect_rerank_threshold_or_query_variants",
    }
