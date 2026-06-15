from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.tools import debug_generation_guardrails as dgg


def test_debug_generation_guardrails_saves_prompt_and_drops_unsupported_sentence(tmp_path: Path) -> None:
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
        answer_language="vi",
        owner_id="debug_owner",
        collection_id="65f000000000000000000080",
        material_id="65f000000000000000000081",
        document_name=None,
        preferred_modality="table",
        top_k=5,
        retrieval_limit=20,
        dense_size=8,
        batch_size=16,
        collection_name="debug_generation_guardrails_test",
        memory_context="",
        max_content_chars=120,
        disable_modality_extra_pass=False,
        disable_modality_routing=False,
        inject_unsupported=True,
    )

    report = asyncio.run(dgg._run(args))
    query_report = report["queries"][0]
    audit = query_report["citation_audit"]

    assert query_report["prompt_file"] == "qa_table.txt"
    assert "<EVIDENCE id=\"1\"" in query_report["prompt"]
    assert query_report["raw_llm_answer"]
    assert query_report["answer_language_ok"] is True
    assert query_report["answer_uses_evidence"] is False
    assert audit["unsupported_claims"]
    assert audit["dropped_count"] == 1
    assert "2030" not in audit["final_answer"]
    assert audit["invalid_citations"] == []
    assert audit["citations"][0]["evidence_block_ids"]
