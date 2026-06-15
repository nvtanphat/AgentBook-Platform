from __future__ import annotations

from pathlib import Path

from backend.tools import debug_table_understanding as dtu


def test_debug_table_understanding_reports_cells_and_query_citations(tmp_path: Path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "Ten san pham,Gia VND,So luong\n"
        "Laptop Dell XPS 13,28500000,32\n"
        "MacBook Air M3,30900000,24\n",
        encoding="utf-8",
    )

    parsed = dtu._parse_document(csv_path, language="vi")
    tables = dtu._extract_tables(parsed)

    assert len(tables) == 1
    debug = dtu._debug_table(tables[0], max_rows=2)

    assert debug["header"] == ["Ten san pham", "Gia VND", "So luong"]
    assert debug["row_indices_preview"] == [2, 3]
    assert debug["html_grid_ok"] is True
    assert debug["verbalized_row_count"] == 2

    lookup = debug["query_tests"]["lookup"]
    assert lookup["cell_value"] == "28500000"
    assert lookup["citation"]["row_index"] == 2
    assert lookup["citation"]["column_name"] == "Gia VND"

    compare = debug["query_tests"]["compare_2_rows"]
    assert compare["a_value"] == "28500000"
    assert compare["b_value"] == "30900000"
    assert [c["row_index"] for c in compare["citations"]] == [2, 3]

    aggregations = {
        item["result"]["operation"]: item["result"]
        for item in debug["query_tests"]["aggregations"]
        if item["result"] is not None
    }
    assert aggregations["max"]["value"] == 30900000.0
    assert aggregations["max"]["arg_label"] == "MacBook Air M3"
    assert aggregations["min"]["value"] == 28500000.0
    assert aggregations["sum"]["value"] == 59400000.0
    assert aggregations["avg"]["value"] == 29700000.0
