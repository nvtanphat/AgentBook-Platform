# -*- coding: utf-8 -*-
"""Tests for the deterministic table aggregation executor (Phase 3)."""
from types import SimpleNamespace

from src.processing import table_executor as te
from src.processing import table_serializer as ts


def _table_block(header, rows, sheet="San_pham", bid="blk-1"):
    return SimpleNamespace(
        block_id=bid,
        content=ts.to_html(header, rows),
        extra={"block_kind": "table_block", "sheet_name": sheet, "columns": header},
    )


HEADER = ["Ten san pham", "Gia VND", "So luong"]
ROWS = [
    ["Laptop Dell XPS 13", "28500000", "32"],
    ["MacBook Air M3", "30900000", "24"],
    ["iPhone 15 Pro", "28900000", "48"],
]


class TestParseHtmlTable:
    def test_roundtrip(self):
        html = ts.to_html(["A", "B"], [["1", "2"], ["3", "4"]])
        assert ts.parse_html_table(html) == (["A", "B"], [["1", "2"], ["3", "4"]])

    def test_non_table_returns_none(self):
        assert ts.parse_html_table("plain text") is None


class TestDetectOperation:
    def test_max(self):
        assert te.detect_operation("Sản phẩm nào có giá cao nhất?") == "max"

    def test_sum(self):
        assert te.detect_operation("Tổng số lượng tất cả sản phẩm?") == "sum"

    def test_avg(self):
        assert te.detect_operation("Giá trung bình là bao nhiêu?") == "avg"

    def test_none(self):
        assert te.detect_operation("iPhone 15 Pro giá bao nhiêu?") is None


class TestExecute:
    def test_max_returns_value_and_label(self):
        blocks = [_table_block(HEADER, ROWS)]
        r = te.execute(blocks=blocks, query="Sản phẩm nào có giá (Gia VND) cao nhất?")
        assert r is not None
        assert r.operation == "max"
        assert r.value == 30900000.0
        assert r.arg_label == "MacBook Air M3"
        assert r.n_rows == 3
        assert r.source_block_ids == ["blk-1"]

    def test_sum_over_full_column_across_batches(self):
        # two batched grid blocks of the SAME sheet → all rows summed
        blocks = [
            _table_block(HEADER, ROWS[:2], bid="blk-1"),
            _table_block(HEADER, ROWS[2:], bid="blk-2"),
        ]
        r = te.execute(blocks=blocks, query="Tổng số lượng (So luong)?")
        assert r is not None and r.operation == "sum"
        assert r.value == 32 + 24 + 48
        assert r.n_rows == 3
        assert set(r.source_block_ids) == {"blk-1", "blk-2"}

    def test_avg_price(self):
        blocks = [_table_block(HEADER, ROWS)]
        r = te.execute(blocks=blocks, query="Giá (Gia VND) trung bình?")
        assert r is not None and r.operation == "avg"
        assert round(r.value) == round((28500000 + 30900000 + 28900000) / 3)

    def test_count_needs_no_column(self):
        blocks = [_table_block(HEADER, ROWS)]
        r = te.execute(blocks=blocks, query="Có bao nhiêu sản phẩm?")
        assert r is not None and r.operation == "count" and r.value == 3.0

    def test_non_numeric_column_falls_back_none(self):
        blocks = [_table_block(HEADER, ROWS)]
        # max over the text name column → not numeric → None (RAG fallback)
        r = te.execute(blocks=blocks, query="Ten san pham lon nhat?")
        assert r is None

    def test_no_table_blocks_returns_none(self):
        blocks = [SimpleNamespace(block_id="x", content="plain", extra={"block_kind": "table_row"})]
        assert te.execute(blocks=blocks, query="tổng giá?") is None


class TestNumberCoercion:
    def test_plain_integer(self):
        assert te._to_number("28500000") == 28500000.0

    def test_thousands_dot(self):
        assert te._to_number("28.500.000") == 28500000.0

    def test_decimal_comma(self):
        assert te._to_number("4,7") == 4.7
