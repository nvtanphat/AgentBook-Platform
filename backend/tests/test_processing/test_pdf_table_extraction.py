"""Unit tests for PDF table extraction from Docling TableItem objects.

Covers the three version-tolerant grid recovery paths (dataframe / cell grid /
markdown), provenance tagging, defensive skipping, and the downstream
enrichment to HTML grid + verbalized rows.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.processing.docling_parser import DoclingParser
from src.processing.types import BlockType


# ── Docling TableItem stubs ─────────────────────────────────────────────────────


class _Bbox:
    def __init__(self, l, t, r, b):
        self.l, self.t, self.r, self.b = l, t, r, b


class _Prov:
    def __init__(self, page, bbox):
        self.page_no, self.bbox = page, bbox


class _Cell:
    def __init__(self, text):
        self.text = text


class _Data:
    def __init__(self, grid):
        self.grid = grid


class _DataFrameTable:
    """Exposes export_to_dataframe (primary path)."""
    self_ref = "#/tables/0"
    prov = [_Prov(6, _Bbox(10, 700, 200, 750))]

    def export_to_dataframe(self):
        return pd.DataFrame({"Model": ["base", "big"], "BLEU": ["27.3", "28.4"]})


class _GridTable:
    """No dataframe export → falls back to data.grid cells."""
    self_ref = "#/tables/1"
    prov = [_Prov(7, _Bbox(10, 600, 200, 650))]
    data = _Data([[_Cell("Year"), _Cell("Score")], [_Cell("2017"), _Cell("41.8")]])


class _EmptyTable:
    """No usable content → must be skipped, not crash."""
    prov = [_Prov(8, None)]
    data = _Data([])


class _BoomTable:
    """Raises during extraction → must be skipped, not crash the parse."""
    prov = [_Prov(9, None)]
    data = None

    def export_to_dataframe(self):
        raise RuntimeError("boom")


class _Doc:
    def __init__(self, tables):
        self.tables = tables


# ── Tests ───────────────────────────────────────────────────────────────────────


def _parser() -> DoclingParser:
    p = DoclingParser()
    p._page_heights = {6: 800.0, 7: 800.0}  # for bbox y-flip
    return p


def test_dataframe_and_grid_tables_extracted_with_provenance():
    blocks = _parser()._pdf_table_blocks(
        _Doc([_DataFrameTable(), _GridTable(), _EmptyTable(), _BoomTable()]),
        language="en",
    )
    # Only the two well-formed tables survive; empty + boom are skipped safely.
    assert len(blocks) == 2
    for b in blocks:
        assert b.block_type == BlockType.TABLE.value
        assert b.extra["table_source"] == "docling"

    df_block = blocks[0]
    assert df_block.page_number == 6
    assert "Model" in df_block.content and "BLEU" in df_block.content  # markdown
    assert df_block.bbox is not None

    grid_block = blocks[1]
    assert grid_block.page_number == 7
    assert "Year" in grid_block.content and "2017" in grid_block.content


def test_no_tables_returns_empty():
    assert _parser()._pdf_table_blocks(_Doc([]), language="en") == []
    assert _parser()._pdf_table_blocks(object(), language="en") == []


def test_enrichment_produces_html_grid_and_verbalized_rows():
    parser = _parser()
    blocks = parser._pdf_table_blocks(_Doc([_DataFrameTable()]), language="en")
    enriched = parser._enrich_table_blocks(blocks, language="en")

    html_tables = [b for b in enriched if b.block_type == BlockType.TABLE.value]
    assert any(b.content.startswith("<table") for b in html_tables)

    rows = [b for b in enriched if (b.extra or {}).get("block_kind") == "table_row"]
    assert len(rows) == 2  # two data rows verbalized
    assert all("of table" in r.content for r in rows)


def test_table_grid_markdown_fallback():
    """When neither dataframe nor grid is available, parse markdown export."""

    class _MdTable:
        prov = [_Prov(1, None)]

        def export_to_markdown(self, *_):
            return "| A | B |\n| --- | --- |\n| 1 | 2 |"

    header, rows = DoclingParser._table_grid(_MdTable(), object())
    assert header == ["A", "B"]
    assert rows == [["1", "2"]]
