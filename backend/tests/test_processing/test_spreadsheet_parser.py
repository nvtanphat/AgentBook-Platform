from __future__ import annotations

from pathlib import Path

import pytest

from src.processing.spreadsheet_parser import SpreadsheetParser


def test_csv_parser_converts_rows_to_table_blocks(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("student,score\nAn,9.5\nBinh,8.0\n", encoding="utf-8")

    parsed = SpreadsheetParser(max_rows_per_block=1).parse(path, language="vi")

    assert parsed.extra["parser"] == "spreadsheet"
    assert parsed.file_type == "csv"
    assert len(parsed.pages) == 1

    blocks = parsed.blocks
    by_kind: dict[str, list] = {}
    for block in blocks:
        by_kind.setdefault(block.extra.get("block_kind", ""), []).append(block)

    assert len(by_kind["table_summary"]) == 1
    assert by_kind["table_summary"][0].extra["columns"] == ["student", "score"]
    assert "2 hàng × 2 cột" in by_kind["table_summary"][0].content

    assert len(by_kind["table_block"]) == 2
    for block in by_kind["table_block"]:
        assert "| student | score |" in block.content
        assert block.block_type == "table"
    assert by_kind["table_block"][0].extra["row_start"] == 2

    assert len(by_kind["table_row"]) == 2
    assert "Hàng 2" in by_kind["table_row"][0].content
    assert "student: An" in by_kind["table_row"][0].content
    assert "score: 9.5" in by_kind["table_row"][0].content
    assert by_kind["table_row"][1].extra["row_index"] == 3


def test_xlsx_parser_preserves_sheet_metadata(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "workbook.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Metrics"
    worksheet.append(["model", "mse"])
    worksheet.append(["baseline", 0.42])
    worksheet.append(["candidate", 0.37])
    workbook.save(path)

    parsed = SpreadsheetParser().parse(path, language="en")

    assert parsed.file_type == "xlsx"
    table_blocks = [b for b in parsed.pages[0].blocks if b.block_type == "table"]
    assert table_blocks
    assert table_blocks[0].extra["sheet_name"] == "Metrics"
    assert "| baseline | 0.42 |" in table_blocks[0].content

    summaries = [b for b in parsed.pages[0].blocks if b.extra.get("block_kind") == "table_summary"]
    assert summaries and "2 rows × 2 columns" in summaries[0].content


def test_xlsx_parser_skips_merged_title_rows(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "merged_title.xlsx"
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "DuLieuTruyVan"
    worksheet.merge_cells("A1:G1")
    worksheet["A1"] = "Hệ thống Multimodal RAG cho thư viện số học tập"
    worksheet.merge_cells("A2:G2")
    worksheet["A2"] = "Sheet dữ liệu dạng bảng để kiểm tra retrieval theo hàng/cột."
    worksheet["A4"] = "ID"
    worksheet["B4"] = "Định dạng"
    worksheet["C4"] = "Nguồn mẫu"
    worksheet["D4"] = "Nội dung chính"
    worksheet["E4"] = "Thuật ngữ"
    worksheet["F4"] = "Chunk kỳ vọng"
    worksheet["G4"] = "Câu hỏi kiểm thử"
    worksheet.append([None, None, None, None, None, None, None])
    worksheet.append([1, "DOCX", "rag_mau_hoc_tap.docx", "Tổng quan RAG", "RAG", "Mục 1-3", "RAG khác chatbot thường ở điểm nào?"])
    worksheet.append([2, "PDF", "rag_mau_hoc_tap.pdf", "Checklist kiểm thử", "OCR", "Trang 1", "OCR được dùng để làm gì?"])
    workbook.save(path)

    parsed = SpreadsheetParser(max_rows_per_block=1).parse(path, language="vi")
    blocks = parsed.pages[0].blocks

    summary = next(b for b in blocks if b.extra.get("block_kind") == "table_summary")
    assert summary.extra["columns"] == [
        "ID",
        "Định dạng",
        "Nguồn mẫu",
        "Nội dung chính",
        "Thuật ngữ",
        "Chunk kỳ vọng",
        "Câu hỏi kiểm thử",
    ]
    assert "Hệ thống Multimodal RAG cho thư viện số học tập" in summary.extra["preamble"]
    assert any("Sheet dữ liệu dạng bảng" in item for item in summary.extra["preamble"])
    assert "Column 2" not in summary.content

    table_rows = [b for b in blocks if b.extra.get("block_kind") == "table_row"]
    assert table_rows and table_rows[0].extra["row_index"] == 6
    assert "Hàng 6" in table_rows[0].content
