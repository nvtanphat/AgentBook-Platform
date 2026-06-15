"""Shared table serialization for table-native RAG.

Produces a structured HTML cell grid (LLM-native for lookup/compare/aggregate)
plus per-row verbalized sentences (keyword/value retrieval), from either:

  * a clean grid ``(header, rows)`` — spreadsheet_parser / DOCX tables, or
  * an existing markdown pipe table string — docling PDF export.

Both docling_parser and spreadsheet_parser route through here so the two
representations stay byte-identical regardless of source. The verbalized-row
format MUST match what inference_engine._filter_substantive_chunks keys off
("Hàng N của bảng '…'" / "Row N of table '…'").
"""

from __future__ import annotations

import html as _html
import re

# Keep HTML compact; cells are escaped. No styling — the LLM reads structure only.
_MAX_CELL_CHARS = 300


def parse_markdown_table(content: str) -> tuple[list[str], list[list[str]]] | None:
    """Recover ``(header, rows)`` from a markdown pipe table, else None.

    Tolerates a leading/trailing pipe and the ``| --- | --- |`` separator row.
    Returns None when the text is not a 2-column-plus pipe table.
    """
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if ln.startswith("|") or "|" in ln]
    if len(pipe_lines) < 2:
        return None

    def split_row(line: str) -> list[str]:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [c.replace("\\|", "|").strip() for c in line.split("|")]

    rows = [split_row(ln) for ln in pipe_lines]
    # Drop the markdown separator row (cells are all dashes/colons)
    rows = [r for r in rows if not all(re.fullmatch(r"[:\-\s]+", c or "-") for c in r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    norm = [(r + [""] * width)[:width] for r in rows]
    return norm[0], norm[1:]


def parse_html_table(content: str) -> tuple[list[str], list[list[str]]] | None:
    """Recover ``(header, rows)`` from an HTML ``<table>`` grid, else None.

    Symmetric with ``parse_markdown_table`` and ``to_html``; used by the table
    executor to reconstruct a column for deterministic aggregation. Tolerant of
    the exact whitespace ``to_html`` emits.
    """
    text = content or ""
    if "<table" not in text.lower():
        return None
    rows: list[list[str]] = []
    for row_html in re.findall(r"<tr>(.*?)</tr>", text, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        rows.append([_unescape(c) for c in cells])
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    if width < 1:
        return None
    norm = [(r + [""] * width)[:width] for r in rows]
    return norm[0], norm[1:]


def _unescape(cell: str) -> str:
    return " ".join(_html.unescape(cell).split())


def _clip(value: str) -> str:
    text = " ".join(str(value or "").split())
    return text[:_MAX_CELL_CHARS]


def to_html(header: list[str], rows: list[list[str]]) -> str:
    """Render an HTML ``<table>`` grid with a header row and escaped cells."""
    width = len(header)
    th = "".join(f"<th>{_html.escape(_clip(c))}</th>" for c in header)
    body_rows = []
    for row in rows:
        cells = (list(row) + [""] * width)[:width]
        tds = "".join(f"<td>{_html.escape(_clip(c))}</td>" for c in cells)
        body_rows.append(f"<tr>{tds}</tr>")
    return f"<table><tr>{th}</tr>{''.join(body_rows)}</table>"


def verbalize_rows(
    header: list[str],
    rows: list[list[str]],
    *,
    table_name: str,
    language: str = "vi",
    start_index: int = 1,
    row_numbers: list[int] | None = None,
) -> list[tuple[int, str]]:
    """One sentence per data row → ``[(row_index, sentence), …]``.

    Skips fully-empty rows. Format matches the spreadsheet verbalizer so the
    downstream substantive-chunk filter recognises these as table rows.
    """
    width = len(header)
    out: list[tuple[int, str]] = []
    for offset, row in enumerate(rows):
        idx = row_numbers[offset] if row_numbers and offset < len(row_numbers) else start_index + offset
        cells = (list(row) + [""] * width)[:width]
        pairs = []
        for col_name, cell in zip(header, cells):
            value = _clip(cell)
            if not value:
                continue
            label = (col_name or "").strip() or "—"
            pairs.append(f"{label}: {value}")
        if not pairs:
            continue
        body = ". ".join(pairs)
        if language == "en":
            out.append((idx, f"Row {idx} of table '{table_name}'. {body}."))
        else:
            out.append((idx, f"Hàng {idx} của bảng '{table_name}'. {body}."))
    return out


def structured_meta(header: list[str], rows: list[list[str]]) -> dict:
    """Structured descriptor stored in block.extra for table-aware prompting."""
    return {
        "columns": [(_clip(c) or f"Column {i + 1}") for i, c in enumerate(header)],
        "n_rows": len(rows),
        "n_cols": len(header),
    }
