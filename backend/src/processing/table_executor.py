"""Deterministic table aggregation (sum/avg/max/min/count) over a FULL table.

RAG retrieves only the top-k rows, so questions like "which product has the
highest price?" cannot be answered from retrieved context — the model refuses.
This executor sidesteps that: it reconstructs the WHOLE column from the stored
table blocks (`get_material_pages` returns every block regardless of retrieval)
and computes the answer with plain Python — no LLM, no hallucination risk.

On any ambiguity (column not found, non-numeric column for a numeric op, no
table rows) it returns None and the caller falls back to the normal RAG path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.processing import table_serializer
from src.processing.slug import ascii_fold

logger = logging.getLogger(__name__)

Operation = str  # "sum" | "avg" | "max" | "min" | "count"

_OP_PATTERNS: list[tuple[Operation, re.Pattern[str]]] = [
    ("avg", re.compile(r"\b(trung binh|binh quan|average|mean)\b", re.I)),
    ("max", re.compile(r"\b(lon nhat|cao nhat|max|maximum|highest|dat nhat|nhieu nhat)\b", re.I)),
    ("min", re.compile(r"\b(nho nhat|thap nhat|min|minimum|lowest|re nhat|it nhat)\b", re.I)),
    ("sum", re.compile(r"\b(tong|tong cong|cong lai|sum|total)\b", re.I)),
    # count requires a countable noun ("bao nhieu san pham") — bare "bao nhieu"
    # is a value lookup ("giá bao nhiêu"), not a count.
    ("count", re.compile(r"\b(dem so|how many|\bcount\b|bao nhieu (san pham|dong|hang|muc|mau|loai))\b", re.I)),
]

_NUMERIC_RE = re.compile(r"-?\d[\d.,]*")


@dataclass
class AggregationResult:
    operation: Operation
    column: str
    value: float
    n_rows: int
    sheet_name: str
    source_block_ids: list[str] = field(default_factory=list)
    arg_label: str | None = None   # for max/min: the row's identifying label
    label_column: str | None = None


# ── Query parsing ─────────────────────────────────────────────────────────────

def detect_operation(query: str) -> Operation | None:
    text = ascii_fold(query).lower()
    for op, pat in _OP_PATTERNS:
        if pat.search(text):
            return op
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", ascii_fold(s).lower()).strip()


def _match_column(query: str, header: list[str], allowed: set[int] | None = None) -> int | None:
    """Pick the header column the query asks about (token overlap).

    `allowed` restricts candidates (e.g. only numeric columns for numeric ops),
    preventing a name column with stray digits from being aggregated.
    """
    q_tokens = set(_norm(query).split())
    best_idx, best_overlap = None, 0
    for idx, col in enumerate(header):
        if allowed is not None and idx not in allowed:
            continue
        col_tokens = set(_norm(col).split())
        if not col_tokens:
            continue
        overlap = len(q_tokens & col_tokens)
        if overlap > best_overlap:
            best_idx, best_overlap = idx, overlap
    return best_idx if best_overlap > 0 else None


def _numeric_columns(header: list[str], rows: list[list[str]], *, min_ratio: float = 0.6) -> set[int]:
    """Indices of columns whose cells are mostly parseable numbers."""
    numeric: set[int] = set()
    for idx in range(len(header)):
        cells = [row[idx] for row in rows if idx < len(row) and (row[idx] or "").strip()]
        if not cells:
            continue
        hits = sum(1 for c in cells if _to_number(c) is not None)
        if hits / len(cells) >= min_ratio:
            numeric.add(idx)
    return numeric


def _to_number(cell: str) -> float | None:
    # A real numeric cell is digits + separators (optionally a currency unit),
    # NOT a name that merely contains a digit ("XPS 13"). Reject anything with
    # letters after stripping a currency token.
    s = ascii_fold(cell or "").strip().lower()
    s = re.sub(r"(vnd|usd|dong|eur|gbp|jpy|%)", "", s)
    s = re.sub(r"[$₫€£¥]", "", s).strip()
    if not s or re.search(r"[a-z]", s):
        return None
    m = _NUMERIC_RE.fullmatch(s)
    if not m:
        return None
    raw = m.group(0)
    # Heuristic separator handling: if both '.' and ',' present, the last one is
    # the decimal sep; otherwise treat a lone separator with 3-digit groups as
    # thousands. Vietnamese product data here uses plain integers ("28500000").
    if "." in raw and "," in raw:
        dec = max(raw.rfind("."), raw.rfind(","))
        intpart = re.sub(r"[.,]", "", raw[:dec])
        raw = intpart + "." + re.sub(r"[.,]", "", raw[dec + 1:])
    else:
        sep = "." if "." in raw else ("," if "," in raw else "")
        if sep and all(len(g) == 3 for g in raw.split(sep)[1:]):
            raw = raw.replace(sep, "")     # thousands grouping
        else:
            raw = raw.replace(",", ".")    # decimal comma
    try:
        return float(raw)
    except ValueError:
        return None


# ── Column reconstruction + computation ───────────────────────────────────────

def reconstruct_table(blocks: list, sheet_name: str | None) -> tuple[list[str], list[list[str]], list[str]]:
    """Concatenate all HTML grid blocks of one table into (header, rows, block_ids).

    `blocks` are MaterialBlock-like objects with `.content` and `.extra`. When
    `sheet_name` is given, only that table's blocks are used.
    """
    header: list[str] = []
    rows: list[list[str]] = []
    block_ids: list[str] = []
    for b in blocks:
        extra = getattr(b, "extra", {}) or {}
        if extra.get("block_kind") != "table_block":
            continue
        if sheet_name and extra.get("sheet_name") != sheet_name:
            continue
        parsed = table_serializer.parse_html_table(getattr(b, "content", "") or "")
        if parsed is None:
            continue
        h, r = parsed
        if not header:
            header = h
        rows.extend(r)
        block_ids.append(getattr(b, "block_id", ""))
    return header, rows, [bid for bid in block_ids if bid]


def aggregate(
    *, header: list[str], rows: list[list[str]], block_ids: list[str],
    query: str, operation: Operation, sheet_name: str,
) -> AggregationResult | None:
    if not header or not rows:
        return None

    if operation == "count":
        # count needs no numeric column
        return AggregationResult(operation="count", column="*", value=float(len(rows)),
                                 n_rows=len(rows), sheet_name=sheet_name, source_block_ids=block_ids)

    # Numeric ops only consider numeric columns, so a name column with stray
    # digits ("XPS 13") is never aggregated.
    numeric_cols = _numeric_columns(header, rows)
    col_idx = _match_column(query, header, allowed=numeric_cols)
    if col_idx is None:
        return None

    # Label = the first non-numeric (identifying) column, e.g. the product name.
    label_idx = next((i for i in range(len(header)) if i not in numeric_cols), None)
    if label_idx is None:
        label_idx = next((i for i in range(len(header)) if i != col_idx), 0)
    pairs: list[tuple[float, str]] = []
    for row in rows:
        if col_idx >= len(row):
            continue
        num = _to_number(row[col_idx])
        if num is None:
            continue
        label = row[label_idx] if label_idx < len(row) else ""
        pairs.append((num, label))
    if not pairs:
        return None  # column wasn't numeric → fall back to RAG

    values = [v for v, _ in pairs]
    result = AggregationResult(
        operation=operation, column=header[col_idx], value=0.0,
        n_rows=len(values), sheet_name=sheet_name, source_block_ids=block_ids,
        label_column=header[label_idx],
    )
    if operation == "sum":
        result.value = sum(values)
    elif operation == "avg":
        result.value = sum(values) / len(values)
    elif operation == "max":
        result.value, result.arg_label = max(pairs, key=lambda p: p[0])
    elif operation == "min":
        result.value, result.arg_label = min(pairs, key=lambda p: p[0])
    else:
        return None
    return result


def execute(*, blocks: list, query: str, sheet_name: str | None = None) -> AggregationResult | None:
    """Top-level: detect op, reconstruct the table, compute. None ⇒ RAG fallback."""
    operation = detect_operation(query)
    if operation is None:
        return None
    header, rows, block_ids = reconstruct_table(blocks, sheet_name)
    if not rows:
        return None
    used_sheet = sheet_name or ""
    if not used_sheet:
        for b in blocks:
            extra = getattr(b, "extra", {}) or {}
            if extra.get("block_kind") == "table_block" and extra.get("sheet_name"):
                used_sheet = extra["sheet_name"]
                break
    try:
        return aggregate(header=header, rows=rows, block_ids=block_ids,
                         query=query, operation=operation, sheet_name=used_sheet)
    except Exception as exc:
        logger.warning("Table aggregation failed", extra={"error": str(exc), "op": operation})
        return None
