"""Isolated subprocess entry point for Docling parsing.

Docling's heavy models can silently degrade inside a long-running server
process: ``convert()`` starts returning an empty document, so every page falls
back to full-page OCR and all table/figure/equation structure is lost. Running
each parse in a *fresh* subprocess sidesteps this — a clean process reliably
yields the full layout (verified: same file parses to 4 tables standalone vs 0
in the server process).

Usage:
    python -m src.processing.parse_worker <input_path> <language> <output_json>

Writes ``ParsedDocument.model_dump_json()`` to ``output_json`` and exits 0.
Any failure exits non-zero; the caller falls back to an in-process parse.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        sys.stderr.write("usage: parse_worker <input_path> <language> <output_json>\n")
        return 2
    input_path, language, output_json = argv[1], argv[2], argv[3]

    from src.processing.docling_parser import DoclingParser

    parsed = DoclingParser().parse(Path(input_path), language=language)
    Path(output_json).write_text(parsed.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
