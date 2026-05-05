"""
Chunk quality check — parses every file in a folder and reports chunk stats.
Usage: python -m scripts.chunk_quality_check [folder]
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from src.core.config import get_settings
from src.processing.chunking import build_chunker
from src.processing.types import EvidenceBlock, EvidenceMap


def build_evidence_map(parsed_doc, settings) -> EvidenceMap:
    from src.processing.types import EvidenceBlock, EvidenceMap, BBox
    from uuid import uuid4

    blocks: list[EvidenceBlock] = []
    for page in parsed_doc.pages:
        for blk in page.blocks:
            bbox = None
            if blk.bbox:
                bbox = blk.bbox
            blocks.append(EvidenceBlock(
                owner_id="test",
                collection_id="test_col",
                material_id="test_mat",
                document_name=parsed_doc.source_path,
                page=blk.page_number,
                block_id=blk.block_id,
                block_type=blk.block_type,
                snippet_original=blk.content,
                source_language=blk.language or parsed_doc.language,
                bbox=bbox,
                confidence=blk.ocr_confidence,
            ))

    return EvidenceMap(
        owner_id="test",
        collection_id="test_col",
        material_id="test_mat",
        document_name=parsed_doc.source_path,
        blocks=blocks,
    )


def chunk_stats(chunks) -> dict:
    if not chunks:
        return {"count": 0}
    token_counts = [c.token_count for c in chunks]
    tiny   = sum(1 for t in token_counts if t < 50)
    small  = sum(1 for t in token_counts if 50 <= t < 150)
    good   = sum(1 for t in token_counts if 150 <= t <= 512)
    large  = sum(1 for t in token_counts if t > 512)
    sorted_t = sorted(token_counts)
    n = len(sorted_t)
    median = sorted_t[n // 2]
    avg = sum(sorted_t) // n
    p90 = sorted_t[int(n * 0.9)]
    return {
        "count": n,
        "tiny_pct":  f"{tiny / n * 100:.1f}%  ({tiny})",
        "small_pct": f"{small / n * 100:.1f}% ({small})",
        "good_pct":  f"{good / n * 100:.1f}%  ({good})",
        "large_pct": f"{large / n * 100:.1f}% ({large})",
        "min": sorted_t[0],
        "max": sorted_t[-1],
        "avg": avg,
        "median": median,
        "p90": p90,
    }


def print_stats(name, stats, strategy=""):
    label = f"{name}"
    if strategy:
        label += f" [{strategy}]"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if stats["count"] == 0:
        print("  (no chunks produced)")
        return
    print(f"  Total chunks : {stats['count']}")
    print(f"  Tiny  <50t   : {stats['tiny_pct']}")
    print(f"  Small 50-150 : {stats['small_pct']}")
    print(f"  Good  150-512: {stats['good_pct']}")
    print(f"  Large >512   : {stats['large_pct']}")
    print(f"  min/avg/med/p90/max : {stats['min']}/{stats['avg']}/{stats['median']}/{stats['p90']}/{stats['max']}")


def parse_file(file_path: Path, settings):
    ext = file_path.suffix.lower().lstrip(".")
    if ext in ("pdf", "pptx", "docx"):
        from src.processing.docling_parser import DoclingParser
        parser = DoclingParser()
        return parser.parse(file_path)
    elif ext in ("xlsx", "xls", "csv"):
        from src.processing.spreadsheet_parser import SpreadsheetParser
        parser = SpreadsheetParser()
        return parser.parse(file_path, display_name=file_path.name)
    elif ext in ("png", "jpg", "jpeg"):
        from src.processing.ocr_engine import EasyOCREngine
        engine = EasyOCREngine(lang="vi")
        return engine.parse_image(file_path)
    else:
        print(f"  [skip] unsupported extension: {ext}")
        return None


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "test data"
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    settings = get_settings()
    chunker = build_chunker(settings, embedder=None)  # layout chunker (no embedder available in script)

    files = sorted(folder.iterdir())
    files = [f for f in files if f.is_file()]

    print(f"\nChunk quality check — {len(files)} files in: {folder}")
    print(f"Chunker strategy    : {settings.chunk_strategy}  (running as layout; no embedder in script)")
    print(f"Target token count  : {settings.chunk_target_token_count}")
    print(f"Min token count     : {settings.chunk_min_token_count}")
    print(f"Overlap             : {settings.chunk_overlap_token_count}")

    totals = {"count": 0, "tiny": 0, "small": 0, "good": 0, "large": 0, "tokens": []}

    for file_path in files:
        print(f"\n>> Parsing: {file_path.name}")
        try:
            parsed = parse_file(file_path, settings)
        except Exception as exc:
            print(f"  [ERROR] parse failed: {exc}")
            continue
        if parsed is None:
            continue

        block_count = sum(len(p.blocks) for p in parsed.pages)
        print(f"  Pages: {len(parsed.pages)}, Blocks: {block_count}, Lang: {parsed.language}")

        if block_count == 0:
            print("  [skip] no blocks extracted")
            continue

        try:
            emap = build_evidence_map(parsed, settings)
            chunks = chunker.build_chunks(emap)
        except Exception as exc:
            print(f"  [ERROR] chunking failed: {exc}")
            import traceback; traceback.print_exc()
            continue

        stats = chunk_stats(chunks)
        print_stats(file_path.name, stats)

        # accumulate totals
        for c in chunks:
            t = c.token_count
            totals["count"] += 1
            totals["tokens"].append(t)
            if t < 50: totals["tiny"] += 1
            elif t < 150: totals["small"] += 1
            elif t <= 512: totals["good"] += 1
            else: totals["large"] += 1

        # print 3 sample chunks (smallest, middle, largest)
        sorted_chunks = sorted(chunks, key=lambda c: c.token_count)
        samples = [sorted_chunks[0], sorted_chunks[len(sorted_chunks) // 2], sorted_chunks[-1]]
        print("\n  Sample chunks (smallest / median / largest):")
        for i, c in enumerate(samples):
            label = ["smallest", "median ", "largest"][i]
            preview = c.content[:120].replace("\n", " ")
            print(f"    [{label}] {c.token_count}t | p{c.source_pages} | {preview!r}")

    # Summary
    if totals["count"] > 0:
        n = totals["count"]
        tokens = sorted(totals["tokens"])
        print(f"\n{'=' * 60}")
        print(f"  TOTAL ACROSS ALL FILES")
        print(f"{'=' * 60}")
        print(f"  Total chunks : {n}")
        print(f"  Tiny  <50t   : {totals['tiny'] / n * 100:.1f}%  ({totals['tiny']})")
        print(f"  Small 50-150 : {totals['small'] / n * 100:.1f}% ({totals['small']})")
        print(f"  Good  150-512: {totals['good'] / n * 100:.1f}%  ({totals['good']})")
        print(f"  Large >512   : {totals['large'] / n * 100:.1f}% ({totals['large']})")
        print(f"  avg/median/p90: {sum(tokens)//n}/{tokens[n//2]}/{tokens[int(n*0.9)]}")
    print()


if __name__ == "__main__":
    main()
