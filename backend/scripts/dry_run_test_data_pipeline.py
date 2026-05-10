"""Dry-run parsing and chunking for files in data/test data.

This does not write MongoDB/Qdrant. It exercises the production parser routing
so image files can use VLM-first with OCR fallback, while DOCX/PPTX/PDF use
Docling and spreadsheets use the spreadsheet parser.
"""
from __future__ import annotations

import sys
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from src.core.config import get_settings  # noqa: E402
from src.processing.evidence_mapper import EvidenceMapper  # noqa: E402
from src.processing.layout_normalizer import LayoutNormalizer  # noqa: E402
from src.processing.chunking import build_chunker  # noqa: E402
from src.services.parse_index_pipeline import ParseIndexPipeline  # noqa: E402


SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"}


def token_stats(chunks) -> dict:
    counts = [chunk.token_count for chunk in chunks]
    if not counts:
        return {"count": 0}
    counts_sorted = sorted(counts)
    return {
        "count": len(counts),
        "tiny": sum(1 for count in counts if count < 50),
        "small": sum(1 for count in counts if 50 <= count < 100),
        "over": sum(1 for count in counts if count > 512),
        "min": counts_sorted[0],
        "avg": sum(counts_sorted) // len(counts_sorted),
        "median": counts_sorted[len(counts_sorted) // 2],
        "max": counts_sorted[-1],
    }


def print_result(path: Path, parsed, normalized, chunks) -> None:
    extra = parsed.extra or {}
    blocks = normalized.blocks
    block_types = Counter(block.block_type for block in blocks)
    sources = Counter(block.source for block in blocks)
    caption_sources = Counter(
        block.extra.get("caption_source")
        for block in blocks
        if block.extra.get("caption_source")
    )
    stats = token_stats(chunks)
    ocr_quality = extra.get("ocr_quality") or {}

    print("\n" + "=" * 88)
    print(path.name)
    print("=" * 88)
    print(f"parser          : {extra.get('parser') or extra.get('parse_method') or 'unknown'}")
    print(f"parse_method    : {extra.get('parse_method') or 'n/a'}")
    print(f"fallback_reason : {extra.get('fallback_reason') or 'n/a'}")
    print(f"vlm_model       : {extra.get('vlm_model') or 'n/a'}")
    print(f"ocr_score       : {round(float(ocr_quality['score']), 3) if 'score' in ocr_quality else 'n/a'}")
    print(f"pages/blocks    : {len(normalized.pages)} / {len(blocks)}")
    print(f"block_types     : {dict(block_types)}")
    print(f"sources         : {dict(sources)}")
    print(f"caption_sources : {dict(caption_sources) if caption_sources else 'n/a'}")

    if stats["count"] == 0:
        print("chunks          : 0")
        return

    print(
        "chunks          : "
        f"{stats['count']} | tiny<50={stats['tiny']} | small50-99={stats['small']} | "
        f"over512={stats['over']} | min/avg/med/max={stats['min']}/{stats['avg']}/{stats['median']}/{stats['max']}"
    )
    print("sample_chunks   :")
    for chunk in sorted(chunks, key=lambda item: item.token_count)[:3]:
        preview = " ".join((chunk.content or "").split())[:180]
        print(f"  - {chunk.token_count:>4}t p{chunk.source_pages} {preview!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(ROOT / "data" / "test data"))
    parser.add_argument("--ocr-only", action="store_true", help="Disable VLM probing/captioning for a fast OCR/layout run.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files for quick checks.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    if args.ocr_only:
        from src.processing.figure_captioner import FigureCaptioner

        FigureCaptioner._detect_available_model = lambda self: None  # type: ignore[method-assign]

    settings = get_settings()
    pipeline = ParseIndexPipeline(settings=settings)
    normalizer = LayoutNormalizer()
    mapper = EvidenceMapper()
    chunker = build_chunker(settings, embedder=None)
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED)
    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Dry-run folder  : {folder}")
    print(f"Files           : {len(files)}")
    print(f"VLM             : {'disabled' if args.ocr_only else 'enabled'}")
    print(f"Chunk strategy  : {settings.chunk_strategy} (dry-run uses layout fallback when no embedder)")
    print(f"Target/min      : {settings.chunk_target_token_count}/{settings.chunk_min_token_count}")

    total_chunks = 0
    total_blocks = 0
    failures: list[tuple[str, str]] = []
    for path in files:
        material = SimpleNamespace(
            file_type=path.suffix.lower().lstrip("."),
            language="unknown",
            extra_metadata={"source_type": "dry_run_test_data"},
            modality="mixed",
            storage_path=str(path),
            original_name=path.name,
        )
        try:
            parsed = pipeline._parse_material(material)
            parsed = pipeline._caption_figures(parsed, material.language)
            normalized = normalizer.normalize(parsed)
            evidence_map = mapper.build(
                parsed=normalized,
                owner_id="dry_run",
                collection_id="65f000000000000000000002",
                material_id="65f000000000000000000001",
                document_name=path.name,
            )
            chunks = chunker.build_chunks(evidence_map)
            chunks = [chunk for chunk in chunks if len((chunk.content or "").strip()) >= 50]
            print_result(path, parsed, normalized, chunks)
            total_chunks += len(chunks)
            total_blocks += len(normalized.blocks)
        except Exception as exc:
            failures.append((path.name, f"{type(exc).__name__}: {exc}"))
            print("\n" + "=" * 88)
            print(path.name)
            print("=" * 88)
            print(f"FAILED: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 88)
    print("TOTAL")
    print("=" * 88)
    print(f"files={len(files)} blocks={total_blocks} chunks={total_chunks} failures={len(failures)}")
    for name, error in failures:
        print(f"  - {name}: {error}")


if __name__ == "__main__":
    main()
