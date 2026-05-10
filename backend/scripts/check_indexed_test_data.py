"""Check indexed chunk coverage for the current sample test-data files."""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from beanie import PydanticObjectId, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import get_settings  # noqa: E402
from src.database import DOCUMENT_MODELS  # noqa: E402
from src.models.chunk import Chunk  # noqa: E402
from src.models.material import Material  # noqa: E402


TEST_DATA_NAMES = [
    "ML_Metrics_CheatSheet.png",
    "ML_Roadmap_Infographic.png",
    "ML_Starter_Pack_Slides.pptx",
    "ML_Study_Workbook.xlsx",
    "ML_Tai_lieu_hoc_20_trang.docx",
    "rag_mau_hoc_tap.pdf",
]
OWNER_ID = "user_demo"
COLLECTION_ID = "69fc3c0949fae4625be50223"


def qdrant_count(client: QdrantClient, collection_name: str, material_id: str) -> int | str:
    try:
        result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="material_id",
                        match=models.MatchValue(value=material_id),
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def _read_processed_artifact(material: Material, settings) -> dict:
    rel_path = (material.extra_metadata or {}).get("parsed_artifact_path")
    candidates: list[Path] = []
    if isinstance(rel_path, str) and rel_path:
        candidates.append(settings.data_dir / rel_path)
    candidates.append(
        settings.processed_data_dir
        / material.owner_id
        / str(material.collection_id)
        / f"{material.id}.parsed.json"
    )
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"artifact_error": f"{type(exc).__name__}: {exc}", "artifact_path": str(path)}
    return {"artifact_error": "missing"}


def _artifact_summary(artifact: dict) -> dict[str, str | int | float]:
    pages = artifact.get("pages") or []
    blocks = [block for page in pages for block in (page.get("blocks") or [])]
    extra = artifact.get("extra") or {}
    caption_sources = Counter(
        (block.get("extra") or {}).get("caption_source")
        for block in blocks
        if (block.get("extra") or {}).get("caption_source")
    )
    parse_methods = Counter(
        (block.get("extra") or {}).get("parse_method") or block.get("source") or extra.get("parse_method") or extra.get("parser")
        for block in blocks
    )
    block_types = Counter(block.get("block_type") for block in blocks)
    ocr_quality = extra.get("ocr_quality") or {}
    return {
        "file_type": artifact.get("file_type") or "unknown",
        "parser": extra.get("parser") or extra.get("parse_method") or "unknown",
        "parse_methods": ",".join(f"{key}:{value}" for key, value in sorted(parse_methods.items()) if key) or "n/a",
        "caption_sources": ",".join(f"{key}:{value}" for key, value in sorted(caption_sources.items())) or "n/a",
        "block_types": ",".join(f"{key}:{value}" for key, value in sorted(block_types.items()) if key) or "n/a",
        "pages": len(pages),
        "blocks": len(blocks),
        "ocr_score": round(float(ocr_quality["score"]), 3) if isinstance(ocr_quality, dict) and "score" in ocr_quality else "n/a",
        "fallback_reason": extra.get("fallback_reason") or "n/a",
        "vlm_model": extra.get("vlm_model") or "n/a",
    }


def _quality_note(file_type: str, chunk_count: int, tiny_count: int, small_count: int, empty_count: int, over_budget_count: int) -> str:
    if empty_count or over_budget_count:
        return "FAIL"
    if chunk_count <= 0:
        return "FAIL:no_chunks"
    tiny_pct = tiny_count / chunk_count
    small_pct = small_count / chunk_count
    if file_type == "pptx":
        if tiny_pct > 0.20 or small_pct > 0.80:
            return "WARN:slide_small_chunks"
        return "OK:slide"
    if file_type in {"docx", "pdf"}:
        if tiny_pct > 0.05 or small_pct > 0.50:
            return "WARN:text_small_chunks"
        return "OK:text"
    if file_type in {"xlsx", "xls", "csv"}:
        return "OK:table"
    if file_type in {"png", "jpg", "jpeg"}:
        return "OK:image"
    return "OK"


async def main() -> None:
    settings = get_settings()
    mongo = AsyncIOMotorClient(settings.mongodb_uri)
    await init_beanie(database=mongo[settings.mongodb_database], document_models=DOCUMENT_MODELS)

    collection_oid = PydanticObjectId(COLLECTION_ID)
    materials = await Material.find(
        {
            "owner_id": OWNER_ID,
            "collection_id": collection_oid,
            "original_name": {"$in": TEST_DATA_NAMES},
        }
    ).to_list()
    by_name = {material.original_name: material for material in materials}

    qdrant = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    print(f"Mongo database: {settings.mongodb_database}")
    print(f"Qdrant: {settings.qdrant_url} / {settings.qdrant_collection_name}")
    print(f"Scope: owner={OWNER_ID}, collection={COLLECTION_ID}")
    print()

    total_chunks = 0
    total_points = 0
    for name in TEST_DATA_NAMES:
        material = by_name.get(name)
        if material is None:
            print(f"{name}\tMISSING")
            continue

        chunks = await Chunk.find(Chunk.material_id == material.id).to_list()
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        tiny_count = sum(1 for count in token_counts if count < 50)
        small_count = sum(1 for count in token_counts if 50 <= count < 100)
        over_budget_count = sum(1 for count in token_counts if count > settings.chunk_target_token_count)
        empty_count = sum(1 for chunk in chunks if not (chunk.content or "").strip())
        token_summary = "n/a"
        if token_counts:
            token_summary = f"{min(token_counts)}/{sum(token_counts) // len(token_counts)}/{max(token_counts)}"
        strategies = ",".join(sorted({chunk.chunk_strategy for chunk in chunks})) or "n/a"
        versions = ",".join(sorted({chunk.chunker_version for chunk in chunks})) or "n/a"
        artifact = _read_processed_artifact(material, settings)
        summary = _artifact_summary(artifact)
        chunks_per_page = round(len(chunks) / max(1, int(summary["pages"])), 2)
        quality_note = _quality_note(
            str(summary["file_type"]),
            len(chunks),
            tiny_count,
            small_count,
            empty_count,
            over_budget_count,
        )
        points = qdrant_count(qdrant, settings.qdrant_collection_name, str(material.id))
        if isinstance(points, int):
            total_points += points
        total_chunks += len(chunks)
        print(
            f"{name}\tstatus={material.status}\tmongo_chunks={len(chunks)}"
            f"\tqdrant_points={points}\ttokens_min_avg_max={token_summary}"
            f"\ttiny_lt50={tiny_count}\tsmall_50_99={small_count}"
            f"\tover_{settings.chunk_target_token_count}={over_budget_count}\tempty={empty_count}"
            f"\tfile_type={summary['file_type']}\tparser={summary['parser']}"
            f"\tparse_methods={summary['parse_methods']}\tcaption_sources={summary['caption_sources']}"
            f"\tocr_score={summary['ocr_score']}\tvlm_model={summary['vlm_model']}"
            f"\tfallback_reason={summary['fallback_reason']}\tchunks_per_page={chunks_per_page}"
            f"\tquality={quality_note}"
            f"\tstrategies={strategies}\tchunker_versions={versions}"
        )

    print()
    print(f"TOTAL\tmongo_chunks={total_chunks}\tqdrant_points={total_points}")


if __name__ == "__main__":
    asyncio.run(main())
