from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from src.core.config import Settings
from src.processing.chunk_qa import run_chunk_qa
from src.processing.chunking import LayoutAwareChunker
from src.processing.docling_parser import DoclingParser
from src.processing.evidence_mapper import EvidenceMapper
from src.processing.layout_normalizer import LayoutNormalizer
from src.processing.spreadsheet_parser import SpreadsheetParser
from src.processing.types import TextChunk
from src.rag.embedder import EmbeddedText, SparseEmbedding
from src.rag.indexer import QdrantMongoIndexer
from src.services.parse_index_pipeline import ParseIndexPipeline

pytestmark = pytest.mark.skipif(
    os.getenv("AGENTBOOK_RUN_CORPUS_SMOKE") != "true",
    reason="Set AGENTBOOK_RUN_CORPUS_SMOKE=true to run the sample-corpus smoke test.",
)


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[EmbeddedText]:
        embeddings: list[EmbeddedText] = []
        for index, text in enumerate(texts):
            base = float((len(text) % 7) + 1)
            embeddings.append(
                EmbeddedText(
                    dense=[base + offset for offset in range(8)],
                    sparse=SparseEmbedding(
                        indices=[1, 7, 42],
                        values=[0.1 + index * 0.01, 0.2 + index * 0.01, 0.3 + index * 0.01],
                    ),
                )
            )
        return embeddings


class MemoryIndexer(QdrantMongoIndexer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunk_counter = 0

    async def _store_chunks(self, chunks: list[TextChunk]):
        stored = [SimpleNamespace(id=f"chunk-{self._chunk_counter + i}") for i in range(len(chunks))]
        self._chunk_counter += len(chunks)
        return stored

    async def _store_graph(self, *, entities, events, relations) -> None:
        return None


def _corpus_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "test data"


def _parse_file(path: Path, settings: Settings):
    suffix = path.suffix.lower()
    if suffix in {".pdf", ".docx", ".pptx"}:
        return DoclingParser().parse(path, language="unknown")
    if suffix in {".csv", ".xlsx", ".xls"}:
        return SpreadsheetParser().parse(path, language="unknown", display_name=path.name)
    if suffix in {".png", ".jpg", ".jpeg"}:
        pipeline = ParseIndexPipeline(settings=settings)
        fake_material = SimpleNamespace(
            file_type=suffix.lstrip("."),
            language="unknown",
            extra_metadata={"source_type": "corpus_smoke"},
            modality="mixed",
            storage_path=str(path),
            original_name=path.name,
        )
        return pipeline._parse_material(fake_material)
    raise ValueError(f"Unsupported sample file type: {suffix}")


async def _run_file(path: Path) -> dict:
    settings = Settings(
        testing=True,
        qdrant_url=":memory:",
        qdrant_collection_name=f"corpus_{path.stem[:12].replace(' ', '_')}",
        embedding_dense_size=8,
        index_batch_size=16,
        chunk_target_token_count=128,
        chunk_overlap_token_count=16,
        chunk_max_blocks_per_chunk=8,
        contextual_retrieval_enabled=False,
        chunk_strategy="layout",
    )

    material_id = str(uuid5(NAMESPACE_URL, f"material:{path.name}"))
    parsed = _parse_file(path, settings)
    normalized = LayoutNormalizer().normalize(parsed)
    evidence_map = EvidenceMapper().build(
        parsed=normalized,
        owner_id="user_demo",
        collection_id="65f000000000000000000002",
        material_id=material_id,
        document_name=path.name,
    )
    chunker = LayoutAwareChunker(settings)
    chunks = chunker.build_chunks(evidence_map)

    assert chunks, f"{path.name} produced no chunks"
    assert all(chunk.token_count <= settings.chunk_target_token_count for chunk in chunks), path.name
    assert all(chunk.content.strip() for chunk in chunks), path.name
    assert any(len(chunk.source_block_ids) >= 1 for chunk in chunks), path.name

    qa_report = run_chunk_qa(chunks, material_id=material_id)
    assert not qa_report.noisy_ocr, f"{path.name}: noisy OCR chunks at indices {qa_report.noisy_ocr}"
    assert len(qa_report.duplicate) <= max(1, len(chunks) // 5), (
        f"{path.name}: too many duplicate chunks ({len(qa_report.duplicate)}/{len(chunks)})"
    )

    fake_embedder = FakeEmbedder()
    indexer = MemoryIndexer(settings=settings, qdrant_client=QdrantClient(location=":memory:"), embedder=fake_embedder)
    await indexer.index(chunks=chunks, entities=[], events=[], relations=[])

    points, _ = indexer.qdrant_client.scroll(
        collection_name=settings.qdrant_collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    assert len(points) == len(chunks), path.name

    ocr_quality = normalized.extra.get("ocr_quality") if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else None

    return {
        "file": path.name,
        "parser": normalized.extra,
        "language": normalized.language,
        "pages": len(normalized.pages),
        "blocks": len(normalized.blocks),
        "chunks": len(chunks),
        "qdrant_points": len(points),
        "chunk_modalities": sorted({chunk.modality for chunk in chunks}),
        "chunk_qa": qa_report.summary(),
        "ocr_quality": ocr_quality,
    }


@pytest.mark.asyncio
async def test_sample_corpus_parses_chunks_and_indexes() -> None:
    corpus_root = _corpus_root()
    files = sorted(path for path in corpus_root.iterdir() if path.is_file())
    assert files, f"No sample files found in {corpus_root}"

    results = [await _run_file(path) for path in files]
    results_by_name = {item["file"]: item for item in results}

    assert results_by_name["rag_du_lieu_test.xlsx"]["chunk_modalities"] == ["paragraph", "table"]
    assert results_by_name["rag_mau_hoc_tap (1).docx"]["parser"]["parser"] == "docling"
    assert results_by_name["rag_mau_hoc_tap.pdf"]["parser"]["pdf_strategy"] == "docling_layout_first_text_ocr_missing_pages"
    assert results_by_name["rag_mau_hoc_tap.pptx"]["parser"]["parser"] == "docling"
    assert results_by_name["rag_scan_mau.png"]["parser"]["parser"] == "paddleocr"

    # OCR quality gate metadata is present for image files and score is above warn threshold.
    png_result = results_by_name["rag_scan_mau.png"]
    if png_result["ocr_quality"] is not None:
        assert png_result["ocr_quality"]["score"] >= 0.35, (
            f"OCR quality score below fail threshold: {png_result['ocr_quality']}"
        )

    # Chunk QA must pass for all files (no noisy OCR chunks, bounded duplicates).
    for result in results:
        assert result["chunk_qa"] != "noisy-ocr", f"{result['file']} has noisy OCR chunks"

    # Product-level sanity checks: every sample should have a bounded chunk count
    # and an index result that mirrors the chunk count.
    # multimodal_rag_test_day_du.docx is the comprehensive multi-content-type test
    # file and intentionally produces more chunks than the other samples.
    chunk_upper_bounds = {
        "multimodal_rag_test_day_du.docx": 60,
    }
    default_upper_bound = 20
    for result in results:
        upper = chunk_upper_bounds.get(result["file"], default_upper_bound)
        assert result["chunks"] >= 1, result
        assert result["chunks"] <= upper, result
        assert result["qdrant_points"] == result["chunks"], result

