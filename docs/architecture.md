# AgentBook Architecture

AgentBook is a FastAPI + MongoDB Atlas + Qdrant Graph RAG system for grounded document intelligence over study materials.

## Runtime Services

- FastAPI backend exposes upload, query, evidence, graph, and admin APIs.
- MongoDB Atlas stores collections, material metadata, parsed pages/blocks, chunks, graph entities/events/relations, query logs, and feedback.
- Qdrant stores dense BGE-M3 vectors and BGE-M3 sparse vectors for hybrid retrieval.
- Redis + Celery run parse/index jobs asynchronously.
- Vite + React dashboard provides upload, chat, evidence, compare, graph, and mindmap views.

## Pipeline

1. Upload validates file type, size, MIME, magic bytes, checksum filename, and scoped storage.
2. Celery parses with an accuracy-first router: Docling layout/table parsing for PDF/DOCX/PPTX, pypdf text fallback and PaddleOCR only for missing PDF pages, SpreadsheetParser for CSV/XLS/XLSX, and PaddleOCR for printed images. Handwriting passes an image quality and confidence gate before it can become evidence.
3. Layout normalizer preserves page, block, bbox, source language, and confidence.
4. Evidence mapper creates block-level evidence refs.
5. Chunker creates layout-aware chunks while preserving `owner_id`, `collection_id`, `material_id`, `page`, `block_id`, `bbox`, and `snippet_original`.
6. Entity/event/relation extraction builds a lightweight MongoDB graph. Following the RAG-Anything pattern, the graph includes text-semantic edges plus block-level cross-modal edges such as entity-to-block mentions, heading-to-content containment, and adjacency/caption links between text, tables, figures, equations, OCR text, and handwriting.
7. BGE-M3 embeddings are upserted into Qdrant.
8. Query path uses scoped dense + sparse retrieval, optional graph expansion, reranking, refusal guardrails, and grounded synthesis.

## Safety

All retrieval requires `owner_id` plus `collection_id` or explicit `material_ids`. Answers are generated only from retrieved evidence. Citations always point to original source text, page, block, and bbox when available.
