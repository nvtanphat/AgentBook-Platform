"""
Quick eval: retrieval quality + citation coverage, no LLM generation.

Chạy retrieval cho từng câu hỏi, kiểm tra:
- Top chunks có relevant không (human review)
- Citation sources có đúng document không
- Confidence score của reranker

Usage:
    cd backend
    python scripts/quick_eval.py \
        --owner-id user_demo \
        --collection-id 69fc3c0949fae4625be50223
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import get_settings
from src.database import init_database
from src.models.chunk import Chunk
from src.rag.embedder import BGEM3Embedder
from src.rag.query_processor import QueryProcessor
from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.types import RetrievalScope
from src.rag.vector_store import get_qdrant_client_for_settings
from beanie import PydanticObjectId

# Câu hỏi cố định theo topic ML (không cần LLM sinh)
ML_QUESTIONS = [
    # Factual
    ("factual",    "Regularization là gì và tại sao cần dùng?"),
    ("factual",    "L1 và L2 regularization khác nhau như thế nào?"),
    ("factual",    "Dropout hoạt động như thế nào trong quá trình training?"),
    ("factual",    "Overfitting là gì?"),
    ("factual",    "What is gradient descent?"),
    # Summarization
    ("summarization", "Tóm tắt các kỹ thuật regularization phổ biến"),
    ("summarization", "Nêu các bước trong quá trình training một mô hình ML"),
    # Comparison
    ("comparison", "So sánh L1 và L2 regularization"),
    ("comparison", "Dropout và Batch Normalization khác nhau như thế nào?"),
    # Graph/relation
    ("graph_relation", "Dropout ảnh hưởng như thế nào đến overfitting?"),
    ("graph_relation", "Regularization và generalization có liên quan như thế nào?"),
    # Claim check
    ("claim_check", "Dropout có giúp giảm overfitting không?"),
    ("claim_check", "L1 regularization tạo ra sparse weights đúng không?"),
    # Off-topic (phải từ chối)
    ("off_topic_should_refuse", "Thủ đô của nước Pháp là gì?"),
    ("off_topic_should_refuse", "Hôm nay thời tiết thế nào?"),
    # False premise (phải correct)
    ("false_premise", "Tại sao L2 regularization tạo ra sparse weights?"),
    ("false_premise", "Vì sao dropout làm tăng overfitting?"),
    # Cross-lingual
    ("cross_lingual", "What is the difference between L1 and L2 regularization?"),
    ("cross_lingual", "How does dropout prevent overfitting?"),
    # Anaphora
    ("anaphora", "Nó có ưu điểm gì so với phương pháp trước?"),
]


async def retrieve_for_query(
    *,
    query: str,
    scope: RetrievalScope,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_k: int = 5,
) -> list[dict]:
    processor = QueryProcessor()
    processed = processor.process(query)

    all_chunks = []
    for rq in processed.retrieval_queries:
        chunks = await retriever.retrieve(query=rq, scope=scope, limit=20)
        all_chunks.extend(chunks)

    # Dedupe by chunk_id
    seen, deduped = set(), []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    if reranker and deduped:
        reranked = reranker.rerank(query=query, chunks=deduped, limit=top_k)
    else:
        reranked = deduped[:top_k]

    return [
        {
            "chunk_id": c.chunk_id,
            "document_name": c.document_name,
            "page": c.source_pages[:1][0] if c.source_pages else None,
            "score": round(c.fused_score or 0.0, 4),
            "reranker_score": round(getattr(c, "reranker_score", None) or 0.0, 4),
            "content_preview": (c.content or "")[:500],
        }
        for c in reranked
    ]


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    print(f"\n{'='*60}")
    print(f"  Noelys Quick Retrieval Eval")
    print(f"  Owner:      {args.owner_id}")
    print(f"  Collection: {args.collection_id}")
    print(f"{'='*60}\n")

    await init_database(settings)

    scope = RetrievalScope(
        owner_id=args.owner_id,
        collection_id=args.collection_id,
    )

    qdrant_client = get_qdrant_client_for_settings(settings)
    embedder = BGEM3Embedder(settings)
    retriever = HybridRetriever(settings=settings, qdrant_client=qdrant_client, embedder=embedder)
    reranker = None  # skip reranker in eval script to avoid OOM/segfault

    results = []
    for i, (qtype, query) in enumerate(ML_QUESTIONS):
        print(f"[{i+1:>2}/{len(ML_QUESTIONS)}] [{qtype}] {query[:60]}")
        try:
            chunks = await retrieve_for_query(
                query=query,
                scope=scope,
                retriever=retriever,
                reranker=reranker,
            )
            top_score = chunks[0]["score"] if chunks else 0.0
            docs = list({c["document_name"] for c in chunks})
            print(f"        fused_score={top_score:.3f}  chunks={len(chunks)}  docs={docs}")
        except Exception as exc:
            print(f"        ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            chunks = []

        results.append({
            "id": f"q{i+1:03d}",
            "query_type": qtype,
            "query": query,
            "retrieved_chunks": chunks,
            "top_reranker_score": chunks[0]["score"] if chunks else 0.0,
            "retrieved_docs": list({c["document_name"] for c in chunks}),
            # Fill after review:
            "retrieval_ok": None,   # true / false / partial
            "notes": "",
        })

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick stats
    has_results = [r for r in results if r["retrieved_chunks"]]
    avg_score = sum(r["top_reranker_score"] for r in has_results) / len(has_results) if has_results else 0
    off_topic = [r for r in results if "off_topic" in r["query_type"]]
    off_topic_retrieved = [r for r in off_topic if r["retrieved_chunks"]]

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL STATS (trước human review)")
    print(f"{'─'*60}")
    print(f"  Queries run:         {len(results)}")
    print(f"  Got results:         {len(has_results)}/{len(results)}")
    print(f"  Avg top score:       {avg_score:.3f}")
    print(f"  Off-topic retrieved: {len(off_topic_retrieved)}/{len(off_topic)}  (lý tưởng = 0)")
    print(f"{'─'*60}")
    print(f"  Saved: {out.resolve()}")
    print(f"\n  Tiếp theo: mở file, điền 'retrieval_ok': true/false/partial")
    print(f"  Chạy: python scripts/score_retrieval_eval.py --input {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--output", default="eval_results/retrieval_eval.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args))
