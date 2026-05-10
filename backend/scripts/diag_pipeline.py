"""
Deep pipeline diagnostic: chunk quality, embedding, retrieval, reranker.
Run from backend/: python scripts/diag_pipeline.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter, defaultdict

OWNER_ID = "user_demo"
COLLECTION_ID = "69fc3c0949fae4625be50223"

TEST_QUERIES = [
    "L1 regularization tạo ra sparse weights",
    "Dropout giảm overfitting như thế nào",
    "What is gradient descent",
]

async def main():
    import motor.motor_asyncio
    from beanie import init_beanie
    from qdrant_client import QdrantClient
    from src.core.config import get_settings
    from src.models.chunk import Chunk
    from src.models.material import Material
    from src.rag.embedder import BGEM3Embedder
    from src.rag.reranker import CrossEncoderReranker
    from src.rag.retriever import HybridRetriever
    from src.rag.types import RetrievalScope

    settings = get_settings()
    client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
    from src.database import DOCUMENT_MODELS
    await init_beanie(database=client[settings.mongodb_database], document_models=DOCUMENT_MODELS)

    from beanie import PydanticObjectId
    col_oid = PydanticObjectId(COLLECTION_ID)

    # ── 1. CORPUS STATS ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  1. CORPUS STATS")
    print("="*60)
    chunks = await Chunk.find(
        Chunk.owner_id == OWNER_ID,
        Chunk.collection_id == col_oid,
    ).to_list()
    print(f"Total chunks in MongoDB: {len(chunks)}")

    # Content length distribution
    lengths = [len(c.content or "") for c in chunks]
    print(f"Content length — min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)}")

    # Blocks per chunk
    block_counts = [len(c.source_block_ids) for c in chunks]
    print(f"Blocks/chunk — min={min(block_counts)} max={max(block_counts)} avg={sum(block_counts)//len(block_counts)}")
    heavy = [c for c in chunks if len(c.source_block_ids) > 20]
    print(f"Chunks with >20 blocks (noisy): {len(heavy)}")

    # Per-document chunk counts
    mat_ids = list({str(c.material_id) for c in chunks})
    mats = await Material.find({"_id": {"$in": [PydanticObjectId(m) for m in mat_ids]}}).to_list()
    mat_names = {str(m.id): m.original_name for m in mats}
    by_doc = Counter(str(c.material_id) for c in chunks)
    print("\nChunks per document:")
    for mid, count in by_doc.most_common():
        name = mat_names.get(mid, mid)[:50]
        print(f"  {count:3d}  {name}")

    # ── 2. CHUNK QUALITY SAMPLE ────────────────────────────────────────
    print("\n" + "="*60)
    print("  2. CHUNK QUALITY SAMPLE (short content chunks)")
    print("="*60)
    short_chunks = sorted(chunks, key=lambda c: len(c.content or ""))[:5]
    for c in short_chunks:
        name = mat_names.get(str(c.material_id), "?")[:30]
        print(f"  len={len(c.content):4d} blocks={len(c.source_block_ids):2d} doc={name}")
        print(f"    content: {repr((c.content or '')[:120])}")

    # ── 3. QDRANT STATS ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  3. QDRANT COLLECTION STATS")
    print("="*60)
    qdrant = QdrantClient(url=settings.qdrant_url)
    info = qdrant.get_collection(settings.qdrant_collection_name)
    print(f"Vectors count:   {info.vectors_count}")
    print(f"Points count:    {info.points_count}")
    print(f"Indexed vectors: {info.indexed_vectors_count}")
    print(f"Status:          {info.status}")

    # ── 4. EMBEDDING TEST ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  4. EMBEDDING QUALITY")
    print("="*60)
    embedder = BGEM3Embedder(settings)
    test_texts = [
        "L1 regularization tạo ra sparse weights",
        "L1 regularization creates sparse weights by driving small weights to zero",
        "Hôm nay trời đẹp",  # off-topic
    ]
    embeddings = embedder.encode(test_texts)
    if embeddings and len(embeddings) == 3:
        import numpy as np
        def cosine(a, b):
            a, b = np.array(a), np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        vi_en = cosine(embeddings[0].dense, embeddings[1].dense)
        vi_off = cosine(embeddings[0].dense, embeddings[2].dense)
        print(f"Dense dim: {len(embeddings[0].dense)}")
        print(f"Sparse nnz (VI query): {len(embeddings[0].sparse.indices)}")
        print(f"Cosine(VI query, EN equivalent): {vi_en:.3f}  (should be >0.7)")
        print(f"Cosine(VI query, off-topic):     {vi_off:.3f}  (should be <0.5)")
    else:
        print("Embedding failed!")

    # ── 5. RETRIEVAL + RERANKER QUALITY ──────────────────────────────
    print("\n" + "="*60)
    print("  5. RETRIEVAL + RERANKER QUALITY")
    print("="*60)
    retriever = HybridRetriever(settings=settings, qdrant_client=qdrant, embedder=embedder)
    reranker = CrossEncoderReranker(settings)
    scope = RetrievalScope(owner_id=OWNER_ID, collection_id=COLLECTION_ID)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query!r}")
        chunks_retrieved = await retriever.retrieve(query=query, scope=scope, limit=8)
        print(f"  Retrieved: {len(chunks_retrieved)} chunks (pre-rerank)")
        for i, c in enumerate(chunks_retrieved[:3], 1):
            name = c.document_name[:30]
            preview = (c.content or "")[:80].replace("\n", " ")
            print(f"  [{i}] fused={c.fused_score:.4f}  doc={name}")
            print(f"       content: {preview!r}")
            noise = sum(1 for e in c.evidence if len((e.snippet_original or "").strip()) < 40)
            print(f"       ev_blocks={len(c.evidence)} noise={noise}")

        # Rerank
        reranked = reranker.rerank(query=query, chunks=chunks_retrieved, limit=5)
        print(f"  After rerank (top 3):")
        for i, c in enumerate(reranked[:3], 1):
            name = c.document_name[:30]
            score = c.rerank_score
            preview = (c.content or "")[:80].replace("\n", " ")
            print(f"  [{i}] rerank={score:.4f}  doc={name}")
            print(f"       content: {preview!r}")

    # ── 6. RECALL CHECK ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  6. KEYWORD RECALL — does relevant content exist in index?")
    print("="*60)
    keywords = {
        "sparse weights": ["L1", "sparse"],
        "gradient descent": ["gradient", "descent"],
        "dropout": ["dropout", "Dropout"],
        "batch normalization": ["Batch Normalization", "batch norm"],
    }
    for topic, kws in keywords.items():
        hits = [c for c in chunks if any(kw.lower() in (c.content or "").lower() for kw in kws)]
        print(f"  {topic!r:25s}: {len(hits):3d} chunks contain keywords {kws}")
        if hits:
            best = max(hits, key=lambda c: sum(kw.lower() in (c.content or "").lower() for kw in kws))
            preview = (best.content or "")[:120].replace("\n", " ")
            print(f"    best match: {preview!r}")

asyncio.run(main())
