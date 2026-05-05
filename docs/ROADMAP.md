# 🚀 AGENTBOOK - ROADMAP CẢI THIỆN

## Tổng quan
Roadmap này được xây dựng dựa trên phân tích kỹ thuật sâu của Principal AI Engineer, tập trung vào **ROI cao** và **khả năng mở rộng**.

---

## 📊 HIỆN TRẠNG

### Điểm mạnh (8.7/10)
- ✅ Hybrid Retrieval SOTA (Dense + Sparse + Lexical)
- ✅ Evidence Tracing hoàn hảo
- ✅ Semantic Chunking với tokenizer-accurate
- ✅ Multi-layer Guardrails
- ✅ Cross-lingual Support (VI → EN)

### Giới hạn hiện tại
- ⚠️ Capacity: 100K documents (realistic)
- ⚠️ Throughput: 450 queries/hour (single worker)
- ⚠️ Latency: 2-8s per query
- ⚠️ No request batching
- ⚠️ CPU-only inference

---

## 🎯 PHASE 1: Quick Wins (1-2 tuần)

### 1.1 Query Result Cache ✅ IMPLEMENTED
**File:** `backend/src/services/query_cache.py`

**Impact:**
- ⬇️ 90% latency cho repeated queries
- ⬆️ 10x throughput cho popular queries
- Redis-backed với TTL 1 hour

**Usage:**
```python
from src.services.query_cache import QueryResultCache

cache = QueryResultCache(redis_url=settings.redis_url)

# Check cache first
cached = cache.get(query, scope)
if cached:
    return cached

# Generate response
response = await inference_engine.answer(query, scope)

# Cache result
cache.set(query, scope, response)
```

### 1.2 Request Batching (TODO)
**Impact:** ⬆️ 3-5x throughput

**Implementation:**
```python
# backend/src/core/batch_llm.py
class BatchedLLM:
    async def generate(self, prompt: str) -> str:
        # Queue requests
        # Flush when batch_size reached or timeout
        # Batch inference
```

### 1.3 Adaptive Context Window (TODO)
**Impact:** ⬇️ 30% token waste, prevent overflow

**Implementation:**
```python
def build_adaptive_prompt(query, chunks, max_tokens):
    # Select top chunks until 90% of max_tokens
    # Prioritize by rerank_score
```

### 1.4 Async Parallel Processing (TODO)
**Impact:** ⬆️ 2x throughput

**Implementation:**
```python
# Parallel contextual enrichment
tasks = [enrich_one(chunk, sem) for chunk in chunks]
results = await asyncio.gather(*tasks)
```

**Estimated Results:**
- Throughput: 450 → 2,700 queries/hour (6x)
- Latency: 2-8s → 0.2-8s (90% cache hit)
- Cost: ⬇️ 40% (less redundant processing)

---

## ⚡ PHASE 2: Quality Improvements (2-3 tuần)

### 2.1 HyDE Query Expansion
**Impact:** ⬆️ 20-30% recall on abstract queries

**Implementation:**
```python
async def hyde_expand(query: str) -> List[str]:
    prompt = f"Write a factual answer to: {query}"
    hypothetical_doc = await llm.generate(prompt)
    return [query, hypothetical_doc]
```

### 2.2 Few-Shot Prompting
**Impact:** ⬆️ 15-20% answer quality

**Implementation:**
```python
if route_type == RouteType.COMPARISON:
    prompt = f"""
    Example 1: Compare L1 and L2...
    Example 2: Compare dropout and early stopping...
    
    Now answer: {query}
    """
```

### 2.3 Graph Centrality Caching
**Impact:** ⬆️ 30% graph retrieval speed

**Implementation:**
```python
# Pre-compute PageRank scores
pagerank = nx.pagerank(graph)
await redis.hset(f"centrality:{collection_id}", mapping=pagerank)
```

### 2.4 RAGAS Evaluation
**Impact:** Measurable end-to-end quality

**Implementation:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy],
)
```

**Estimated Results:**
- Recall@5: 80% → 90%
- Answer Quality: +20%
- Graph Retrieval: 2x faster

---

## 🎯 PHASE 3: Scalability (3-4 tuần)

### 3.1 ColBERT Multi-Vector Retrieval
**Impact:** ⬆️ 15-20% recall, better for long chunks

**Architecture:**
```python
# Store N vectors per chunk (sentence-level)
for sentence in chunk.sentences:
    vector = embedder.encode(sentence)
    qdrant.upsert(f"{chunk_id}:{i}", vector)

# MaxSim scoring
score = mean([max(sim(q_vec, d_vec) for d_vec in doc_vecs) 
              for q_vec in query_vecs])
```

### 3.2 vLLM Inference Engine
**Impact:** ⬆️ 5-10x throughput

**Implementation:**
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-3B",
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

# Continuous batching automatically
outputs = llm.generate(prompts, sampling_params)
```

### 3.3 Distributed Embedding Pipeline
**Impact:** ⬆️ 10x indexing speed

**Implementation:**
```python
from ray import serve

@serve.deployment(num_replicas=4)
class EmbeddingService:
    def __init__(self):
        self.embedder = BGEM3Embedder(device="cuda")
    
    async def embed_batch(self, texts: List[str]):
        return self.embedder.encode(texts)
```

### 3.4 Qdrant Sharding
**Impact:** Support 1M+ documents

**Implementation:**
```python
qdrant_cluster = [
    QdrantClient("qdrant-shard-1:6333"),
    QdrantClient("qdrant-shard-2:6333"),
    QdrantClient("qdrant-shard-3:6333"),
]

def get_shard(collection_id: str) -> QdrantClient:
    return qdrant_cluster[hash(collection_id) % len(qdrant_cluster)]
```

**Estimated Results:**
- Capacity: 100K → 1M+ documents
- Throughput: 2,700 → 18,000 queries/hour
- Indexing: 100 docs/hour → 1,000 docs/hour

---

## 🔬 PHASE 4: Advanced Features (4-6 tuần)

### 4.1 Neo4j Migration
**Impact:** Advanced graph algorithms

**Features:**
- PageRank, community detection
- Cypher query language
- Native graph traversal

### 4.2 Agentic Chunking
**Impact:** Better semantic coherence

**Implementation:**
```python
# LLM decides chunk boundaries
boundaries = await llm.analyze_document_structure(blocks)
chunks = build_from_boundaries(blocks, boundaries)
```

### 4.3 Multi-Modal Retrieval
**Impact:** Support images, tables, figures

**Implementation:**
```python
# CLIP for image retrieval
image_embedding = clip.encode_image(image)
text_embedding = clip.encode_text(query)
similarity = cosine_similarity(image_embedding, text_embedding)
```

### 4.4 Active Learning
**Impact:** Continuous improvement

**Implementation:**
```python
# Collect user feedback
feedback = await collect_feedback(query, response)

# Retrain reranker
if len(feedback_dataset) > 1000:
    fine_tune_reranker(feedback_dataset)
```

---

## 📊 EXPECTED OUTCOMES

### After Phase 1 (2 tuần)
- Throughput: 450 → 2,700 queries/hour (6x)
- Latency: 2-8s → 0.2-8s (with cache)
- Cost: ⬇️ 40%

### After Phase 2 (1 tháng)
- Recall@5: 80% → 90%
- Answer Quality: +20%
- Graph Retrieval: 2x faster

### After Phase 3 (2 tháng)
- Capacity: 100K → 1M+ documents
- Throughput: 2,700 → 18,000 queries/hour
- Indexing: 100 → 1,000 docs/hour

### After Phase 4 (3 tháng)
- Multi-modal support
- Advanced graph reasoning
- Continuous learning

---

## 🎯 RECOMMENDED NEXT STEPS

### Week 1-2: Quick Wins
1. ✅ Integrate Query Result Cache
2. ⬜ Implement Request Batching
3. ⬜ Add Adaptive Context Window
4. ⬜ Test with load testing (Apache Bench)

### Week 3-4: Quality
5. ⬜ Implement HyDE expansion
6. ⬜ Add Few-shot examples
7. ⬜ Setup RAGAS evaluation
8. ⬜ Benchmark improvements

### Month 2: Scalability
9. ⬜ Setup vLLM (if GPU available)
10. ⬜ Implement distributed embedding
11. ⬜ Add Qdrant sharding
12. ⬜ Load test with 1M documents

### Month 3: Advanced
13. ⬜ Evaluate Neo4j migration
14. ⬜ Prototype ColBERT retrieval
15. ⬜ Design multi-modal pipeline
16. ⬜ Plan active learning system

---

## 💡 QUICK START

### Test Query Cache
```bash
cd backend
python -c "
from src.services.query_cache import QueryResultCache
cache = QueryResultCache('redis://localhost:6379/0')
print(cache.stats())
"
```

### Benchmark Current System
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test throughput
ab -n 1000 -c 10 -p query.json -T application/json \
   http://localhost:8000/api/v1/query
```

### Monitor Performance
```bash
# Redis stats
redis-cli info stats

# Qdrant metrics
curl http://localhost:6333/metrics
```

---

## 📚 REFERENCES

- **HyDE**: Gao et al. 2022 - "Precise Zero-Shot Dense Retrieval"
- **ColBERT**: Khattab & Zaharia 2020 - "ColBERT: Efficient and Effective Passage Search"
- **vLLM**: Kwon et al. 2023 - "Efficient Memory Management for LLM Serving"
- **RAGAS**: Explodinggradients 2023 - "RAG Assessment Framework"

---

**Last Updated:** 2026-05-02
**Status:** Phase 1 in progress (Query Cache ✅)
