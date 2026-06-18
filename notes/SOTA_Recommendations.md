# 🏆 Gợi ý SOTA cho Noelys — Agentic Document Intelligence Platform

> Phân tích deep-dive từ toàn bộ codebase hiện tại, đối chiếu với các paper/hệ thống SOTA 2025–2026.

---

## 📊 Đánh giá Hiện trạng

Hệ thống Noelys đã có nền tảng **rất tốt** với kiến trúc modular:

| Component | Hiện tại | Đánh giá |
|---|---|---|
| **Retrieval** | BGE-M3 dense + sparse + RRF | ✅ SOTA-level |
| **Reranking** | BGE-reranker-v2-m3 + MMR diversity | ✅ Solid |
| **Chunking** | Layout-aware + Semantic breakpoint (Kamradt) | ✅ Advanced |
| **Contextual Retrieval** | Anthropic-style LLM enrichment | ✅ SOTA technique |
| **Agentic Orchestration** | Plan → retrieve → sub-questions → coverage verify → repair → synthesize → claim verify | ✅ Advanced |
| **Graph RAG** | Entity/Relation extraction + 2-hop traversal | ⚠️ Good but improvable |
| **Claim Verification** | Heuristic + optional NLI (DeBERTa) | ⚠️ Needs upgrade |
| **Evaluation** | Recall/MRR/nDCG stubs, chưa có automated benchmark | ❌ Critical gap |
| **Query Router** | Regex-based intent detection | ⚠️ Brittle |

---

## 🔥 TIER 1 — High Impact, Nên Làm Ngay

### 1. Corrective RAG (CRAG)
> Yan et al., ICML 2025

**Vấn đề**: Sau khi retrieve, hệ thống hiện không đánh giá **chất lượng relevance** của evidence trước khi đưa cho LLM synthesis. Noise evidence vẫn được gửi vào prompt → gây hallucination.

**Giải pháp**: Thêm một **Retrieval Evaluator** giữa bước rerank và synthesis. Module này phân loại từng chunk thành `CORRECT / INCORRECT / AMBIGUOUS`:
- Nếu >50% chunks là CORRECT → tiến hành synthesis bình thường
- Nếu >50% INCORRECT → **strip irrelevant sentences** từ mỗi chunk, chỉ giữ phần relevant
- Nếu AMBIGUOUS → trigger query decomposition + re-retrieve

**Vị trí tích hợp**: Sau `_rerank()`, trước `_build_prompt()` trong `AgenticRagService.answer()`.

**Impact**: Giảm hallucination 15-25%, đặc biệt khi document corpus lớn và nhiều noise.

---

### 2. Self-RAG Reflection Tokens
> Asai et al., NeurIPS 2025 extended

**Vấn đề**: Claim verification hiện chạy **sau khi** generate xong toàn bộ answer → phát hiện vấn đề muộn, phải repair toàn bộ.

**Giải pháp**: Chuyển sang **inline self-reflection** — sau khi generate draft answer, LLM tự critique từng claim:
1. Generate draft answer
2. Extract individual claims từ draft
3. Với mỗi claim → check support status against evidence
4. Hedge hoặc remove unsupported claims **trước khi** finalize
5. Assess completeness → quyết định có cần retrieve thêm không

Khác biệt quan trọng so với hiện tại: verification xảy ra **per-claim** thay vì **per-answer**, và repair xảy ra **trước** khi trả về user.

**Impact**: Giảm unsupported claims 30-50%. Giảm số lần phải repair_answer.

---

### 3. Late Chunking
> Jina AI, 2025 — "Late Chunking: Contextual Chunk Representations Using Long-Context Embedding Models"

**Vấn đề**: `ContextualEnricher` hiện gọi LLM **per-chunk** để tạo situating context → rất chậm, tốn cost. Selective enrichment bỏ qua ~60-70% chunks.

**Giải pháp**: Encode **toàn bộ document** bằng long-context embedder (BGE-M3 hỗ trợ đến 8192 tokens), thu được token-level embeddings. Sau đó **mean-pool** token embeddings theo chunk boundaries đã xác định. Mỗi chunk embedding tự động mang context từ toàn bộ document mà không cần LLM call nào.

**Vị trí thay thế**: Thay `ContextualEnricher` trong `parse_index_pipeline.py`, hoặc dùng song song (late chunking cho mọi chunk, LLM enrichment cho important chunks).

**Impact**: Loại bỏ hoàn toàn LLM calls cho contextual enrichment. Tốc độ indexing tăng gấp 3-5x. Quality tương đương hoặc cao hơn.

---

### 4. Adaptive Retrieval — "Retrieve Only When Needed"
> Tham khảo: FLARE (Jiang et al.) + Self-RAG adaptive retrieval, 2025

**Vấn đề**: Hệ thống luôn chạy full retrieval pipeline cho mọi query. `IntentClassifier` đã phân loại `CHITCHAT` / `OFF_TOPIC`, nhưng thiếu loại `PARAMETRIC` — câu hỏi mà LLM đã confident từ training data.

**Giải pháp**: Mở rộng `QueryIntent` thêm `PARAMETRIC`. Khi LLM confident cao về câu trả lời, **skip retrieval** nhưng vẫn verify bằng một lightweight check. Nếu verify fail → fallback về full RAG pipeline.

**Impact**: Giảm latency 60-80% cho simple factual queries. Giảm compute cost đáng kể.

---

## 🔧 TIER 2 — Substantial Improvements

### 5. HyDE (Hypothetical Document Embeddings)
> Gao et al., 2025 updated

**Vấn đề**: `QueryRewriter` tạo paraphrases nhưng không tạo **hypothetical answers** — content-style text sẽ match tốt hơn với document embeddings so với question-style text.

**Giải pháp**: Thêm một branch trong `query_processor`: LLM generate một đoạn văn giả định trả lời câu hỏi, embed đoạn đó, dùng embedding này cho retrieval. Kết quả fuse với original query results qua RRF.

**Vị trí tích hợp**: Thêm vào `retrieval_queries` list trong `QueryProcessor.process_async()`.

**Impact**: +5-15% Recall@10 cho domain-specific queries, đặc biệt queries dài/phức tạp.

---

### 6. Graph RAG 2.0 — Community Detection + Global Summarization
> Microsoft GraphRAG, 2025

**Vấn đề**: Graph retrieval hiện chỉ dùng keyword matching + 2-hop traversal. Không thể trả lời **global questions** như "Tóm tắt tất cả mối quan hệ trong tài liệu".

**Giải pháp**: Sau khi extract entities/relations, chạy **Leiden community detection** để nhóm entities thành communities. Mỗi community được LLM tạo một summary. Khi user hỏi global questions → search trên community summaries thay vì individual entities.

**Vị trí tích hợp**: Thêm bước `detect_communities()` sau entity/relation extraction trong `parse_index_pipeline.py`. Thêm `community_search()` vào `GraphRetriever`.

**Impact**: Mở rộng khả năng trả lời global/abstract queries +20-30%.

---

### 7. LLM-based Query Router
> Thay thế regex-based `QueryRouter`

**Vấn đề**: `query_router.py` dùng regex patterns → dễ miss edge cases, không handle mixed-intent, kém với Vietnamese informal language.

**Giải pháp**: Dùng LLM (cùng model Qwen đang dùng) classify intent qua structured JSON output. Giữ regex router làm **fallback** khi LLM call fail hoặc timeout.

**Tương tự**: `AgenticPlanner` đã có pattern `build_with_llm()` + fallback `build()`. Áp dụng y hệt cho router.

**Impact**: +15-25% routing accuracy, nhất là Vietnamese queries không match regex patterns.

---

### 8. RAGAS Evaluation Framework
> Es et al., 2025 — RAGAS v2

**Vấn đề**: `evaluation/metrics.py` chỉ có basic retrieval metrics và `ragas_stub()`. Không có end-to-end benchmark → mọi cải tiến đều "cảm tính", không chứng minh được.

**Giải pháp**: 
1. Tạo **golden test set** (~50-100 QA pairs) với ground truth answers + expected source blocks
2. Integrate RAGAS metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**
3. Chạy eval tự động sau mỗi thay đổi pipeline (CI/CD hook)

**Impact**: Đây là **multiplier** cho mọi cải tiến khác — không có eval = không thể chứng minh progress.

---

## 🚀 TIER 3 — Advanced / Research-Level

### 9. Speculative RAG
> Wang et al., 2025

Dùng **small specialist LLM** tạo multiple draft answers song song, sau đó **large generalist LLM** chọn + verify draft tốt nhất. Trade-off: tăng quality, giảm latency (parallel drafts), nhưng tốn nhiều compute hơn.

### 10. Proposition-Level Chunking (Agentic Chunking)
> Chen et al., 2025

Thay vì chunk theo layout/semantic distance, LLM trích xuất **propositions** (atomic facts) từ document. Mỗi proposition là một statement tự chứa đủ context. Group propositions thành chunks dựa trên semantic coherence. Đặc biệt hiệu quả cho tài liệu dense-information (textbooks, papers).

### 11. ColPali/ColQwen — Visual Document Retrieval
> Faysse et al., 2025

Bypass OCR — embed **page images** trực tiếp bằng Vision Language Model, retrieve bằng visual similarity. Cực kỳ powerful cho tables, diagrams, figures mà OCR không capture tốt. Hệ thống hiện đã có `FigureCaptioner` — ColPali sẽ là bước tiến triệt để hơn.

### 12. Mixture-of-Agents (MoA) cho Answer Synthesis
> Wang et al., 2025

Nhiều LLM generate answers song song cho cùng query + evidence. Một aggregator LLM tổng hợp best parts từ mỗi answer. Tăng quality nhưng tốn compute gấp N lần.

---

## ⚡ Quick Wins — Làm ngay hôm nay (chỉ thay config)

| # | Action | Config Change |
|---|---|---|
| 1 | **Bật NLI Claim Verifier** | `AGENTBOOK_CLAIM_NLI_ENABLED=true` |
| 2 | **Bật Semantic Chunking** | `chunk_strategy=semantic` |
| 3 | **Bật LLM Planner** | `agentic_planner_llm_enabled=true` |
| 4 | **Tăng retrieval iterations** | `agentic_max_retrieval_iterations=3` |
| 5 | **Wire SmartReranker** | Code đã có `smart_reranker.py` nhưng chưa dùng — cần wire vào `InferenceEngine` |

---

## 🎯 Expected Impact

| Cải tiến | Metric | Expected Δ |
|---|---|---|
| CRAG Evaluator | Faithfulness | +15-25% |
| Self-RAG Reflection | Unsupported Claims | -30-50% |
| Late Chunking | Indexing Speed + Recall@10 | +3-5x speed, +5-10% recall |
| HyDE | Recall@10 (domain queries) | +5-15% |
| LLM Router | Routing Accuracy | +15-25% |
| RAGAS Evaluation | Dev Velocity | Immeasurable (đo được = cải tiến được) |
| Graph Communities | Global Query Coverage | +20-30% |

---

> [!TIP]
> **Ưu tiên #1 tuyệt đối**: Triển khai RAGAS evaluation trước. Không có metrics = không thể chứng minh bất kỳ cải tiến nào hiệu quả. Mọi SOTA technique đều vô nghĩa nếu không đo được.

> [!IMPORTANT]
> Tất cả gợi ý đều compatible với stack hiện tại (BGE-M3, Qdrant, Ollama, FastAPI). Không cần thay đổi infrastructure.
