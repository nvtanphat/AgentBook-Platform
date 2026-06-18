# Research RAG mới nhất (09/2025 → 05/2026)

> Nguồn: web search 05/2026 — cập nhật cho đồ án Noelys

---

## 🆕 Phát kiến quan trọng từ 09/2025

| Năm/tháng | Tech | Đóng góp | Apply cho Noelys? |
|---|---|---|---|
| **06/2025** | **LazyGraphRAG** (MSFT) | Giảm chi phí indexing GraphRAG xuống **0.1%** của bản gốc, giữ nguyên chất lượng | ✅ Thay graph_builder hiện tại |
| **07/2025** | **Practical GraphRAG** (arXiv 2507.03226) | Dependency-based extraction đạt **94%** chất lượng của LLM-based mà rẻ hơn rất nhiều | ✅ Replace relation_extractor |
| **10/2025** | **LinearRAG** | Relation-free graph construction — bỏ qua relation extraction, vẫn multi-hop được | ✅ Hoặc thay thế GraphRAG hẳn |
| **2026** | **A-RAG** (Hierarchical Retrieval Interfaces) | Agent có 3 tools: keyword search, semantic search, chunk read. SOTA HotpotQA/2WikiMQA với GPT-5-mini | ⚡ Replace fixed pipeline |
| **2026** | **MMOA-RAG** | Mỗi component RAG = 1 RL agent, optimize joint F1 thay vì độc lập | 🔬 Long-term |
| **2026** | **ICLR'26 GraphRAG-Bench** | Standardized benchmark cho graph-based RAG, vừa được accept | ✅ Dùng eval |
| **05/2025** | **ViDoRe V2** | Multimodal RAG benchmark mới, harder + multilingual | ✅ Eval visual retrieval |
| **2025-2026** | **ColQwen2.5** | ColPali với Qwen2.5-VL backbone — top ViDoRe V2 leaderboard | ✅ Cho PDF có layout |
| **2025-2026** | **Cohere Embed 4, voyage-multimodal-3.5** | Single-vector multimodal — alternative cho ColPali multi-vector | 🔬 Trade-off cost/quality |
| **2026** | **Search-o1** | RAG như một test-time compute method, integrate với reasoning models | ⚡ Hot trend |

---

## 🔥 3 trends nóng nhất 09/2025 - 05/2026

### 1. Agentic RAG đã thay thế "pipeline RAG"

- Survey [arXiv 2603.07379 SoK Agentic RAG](https://arxiv.org/abs/2603.07379) systematize toàn bộ field
- Pattern chính: **reflection + planning + tool use + multi-agent**
- A-RAG framework: model tự chọn keyword vs semantic vs chunk read theo từng step
- Pipeline hard-coded (như Noelys hiện tại) đang **outdated** — không thua kém Self-RAG nhưng kém flexibility

### 2. Reasoning models (DeepSeek-R1, o3, Gemini 3) + RAG

- R1-Zero generate hàng nghìn reasoning tokens, retrieve liên tục giữa các bước think
- IBM watsonx tutorial: "Improve DeepSeek-R1 Reasoning with RAG"
- Search-o1: 1 trong 5 phương pháp scale test-time compute
- Implication: LLM 4B local như Qwen3 không đủ — cần model có thinking tag

### 3. Late interaction trở thành mainstream

- Late Interaction Workshop @ **ECIR 2026** (event riêng) → cộng đồng đã consolidate
- **PLAID**: ColBERTv2 nhanh 7× GPU / 45× CPU
- Pattern chuẩn 2026: **bi-encoder (BGE-M3) → top-100 → ColBERT rerank → top-k → LLM**
- Hybrid SPLADE ∪ BM25 ∪ dense ∪ ColBERT đang là default

---

## 🎯 Recommendations cập nhật cho Noelys

### Phase 1 — Quick wins (1-2 tuần)

1. **Replace BGE reranker bằng ColQwen2** hoặc giữ BGE nhưng add ColBERT layer ([ColBERT github](https://github.com/stanford-futuredata/ColBERT))
2. **Switch graph_builder sang LazyGraphRAG hoặc LinearRAG** — fix vấn đề MongoDB regex slow (cut 25s timeout)
3. **Adopt RAGAS official lib** (400K+ monthly downloads) — replace regex-based faith metric

### Phase 2 — Architecture refactor (1 tháng)

4. **Migrate sang Agentic RAG pattern (A-RAG style)** — cho LLM 3 tools (keyword/semantic/graph) thay vì hard-coded route + retrieve + rerank pipeline
5. **Adopt ColPali/ColQwen2.5 cho visual chunks** — Noelys đang có visual_embedder nhưng chưa SOTA
6. **Tích hợp DeepSeek-R1 hoặc Qwen3-Reasoning** thay Qwen3 4B base — gain test-time compute benefits

### Phase 3 — Research-grade (2-3 tháng)

7. **GraphRAG-Bench eval** — submit benchmark khi ICLR'26 release
8. **MMOA-RAG approach** — RL joint training các components
9. **Hierarchical retrieval (RAPTOR successor)** — chunking-aware reasoning

---

## 📚 Reading list

- [SoK: Agentic RAG (arXiv 2603.07379)](https://arxiv.org/abs/2603.07379) — must-read survey
- [A-RAG paper + code (arXiv 2602.03442)](https://arxiv.org/abs/2602.03442) + [GitHub](https://github.com/Ayanami0730/arag)
- [Practical GraphRAG (arXiv 2507.03226)](https://arxiv.org/abs/2507.03226)
- [VoltAgent's curated 2026 agent papers](https://github.com/VoltAgent/awesome-ai-agent-papers)
- [Awesome-GraphRAG (DEEP-PolyU)](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
- [Late Interaction overview (Weaviate)](https://weaviate.io/blog/late-interaction-overview)
- [GitHub aishwaryanr — RAG research updates table](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md)

---

## ✅ Ưu tiên cho đồ án

| # | Thay đổi | Lý do |
|---|---|---|
| 1 | **A-RAG pattern** với 3 retrieval tools cho LLM | Direction mainstream 2026, Qwen3 4B đủ demo (chỉ cần tool calling) |
| 2 | **LazyGraphRAG** | Giải quyết vấn đề KG slow (đang phải timeout 25s) |
| 3 | **RAGAS official + GraphRAG-Bench eval** | Metrics chuẩn industry, defendable trong báo cáo |

---

## 📎 Sources (full)

- [Retrieval-Augmented Generation: A Comprehensive Survey (arXiv 2506.00054)](https://arxiv.org/html/2506.00054v1)
- [SoK: Agentic RAG (arXiv 2603.07379)](https://arxiv.org/abs/2603.07379)
- [A-RAG: Hierarchical Retrieval Interfaces (arXiv 2602.03442)](https://arxiv.org/abs/2602.03442)
- [Next-Gen Agentic RAG with LangGraph 2026](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8)
- [Practical GraphRAG (arXiv 2507.03226)](https://arxiv.org/abs/2507.03226)
- [GraphRAG in 2026 Practical Buyer's Guide](https://medium.com/@tongbing00/graphrag-in-2026-a-practical-buyers-guide-to-knowledge-graph-augmented-rag-43e5e72d522d)
- [RAG Evaluation 2026 Metrics](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
- [Top 5 RAG Evaluation Platforms 2026](https://www.getmaxim.ai/articles/top-5-rag-evaluation-platforms-in-2026-2/)
- [Multimodal RAG in 2026: ColPali landscape](https://bigdataboutique.com/blog/multimodal-rag-retrieval-over-images-pdfs-and-text)
- [Beyond OCR — ColPali changing RAG](https://medium.com/@mudassar.hakim/beyond-ocr-why-colpali-is-changing-how-we-build-rag-for-documents-2ebeb853e400)
- [Late Interaction Workshop ECIR 2026](https://www.lateinteraction.com/)
- [DeepSeek-R1 + RAG (IBM watsonx)](https://www.ibm.com/think/tutorials/deepseek-reasoning-improvements-with-rag-watsonx-ai)
- [Awesome-GraphRAG (DEEP-PolyU)](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
- [Top 10 Open-source Reasoning Models 2026](https://www.clarifai.com/blog/top-10-open-source-reasoning-models-in-2026)
