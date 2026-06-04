<div align="center">
  <h1>🚀 AgentBook</h1>
  <p><strong>Evidence-grounded multi-domain assistant for complex documents — agentic RAG with verified citations across PDFs, slides, tables, scans, handwriting, and audio.</strong></p>

  [![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
  [![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)](https://react.dev/)
  [![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
  [![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-EF4444?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
  [![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
  [![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?style=flat-square&logo=ollama&logoColor=black)](https://ollama.com/)
  [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
</div>

AgentBook turns heterogeneous documents into a queryable, citation-backed knowledge surface. Every sentence in an answer is traceable to a chunk on a specific page, block, or audio segment of the source document. The system is built for bilingual Vietnamese–English cross-domain settings: users can ask in Vietnamese over English documents (or vice versa) and receive grounded answers without losing the original-language evidence.


---

## Highlights

- **Universal ingestion** — PDF, DOCX, PPTX, XLSX, CSV, PNG/JPG (printed + handwritten), and audio (MP3/WAV/M4A/FLAC/OGG/WebM) flow through a single block-level evidence schema. Bounding boxes for text/figures, page numbers, table row/column anchors, and audio start/end timestamps are preserved end-to-end so every citation can be rendered visually.
- **Hybrid retrieval** — BGE-M3 produces dense + sparse vectors in one pass; both are queried against Qdrant and fused with Reciprocal Rank Fusion (RRF). A BGE cross-encoder reranks the top-30 candidates. Multi-query rewriting kicks in for hard-recall questions, and graph traversal kicks in when the query mentions relations (`liên quan`, `tác động`, `phụ thuộc`, `causes`, `depends on`, …).
- **Agentic reasoning** — A bounded state-machine pipeline (Planner → Director → CRAG Critic → Reranker → Synthesizer → Guardrails → Sentence-Level Coverage Gate) replaces free-form ReAct. Each stage has a single responsibility, a typed contract, and a measurable verdict written back to the shared `AgentState` blackboard, which is what the UI renders as the *Agent Trace* panel.
- **Cross-lingual robustness** — Native VI↔EN routing: original-language query keeps recall, a translated variant catches paraphrases, RRF fuses both. The claim verifier auto-skips when answer-language ≠ chunk-language (token-overlap NLI is meaningless across languages), so EN queries over a VN corpus return grounded EN answers instead of spurious refusals.
- **Calibrated refusal** — Off-topic and chitchat are refused in 2–10 s via an intent-classifier shortcut. On-topic questions with weak grounding surface a *partial* badge through the SLEC gate instead of being silently fabricated; only the floor case (weighted SLEC < `refuse_below`) becomes a hard refusal.
- **Pixel-accurate evidence UI** — Citations carry `doc_id`, `page`, `block_id`, `bbox`, `snippet_original` and (optionally) `snippet_translated`. Clicking `[1]` scrolls the PDF / slide viewer to the exact region, plays an audio segment, or highlights a table row.
- **Anti-hallucination post-processing** — Acronym expansions invented by the LLM are stripped when not verbatim in evidence (catches both `RAG (Relevant Answer Generation)` and `RAG là viết tắt của …` patterns). Sentences whose detected language drifts away from the requested answer language are dropped before citation injection.
- **Knowledge graph + mindmap** — Entities (concept / model / dataset / metric / framework) and typed relations (`references`, `mentioned_in_block`, `co_located_with`, `section_contains`) are extracted at ingestion. A concept graph and an LLM-summarised topical mindmap are derived from this layer per collection.

---

## Architecture

```
┌──────────────────────── Ingestion ────────────────────────┐
│  Upload  →  Docling parse  →  OCR / handwriting / audio   │
│           →  layout normalize  →  chunking (token-aware)  │
│           →  contextual enrichment  →  entity + relation  │
│             extraction  →  embedding (BGE-M3 dense+sparse)│
│           →  Qdrant index  +  Mongo evidence store        │
└────────────────────────────────────────────────────────────┘
                                │
┌──────────────────────── Query  ───────────────────────────┐
│  Intent classifier  (chitchat/off-topic shortcut)         │
│        │                                                  │
│  Query processor  (anaphora, language detect, multi-query)│
│        │                                                  │
│  Planner agent  →  Director agent  (text / graph / per-   │
│                                     source retrieval)     │
│        │                                                  │
│  CRAG Critic  →  Cross-encoder rerank  →  Synthesizer     │
│        │                                                  │
│  Guardrails  (NLI claim verifier, contradiction detector) │
│        │                                                  │
│  Sentence-Level Coverage  (SLEC drop / hedge / refuse)    │
│        │                                                  │
│  Response parser  (citation inject, acronym strip,        │
│                    language-drift strip)                  │
└────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Ingestion pipeline

1. **Upload + safety gate** — MIME / magic-byte validation, path-traversal guard, per-owner storage scoping, configurable size limit (`max_file_size_mb`, currently 100 MB to accommodate audio).
2. **Parse** — Docling handles PDF/DOCX/PPTX into typed blocks (text / table / figure / equation). Spreadsheets pass through a dedicated row-to-sentence parser. Audio is transcribed by Faster-Whisper with per-segment timestamps.
3. **OCR / handwriting** — Scanned pages are routed to EasyOCR (printed) or a VLM fallback (handwriting). Image-quality and OCR-confidence gates label low-quality outputs so the answer pipeline can refuse instead of citing noise.
4. **Layout normalize** — Reading order, multi-column merge, header/footer stripping, table-row deduplication, figure-caption pairing.
5. **Chunking** — Token-aware semantic chunking (BGE-M3 tokenizer); chunks carry `source_block_ids`, `source_pages`, `bbox` aggregates, and `language`. A chunking QA pass drops empty / near-duplicate chunks.
6. **Contextual enrichment** — Each chunk gets a one-sentence neighborhood summary prepended (improves recall on definitional queries).
7. **Entity + relation extraction** — LLM-driven extractor produces typed entities (concept, model, dataset, metric, …) and relations (`references`, `co_located_with`, …) with evidence-block back-references. A quality gate drops single-char, all-numeric, or noise labels.
8. **Embedding + index** — BGE-M3 dense (1024-d) + sparse (sparse-impact) vectors written to Qdrant in a single hybrid collection. Mongo holds the chunks, blocks, entities, relations, and material metadata.

### Query pipeline

Each step writes a typed `AgentTraceStep` to the shared state and is visible in the UI's *Agent Trace* panel.

| Step | Owner | Decision |
|---|---|---|
| `classify_intent` | `IntentClassifier` | knowledge / chitchat / off-topic — off-topic short-circuits to a refusal in ≈2 s |
| `process_query` | `QueryProcessor` | Language detect, anaphora resolution, optional translation (gated when output would lose VN diacritics), build retrieval-query list |
| `plan_query` | `PlannerAgent` | Plan type (general / comparison / summarization / claim_check / relation_trace), optional sub-questions, multi-query flag |
| `retrieve_evidence` | `RetrieverDirector` | Routes each query to hybrid text search, per-source search, or graph traversal; merges + dedupes |
| `crag_triage` | `CRAGCriticAgent` | Pass-through when only RRF scores are available; tags candidates as CORRECT / AMBIGUOUS / INCORRECT once rerank scores exist |
| `rerank_evidence` | `CrossEncoderReranker` | BGE reranker (optional MMR) trims to `final_top_k` |
| `synthesize_answer` | `SynthesizerAgent` | Grounded answer with `[N]` citation markers; followed by acronym-strip and language-drift-strip postprocessing |
| `verify_claims` | `GuardrailsAgent` (NLI) | SUPPORTED / NOT_ENOUGH_EVIDENCE / CONTRADICTED — auto-skipped on cross-lingual answers |
| `repair_answer` | `SynthesizerAgent` (repair mode) | Re-synthesize with explicit warning when guardrails flag grounding gaps |
| `slec_gate` | `SentenceCoverageGate` | Per-sentence rerank-style score; drops UNSUPPORTED, hedges PARTIAL, refuses if weighted coverage < floor |
| `build_response` | `ResponseParser` | Inject missing citations, drop orphan `[N]` markers, finalize evidence list |

### Evidence schema (citation payload)

```jsonc
{
  "doc_id": "6a1248b548a253a8162ee173",
  "doc_name": "lecture_notes.pdf",
  "page": 3,
  "pages": [3],
  "block_id": "b_034",
  "block_type": "text",
  "snippet_original": "WAPE được chọn làm chỉ số chính vì phù hợp hơn MAPE trong chuỗi có giá trị gần 0…",
  "snippet_translated": null,
  "bbox": { "x0": 72.4, "y0": 612.1, "x1": 540.0, "y1": 668.3 },
  "role": "primary",
  "source_language": "vi",
  "confidence": 0.74,
  "evidence_blocks": [ /* every block in the parent chunk, with its own bbox/page/audio segment */ ]
}
```

### Refusal taxonomy

| Trigger | Stage | Typical latency |
|---|---|---|
| Off-topic / chitchat | `IntentClassifier` shortcut | ≤ 10 s |
| No / low-quality evidence | `RefusalPolicy` (`min_rerank_score`, `min_confidence_threshold`) | retrieval-time |
| Majority-contradicted claims | `ClaimVerifier` (NLI) when ≥ `contradicted_majority_fraction` claims are flagged | post-synthesis |
| Coverage floor | `SentenceCoverageGate` when weighted score < `refuse_below` | post-synthesis |
| False premise | LLM emits `Tiền đề không chính xác:` prefix and corrects in-place | answers, not refuses |

---

## API Surface

All routes are mounted under `/api/v1` and rate-limited to **15 requests / minute / IP** (Slowapi). JSON responses follow `APIResponse[T]` with `success`, `message`, `data`, `error`.

### Query

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query/ask` | Grounded Q&A (returns `QueryResponse` with answer, citations, agent trace, SLEC report) |
| `POST` | `/query/ask-stream` | SSE-streamed version of `/ask` (emits `agent_step` events as the pipeline progresses) |
| `POST` | `/query/ask-graph` | Graph-anchored query: caller supplies `entity_ids` / `relation_ids` as the retrieval seed |
| `POST` | `/query/ask-image` | Multipart image-as-query (multimodal) |
| `POST` | `/query/compare` | Cross-document comparison matrix on a given `topic` × `dimensions` |
| `POST` | `/query/summarize` | Material-level grounded summary |
| `POST` | `/query/study-guide` | Outline + key concepts + citations for review purposes |

### Materials

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/materials?owner_id=…` | List materials for an owner / collection |
| `POST` | `/materials/upload` | Multipart upload + auto-trigger ingestion |
| `POST` | `/materials/batch_upload` | Multi-file upload (207 multi-status) |
| `GET` | `/materials/{id}/status?owner_id=…` | Indexing status |
| `GET` | `/materials/{id}/debug?owner_id=…` | Per-page blocks + chunks for inspection |
| `GET` | `/materials/{id}/raw?owner_id=…` | Original file download |
| `POST` | `/materials/{id}/retry` | Retry failed ingestion stages |
| `DELETE` | `/materials/{id}?owner_id=…` | Delete material + cascade chunks/vectors |

### Collections & graph

| Method | Path | Purpose |
|---|---|---|
| `POST` / `GET` / `PATCH` / `DELETE` | `/collections[/…]` | Standard CRUD + dashboard |
| `POST` | `/graph` | Concept graph (nodes + edges, scope-filtered) |
| `GET` | `/graph/entity/{id}/subgraph` | K-hop subgraph around an entity |
| `POST` | `/graph/mindmap` | LLM-summarised topical mindmap for a collection |
| `GET` | `/evidence/{doc_id}/{page}` | Page-level evidence drilldown |

### Auth, admin, evaluation

`/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/me` (JWT, only when `API_AUTH_ENABLED=true`).  
`/admin/settings`, `/admin/metrics`, `/admin/feedback`.  
`/evaluation/embed`, `/evaluation/ragas` (RAGAS faithfulness / answer-relevance / context-precision over a labelled set).

---

## Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "owner_id": "demo_user",
    "collection_id": "6a0ed9e4455165de1b01120d",
    "query": "Tại sao chọn WAPE thay vì MAPE?",
    "top_k": 5
  }'
```

```jsonc
{
  "success": true,
  "data": {
    "answer": "WAPE được chọn làm chỉ số chính thay vì MAPE vì dữ liệu phụ tải có những đoạn bằng 0 hoặc gần 0…[1] Ở MAPE, sai số tại mỗi thời điểm được chia cho giá trị thực tế…[1]",
    "answer_language": "vi",
    "query_language": "vi",
    "citations": [ /* CitationSchema × 5 */ ],
    "confidence": 0.68,
    "was_refused": false,
    "sentence_coverage": {
      "enabled": true,
      "weighted_score": 0.72,
      "sentences": [
        { "status": "supported", "score": 0.81, "text": "WAPE được chọn…" },
        { "status": "supported", "score": 0.63, "text": "Ở MAPE, sai số…" }
      ]
    },
    "agent_trace": {
      "plan_type": "general",
      "steps": [ /* AgentTraceStep × N */ ],
      "verification": { "verdict": "supported", "confidence": 0.92 }
    }
  }
}
```

---

## Performance & Limits

| Aspect | Observed (Qwen 2.5 3B local, Ryzen / no GPU) |
|---|---|
| Cold start (model load) | ~10–15 s |
| Single-PDF ingestion (≤ 10 pp.) | 30–120 s |
| `/query/ask` end-to-end | 120–250 s typical, 2–10 s for off-topic refusal |
| `/query/ask-stream` time-to-first-event | 1–3 s |
| Hybrid retrieval (dense + sparse + RRF) | ≈ 0.3–0.6 s |
| Cross-encoder rerank (20 candidates) | ≈ 25–45 s |
| Concept graph / mindmap render | < 3 s |
| Rate limit | 15 requests / minute / IP |
| Per-file upload cap | 100 MB (`upload.max_file_size_mb`) |
| Per-query input cap | 4 000 characters |

Switching `AGENTBOOK_LLM_DEFAULT_PROVIDER=openai` and pointing at GPT-4o-mini / Claude cuts the synthesis stage from tens of seconds to ~1–3 s and lifts answer quality measurably (the rest of the pipeline is unchanged).

---

## Tech Stack

| Layer | Components |
|---|---|
| **API** | FastAPI (async), Pydantic v2, Beanie ODM, Slowapi rate limiting |
| **Storage** | MongoDB (documents + evidence + graph), Qdrant (hybrid vectors), Redis (cache + Celery broker) |
| **Embeddings** | BGE-M3 dense + sparse (FlagEmbedding), BGE reranker |
| **Parsing** | Docling (PDF/DOCX/PPTX), EasyOCR (printed scans), VLM fallback (handwriting), Faster-Whisper (audio) |
| **LLM** | Local Ollama (`qwen2.5:3b`, `qwen2.5-vl:7b`) with OpenAI-compatible cloud fallback |
| **Background** | Celery (eager mode supported for local dev), structured background tasks |
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS, React Flow, Zustand |
| **Tooling** | Pytest (15 test suites), Ruff, MyPy-friendly typing |

---

## Quick Start

### Prerequisites

- Python 3.12 (3.11+ supported)
- Node.js 18+
- Docker Desktop (for Qdrant)
- A MongoDB instance (local Docker or Atlas — connection string only)
- [Ollama](https://ollama.com) running locally with the LLM models pulled:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull qwen2.5-vl:7b
  ```
  BGE-M3 embeddings and the BGE reranker download automatically on first use through FlagEmbedding.

### 1. Configure environment

Create `backend/.env` from `backend/.env.example` and set at least:

```env
MONGODB_URI=mongodb://localhost:27017      # or your Atlas URI
AGENTBOOK_MONGODB_DATABASE=agentbook
AGENTBOOK_QDRANT_URL=http://localhost:6333
AGENTBOOK_LLM_DEFAULT_PROVIDER=local
AGENTBOOK_LLM_LOCAL_MODEL=qwen2.5:3b
AGENTBOOK_OLLAMA_BASE_URL=http://localhost:11434
AGENTBOOK_CELERY_TASK_ALWAYS_EAGER=true    # run pipelines inline (no broker needed)
```

Per-feature thresholds, top-k, prompts, and routing live in [`config/*.yaml`](config/) — never edit them in code.

### 2. Start the platform

```powershell
.\start_all.ps1
```

The script boots Qdrant (Docker), the FastAPI backend on `:8000`, and the Vite dev server on `:5173`.

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

### 3. First query

1. Open the frontend, create a collection, and upload a document.
2. Wait for the status to flip to *indexed* (typically 30–120 s for a typical PDF on a laptop CPU).
3. Ask a question — every claim in the answer carries a `[N]` marker linking back to the source chunk.

---

## Repository Layout

```
.
├── backend/
│   ├── src/
│   │   ├── agentic/              # Bounded multi-agent orchestration
│   │   │   ├── agents/           # Planner, Director, CRAG Critic, Synthesizer, Guardrails, Critic
│   │   │   ├── tools/            # Hybrid text search, graph search, NLI verifier
│   │   │   ├── planner.py        # Rule-based plan builder
│   │   │   ├── service.py        # Coordinator (state machine over the blackboard)
│   │   │   └── state.py          # Shared AgentState
│   │   ├── api/v1/endpoints/     # FastAPI routes (query, materials, collections, graph, …)
│   │   ├── core/                 # Settings, LLM factory, rate limit, security
│   │   ├── evaluation/           # RAGAS evaluator (faithfulness, relevance, precision)
│   │   ├── guardrails/           # Claim verifier, sentence-coverage gate, refusal policy
│   │   ├── inference/            # Inference engine, response parser, reasoning-path builder
│   │   ├── models/               # Beanie documents (Material, Chunk, Entity, Relation, …)
│   │   ├── processing/           # Docling, OCR, handwriting, chunking, entity / relation extraction
│   │   ├── prompts/              # System + task prompts (qa_grounded, summarization, claim_check, …)
│   │   ├── rag/                  # Embedding, Qdrant store, hybrid retriever, rerankers, CRAG
│   │   ├── schemas/              # Pydantic request / response schemas
│   │   ├── services/             # Material / query / summary / study-guide / memory orchestration
│   │   └── tasks/                # Celery task definitions
│   └── tests/                    # Pytest: agentic, api, evaluation, guardrails, integration, …
├── frontend/
│   └── src/
│       ├── api/                  # Typed API client
│       ├── components/           # Workspace UI, GraphCanvas, AudioCitationPlayer, EvidencePanel, …
│       ├── pages/                # WorkspacePage and route shells
│       └── state/                # Workspace store
├── config/                       # YAML configs (model, retrieval, guardrails, logging)
├── scripts/                      # reindex_material.py, smoke_test_api.py
├── data/test data/               # Sample corpora + scripted test scenarios
├── docker-compose.yml            # Qdrant container
└── start_all.ps1                 # Unified dev launcher
```

---

## Configuration Surface

All quality-affecting knobs live in [`config/*.yaml`](config/) and can be overridden by environment variables. The most relevant for tuning:

| File | Knob | Purpose |
|---|---|---|
| `retrieval_config.yaml` | `dense_top_k`, `sparse_top_k`, `rerank_input_k`, `final_top_k` | Recall / context width |
| `retrieval_config.yaml` | `agentic_rag_enabled`, `agentic_critic_enabled`, `agentic_max_retrieval_iterations` | Agentic pipeline shape |
| `retrieval_config.yaml` | `multi_query_enabled`, `crag.evaluator_enabled` | Query expansion + CRAG triage |
| `guardrails_config.yaml` | `sentence_coverage.supported_threshold` | SLEC strictness |
| `guardrails_config.yaml` | `claim_verification.contradicted_majority_fraction` | NLI tolerance |
| `guardrails_config.yaml` | `refusal.min_rerank_score`, `min_confidence_threshold` | Refusal floors |
| `model_config.yaml` | `local_model`, `provider`, `temperature` | LLM routing |

---

## Testing

```bash
cd backend
pytest                            # full suite
pytest tests/test_agentic         # agentic-only
pytest -k "test_retrieval"        # by name
```

End-to-end API smoke and ablation scripts (LEGACY vs AGENTIC paths, multimodal coverage, refusal correctness) live in [`scripts/smoke_test_api.py`](scripts/smoke_test_api.py).

---

## License

Apache License 2.0.
