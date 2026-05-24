<div align="center">
  <h1>🚀 AgentBook</h1>
  <p><strong>Evidence-grounded learning assistant for university study materials — agentic RAG with verified citations across PDFs, slides, tables, scans, handwriting, and audio.</strong></p>

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

AgentBook turns heterogeneous course materials into a queryable, citation-backed knowledge surface. Every sentence in an answer is traceable to a chunk on a specific page, block, or audio segment of the source document. The system is built for the bilingual Vietnamese–English educational setting: students can ask in Vietnamese over English textbooks (or vice versa) and receive grounded answers without losing the original-language evidence.


---

## Highlights

- **Universal ingestion** — PDF, DOCX, PPTX, XLSX, CSV, PNG/JPG (printed + handwritten), and audio (MP3/WAV/M4A/FLAC/OGG/WebM) flow through a single block-level evidence schema with bounding boxes, page numbers, and audio timestamps preserved end-to-end.
- **Hybrid retrieval** — BGE-M3 dense + sparse vectors fused via RRF in Qdrant, smart reranking with a cross-encoder, optional multi-query rewriting for hard recall, and conditional graph traversal for relation questions.
- **Agentic reasoning** — A bounded multi-agent pipeline (Planner → Director → CRAG Critic → Reranker → Synthesizer → Guardrails → Sentence-Level Coverage Gate) orchestrates retrieval and answer generation with explicit safety gates instead of free-form ReAct loops.
- **Cross-lingual robustness** — Native VI↔EN handling: queries are routed in both languages, evidence is kept in its source language, and the claim verifier is automatically bypassed when answer-language differs from chunk-language to avoid spurious refusals.
- **Calibrated refusal** — Off-topic / chitchat / low-confidence questions are refused in seconds via an intent classifier shortcut; on-topic questions with weak grounding surface a *partial* badge instead of fabricating content.
- **Pixel-accurate evidence UI** — Citations link back to the original PDF region, slide block, table row, or audio segment; clicking a citation scrolls and highlights the source.
- **Knowledge graph + mindmap** — Entities and relations extracted at ingestion time power a navigable concept graph and an LLM-summarised topical mindmap per collection.

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

See [`AgentBook_Implementation_Plan.md`](AgentBook_Implementation_Plan.md) for the full pipeline specification and [`CLAUDE.md`](CLAUDE.md) for engineering invariants.

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

## Documentation

- [**Implementation Plan**](AgentBook_Implementation_Plan.md) — product scope, API contracts, data schemas
- [**CLAUDE.md**](CLAUDE.md) — engineering invariants (evidence trace, owner / collection isolation, no hardcoded thresholds)
- [**SOTA Recommendations**](SOTA_Recommendations.md) — research notes on retrieval and reasoning upgrades

---

## License

Apache License 2.0.
