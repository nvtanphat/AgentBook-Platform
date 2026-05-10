# Noelys - AgentBook Platform

Noelys is a local-first document intelligence and learning workspace. The app lets users upload learning materials, index them into a hybrid retrieval system, ask grounded questions, inspect citations/evidence, compare sources, generate summaries/study guides, and visualize knowledge graphs or mind maps.

This repository contains the full stack:

- `backend/`: FastAPI API, ingestion pipeline, Hybrid RAG, Graph RAG, optional Agentic RAG, citation/evidence logic.
- `frontend/`: Vite + React + TypeScript workspace UI.
- `config/`: model, retrieval, and guardrail configuration.
- `data/`: local raw/processed/vector data used during development.
- `scripts/`: utility scripts for reindexing, migration, and diagnostics.

The current development setup is optimized for Windows + PowerShell, local Qdrant via Docker, MongoDB, and local LLM inference through Ollama.

---

## Current Status

The current default Q&A path is:

1. User asks a question in the workspace.
2. Backend routes the query by intent/type.
3. Hybrid retrieval searches Qdrant using dense + sparse BGE-M3 vectors.
4. For relationship queries, Graph RAG is used in the normal inference path.
5. Evidence is reranked and filtered.
6. The LLM generates a grounded answer with inline citations.
7. The frontend shows citations, highlighted evidence blocks, reasoning path, and graph/evidence panels.

Important current behavior:

- `AGENTBOOK_AGENTIC_RAG_ENABLED=false` in local `.env` is intentional for faster default Q&A.
- Graph RAG is still active for relationship-style questions through the normal inference engine.
- Agentic RAG exists and can be enabled per request or via env, but it is heavier because it performs planning, sub-question retrieval, coverage checks, and answer verification.
- Prompting now includes a language lock so `answer_language=vi` should produce Vietnamese final answers even when the query or examples are in English.

---

## Main Features

### Document Ingestion

Supported upload formats:

- PDF
- DOCX
- PPTX
- PNG/JPG/JPEG
- CSV
- XLSX/XLS

The ingestion pipeline handles:

- file validation and checksum tracking
- parsing by document type
- OCR fallback for scanned/visual content
- page/block extraction
- evidence block construction
- semantic chunking
- embedding and Qdrant indexing
- entity/relation extraction for graph features

### Hybrid RAG

The core retrieval path combines:

- BGE-M3 dense vectors
- BGE-M3 sparse vectors
- Reciprocal Rank Fusion
- optional cross-encoder reranking with `BAAI/bge-reranker-v2-m3`
- source/evidence-aware citation generation

### Graph RAG

Graph RAG is used for relationship, cause/effect, dependency, and connection questions.

The graph layer stores:

- entities
- relations
- evidence references
- graph paths hydrated back into source evidence blocks

Reasoning paths can show:

- `retrieve`
- `traverse`
- `synthesize`

The frontend also has graph and mindmap views through:

- `/api/v1/graph`
- `/api/v1/graph/mindmap`

### Evidence Workspace

The right panel in the workspace shows:

- cited document
- page and block metadata
- primary/supporting evidence
- selected evidence state
- inline snippet highlighting
- copy action
- source confidence/coverage indicators

Recent behavior to note:

- Citation markers render as `[1]`, `[2]`, etc.
- Evidence selection now prefers exact or focused evidence blocks instead of blindly selecting the first block in a chunk.
- Repeated citation markers inside a paragraph are collapsed where possible.

### Optional Agentic RAG

Agentic RAG is implemented in `backend/src/agentic/`.

When enabled, it can perform:

- query planning
- route-specific sub-question generation
- text retrieval
- per-source retrieval for comparison/multi-source workflows
- graph tracing for relation queries
- evidence quality checks
- retrieval repair
- reranking
- answer synthesis
- claim verification and answer repair

Current recommendation:

- Keep it disabled by default for normal local development unless you are testing agentic behavior.
- Use it selectively with request flag `rag_flags.agentic_rag_enabled=true`.

---

## Tech Stack

### Backend

- Python 3.11+
- FastAPI
- Pydantic v2
- Beanie + Motor for MongoDB
- Qdrant for vector search
- BGE-M3 embeddings
- BGE reranker
- Ollama local LLM by default
- Celery/Redis support for async ingestion workers

### Frontend

- React 18
- TypeScript
- Vite
- Tailwind CSS
- React Flow
- Lucide React icons
- resizable panels

### Infrastructure

- Docker Compose for Qdrant and Redis
- MongoDB Atlas or local MongoDB
- Ollama for local LLM inference

---

## Repository Structure

```text
.
├── backend/
│   ├── src/
│   │   ├── agentic/          # Agentic RAG planner/service
│   │   ├── api/              # FastAPI routes
│   │   ├── core/             # settings, LLM clients, rate limits
│   │   ├── guardrails/       # claim/contradiction validation
│   │   ├── inference/        # main answer engine, prompts, citations
│   │   ├── models/           # Beanie documents
│   │   ├── processing/       # parsing, OCR, chunking, graph extraction
│   │   ├── rag/              # retrievers, vector store, graph retriever
│   │   ├── schemas/          # API schemas
│   │   ├── services/         # business services
│   │   └── tasks/            # Celery tasks
│   ├── tests/
│   ├── scripts/
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── config/
│   ├── model_config.yaml
│   ├── retrieval_config.yaml
│   └── guardrails_config.yaml
├── data/
├── scripts/
├── docker-compose.yml
├── start_all.ps1
└── README.md
```

---

## Prerequisites

Install:

- Python 3.11 or newer
- Node.js 18 or newer
- Docker Desktop
- MongoDB Atlas or a reachable MongoDB instance
- Ollama
- PowerShell on Windows

Pull a local model for Ollama. The checked-in model config uses `qwen3:4b`, while the local `.env` may override it to `qwen2.5:3b` for faster development.

```powershell
ollama pull qwen2.5:3b
ollama pull qwen3:4b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

---

## Environment Configuration

Create or update `backend/.env`.

Minimal local development example:

```env
AGENTBOOK_APP_ENV=development

MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority
AGENTBOOK_MONGODB_DATABASE=agentbook

AGENTBOOK_QDRANT_URL=http://localhost:6333
AGENTBOOK_CELERY_TASK_ALWAYS_EAGER=true

AGENTBOOK_LLM_DEFAULT_PROVIDER=local
AGENTBOOK_LLM_LOCAL_MODEL=qwen2.5:3b
AGENTBOOK_OLLAMA_BASE_URL=http://localhost:11434

AGENTBOOK_RERANKER_ENABLED=true
AGENTBOOK_AGENTIC_RAG_ENABLED=false
AGENTBOOK_CONTEXTUAL_RETRIEVAL_ENABLED=false
```

Useful flags:

| Variable | Current Use |
|---|---|
| `MONGODB_URI` | MongoDB connection string |
| `AGENTBOOK_MONGODB_DATABASE` | Mongo database name |
| `AGENTBOOK_QDRANT_URL` | Qdrant URL, usually `http://localhost:6333` locally |
| `AGENTBOOK_LLM_DEFAULT_PROVIDER` | `local` for Ollama |
| `AGENTBOOK_LLM_LOCAL_MODEL` | Ollama model name |
| `AGENTBOOK_OLLAMA_BASE_URL` | Ollama API base URL |
| `AGENTBOOK_RERANKER_ENABLED` | Enable/disable reranker |
| `AGENTBOOK_AGENTIC_RAG_ENABLED` | Enable full agentic RAG by default |
| `AGENTBOOK_AGENTIC_PLANNER_LLM_ENABLED` | Use LLM planner instead of deterministic planner |
| `AGENTBOOK_CONTEXTUAL_RETRIEVAL_ENABLED` | Enable contextual retrieval enrichment |
| `AGENTBOOK_CELERY_TASK_ALWAYS_EAGER` | Run ingestion tasks synchronously in local dev |

Do not commit real secrets from `backend/.env`.

---

## Configuration Files

### `config/model_config.yaml`

Controls:

- LLM provider/model
- Ollama URL
- OpenAI fallback config
- embedding model and dimensions
- reranker model
- OCR model settings
- parser/chunk/index version strings

Current important defaults:

```yaml
llm:
  default_provider: "local"
  local_model: "qwen3:4b"

embedding:
  model_name: "BAAI/bge-m3"
  dense_size: 1024

reranker:
  enabled: true
  model_name: "BAAI/bge-reranker-v2-m3"
```

### `config/retrieval_config.yaml`

Controls:

- Qdrant collection
- dense/sparse top-k
- graph top-k
- final top-k
- chunking parameters

Current important defaults:

```yaml
retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  graph_top_k: 10
  final_top_k: 5
  rerank_input_k: 8
  graph_max_hops: 2
```

---

## Running Locally

### Option 1: Start Everything With PowerShell

Recommended on Windows:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\start_all.ps1
```

The script:

1. Stops old Python/frontend processes on ports `8000` and `5173`.
2. Starts Qdrant with Docker Compose.
3. Waits for Qdrant.
4. Starts FastAPI backend on port `8000`.
5. Starts Vite frontend on port `5173`.
6. Writes logs to root-level `backend.err.log`, `backend.out.log`, `frontend.err.log`, and `frontend.out.log`.

Open:

- App: `http://localhost:5173`
- Backend health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`
- Qdrant dashboard: `http://localhost:6333/dashboard`

### Option 2: Manual Startup

Start Qdrant:

```powershell
docker compose up -d qdrant
```

Optionally start Redis:

```powershell
docker compose up -d redis
```

Start backend:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Start frontend:

```powershell
cd frontend
npm install
npm run dev
```

---

## Docker Compose

`docker-compose.yml` defines:

- `api`
- `worker`
- `qdrant`
- `redis`

For local development, it is common to run only Qdrant/Redis through Docker and run backend/frontend directly on the host.

```powershell
docker compose up -d qdrant redis
```

Full Docker backend mode is available, but check the model and Ollama host settings first. The compose file points Ollama to `http://host.docker.internal:11434` for containers.

---

## API Overview

Base URL:

```text
http://127.0.0.1:8000/api/v1
```

### Collections

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/collections` | Create collection |
| `GET` | `/collections` | List collections |
| `GET` | `/collections/{collection_id}/dashboard` | Collection dashboard/status |
| `PATCH` | `/collections/{collection_id}` | Update collection |
| `DELETE` | `/collections/{collection_id}` | Delete collection |

### Materials

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/materials` | List materials |
| `POST` | `/materials/upload` | Upload one material |
| `POST` | `/materials/batch_upload` | Upload multiple materials |
| `GET` | `/materials/{material_id}/status` | Check material status |
| `GET` | `/materials/{material_id}/debug` | Debug parsed/indexed material |
| `GET` | `/materials/{material_id}/raw` | Download/view raw material |
| `POST` | `/materials/{material_id}/retry` | Retry failed ingestion |
| `DELETE` | `/materials/{material_id}` | Delete material |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query/ask` | Ask grounded RAG question |
| `POST` | `/query/ask-stream` | Streaming ask endpoint |
| `POST` | `/query/compare` | Compare topic/dimensions across sources |
| `POST` | `/query/summarize` | Summarize collection/material |
| `POST` | `/query/study-guide` | Build study guide |

### Evidence

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/evidence/{doc_id}/{page}` | Fetch evidence page data |

### Graph

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/graph` | Load graph nodes/edges |
| `POST` | `/graph/mindmap` | Generate concept mindmap |

### Admin/Evaluation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/admin/settings` | Read runtime settings |
| `PATCH` | `/admin/settings` | Update admin settings |
| `GET` | `/admin/metrics` | Admin metrics |
| `POST` | `/admin/feedback` | Submit feedback |
| `POST` | `/evaluation/embed` | Embedding diagnostic |
| `POST` | `/evaluation/ragas` | RAGAS evaluation helper |

---

## Example Ask Request

```powershell
$body = @{
  owner_id = "user_demo"
  collection_id = "<collection_id>"
  query = "dropout là gì"
  top_k = 5
  answer_language = "vi"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/query/ask" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Force Agentic RAG for one request:

```json
{
  "owner_id": "user_demo",
  "collection_id": "<collection_id>",
  "query": "how is dropout related to overfitting?",
  "top_k": 5,
  "answer_language": "vi",
  "rag_flags": {
    "agentic_rag_enabled": true
  }
}
```

For normal local use, leave `agentic_rag_enabled` false unless you specifically need the full trace/repair workflow.

---

## Ingestion Flow

High-level ingestion:

1. Upload material through `/materials/upload` or batch upload.
2. Validate extension, size, and metadata.
3. Store raw file under `data/raw`.
4. Parse document pages and blocks.
5. Run OCR or image captioning when needed.
6. Build `MaterialPageDocument` records.
7. Chunk content.
8. Build evidence blocks.
9. Extract entities/relations for graph.
10. Embed chunks with BGE-M3.
11. Store vectors in Qdrant.
12. Mark material as indexed.

If upload appears stuck, check:

- `backend.err.log`
- material status endpoint
- collection dashboard endpoint
- Qdrant health
- MongoDB connectivity

---

## Prompting and Language

Prompt templates live in:

```text
backend/src/prompts/
```

Main templates:

- `qa_grounded.txt`
- `graph_relation.txt`
- `comparison.txt`
- `claim_check.txt`
- `summarization.txt`
- `multi_source.txt`
- `chitchat.txt`
- `off_topic.txt`

The prompt builder is in:

```text
backend/src/inference/inference_engine.py
```

The project currently keeps most instruction text in English because local models generally follow formatting/citation rules more consistently in English. To avoid English answers when the requested answer language is Vietnamese, the prompt builder adds a language lock:

- final answer must use `answer_language`
- Vietnamese output is enforced for `answer_language=vi`
- technical terms may remain as standard terms
- examples should not be copied into the final answer

---

## Testing

Run focused Agentic RAG tests:

```powershell
python -m pytest backend/tests/test_agentic/test_agentic_service.py -q
```

Compile backend:

```powershell
python -m compileall -q backend/src
```

Build frontend:

```powershell
npm.cmd --prefix frontend run build
```

Run broader backend tests:

```powershell
python -m pytest backend/tests -q
```

Note: broader tests may require local services or test fixtures depending on the test group.

---

## Troubleshooting

### Backend does not start

Check:

```powershell
Get-Content backend.err.log -Tail 100
Invoke-RestMethod http://127.0.0.1:8000/health
```

Common causes:

- MongoDB URI is missing or DNS cannot resolve MongoDB Atlas.
- Qdrant is not running.
- Ollama is not running.
- Port `8000` is already occupied.

### Frontend is offline

Check:

```powershell
Get-Content frontend.err.log -Tail 100
netstat -ano | Select-String ':5173'
```

Then restart:

```powershell
cd frontend
npm run dev
```

### Qdrant is not reachable

```powershell
docker compose ps
Invoke-WebRequest http://127.0.0.1:6333/readyz -UseBasicParsing
```

Start again:

```powershell
docker compose up -d qdrant
```

### Ollama/model errors

```powershell
ollama list
ollama pull qwen2.5:3b
ollama serve
```

Make sure `AGENTBOOK_OLLAMA_BASE_URL=http://localhost:11434`.

### Redis warnings

You may see Redis embedding cache warnings during local development. These are usually non-fatal if retrieval still works. Start Redis if you want the cache:

```powershell
docker compose up -d redis
```

### Vietnamese text looks broken

Use UTF-8 everywhere:

- source files
- prompt files
- terminal encoding
- JSON request content type

When calling APIs with Vietnamese text from PowerShell, prefer:

```powershell
-ContentType "application/json; charset=utf-8"
```

---

## Development Notes

- The backend may have a dirty worktree during active development; avoid resetting unrelated files.
- `backend/.env` is local and can differ from `config/*.yaml`.
- `config/model_config.yaml` currently defaults to `qwen3:4b`, but local `.env` often overrides to `qwen2.5:3b`.
- Full Agentic RAG can be slower than standard RAG; this is expected.
- Graph quality depends heavily on entity/relation extraction quality during ingestion.
- Frontend build currently emits a Vite chunk-size warning; this does not block production build.

---

## Useful Commands

Check ports:

```powershell
netstat -ano | Select-String ':8000'
netstat -ano | Select-String ':5173'
netstat -ano | Select-String ':6333'
```

Stop a process:

```powershell
Stop-Process -Id <PID> -Force
```

Restart backend manually:

```powershell
cd backend
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Rebuild frontend:

```powershell
npm.cmd --prefix frontend run build
```

---

## License

No license file is currently documented in this repository. Add a `LICENSE` file before publishing or distributing the project.
