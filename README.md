<div align="center">
  <h1>🚀 Noelys - AgentBook Platform</h1>
  <p><strong>A Local-First Document Intelligence and Agentic RAG Workspace</strong></p>
  
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
  [![Vite + React](https://img.shields.io/badge/Vite+React-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
  [![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-EF4444?style=flat&logo=qdrant)](https://qdrant.tech/)
  [![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-white?style=flat&logo=ollama)](https://ollama.com/)
</div>

---

Noelys is an advanced, local-first document intelligence and learning workspace. It empowers users to upload learning materials, index them into a hybrid retrieval system, ask grounded questions, inspect citations/evidence, compare sources, generate study guides, and visualize knowledge via interactive graphs and mind maps.

## 📖 Table of Contents
- [✨ Key Features](#-key-features)
- [🏗️ Tech Stack](#️-tech-stack)
- [⚙️ Prerequisites](#️-prerequisites)
- [🚀 Quick Start](#-quick-start)
- [🧠 Retrieval & AI Capabilities](#-retrieval--ai-capabilities)
- [📂 Repository Structure](#-repository-structure)
- [🛠️ Configuration & Environment](#️-configuration--environment)
- [📚 API Overview](#-api-overview)
- [🛠️ Troubleshooting](#️-troubleshooting)

---

## ✨ Key Features

- **📄 Document Ingestion**: Supports PDF, DOCX, PPTX, PNG/JPG, CSV, XLSX with integrated OCR and semantic chunking.
- **🔍 Hybrid RAG**: Combines dense BGE-M3 vectors, sparse vectors, Reciprocal Rank Fusion, and Cross-Encoder reranking for high-precision retrieval.
- **🕸️ Graph RAG**: Captures entities and relationships to answer complex dependency or cause/effect queries.
- **🤖 Optional Agentic RAG**: Implements an advanced reasoning pipeline with query planning, sub-question generation, coverage checking, and claim verification.
- **📑 Transparent Evidence**: UI highlights source snippets, primary/supporting evidence, and inline citations.
- **🗺️ Visualizations**: Explore your document corpus using interactive mind maps and graph views.

---

## 🏗️ Tech Stack

### Backend
- **Framework**: Python 3.11+, FastAPI, Pydantic v2
- **Data & State**: MongoDB (Beanie + Motor), Redis, Celery
- **Vector Search**: Qdrant
- **AI Models**: Local inference via [Ollama](https://ollama.com/) (e.g., `qwen2.5:3b`, `qwen3:4b`), BGE-M3 for embeddings, BGE for reranking

### Frontend
- **Framework**: React 18, TypeScript, Vite
- **Styling & UI**: Tailwind CSS, Lucide React Icons
- **Visuals**: React Flow for graph visualizations

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** 18+
- **Python** 3.11+
- **Docker Desktop** (for Qdrant & Redis)
- **MongoDB** (Local instance or Atlas)
- **Ollama** (for local LLMs)
- **PowerShell** (if running on Windows)

### Model Setup
Pull the required LLMs via Ollama. The system is optimized for Qwen models:
```powershell
ollama pull qwen2.5:3b
ollama pull qwen3:4b
```
Ensure the Ollama server is running: `ollama serve`

---

## 🚀 Quick Start

### 1. Environment Setup
Create a `backend/.env` file with your local settings. Example:
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
```

### 2. Launch the Platform (Windows)
We provide an integrated script to start the entire stack (FastAPI, Vite, and Qdrant via Docker):
```powershell
powershell.exe -ExecutionPolicy Bypass -File .\start_all.ps1
```

Once started, access the following endpoints:
- **Web App**: [http://localhost:5173](http://localhost:5173)
- **API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

*For manual startup instructions using Docker Compose and separate terminal windows, please explore the `/scripts` or `docker-compose.yml` configurations.*

---

## 🧠 Retrieval & AI Capabilities

### Hybrid & Graph RAG
The platform utilizes a multi-layered approach to document retrieval:
1. **Hybrid Retrieval**: Fuses dense semantic search with sparse keyword matching. Results are refined using `BAAI/bge-reranker-v2-m3`.
2. **Graph RAG**: For relationship-based queries, the system traverses a knowledge graph of entities and dependencies, combining retrieved edges with document chunks.

### Agentic Pipeline
The Agentic RAG pipeline (enabled via `AGENTBOOK_AGENTIC_RAG_ENABLED=true` or request flags) introduces a robust planner-executor model:
- Formulates multi-step query plans.
- Issues per-source sub-queries.
- Identifies retrieval gaps and performs self-correction.
- Employs a **Claim Verifier** to ensure generated answers are factually grounded in the source text.

---

## 📂 Repository Structure

```text
.
├── backend/
│   ├── src/
│   │   ├── agentic/      # Agentic Planner, Claim Validator & Service
│   │   ├── api/          # FastAPI Routes & Endpoints
│   │   ├── core/         # Settings, LLM connections, App configs
│   │   ├── inference/    # Core Reasoning Engine & Response Parsers
│   │   ├── rag/          # Hybrid, Graph, and Cross-Encoder Retrievers
│   │   └── ...           # Processing pipelines, MongoDB schemas
│   └── tests/            # Pytest test suites
├── frontend/             # React/Vite workspace UI
├── config/               # Model, retrieval, and guardrail YAML configs
├── data/                 # Local data storage (raw files, vectors)
├── scripts/              # Evaluation & diagnostic utilities
├── docker-compose.yml    # Infrastructure configuration
└── start_all.ps1         # Windows quick-start script
```

---

## 📚 API Overview

The RESTful API is available at `http://127.0.0.1:8000/api/v1`. 

**Key Endpoints:**
- **Collections**: `GET /collections`, `POST /collections`
- **Materials (Ingestion)**: `POST /materials/upload`, `POST /materials/batch_upload`
- **Query (RAG)**: 
  - `POST /query/ask` (Standard Q&A)
  - `POST /query/ask-stream` (Streaming Q&A)
  - `POST /query/compare` (Multi-source comparison)
- **Graph Visualization**: `POST /graph`, `POST /graph/mindmap`

---

## 🛠️ Troubleshooting

- **Backend won't start?** Verify MongoDB connectivity and ensure Qdrant & Ollama are running. Check `backend.err.log`.
- **Vietnamese Characters Broken?** Ensure `charset=utf-8` is passed in API request headers and terminal encodings.
- **Docker Issues?** Run `docker compose up -d qdrant redis` manually. Check `docker ps`.

For detailed API usage and advanced configuration, see the `config/` directory.

---
<div align="center">
  <i>Built with ❤️ for advanced document understanding.</i>
</div>
