# 🚀 AgentBook

An advanced educational RAG (Retrieval-Augmented Generation) assistant that enables students and educators to upload learning materials (PDFs, documents, slides, images, and audio files) into organized collections and query them with precise, visual citation evidence.

---

## 📖 Overview

**AgentBook** is a local-first, highly robust document intelligence platform designed to handle complex, heterogeneous university materials. It processes and indexes diverse formats into a unified evidence schema, delivering millisecond-level retrieval and pixel-level citation accuracy on the UI.

The platform is specially optimized for **bilingual (Vietnamese & English) higher-education contexts**, enabling users to ask questions in Vietnamese over English documents without losing context or encountering translation hallucinations.

---

## ✨ Key Features

- **📄 Universal Document Ingestion:** Normalizes multiple document layouts into an invariant coordinate and evidence citation schema:
  - **PDF Documents:** Extracts precise pixel-level bounding boxes (`bbox`) to highlight cited text/figures directly on the PDF viewer.
  - **Word Documents (DOCX):** Parses reading-order context and hierarchical document structures.
  - **Lecture Slides (PPTX):** Aggregates scattered text blocks into slide-level contextual chunks.
  - **Tabular Data (XLSX/CSV):** Converts data rows into structured natural sentences while maintaining column headers.
  - **Images & Diagrams:** Extracts visual information, diagrams, and equations using OCR and vision-language models.
  - **Audio Lectures:** Transcribes and segments lecture recordings. Clicking an audio citation automatically plays the exact segment in the UI player.
- **⚡ Fast Progressive Enrichment:** Decouples synchronous text indexing (which takes less than 5 seconds) from background asynchronous visual captioning and deep processing, allowing instant searchability upon upload.
- **🛡️ Bilingual Quality Gate:** Fuses cross-lingual retrieval and Vietnamese query translation to search both languages, rank them using Reciprocal Rank Fusion (RRF), and formulate answers in Vietnamese using original English snippets.
- **🤖 Deterministic Multi-Agent System:** Orchestrates specialized agents (Planner, Director, Critic, Guardrails) with bounded reasoning loops to ensure low-latency, deterministic, and safe responses.

---

## 🏗️ Technical Stack

- **Backend Framework**: Python 3.11+, FastAPI (Async), Pydantic v2
- **Database & State**: MongoDB (via Beanie ODM), Redis, Celery (Distributed task queue)
- **Vector Search**: Qdrant (Fusing dense semantic embeddings and sparse lexical vectors)
- **Inference & Models**: 
  - Local [Ollama](https://ollama.com/) running `qwen2.5:3b` and `qwen2.5-vl:7b`
  - BGE-M3 (dense + sparse embedding), BGE Reranker
  - Docling, EasyOCR, Whisper (audio transcription)
- **Frontend App**: React 18, TypeScript, Vite, React Flow, TailwindCSS

---

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js** v18+ & **Python** 3.11+
- **Docker Desktop** (running Qdrant, MongoDB, Redis)
- **Ollama** (installed and running with `qwen2.5:3b` and `qwen2.5-vl:7b`)

### 2. Configure Environment Variables
Create a `backend/.env` file with your configuration:
```env
AGENTBOOK_APP_ENV=development
MONGODB_URI=mongodb://localhost:27017
AGENTBOOK_MONGODB_DATABASE=agentbook
AGENTBOOK_QDRANT_URL=http://localhost:6333
AGENTBOOK_LLM_DEFAULT_PROVIDER=local
AGENTBOOK_LLM_LOCAL_MODEL=qwen2.5:3b
AGENTBOOK_OLLAMA_BASE_URL=http://localhost:11434
AGENTBOOK_RERANKER_ENABLED=true
AGENTBOOK_AGENTIC_RAG_ENABLED=true
```

### 3. Start the Platform
Run the unified start script in PowerShell to boot up the entire local infrastructure and applications:
```powershell
.\start_all.ps1
```
Access the application dashboard at: **[http://localhost:5173](http://localhost:5173)**.

---

## 🧪 Evaluation Dataset Generation
AgentBook includes an evolutionary dataset synthesis pipeline to generate golden benchmark datasets. 

To automatically generate a benchmark dataset of Q&A pairs (factual, reasoning, and multi-hop questions) from your database, run:
```powershell
python scripts/generate_testset.py
```
The output dataset will be saved in `benchmarks/vn_edurag_2000.json`.

---

## 📂 Repository Structure

```text
.
├── backend/
│   ├── src/
│   │   ├── agentic/      # Multi-Agentic Blackboard Orchestration
│   │   ├── api/          # FastAPI Routes & Endpoints
│   │   ├── core/         # Settings, LLM Providers, App configs
│   │   ├── inference/    # Core Reasoning Engine & Response Parsers
│   │   ├── rag/          # Hybrid Retrievers & Vector Search
│   │   └── ...           # Ingestion pipelines (Docling, Whisper)
│   └── tests/            # Pytest test suites
├── frontend/             # React/Vite UI with AudioCitationPlayer & ReactFlow Graphs
├── config/               # Model, retrieval, and guardrail YAML configs
├── benchmarks/           # Generated Q&A benchmark datasets
├── scripts/              # Dataset generator scripts
├── docker-compose.yml    # Infrastructure configuration
└── start_all.ps1         # Unified startup script
```

---

## 🛡️ License

This project is licensed under the **Apache License 2.0**.
