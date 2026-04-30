<div align="center">
  <img src="docs/assets/logo.png" width="200" alt="AgentBook Logo">
  <h1>🚀 AgentBook-Platform</h1>
  <p><b>Graph RAG Document Intelligence for Advanced Cross-Document Reasoning</b></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18+-61dafb.svg?style=flat-square&logo=react&logoColor=black)](https://reactjs.org/)
  [![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-ff4b4b.svg?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech/)
  [![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248.svg?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
</div>

---

## 🌟 Overview

**AgentBook** is a state-of-the-art Document Intelligence platform that goes beyond simple keyword search. By leveraging **Graph RAG** technology, it transforms static documents (PDFs, PPTXs, Images) into a dynamic Knowledge Graph. This allows users to perform **Cross-Document Reasoning**, trace evidence with pinpoint accuracy, and explore complex topics through interactive mindmaps.

Designed specifically for the Vietnamese academic and research context, AgentBook handles bilingual (EN-VI) sources, scanned documents, and even clear handwritten notes.

## ✨ Key Features

- **🔍 Hybrid & Graph Retrieval**: Combines BGE-M3 Dense/Sparse retrieval with an Evidence Graph for deep, multi-hop question answering.
- **📄 Multimodal Document Parsing**: High-fidelity parsing of layouts, tables, and formulas using `Docling` and `PaddleOCR`.
- **🖇️ Evidence Tracing & Citations**: Every AI response includes verifiable citations with document names, page numbers, and snippet locations.
- **⚖️ Cross-Document Comparison**: Automatically generate comparison tables and detect contradictions between multiple sources (e.g., Slides vs. Textbooks).
- **🧠 Interactive Mindmaps**: Visualize your knowledge base as a dynamic mindmap powered by `React Flow`.
- **🛡️ Guardrails & Safe Refusal**: Built-in verification gates to prevent hallucinations and ensure responses are strictly grounded in your data.

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        CLIENT["React Dashboard / CLI"]
        API["FastAPI Gateway"]
    end
    
    subgraph "AI & RAG Service Layer"
        INGEST["Ingestion Pipeline"]
        PARSER["Docling/OCR Engine"]
        GRAPH["Knowledge Graph Builder"]
        RETRIEVER["Hybrid Retriever (Dense+Sparse)"]
        RERANKER["Cross-Encoder Reranker"]
        LLM["Grounded QA Engine (Qwen/Llama)"]
    end
    
    subgraph "Data Persistence"
        MONGO["MongoDB Atlas (Metadata)"]
        QDRANT["Qdrant (Vector Embeddings)"]
        REDIS["Redis (Task Queue)"]
    end
    
    CLIENT --> API
    API --> INGEST --> PARSER --> MONGO
    PARSER --> GRAPH --> MONGO
    API --> RETRIEVER --> RERANKER --> LLM
    LLM --> API
    RETRIEVER --> QDRANT
```

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Task Management**: Celery + Redis
- **RAG Core**: BGE-M3 (Dense + Sparse), Cross-Encoders for Reranking
- **LLM**: Qwen3 / Llama (Local via Ollama or API Fallback)
- **Parsing**: IBM Docling, PaddleOCR

### Databases
- **Vector DB**: Qdrant (Docker / Cloud)
- **Metadata DB**: MongoDB Atlas + Beanie ODM
- **Cache**: Redis

### Frontend
- **Framework**: Vite + React + TypeScript
- **State**: Zustand + TanStack Query
- **Visualization**: React Flow (for Graphs & Mindmaps)

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- MongoDB Atlas account (or local MongoDB)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nvtanphat/AgentBook-Platform.git
   cd AgentBook-Platform
   ```

2. **Configure Environment**:
   Create a `.env` file in the `backend/` directory based on `.env.example`.
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your MONGODB_URI and API keys
   ```

3. **Spin up Infrastructure**:
   ```bash
   docker compose up -d
   ```

4. **Start the Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn src.main:app --reload
   ```

5. **Start the Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📊 Evaluation & Benchmarking

AgentBook includes a robust evaluation suite to ensure retrieval accuracy and answer quality:
- **Retrieval**: Recall@k, MRR, nDCG via `evaluation/run_eval.py`.
- **RAG Performance**: Faithfulness and Relevancy using RAGAS.
- **Ablation Studies**: Compare Hybrid vs. Vector search, and Flat vs. Layout-aware chunking.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
  Built with ❤️ by the AgentBook Team
</div>
