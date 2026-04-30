# AgentBook-Platform

Graph RAG Document Intelligence for grounded study-material QA, evidence tracing, comparison, and mindmap exploration.

## Stack

- Backend: FastAPI, Beanie, MongoDB Atlas, Qdrant, Redis, Celery
- RAG: BGE-M3 dense + sparse retrieval, RRF fusion, cross-encoder reranker, MongoDB evidence graph
- Frontend: Vite, React, TypeScript, Tailwind CSS, React Flow

## Setup

Create `backend/.env` from `backend/.env.example` and set `MONGODB_URI`.

```bash
docker compose up --build
```

Backend:

```bash
cd backend
uvicorn src.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Tests

```bash
pytest backend/tests -v
python -m compileall backend/src evaluation scripts
npm --prefix frontend run build
```

## Demo Flow

1. Open the frontend dashboard.
2. Upload a PDF/DOCX/PPTX/image with metadata.
3. Wait for parse/index status to reach `indexed`.
4. Ask a question in Chat.
5. Inspect citations in the Evidence panel.
6. Use Compare for cross-document tables.
7. Open Graph or Mindmap to inspect concept nodes and evidence.

## Evaluation

```bash
python evaluation/run_eval.py --config evaluation/ablation_configs/a1_hybrid_vs_vector.yaml
python scripts/run_ablation_suite.py
python scripts/prepare_model_adaptation.py
python scripts/calibrate_thresholds.py --scores evaluation/results/sample_scores.json
```
