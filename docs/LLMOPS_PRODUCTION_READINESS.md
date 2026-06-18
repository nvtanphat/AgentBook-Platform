# Ke hoach LLMOps production readiness cho AgentBook

Ngay audit: 2026-06-11

Pham vi doc repo:
- Backend FastAPI, pipeline ingest, RAG, agentic reasoning, guardrails, evaluation, tests.
- Config YAML, Dockerfile, docker-compose, script dev, docs.
- Frontend React/Vite o muc anh huong production.
- Khong tinh `frontend/node_modules`, file binary, cache, artifact sinh ra tu runtime.

## 1. Ket luan nhanh

AgentBook da co nen tang ung dung RAG kha tot: ingest da dang file, hybrid retrieval, graph/mindmap, citation evidence, guardrails, Celery worker, Qdrant, MongoDB, Redis, FastAPI, va bo test backend kha rong. Tuy nhien muc LLMOps production hien tai moi o khoang **5.5/10**: manh ve logic AI/RAG, con yeu ve van hanh production, quan sat he thong, CI/CD, governance model/prompt, cost control, bao mat mac dinh, va benchmark gate tu dong.

Neu muon san sang production, khong nen chi "deploy Docker". Can bien he thong thanh mot vong doi LLMOps day du:

1. Quan sat duoc tung request, tung job, tung model call.
2. Do duoc chat luong RAG bang benchmark vang.
3. Chan merge/deploy khi retrieval, citation, refusal, hallucination bi tut.
4. Version hoa model, prompt, config, index, dataset.
5. Co rollout/canary/rollback.
6. Co quota, cost, auth, audit, backup, runbook.

## 2. Hien trang da co

### Ung dung va pipeline

- FastAPI co lifespan khoi tao MongoDB, Qdrant, payload index, recover job bi ket, va health endpoint.
- Ingestion pipeline co parse, OCR, handwriting/VLM fallback, audio, spreadsheet, layout normalization, chunking, QA chunk, entity/relation extraction, graph quality gate, contextual enrichment, text index, visual index.
- Query pipeline co intent classifier, query processor, hybrid retriever, graph retriever, reranker, agentic orchestration, SLEC, claim verifier, response parser, citation injection.
- API surface da co upload, batch upload, material status/debug/raw/retry/delete, collection dashboard, graph, entity subgraph, auto viz, mindmap, ask/stream/graph/image/compare/summarize/study-guide.
- Celery worker da co retry, `acks_late`, `reject_on_worker_lost`, JSON serializer, queue ingest.

### Chat luong va safety

- Da co refusal policy, claim verifier, contradiction detector, sentence coverage gate, off-topic/chitchat shortcut.
- Test bao phu nhieu diem quan trong: guardrails, RAG, query router, reranker, graph retriever, visual embedding, OCR, docling parser, spreadsheet parser, chunking, graph quality gate, API endpoint, agentic service.
- Evaluation folder co metrics recall/precision/MRR/NDCG/citation accuracy, ablation config, dataset mau, model adaptation builder, hard negative mining, calibration.

### Trien khai

- `docker-compose.yml` co API, worker, Qdrant, Redis, volume va healthcheck.
- `backend/Dockerfile` build duoc API Python 3.12 slim.
- `start_all.ps1` ho tro local development.

## 3. Khoang trong production readiness

### 3.1 Observability chua du manh

Hien tai logging con thien ve console/dev. Can bo sung:

- Structured JSON logs.
- `request_id`, `trace_id`, `owner_id`, `collection_id`, `material_id`, `job_id`, `model_provider`, `model_name`, `prompt_version`, `index_version`.
- OpenTelemetry distributed tracing cho API -> Celery -> Mongo/Qdrant -> LLM/OCR/embedding/reranker.
- Prometheus metrics:
  - API latency theo route.
  - Ingest stage latency.
  - Query latency theo stage: intent, retrieve, rerank, synthesize, verify, SLEC.
  - Token in/out, model call count, error rate, timeout rate.
  - Retrieval recall proxy: top score, rerank score, citation count, refusal rate.
  - Queue length, job duration, retry count, dead-letter count.
- Dashboard Grafana va alert: p95 latency, job failure rate, hallucination/eval regression, cost spike.

### 3.2 CI/CD va eval gate con thieu

Repo co nhieu test, nhung `.github` chua co workflow production gate. Can them:

- CI lint/type/test/build Docker.
- Integration test co service containers Mongo/Qdrant/Redis.
- Benchmark gate cho RAG:
  - retrieval recall@k/MRR/NDCG.
  - citation accuracy.
  - faithfulness/groundedness.
  - refusal accuracy.
  - false premise handling.
  - graph/mindmap correctness.
  - multimodal evidence coverage.
- Gate theo nguong: neu metric giam qua nguong thi fail PR.
- Nightly eval voi OpenAI judge hoac model judge co rubric co dinh.

### 3.3 Benchmark dataset chua du SOTA

Dataset hien co la mau nho. De dat muc SOTA can xay benchmark vang da mien:

- File dau vao cong khai tai tu internet, khong dung file noi bo du an.
- Domain: phap ly, tai chinh, y sinh, giao duc, khoa hoc, ky thuat, hanh chinh cong, bao cao doanh nghiep, slide bai giang, bang tinh, scan/OCR, handwriting, audio.
- Format: PDF text, PDF scan, DOCX, PPTX, XLSX, CSV, PNG/JPG, MP3/WAV, file dai, file nhieu cot, file co bang/bieu do/hinh.
- QA types:
  - extractive single-hop.
  - multi-hop cross-page.
  - cross-document comparison.
  - table reasoning.
  - chart/figure reasoning.
  - timestamp/audio citation.
  - Vietnamese query over English source va nguoc lai.
  - false premise.
  - unanswerable/no evidence.
  - contradiction between documents.
  - graph relation/path query.
  - mindmap/topic structure.
- Metadata QA bat buoc:
  - `question_id`, `domain`, `file_id`, `file_url`, `license`, `format`, `language`, `answer_language`.
  - `question_type`, `difficulty`, `expected_answer`.
  - `evidence_refs`: page/block/bbox/table row/audio timestamp.
  - `expected_behavior`: answer/refuse/partial/compare/graph/mindmap.
  - `rubric`: factuality, citation, completeness, refusal.
  - `gold_entities`, `gold_relations` cho graph.
  - `gold_mindmap_nodes`, `gold_mindmap_edges` cho mindmap.

### 3.4 Bao mat mac dinh chua production

- `api_auth_enabled` mac dinh false la hop ly cho dev, nhung prod phai bat.
- Auth hien dang don gian, dung JWT HS256 va `api_key` lam secret. Production can:
  - secret manager.
  - RBAC/tenant role.
  - token rotation/refresh revocation.
  - audit log.
  - rate limit theo owner/user/API key, khong chi IP.
  - CORS allowlist chat.
  - TLS/reverse proxy.
  - upload antivirus/malware scan.
  - PII redaction va data retention.
  - signed URL cho raw file/evidence artifact neu public.

### 3.5 Deployment chua hardened

- Dockerfile chua co non-root user, chua multi-stage, chua resource constraint ro.
- Compose production chua tach rieng dev/prod, chua co reverse proxy/TLS.
- MongoDB khong nam trong compose, co the dung managed Atlas nhung can doc/runbook backup/restore.
- `start_all.ps1` la script dev, khong duoc dung prod vi co hanh vi kill process va sua `.env`.
- Chua co Kubernetes/Helm, autoscaling, canary, blue-green, rollback.

### 3.6 Reliability va scalability

- QueryService singleton trong API load nhieu thanh phan nang; can tinh den warmup, memory, concurrency, timeout.
- Local model/reranker/OCR tren CPU rat cham; production can tach worker pool theo workload/GPU.
- Celery moi co queue ingest chung; can tach:
  - `ingest_parse`
  - `ingest_ocr`
  - `embed`
  - `visual_embed`
  - `graph_extract`
  - `eval`
- Can idempotency key/distributed lock cho job retry, upload duplicate, delete-during-index.
- Can circuit breaker/fallback cho OpenAI/Ollama/Qdrant/Mongo/Redis.
- Can dead-letter queue va admin reprocess.

### 3.7 Model, prompt, index governance con thieu

Config da co model names va YAML, nhung production can registry bat bien:

- Model registry: provider, model, version, embedding dim, context window, cost, safety note.
- Prompt registry: prompt id, version, checksum, owner, changelog, eval result.
- Retriever config registry: dense/sparse/rerank/top_k/MMR/threshold version.
- Index version: collection schema, vector dim, payload schema, migration/rollback.
- Dataset version: source manifest, checksum, license, split train/dev/test, hidden test.
- Experiment tracking: run id, git SHA, config hash, dataset hash, metrics.

### 3.8 Cost governance chua co

Can do va gioi han:

- token input/output moi request.
- cost theo provider/model/owner/collection.
- cache hit rate cho translation/LLM/contextual enrichment.
- monthly budget, per-user quota, alert khi vuot nguong.
- fallback model theo chinh sach cost/latency/quality.

### 3.9 Frontend production

- Vite dev tot cho local, nhung production can:
  - build static artifact va serve qua CDN/Nginx.
  - Sentry/frontend error telemetry.
  - auth token refresh/logout hardened.
  - Playwright e2e cho upload-query-citation-graph-mindmap.
  - bundle size budget.

## 4. Roadmap trien khai LLMOps

### P0 - Bat buoc truoc production pilot (1-2 tuan)

1. Tao production profile
   - `backend/.env.production.example`.
   - `docker-compose.prod.yml`.
   - bat `API_AUTH_ENABLED=true`.
   - tach local/dev config va prod config.
   - cam dung `start_all.ps1` cho prod.

2. Harden Docker va runtime
   - non-root user.
   - health/readiness endpoint rieng.
   - resource limits.
   - graceful shutdown API/worker.
   - pin dependency rui ro cao neu can reproduce.

3. CI co ban
   - `.github/workflows/ci.yml`.
   - run backend pytest.
   - build backend Docker.
   - run frontend typecheck/build.
   - fail neu test guardrails/retrieval/API fail.

4. Structured logging va request context
   - them middleware sinh `request_id`.
   - JSON log cho API va Celery.
   - log stage ingest/query voi latency.

5. Metrics co ban
   - `/metrics` Prometheus.
   - API request count/latency/error.
   - Celery job count/duration/failure/retry.
   - LLM token/call/error/timeout.
   - retrieval/rerank/synthesis latency.

6. Benchmark smoke gate
   - tao `evaluation/datasets/gold_v1.jsonl`.
   - them `scripts/run_production_eval.py`.
   - gate: recall@5, citation_accuracy, refusal_accuracy, hallucination_rate.

7. Security minimum
   - secret khong commit.
   - CORS production allowlist.
   - rate limit theo owner/API key.
   - upload scan/extension/magic-byte da co thi bo sung file size theo tenant.
   - audit log cho upload/delete/query/admin.

### P1 - Production beta on dinh (2-4 tuan)

1. Full OpenTelemetry
   - trace API -> worker -> Qdrant/Mongo/Redis -> LLM.
   - dashboard Grafana.
   - alert p95 latency, queue backlog, failed jobs, cost spike.

2. Model/prompt/config registry
   - bang Mongo `ModelRun`, `PromptVersion`, `EvalRun`.
   - moi QueryLog gan `model_version`, `prompt_version`, `retrieval_config_hash`, `index_version`.

3. Benchmark SOTA v1
   - 200-500 cau hoi vang.
   - it nhat 50 file cong khai da mien.
   - co graph va mindmap gold.
   - co hidden split de tranh overfit.

4. Queue va worker topology
   - tach queue theo workload.
   - worker GPU cho OCR/VLM/embedding/rerank.
   - worker CPU cho parse/docling/chunking.
   - dead-letter queue.
   - admin re-run stage.

5. Reliability
   - distributed lock cho material/job.
   - idempotency key upload/retry.
   - circuit breaker provider LLM.
   - retry policy rieng cho transient/permanent error.

6. Cost va quota
   - token accounting.
   - budget theo tenant.
   - alert khi vuot quota.
   - model fallback policy.

### P2 - Production scale va governance (1-2 thang)

1. Kubernetes/Helm
   - API deployment.
   - Celery worker pools.
   - HPA theo CPU/GPU/queue length.
   - managed Mongo/Qdrant/Redis.
   - secret manager.

2. Progressive delivery
   - canary model/prompt/retriever.
   - shadow traffic.
   - automatic rollback neu eval online/offline giam.

3. Continuous evaluation
   - nightly eval tren benchmark.
   - online sampling judge.
   - human feedback review queue.
   - drift detection theo domain/language/file type.

4. Compliance
   - encryption at rest.
   - backup/restore drill.
   - retention policy.
   - tenant delete/export.
   - PII detection/redaction.

## 5. Benchmark SOTA de gan vao LLMOps

### 5.1 Cau truc thu muc de xuat

```text
evaluation/
  benchmark/
    manifests/
      files_manifest_v1.jsonl
      dataset_manifest_v1.json
    raw/
      legal/
      finance/
      healthcare/
      education/
      science/
      government/
      business/
      audio/
      scanned/
    gold/
      qa_gold_v1.jsonl
      graph_gold_v1.jsonl
      mindmap_gold_v1.jsonl
      refusal_gold_v1.jsonl
      compare_gold_v1.jsonl
    runs/
      <run_id>/
        predictions.jsonl
        metrics.json
        failures.jsonl
```

### 5.2 File manifest

Moi file tai tu nguon cong khai can co:

```json
{
  "file_id": "finance_001",
  "domain": "finance",
  "format": "pdf",
  "language": "en",
  "source_url": "https://...",
  "license": "public-domain-or-open-license",
  "checksum_sha256": "...",
  "pages_or_duration": 24,
  "contains_tables": true,
  "contains_figures": true,
  "contains_scans": false,
  "local_path": "evaluation/benchmark/raw/finance/finance_001.pdf"
}
```

### 5.3 QA gold schema

```json
{
  "question_id": "qa_000001",
  "domain": "finance",
  "question": "Theo bao cao, doanh thu nam 2023 tang bao nhieu phan tram?",
  "question_language": "vi",
  "answer_language": "vi",
  "question_type": "table_reasoning",
  "difficulty": "medium",
  "file_ids": ["finance_001"],
  "expected_answer": "Doanh thu nam 2023 tang ...",
  "expected_behavior": "answer",
  "evidence_refs": [
    {
      "file_id": "finance_001",
      "page": 12,
      "block_id": "optional_if_known",
      "bbox": null,
      "table_cell": "row=Revenue,col=2023"
    }
  ],
  "rubric": {
    "must_include": ["so lieu", "don vi", "nam so sanh"],
    "must_not_include": ["so lieu khong co trong tai lieu"],
    "citation_required": true
  }
}
```

### 5.4 Metric bat buoc

- Retrieval: Recall@5, Recall@10, MRR@10, NDCG@10.
- Citation: citation accuracy, evidence page accuracy, block accuracy, bbox/timestamp accuracy neu co.
- Generation: faithfulness, answer correctness, completeness, language consistency.
- Refusal: unanswerable refusal accuracy, false-premise correction accuracy, over-refusal rate.
- Graph: entity F1, relation F1, path accuracy, relation evidence accuracy.
- Mindmap: node coverage, edge coverage, hierarchy correctness, citation support.
- Multimodal: table cell accuracy, figure/chart answer accuracy, OCR robustness, audio timestamp accuracy.
- Production: p50/p95 latency, cost/query, token/query, error rate.

## 6. File nen them/sua trong repo

### Them moi

- `.github/workflows/ci.yml`
- `.github/workflows/nightly-eval.yml`
- `docker-compose.prod.yml`
- `backend/src/core/observability.py`
- `backend/src/core/metrics.py`
- `backend/src/core/request_context.py`
- `backend/src/core/model_registry.py`
- `backend/src/core/prompt_registry.py`
- `scripts/run_production_eval.py`
- `scripts/download_benchmark_sources.py`
- `evaluation/benchmark/README.md`
- `docs/runbooks/deploy.md`
- `docs/runbooks/backup_restore.md`
- `docs/runbooks/incident_response.md`
- `docs/runbooks/model_rollback.md`

### Sua can than

- `backend/src/main.py`: gan middleware request context, metrics, readiness.
- `backend/src/dependencies.py`: auth/RBAC/rate-limit theo owner.
- `backend/src/tasks/celery_tasks.py`: metrics, queue routing, dead-letter, job lock.
- `backend/src/services/parse_index_pipeline.py`: stage-level tracing, idempotency, artifact version.
- `backend/src/services/query_service.py`: token/cost tracking, model/prompt/config version.
- `backend/src/inference/inference_engine.py`: trace every stage, circuit breaker, eval hooks.
- `backend/src/agentic/service.py`: persist agent trace metrics, prompt version, failure labels.
- `config/logging_config.yaml`: JSON production logs.
- `config/model_config.yaml`: production provider/model profile.
- `config/retrieval_config.yaml`: frozen benchmark/prod profiles.
- `frontend`: production build, Sentry, Playwright e2e.

## 7. Dinh nghia "production ready" cho du an nay

Co the coi AgentBook san sang production pilot khi dat cac dieu kien:

- CI pass toan bo unit/integration tests.
- Docker image build reproducible.
- Auth bat mac dinh trong prod.
- Metrics/logs/traces hoat dong.
- Dashboard va alert co nguong ro.
- Benchmark v1 co it nhat 200 cau hoi vang va co graph/mindmap/refusal/multimodal.
- Moi deploy chay eval gate va co rollback.
- Backup/restore Mongo/Qdrant da test.
- Cost/token per tenant duoc ghi nhan.
- Raw files/artifacts co retention va delete policy.
- Runbook su co co san.

## 8. Uu tien thuc thi ngay

Neu lam theo thu tu toi uu, nen bat dau bang:

1. CI + pytest + Docker build.
2. Structured logs + Prometheus metrics.
3. Benchmark gold v1 va eval gate.
4. Auth/CORS/secrets production.
5. Docker hardening + prod compose.
6. Model/prompt/config registry.
7. OpenTelemetry + dashboard + alert.
8. Queue split + worker pool.
9. Canary/rollback.

Day la duong ngan nhat de bien he thong tu "demo RAG rat nhieu tinh nang" thanh "LLMOps production co the tin cay, do duoc, rollback duoc".
