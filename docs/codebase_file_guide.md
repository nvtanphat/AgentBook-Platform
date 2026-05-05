# Prism / AgentBook - File Guide & General Workflow

Tài liệu này mô tả các file nguồn chính trong repo, kèm vai trò, kỹ thuật đang dùng, và workflow tổng quát của hệ thống.

Phạm vi:
- Tập trung vào file chạy thực tế trong `backend/` và `frontend/`, cùng các file config/infra quan trọng.
- Không liệt kê `node_modules`, cache, log, build artifact, và bộ test chi tiết.
- Các file `__init__.py` là package marker, không chứa logic đáng kể.

## 1) Root / Infra / Config

- `README.md`: giới thiệu dự án, stack, cách chạy local, và các luồng chính của hệ thống.
- `CLAUDE.md`: bộ quy tắc làm việc cho agent, ưu tiên kiến trúc, an toàn dữ liệu, và testing.
- `docker-compose.yml`: orchestration cho backend API, Celery worker, Qdrant, và Redis.
- `start_all.ps1`, `start_all.bat`: script khởi động local theo một lệnh.
- `backend/Dockerfile`: build image backend cho API và worker.
- `backend/requirements.txt`: dependency backend; gồm FastAPI, Beanie, Qdrant client, Celery, Docling, OCR, RAG stack.
- `backend/.env.example`: mẫu biến môi trường cho backend.
- `config/model_config.yaml`: cấu hình model, embedding, reranker, OCR, PDF render, và versioning pipeline.
- `config/retrieval_config.yaml`: cấu hình Qdrant, top-k, RRF, rerank input, và chiến lược chunking.
- `config/guardrails_config.yaml`: giới hạn upload, ngưỡng refusal, và ngưỡng chất lượng ảnh.
- `config/logging_config.yaml`: cấu hình logging chuẩn cho backend.
- `backend/config/logging_config.yaml`: bản logging config dùng trong backend container/runtime.
- `frontend/package.json`: scripts và dependency cho Vite/React/TypeScript frontend.
- `frontend/vite.config.ts`: cấu hình build/dev server của Vite.
- `frontend/vitest.config.ts`: cấu hình test frontend.
- `frontend/tailwind.config.js`: theme và token CSS utility.
- `frontend/postcss.config.js`: pipeline CSS cho Tailwind.
- `frontend/tsconfig.json`: cấu hình kiểm tra kiểu TypeScript.
- `frontend/index.html`: entry HTML của app.
- `frontend/public/`: static assets cho frontend.

## 2) Backend Runtime Và Nền Tảng

- `backend/src/main.py`: tạo FastAPI app, gắn logging, CORS, rate limit, health check, và đảm bảo Qdrant collection tồn tại.
- `backend/src/database.py`: khởi tạo/đóng MongoDB + Beanie, đăng ký document models.
- `backend/src/dependencies.py`: lớp dependency injection cho FastAPI, gồm service factory, Qdrant client, settings, và kiểm tra quyền owner.
- `backend/src/core/config.py`: nạp `Settings` từ YAML + env; là nơi gom config runtime, ngưỡng, path, và model defaults.
- `backend/src/core/security.py`: validation upload an toàn, kiểm tra magic bytes, MIME allowlist, checksum, và path traversal guard.
- `backend/src/core/rate_limit.py`: limiter dựa trên IP cho FastAPI/SlowAPI.
- `backend/src/core/model_factory.py`: chọn implementation LLM theo provider.
- `backend/src/core/base_llm.py`: interface chung cho LLM backend.
- `backend/src/core/local_llm.py`: LLM local qua Ollama / local runtime.
- `backend/src/core/openai_client.py`: LLM client tương thích OpenAI API.
- `backend/src/core/tokenizer.py`: đếm token để phục vụ chunking và giới hạn prompt.

## 3) API Layer

- `backend/src/api/v1/router.py`: gom toàn bộ router version 1.
- `backend/src/api/v1/endpoints/admin.py`: endpoint metrics/feedback cho admin.
- `backend/src/api/v1/endpoints/collections.py`: tạo, liệt kê, và xóa collection.
- `backend/src/api/v1/endpoints/evidence.py`: trả về trang evidence theo `doc_id` và `page`.
- `backend/src/api/v1/endpoints/graph.py`: sinh graph/mindmap, xử lý node/edge fallback, và truy xuất theo scope.
- `backend/src/api/v1/endpoints/materials.py`: upload, batch upload, status, debug, raw file, và delete material.
- `backend/src/api/v1/endpoints/query.py`: hỏi đáp, so sánh, summary, và study guide.
- `backend/src/api/v1/endpoints/__init__.py`: package marker cho nhóm endpoint.

## 4) Processing Pipeline

- `backend/src/processing/types.py`: kiểu dữ liệu lõi cho block/page/document, evidence map, entity/event/relation, và exception chuyên dụng.
- `backend/src/processing/language_detector.py`: heuristic nhận diện tiếng Việt/Anh ở mức block và document.
- `backend/src/processing/layout_normalizer.py`: chuẩn hóa reading order, gom block, và giữ ổn định cấu trúc layout.
- `backend/src/processing/docling_parser.py`: parser chính cho PDF/DOCX/PPTX; dùng Docling và patch tương thích runtime khi cần.
- `backend/src/processing/spreadsheet_parser.py`: parse CSV/XLS/XLSX thành nội dung có thể truy hồi.
- `backend/src/processing/ocr_engine.py`: engine OCR cho ảnh scan; có tiền xử lý ảnh, deduplicate block, merge dòng, và các nhánh OCR.
- `backend/src/processing/image_quality_checker.py`: chấm chất lượng ảnh theo blur, brightness, contrast, skew.
- `backend/src/processing/ocr_quality_gate.py`: gate quyết định OCR có đủ tin cậy để index hay không.
- `backend/src/processing/handwriting_reader.py`: nhánh đọc chữ viết tay riêng, tách khỏi OCR in ấn.
- `backend/src/processing/figure_captioner.py`: caption figure bằng VLM/LLM qua Ollama hoặc fallback tương thích.
- `backend/src/processing/evidence_mapper.py`: biến parsed block thành evidence có `page`, `block_id`, `bbox`, `snippet_original`.
- `backend/src/processing/chunking.py`: layout-aware chunker và semantic chunker; dùng token budget, breakpoint, và overlap.
- `backend/src/processing/chunk_qa.py`: kiểm tra chất lượng chunk, phát hiện chunk quá rỗng/quá nhiễu.
- `backend/src/processing/contextual_enricher.py`: thêm ngữ cảnh xung quanh chunk bằng LLM khi bật contextual retrieval.
- `backend/src/processing/entity_extractor.py`: trích xuất entity từ evidence map.
- `backend/src/processing/entity_resolution.py`: hợp nhất entity trùng hoặc biến thể tên.
- `backend/src/processing/event_extractor.py`: trích xuất event và relation để xây knowledge graph.
- `backend/src/processing/__init__.py`: package marker.

## 5) RAG Layer

- `backend/src/rag/types.py`: kiểu cho `RetrievalScope`, `RetrievedChunk`, `GraphPath`.
- `backend/src/rag/embedder.py`: BGE-M3 embedder, gồm dense + sparse embedding, cache model, và wrapper runtime.
- `backend/src/rag/vector_store.py`: tạo và cache Qdrant client, đóng client khi shutdown.
- `backend/src/rag/indexer.py`: `QdrantMongoIndexer`, batch upsert chunk/entity/event/relation, và đồng bộ metadata với Qdrant.
- `backend/src/rag/retriever.py`: `HybridRetriever` cho dense + sparse retrieval, lexical fallback, dedupe, và cache embedding.
- `backend/src/rag/graph_retriever.py`: mở rộng kết quả theo graph/evidence path để hỗ trợ multi-hop reasoning.
- `backend/src/rag/query_processor.py`: chuẩn hóa query, xử lý ngôn ngữ, và tạo biến thể truy vấn khi cần.
- `backend/src/rag/query_rewriter.py`: LLM-based query rewriting / multi-query để tăng recall.
- `backend/src/rag/query_router.py`: quyết định đường xử lý truy vấn theo loại intent.
- `backend/src/rag/reranker.py`: cross-encoder reranker để tăng precision trước khi sinh đáp án.
- `backend/src/rag/__init__.py`: package marker.

## 6) Inference Layer

- `backend/src/inference/chitchat_detector.py`: nhận diện câu chào / casual chat và trả instant reply.
- `backend/src/inference/intent_classifier.py`: phân loại intent truy vấn.
- `backend/src/inference/confidence_scorer.py`: tính confidence bằng chuẩn hóa sigmoid và score tổng hợp.
- `backend/src/inference/response_parser.py`: chuyển retrieved chunks thành citations, format evidence cho prompt, và đóng gói response.
- `backend/src/inference/inference_engine.py`: orchestration chính cho hỏi đáp grounded, gồm router, retrieval, rerank, guardrail, và answer synthesis.
- `backend/src/inference/__init__.py`: package marker.

## 7) Guardrails

- `backend/src/guardrails/claim_verifier.py`: kiểm tra claim có đủ chứng cứ hay không.
- `backend/src/guardrails/contradiction_detector.py`: phát hiện mâu thuẫn số liệu và mâu thuẫn ngữ nghĩa giữa nguồn.
- `backend/src/guardrails/__init__.py`: package marker.

## 8) Services

- `backend/src/services/material_service.py`: nghiệp vụ upload/list/delete material và quản lý concurrency cho pipeline.
- `backend/src/services/parse_index_pipeline.py`: pipeline lõi, từ parse -> normalize -> evidence map -> chunk -> enrich -> extract -> index.
- `backend/src/services/query_service.py`: điều phối ask/compare, log query, và cập nhật memory sau mỗi lần hỏi.
- `backend/src/services/summary_service.py`: tạo summary có căn cứ từ collection hoặc document scope.
- `backend/src/services/study_guide_service.py`: tạo study guide dạng outline / key concepts.
- `backend/src/services/memory_service.py`: quản lý memory hội thoại ngắn hạn và context summary.
- `backend/src/services/admin_service.py`: thống kê hệ thống, query metrics, và feedback.
- `backend/src/services/__init__.py`: package marker.

## 9) Models

- `backend/src/models/common.py`: enum dùng chung như `PipelineStatus`, `JobType`, `Modality`, `SourceLanguage`, cùng helper `utc_now`.
- `backend/src/models/collection.py`: model `KnowledgeCollection`.
- `backend/src/models/material.py`: model `Material`, `MaterialPageDocument`, `MaterialPage`, `MaterialBlock`, `BoundingBox`, và helper đọc/ghi pages.
- `backend/src/models/chunk.py`: model `Chunk` lưu chunk đã index.
- `backend/src/models/pipeline_job.py`: model tiến trình pipeline theo stage/status.
- `backend/src/models/knowledge_graph.py`: `Entity`, `Event`, `Relation`, và `EvidenceRef` cho graph reasoning.
- `backend/src/models/query_log.py`: log câu hỏi, citations, confidence, latency, và refusal reason.
- `backend/src/models/chat_memory.py`: summary memory cho hội thoại.
- `backend/src/models/translation_cache.py`: cache bản dịch query hoặc snippet.
- `backend/src/models/feedback.py`: phản hồi người dùng.
- `backend/src/models/__init__.py`: package marker.

## 10) Schemas

- `backend/src/schemas/common.py`: envelope `APIResponse[T]`.
- `backend/src/schemas/admin.py`: response schema cho health, metrics, feedback.
- `backend/src/schemas/collection.py`: request/response cho collection.
- `backend/src/schemas/material.py`: request/response cho upload, status, debug, batch upload.
- `backend/src/schemas/evidence.py`: citation, evidence block, page response, và bounding box schema.
- `backend/src/schemas/graph.py`: graph node/edge/response và request cho mindmap.
- `backend/src/schemas/mindmap.py`: schema riêng cho mindmap.
- `backend/src/schemas/query.py`: request/response cho ask, compare, summary, study guide.
- `backend/src/schemas/__init__.py`: package marker.

## 11) Tasks

- `backend/src/tasks/celery_tasks.py`: định nghĩa Celery app và các task parse/index chạy nền.
- `backend/src/tasks/__init__.py`: package marker.

## 12) Prompt Files

- `backend/src/prompts/chitchat.txt`: prompt cho phản hồi casual.
- `backend/src/prompts/off_topic.txt`: prompt cho trường hợp ngoài phạm vi / refusal.
- `backend/src/prompts/qa_grounded.txt`: prompt cho grounded QA.
- `backend/src/prompts/summarization.txt`: prompt cho summary và study guide.

## 13) Frontend Runtime Và State

- `frontend/src/main.tsx`: root entry; gắn `ErrorBoundary`, `WorkspaceProvider`, router, và route redirect.
- `frontend/src/pages/WorkspacePage.tsx`: layout 3 cột, mobile drawer, bottom tab bar, và deep-link cho evidence/settings.
- `frontend/src/state/workspace.tsx`: state global của workspace, lưu `localStorage`, scoped material IDs, citation selection, và material session cache.
- `frontend/src/api/client.ts`: typed API client, request envelope parsing, XHR upload progress, và toàn bộ API contract với backend.
- `frontend/src/styles.css`: style toàn cục cho app.
- `frontend/src/vite-env.d.ts`: type declaration cho Vite env.

## 14) Frontend Components

- `frontend/src/components/AppShell.tsx`: khung app, header status, owner badge, và mount `SettingsModal`.
- `frontend/src/components/ErrorBoundary.tsx`: fallback khi UI crash.
- `frontend/src/components/StatusBadge.tsx`: badge trạng thái material theo stage pipeline.
- `frontend/src/components/MarkdownRenderer.tsx`: renderer markdown custom; hỗ trợ table, math, citation, code block, list, và inline formatting.
- `frontend/src/components/SnippetRenderer.tsx`: wrapper cho snippet, ưu tiên table preview nếu nội dung là bảng.
- `frontend/src/components/EvidencePanel.tsx`: panel đọc evidence, highlight keyword, render bbox overlay, snippet, và navigation citations.
- `frontend/src/components/GraphCanvas.tsx`: canvas React Flow cho graph/mindmap; có layout force-directed và radial, custom node, và full-screen rendering.

### Workspace Components

- `frontend/src/components/workspace/ChatPanel.tsx`: giao diện hỏi đáp grounded; có local reply cho chitchat, confidence badge, citation footer, và history trong `localStorage`.
- `frontend/src/components/workspace/SourcesPanel.tsx`: quản lý collection, batch upload, progress bar, polling status, delete material/collection, và mở debug modal.
- `frontend/src/components/workspace/StudioPanel.tsx`: khung tab cho Studio/Evidence/Compare/Graph/Mindmap.
- `frontend/src/components/workspace/SettingsModal.tsx`: modal cài đặt workspace, test API, đổi owner/collection/language/top-k.
- `frontend/src/components/workspace/DebugModal.tsx`: debug OCR blocks, chunks, vectors, và bbox overlay.

### Studio Tabs

- `frontend/src/components/workspace/studio/StudioHomeTab.tsx`: tạo summary và study guide, hiển thị artifact card, download `.txt`, và quản lý collapse/delete.
- `frontend/src/components/workspace/studio/CompareTab.tsx`: so sánh nhiều khía cạnh bằng tag input, result card, confidence pill, và conflict section.
- `frontend/src/components/workspace/studio/GraphTab.tsx`: tạo graph/mindmap, chuyển đổi response sang canvas, full-screen overlay, và selected-node footer.

### Frontend phụ trợ

- `frontend/src/components/workspace/studio/StudioHomeTab.tsx`: dùng `SummaryService` và `StudyGuideService` qua API.
- `frontend/src/components/workspace/studio/GraphTab.tsx`: dùng `loadGraph` và `loadMindmap` từ API client.
- `frontend/src/components/workspace/studio/CompareTab.tsx`: dùng `compareDocuments`.
- `frontend/src/components/MarkdownRenderer.tsx`: parser tự viết, không dựa vào markdown library nặng.

## 15) Workflow Tổng Quát

1. Người dùng chọn collection và upload file ở panel Sources.
2. Frontend kiểm tra loại file cơ bản, gửi metadata + file qua API bằng `XHR` để có progress.
3. Backend validate extension, MIME, magic bytes, checksum, và path an toàn trước khi lưu.
4. Tạo `Material` và `PipelineJob`, sau đó đẩy xử lý nặng sang Celery.
5. Pipeline parse file theo loại:
   - PDF/DOCX/PPTX qua Docling.
   - CSV/XLS/XLSX qua spreadsheet parser.
   - Ảnh scan qua OCR.
   - Ảnh viết tay qua handwriting reader + quality gate.
6. Parsed document được normalize để giữ reading order, page/block/bbox, và ngôn ngữ nguồn.
7. Evidence mapper chuyển block thành evidence trace có thể trích dẫn.
8. Chunker tạo layout-aware chunk hoặc semantic chunk, rồi contextual enricher thêm ngữ cảnh nếu bật.
9. Entity/event/relation extractor xây lightweight knowledge graph.
10. Embedder tạo dense + sparse embedding bằng BGE-M3, rồi index sang Qdrant và MongoDB.
11. Khi hỏi đáp, query đi qua query processor, router, và query rewriter nếu cần.
12. Hybrid retriever lấy candidate từ dense, sparse, và graph.
13. Reranker xếp lại candidate, response parser sinh citations, rồi confidence scorer/guardrails quyết định trả lời hay refusal.
14. Inference engine sinh đáp án grounded bằng LLM; query log và memory được cập nhật sau đó.
15. Frontend render câu trả lời, citation, evidence overlay, graph, compare, summary, và study guide từ response đã chuẩn hóa.

## 16) Kỹ Thuật Cốt Lõi Đang Dùng

- FastAPI + Pydantic + Beanie cho API và metadata model.
- MongoDB cho dữ liệu bán cấu trúc; Qdrant cho dense/sparse vector.
- Celery + Redis cho xử lý nền.
- Docling + OCR + spreadsheet parser cho ingest đa định dạng.
- Layout-aware chunking để giữ cấu trúc tài liệu.
- Evidence trace đầy đủ `page`, `block_id`, `bbox`, `snippet_original`.
- BGE-M3 dense + sparse retrieval.
- RRF fusion và cross-encoder reranking.
- Graph retrieval để mở rộng multi-hop reasoning.
- LLM query rewriting và grounded answer synthesis.
- Confidence scoring, claim verification, và contradiction detection làm guardrail.
- React + Vite + React Flow cho workspace, evidence, graph, và mindmap.
- `localStorage` cho state workspace/hội thoại ở frontend.

## 17) Ghi Chú Phạm Vi

- Tài liệu này không mô tả từng file test một, vì test được tổ chức song song theo module và sẽ làm tài liệu quá dài nếu liệt kê toàn bộ.
- Nếu cần mở rộng, cách hợp lý nhất là tạo thêm một phần riêng cho `backend/tests/` theo nhóm: API, RAG, inference, processing, integration, và evaluation.
