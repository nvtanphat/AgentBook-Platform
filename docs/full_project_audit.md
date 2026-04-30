# 🔍 Prism (AgentBook) — Full Project Audit Report

> **Auditor Role**: Senior AI Engineer & Solutions Architect  
> **Date**: 2026-04-28  
> **Scope**: Toàn bộ codebase Backend + Frontend + DevOps + Security + Architecture

---

## 📋 Tổng Quan Dự Án

| Aspect | Detail |
|---|---|
| **Tên dự án** | Prism (formerly AgentBook) |
| **Kiến trúc** | Monorepo: FastAPI backend + React/Vite frontend |
| **Database** | MongoDB Atlas (Beanie ODM) + Qdrant Vector DB |
| **LLM** | Ollama (qwen2.5:3b) / OpenAI-compatible fallback |
| **Embedding** | BAAI/bge-m3 (dense + sparse) |
| **Reranker** | BAAI/bge-reranker-v2-m3 (CrossEncoder) |
| **Task Queue** | Celery + Redis |
| **RAG Pipeline** | Hybrid (Dense + Sparse + Graph) → RRF Fusion → Rerank |

---

## 🏗️ 1. KIẾN TRÚC (Architecture)

### ✅ Điểm Tốt
- Clean separation: `api/` → `services/` → `rag/` → `inference/` → `processing/`
- Strategy pattern cho LLM (`BaseLLM` → `OllamaLLM` / `OpenAICompatibleLLM`)
- Config từ YAML + env, centralized qua `Settings` (pydantic-settings)
- Dependency Injection thông qua FastAPI `Depends()`

### ❌ Vấn Đề Phát Hiện

#### 🔴 P0 — Branding Inconsistency (Chưa rebrand hoàn toàn)

| File | Vấn đề |
|---|---|
| `backend/src/prompts/qa_grounded.txt` L1 | Prompt vẫn ghi `"Bạn là AgentBook"` → phải đổi thành `"Bạn là Prism"` |
| `backend/src/core/config.py` L45 | `app_name: str = "AgentBook"` → nên đổi thành `"Prism"` |
| `backend/src/core/config.py` L40 | `env_prefix="AGENTBOOK_"` — tất cả env vars vẫn dùng prefix `AGENTBOOK_` |
| `frontend/src/state/workspace.tsx` L4 | localStorage keys vẫn là `"agentbook.workspace.v2"` |
| `backend/src/api/v1/endpoints/graph.py` L95 | Mindmap root topic hardcode `"AgentBook Knowledge Map"` |
| `backend/src/tasks/celery_tasks.py` L14 | Celery app name vẫn là `"agentbook"` |
| `backend/src/rag/indexer.py` L123 | UUID namespace vẫn dùng `"agentbook:chunk:"` |

#### 🟡 P1 — Frontend/Backend Schema Mismatch

**StudyGuideResponse mismatch:**
- Backend (`backend/src/schemas/query.py` L78-84): trả về `overview`, `key_concepts`, `outline`
- Frontend (`frontend/src/api/client.ts` L342-347): expect `study_guide`, `sections`
- **Hậu quả**: Study Guide feature sẽ CRASH ở frontend khi nhận response

**SummaryRequest mismatch:**
- Backend (`backend/src/schemas/query.py` L53-58): expect `material_id` (singular), `scope`
- Frontend (`frontend/src/api/client.ts` L321-327): gửi `material_ids` (plural), `max_length`, `answer_language`
- **Hậu quả**: Backend sẽ ignore `material_ids` / `max_length` → Summary có thể sai scope

---

## 🔒 2. BẢO MẬT (Security)

### ✅ Điểm Tốt
- Magic-byte validation cho PDF, PNG, JPEG, DOCX, PPTX, XLSX
- MIME type allowlist
- Path traversal protection (`ensure_child_path`)
- File size limit (20MB)
- Scope-based access control (`owner_id` required trên mọi query)

### ❌ Vấn Đề Phát Hiện

#### 🔴 P0 — MongoDB Credentials Exposed
- `backend/.env` L2: Chứa **MONGODB_URI plaintext** với username + password thật
  - `mongodb+srv://nvtanphat69_db_user:Dambamonli9d@cluster0...`
  - **PHẢI** thêm `.env` vào `.gitignore` ngay lập tức
  - **PHẢI** rotate password MongoDB Atlas

#### 🔴 P0 — Không có Authentication / Authorization
- Toàn bộ API endpoints KHÔNG yêu cầu authentication
- `owner_id` truyền từ client → bất kỳ ai cũng có thể xem/xóa data của user khác
- Admin endpoints (`/admin/metrics`, `/admin/feedback`) không có auth guard
- **Khuyến nghị**: Thêm JWT authentication hoặc API key middleware

#### 🟡 P1 — CORS quá rộng
- `backend/src/core/config.py` L68-71: Chỉ allow localhost, OK cho dev
- Nhưng `allow_methods=["*"]`, `allow_headers=["*"]` → nên restrict khi deploy production

#### 🟡 P1 — Không có Rate Limiting
- Không có rate limiter trên bất kỳ endpoint nào
- LLM endpoints (`/query/ask`, `/query/summarize`) có thể bị spam → tốn resource

---

## ⚙️ 3. BACKEND — FastAPI

### ✅ Điểm Tốt
- Async lifespan management (init/close DB, Qdrant)
- Proper error handling trong pipeline (catch → mark FAILED → re-raise)
- Pydantic v2 schemas với validation
- `APIResponse[T]` envelope pattern nhất quán

### ❌ Vấn Đề Phát Hiện

#### 🔴 P0 — Memory OOM Risk (Full-Table Scans bằng `.to_list()`)
- Rất nhiều API đang load **toàn bộ dữ liệu** vào RAM mà không có pagination:
  - `admin_service.py` L18: `await QueryLog.find_all().to_list()` (Load tất cả logs).
  - `collections.py` L35: `await Chunk.find(...).to_list()` (Load toàn bộ chunks text siêu lớn).
  - `materials.py` L32: `await query.sort("-updated_at").to_list()` (Load toàn bộ tài liệu).
  - **Hậu quả**: Chắc chắn sẽ gây OOM Crash khi user upload nhiều file.
  - **Fix**: BẮT BUỘC dùng Pagination (Limit/Offset) để fetch list, hoặc dùng `.count()` nếu chỉ cần đếm số lượng.

#### 🟡 P1 — Storage Leak (Xóa Material nhưng không xóa file JSON)
- Trong `material_service.py` (`delete_material`, `delete_collection`), hệ thống chỉ gọi `.unlink()` để xóa file raw tải lên (`material.storage_path`).
- File JSON parsed siêu lớn ở `processed_data_dir` (có thể lên tới hàng chục MB) hoàn toàn **KHÔNG BỊ XÓA**.
- **Hậu quả**: Lâu ngày ổ cứng server sẽ bị rác (bloat) nghiêm trọng.
- **Fix**: Xóa thêm `material.extra_metadata["parsed_artifact_path"]` khi delete.

#### 🟡 P1 — Orphaned Graph Nodes (Rác Knowledge Graph)
- Khi gọi `delete_material`, hệ thống chỉ xóa `Chunk` và `PipelineJob`, nhưng **BỎ QUÊN** `Entity`, `Event`, `Relation`.
- **Hậu quả**: Các node trong đồ thị tri thức bị "mồ côi" (orphaned), vẫn xuất hiện khi query graph nhưng trỏ về `material_id` không còn tồn tại, gây crash khi frontend render citation.
- **Fix**: Phải có logic dọn dẹp các Graph Nodes gắn với `material_id` bị xóa (cascade delete hoặc update node weights).

#### 🟡 P1 — Retriever: Sequential N+1 DB queries
- `backend/src/rag/retriever.py` L76-80: Mỗi Qdrant point → `await _hydrate_point()` → query MongoDB cho Chunk + Material
  - 15 points = 30 MongoDB queries (sequential!)
  - **Fix**: Batch fetch chunks và materials bằng `$in` query

#### 🟡 P1 — Graph Retriever: Sequential evidence hydration
- `backend/src/rag/graph_retriever.py` L92-120: Mỗi evidence ref → `await Material.get()` riêng
  - **Fix**: Batch load materials, cache trong scope

#### 🟡 P2 — Double validation trên upload
- `backend/src/api/v1/endpoints/materials.py` L70-77: Gọi `validate_upload_bytes()` ở endpoint
- `backend/src/services/material_service.py` L43-48: Cũng gọi `validate_upload_bytes()` trong service
  - **Fix**: Chỉ validate 1 lần (ở service hoặc endpoint, không cả hai)

#### 🟡 P2 — Unused import
- `backend/src/services/material_service.py` L213: `from beanie.operators import In as BIn` — import `BIn` nhưng không dùng trong `delete_material`, chỉ dùng trong `delete_collection`

---

## 🤖 4. RAG PIPELINE

### ✅ Điểm Tốt
- Hybrid Retrieval: Dense + Sparse + Knowledge Graph → RRF Fusion → Rerank
- Query Rewriting: LLM-based Multi-Query / RAG-Fusion
- Fallback: Dictionary-based VI→EN translation khi LLM unavailable
- Confidence scoring với sigmoid normalization
- Grounded refusal mechanism (refuse khi confidence < threshold)
- Evidence traceability: mỗi câu trả lời có citation đến block + page

### ❌ Vấn Đề Phát Hiện

#### 🟡 P1 — CitationSchema.confidence bị clamp sai
- `backend/src/schemas/evidence.py` L25: `confidence: float = Field(ge=0.0, le=1.0)`
- Nhưng `backend/src/inference/response_parser.py` L44: `confidence=chunk.fused_score` — fused_score từ RRF có thể > 1.0
- **Hậu quả**: Pydantic validation error → 500 Internal Server Error
- **Fix**: Clamp: `confidence=min(1.0, max(0.0, chunk.fused_score))`

#### 🟡 P1 — Guardrails chưa được tích hợp
- `ClaimVerifier` và `ContradictionDetector` đã implement nhưng **KHÔNG được gọi** ở bất kỳ đâu
- Không có endpoint nào dùng claim verification
- Compare endpoint không dùng `ContradictionDetector` (trả về `conflicts=[]` cứng)
- **Fix**: Integrate vào `InferenceEngine.answer()` và `QueryService.compare()`

#### 🟡 P2 — Query Rewriter timeout mismatch
- `backend/src/core/openai_client.py` L17: Hardcode `timeout=60.0` — không dùng `settings.llm_timeout_seconds`
- `backend/src/core/local_llm.py` L14: Dùng `settings.llm_timeout_seconds` đúng
- **Fix**: OpenAI client cũng nên dùng `self.settings.llm_timeout_seconds`

#### 🟡 P2 — Ngưỡng từ chối (Refusal Threshold) quá khắt khe
- Phản hồi từ user cho thấy bot thường xuyên từ chối trả lời ("Tôi không tìm thấy đủ bằng chứng...") dù thông tin có tồn tại.
- Ngưỡng `min_evidence_confidence` đang cấu hình có thể làm giảm trải nghiệm.
- **Fix**: Cân nhắc thêm fallback LLM re-prompting, tinh chỉnh threshold, hoặc cho phép model trả lời một phần kèm cảnh báo thay vì từ chối hoàn toàn.

---

## 🎨 5. FRONTEND — React/Vite

### ✅ Điểm Tốt
- Three-column workspace layout (Sources → Chat → Studio)
- Clean state management via React Context
- Responsive design (mobile Sources panel collapse)
- Citation linking: click [N] → open Evidence panel
- Suggestion chips cho quick actions

### ❌ Vấn Đề Phát Hiện

#### 🔴 P0 — StudyGuideResponse type mismatch (đã nêu ở mục 1)
- Frontend expects `{ study_guide, sections }` nhưng backend trả `{ overview, key_concepts, outline }`
- Feature **Study Guide** sẽ hiển thị sai hoặc crash

#### 🟡 P1 — scopedMaterialIds tính từ local state, không phải server
- `frontend/src/state/workspace.tsx` L123: `scopedMaterialIds: materials.map(item => item.materialId)`
- `materials` là session-local uploads, không phải tất cả materials trong collection
- Khi user reload page → `materials` từ localStorage có thể stale
- **Fix**: Khi có `collectionId`, gửi `collection_id` thay vì `material_ids` cho query

#### 🟡 P1 — WorkspaceProvider nằm NGOÀI RouterProvider
- `frontend/src/main.tsx` L36-38: `WorkspaceProvider` wraps `RouterProvider`
- `useLocation()` trong `EvidenceRedirect` cần Router context
- Hiện tại hoạt động vì `EvidenceRedirect` render bên trong route — nhưng nếu cần `useNavigate` trong `WorkspaceProvider`, sẽ crash

#### 🟡 P2 — Không có loading skeleton / error boundary
- Không có React Error Boundary → unhandled error sẽ blank screen
- Không có skeleton loading states cho panels

#### 🟡 P2 — `package.json` dependency issues
- `@vitejs/plugin-react` nằm trong `dependencies` thay vì `devDependencies`
- `react@18.3.1` nhưng `@types/react@19.2.14` — types version ahead of runtime → có thể gây type errors

#### 🟡 P2 — Trải nghiệm UX còn nhiều ma sát (Friction Points)
- **Upload Progress**: Không có thanh tiến trình khi upload file lớn, chỉ có spinner khiến user không rõ trạng thái.
- **Onboarding Collection**: Dễ khiến user bối rối khi upload/hỏi nhưng chưa chọn Collection (app báo "Không có collection nào được chọn").
- **Giao diện Compare**: Khi văn bản so sánh dài, layout bảng hiển thị chật chội, khó đọc.
- **Fix**: Thêm progress bar cho upload, thêm empty state hướng dẫn onboarding, và cải thiện CSS layout cho Compare.

---

## 🐳 6. DEVOPS / DOCKER

### ✅ Điểm Tốt
- Docker Compose với 4 services (api, worker, qdrant, redis)
- Volume mounts cho data persistence
- `host.docker.internal` cho Ollama access từ container

### ❌ Vấn Đề Phát Hiện

#### 🟡 P1 — Dockerfile không có health check
- Không có `HEALTHCHECK` instruction → Docker/orchestrator không biết container healthy hay không
- **Fix**: Thêm `HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1`

#### 🟡 P1 — Không có frontend Docker build
- Chỉ có backend Dockerfile, không có frontend containerization
- **Fix**: Thêm multi-stage Dockerfile cho frontend (build → nginx serve)

#### 🟡 P2 — Worker và API dùng chung image
- Celery worker dùng cùng Dockerfile với API → image chứa uvicorn không cần thiết cho worker
- Chấp nhận được cho MVP, nhưng nên tách khi scale

#### 🟡 P2 — `.dockerignore` cần review
- Cần verify có exclude `node_modules`, `.env`, `__pycache__`, `data/`

---

## 📊 7. DATA MODELS (MongoDB)

### ✅ Điểm Tốt
- Beanie ODM với typed document models
- Versioning fields trên Material (parse_version, chunk_version, embedding_version, index_version)
- Pipeline status tracking (UPLOADED → PARSING → PARSED → INDEXING → INDEXED / FAILED)

### ❌ Vấn Đề Phát Hiện

#### 🟡 P1 — Thiếu MongoDB indexes
- Không thấy index definitions trên bất kỳ model nào
- Queries filter theo `owner_id`, `collection_id`, `material_id`, `status` — cần compound indexes
- **Fix**: Thêm `class Settings` với `indexes` trong mỗi Beanie Document model

#### 🟡 P2 — Knowledge Graph dùng string cho collection_id
- `Entity`, `Event` dùng `collection_id: str` thay vì `PydanticObjectId`
- `Relation` cũng dùng string IDs (`source_id`, `target_id`)
- Inconsistent với Chunk, Material (dùng `PydanticObjectId`)

---

## 🧪 8. TESTING

### ❌ Vấn Đề Nghiêm Trọng

#### 🔴 P0 — Tests gần như không có
- Test directories tồn tại (`test_api/`, `test_rag/`, `test_inference/`, `test_processing/`, `test_evaluation/`) nhưng chưa xác nhận có test files thực sự
- `conftest.py` chỉ setup sys.path, không có fixtures cho DB/Qdrant
- Không có integration tests
- Không có frontend tests (không có vitest/jest setup)
- **Khuyến nghị**: Viết unit tests cho ít nhất:
  - `QueryProcessor` (language detection, translation)
  - `ConfidenceScorer` (sigmoid, refusal logic)
  - `ResponseParser` (citation injection)
  - `LayoutAwareChunker` (chunking logic)
  - Security module (upload validation)

#### 📝 Quy Trình Đề Xuất: Test Định Dạng Đầu Vào (Input Formats Testing)
Hệ thống xử lý nhiều loại file (PDF, DOCX, PPTX, XLSX, PNG, JPG). Cần một quy trình test chuẩn hóa để đảm bảo pipeline không bị crash khi gặp file dị biệt:
1. **Corpus Test Tĩnh (Static Corpus)**: Tạo thư mục `tests/fixtures/files/` chứa các file mẫu đại diện cho từng trường hợp.
   - File chuẩn (Standard PDF, DOCX, PPTX).
   - File chứa bảng biểu phức tạp, công thức toán học (Complex Tables/Equations).
   - File ảnh độ phân giải thấp, chữ viết tay nhiễu (Noisy OCR/Handwriting).
   - File bị hỏng (Corrupted Files) để test error handling.
   - File giả mạo định dạng (ví dụ: đổi đuôi `.exe` thành `.pdf`) để test Security Magic Bytes.
2. **Unit / Integration Test cho từng Parser**:
   - Viết test `test_docling_parser.py` đảm bảo xuất ra đúng cấu trúc `ParsedDocument` (block list, tables).
   - Viết test `test_spreadsheet_parser.py` kiểm tra trích xuất cell/row/sheet từ XLSX.
   - Viết test `test_ocr_engine.py` đảm bảo nhận diện chính xác tiếng Việt.
3. **End-to-End (E2E) Pipeline Test**: Mock Database/Qdrant, chạy `ParseIndexPipeline.run()` với các file từ thư mục Corpus để đảm bảo toàn bộ luồng xử lý (Upload → Parse → Chunk → Embed) chạy trơn tru, không quăng exception bất ngờ.

---

## 📦 9. DEPENDENCIES

### ❌ Vấn Đề Phát Hiện

#### 🟡 P1 — Requirements không pin versions
- Các package sau **không có version pinning**:
  - `docling`, `pypdf`, `paddleocr`, `openpyxl`, `xlrd`, `opencv-python`, `numpy`, `FlagEmbedding`
- **Rủi ro**: Breaking changes khi cài trên môi trường khác
- **Fix**: Pin tất cả versions

#### 🟡 P1 — `sentence-transformers` missing từ requirements
- `backend/src/rag/reranker.py` L26-27: Import `sentence_transformers.CrossEncoder`
- Nhưng `sentence-transformers` **KHÔNG CÓ** trong `requirements.txt`
- **Fix**: Thêm `sentence-transformers>=2.7.0`

#### 🟡 P2 — Frontend type versions mismatch
- `react@18.3.1` nhưng `@types/react@19.2.14` — types version ahead of runtime
- Có thể gây type errors

---

## 📝 10. CODE QUALITY & MISC

### ❌ Vấn Đề Phát Hiện

#### 🟡 P1 — Logging inconsistency
- Có `config/logging_config.yaml` nhưng **không được load** ở bất kỳ đâu
- Backend dùng `logging.getLogger(__name__)` nhưng không configure root logger
- **Fix**: Load logging config trong `lifespan()` hoặc `create_app()`

#### 🟡 P2 — Log files committed
- `backend.log`, `backend_full.log`, `frontend.log`, `server_debug.log` ở project root
- **Fix**: Thêm `*.log` vào `.gitignore`

#### 🟡 P2 — Stale documentation
- `AgentBook_Implementation_Plan.md` (129KB) — rất lớn, có thể outdated
- `CLAUDE.md` — instructions cho AI assistant, không phải project doc

#### 🟡 P2 — `__pycache__` directories committed
- Multiple `__pycache__` dirs trong repo
- **Fix**: Thêm `__pycache__/` vào `.gitignore`, chạy `git rm -r --cached **/__pycache__`

---

## 📊 Tổng Kết Severity

| Severity | Count | Examples |
|---|---|---|
| 🔴 **P0 — Critical** | 5 | MongoDB creds exposed, No auth, Schema mismatch, No tests, Memory OOM Risk (to_list) |
| 🟡 **P1 — Important** | 17 | Storage Leak, Orphaned Graph Nodes, N+1 queries, Missing dependency, CORS, Index missing |
| 🟢 **P2 — Minor** | 14 | Double validation, Unused imports, Log files, Refusal threshold strict, UX friction points |

---

## 🎯 Recommended Action Plan (Ưu Tiên)

### Phase 1 — Critical Fixes (1-2 ngày)
1. **Rotate MongoDB credentials** + đảm bảo `.env` trong `.gitignore`
2. **Fix StudyGuideResponse** schema mismatch (frontend ↔ backend)
3. **Fix CitationSchema confidence** clamping (fused_score > 1.0)
4. **Hoàn tất rebranding** AgentBook → Prism (prompts, config, localStorage keys, celery name)
5. **Thêm `sentence-transformers`** vào `requirements.txt`

### Phase 2 — Security, Storage & Performance (3-5 ngày)
6. **Thêm Authentication** (JWT hoặc API key)
7. **Fix Memory OOM Risk** — áp dụng Pagination cho tất cả `.to_list()` queries và dùng `.count()` thay thế.
8. **Fix Storage Leak & Orphan Nodes** — dọn dẹp triệt để file `.json` và `Entity/Relation` khi delete material.
9. **Batch hydration** trong Retriever (giảm N+1 MongoDB queries)
10. **Pin all dependency versions**
11. **Thêm MongoDB compound indexes**

### Phase 3 — Quality & Resilience (1 tuần)
12. **Viết unit tests** cho core modules (target 70% coverage)
13. **Integrate GuardRails** (ClaimVerifier, ContradictionDetector)
14. **Thêm Rate Limiting** middleware
15. **Thêm Error Boundary** trong React
16. **Cải thiện UX**: Thêm Upload progress bar, tinh chỉnh Refusal threshold, fix layout Compare
17. **Load logging config** từ YAML
18. **Thêm Dockerfile** cho frontend
19. **Clean up** committed logs, `__pycache__`, stale docs

---

## 🧑‍💻 Appendix: End-User Experience Feedback (Mở Rộng)

> **Vai trò**: Người dùng cuối (End-user) thử nghiệm toàn bộ tính năng  
> **Phương pháp**: Code review toàn bộ UI components + mô phỏng luồng sử dụng thực tế  
> **Ngày cập nhật**: 2026-04-28

---

### 1. 🚀 First Impression & Onboarding

#### ✅ Điểm tốt
* Header đẹp, logo Prism gọn gàng, health status indicator (Online/Offline) cho cảm giác chuyên nghiệp.
* Three-column layout (Sources → Chat → Studio) giống NotebookLM — rất quen thuộc, dễ hiểu.
* Khi chưa có collection, chat panel hiện "No active collection" + badge "No sources" → rõ ràng về trạng thái.

#### ❌ Vấn đề
* **Không có onboarding wizard/tutorial:** User mới mở app lần đầu sẽ KHÔNG BIẾT phải làm gì. Không có hướng dẫn nào giải thích flow "Tạo Collection → Upload file → Chờ index xong → Hỏi". Chỉ có 1 dòng intro chat: *"Upload some sources on the left to get started"* — quá ngắn gọn.
* **Owner ID hardcode `"user_demo"`** trong default settings — user không hiểu field này dùng để làm gì, có cần thay đổi không. Không có trang đăng nhập/đăng ký.
* **Welcome message bằng tiếng Anh** (*"Welcome to Prism!"*) nhưng suggestion chips lại bằng **tiếng Việt** (*"Tóm tắt các ý chính..."*) → lẫn lộn ngôn ngữ, không đồng nhất.

**🎯 Gợi ý**: Thêm empty state lớn hơn ở center panel với step-by-step: "1️⃣ Chọn/tạo Collection → 2️⃣ Upload tài liệu → 3️⃣ Đặt câu hỏi". Đồng nhất ngôn ngữ UI theo `workspace.language`.

---

### 2. 📁 Quản Lý Sources (Panel Trái)

#### ✅ Điểm tốt
* Drag-and-drop upload hoạt động trực quan, hỗ trợ nhiều file cùng lúc.
* File queue hiển thị tên + size trước khi upload, cho phép xóa file khỏi queue.
* Collection selector tự động chọn collection có nhiều chunks nhất khi user chưa chọn gì → smart default.
* Hiện thống kê `X docs / Y chunks` cho mỗi collection → user biết collection có dữ liệu chưa.
* Delete material/collection có confirm dialog 2 bước (Yes/No) → tránh xóa nhầm.

#### ❌ Vấn đề
* **Không có progress bar upload**: Chỉ có spinner + tên file đang upload (`setUploadingName`). Với file PDF 15MB, user chờ 30s+ mà không biết % bao nhiêu.
* **Sau upload, status material chỉ hiện badge (`UPLOADED` / `PARSING` / `INDEXED`)** nhưng KHÔNG TỰ ĐỘNG refresh. User phải bấm nút Refresh thủ công để thấy status chuyển từ PARSING → INDEXED.
* **Không hiện page_count**: Field `page_count` có trong schema nhưng UI không hiển thị. User upload file 200 trang sẽ không biết hệ thống đã nhận đúng bao nhiêu trang.
* **Xóa collection chỉ có text "Delete?" nhỏ xíu** — dễ bấm nhầm vì nút quá nhỏ (icon Trash2 size=12).
* **File queue mất sau khi upload xong**: Nếu upload 5 file, 1 file lỗi → toàn bộ queue dừng (`break`). Files sau file lỗi bị bỏ qua hoàn toàn mà không có thông báo rõ ràng.

**🎯 Gợi ý**: 
- Thêm `XMLHttpRequest.upload.onprogress` hoặc `fetch` progress cho upload.
- Auto-poll status mỗi 5s khi có material ở trạng thái `PARSING`.
- Khi upload batch bị lỗi, tiếp tục upload các file còn lại thay vì `break`.

---

### 3. 💬 Chat & Hỏi Đáp (Panel Giữa)

#### ✅ Điểm tốt — "Wow Factors"
* **Citation buttons `[1]` `[2]` `[3]`** inline trong câu trả lời — bấm vào tự động mở Evidence panel, scroll đến đúng block. Đây là tính năng **ấn tượng nhất** của app.
* **Casual reply handler** rất thông minh: gõ "hi", "bạn là ai" → trả lời local không cần gọi API → response instant.
* **Markdown rendering** hỗ trợ `**bold**`, `` `code` ``, list items → output đẹp, dễ đọc.
* **Suggestion chips** ở dưới input: "Tóm tắt các ý chính", "So sánh...", "Tạo study guide" → giúp user biết app làm được gì.
* **Auto-scroll to bottom** khi có message mới → chuẩn chat UX.
* **Textarea auto-expand** theo số dòng, hỗ trợ Shift+Enter xuống dòng → tốt.

#### ❌ Vấn đề
* **🔴 Không có streaming**: User phải chờ 10-30s cho toàn bộ LLM response. Trong thời gian chờ chỉ thấy spinner *"Thinking and searching documents..."* → cảm giác app bị treo, đặc biệt khi model chậm.
* **🔴 Refused message hiện `refusal_reason` raw** thay vì answer: `"The answer was refused by guardrails."` hoặc `"confidence too low"` → user không hiểu gì. Nên hiện message thân thiện hơn.
* **🟡 Suggestion chips chỉ fill text vào input, KHÔNG auto-submit** → user phải bấm Enter thêm 1 lần. "Tạo study guide" suggestion thực ra cũng chỉ fill text rồi gọi chat API, không gọi trực tiếp `buildStudyGuide()` → kết quả sẽ là chat answer thường thay vì structured study guide.
* **🟡 Chat history mất khi reload trang**: Messages chỉ giữ trong React state, không persist vào localStorage hay database. User reload → mất hết lịch sử.
* **🟡 Không hiển thị confidence score** cho câu trả lời. Backend trả `confidence` nhưng UI không show. User không biết câu trả lời đáng tin cậy đến mức nào.
* **🟡 Không có retry button**: Khi API lỗi, user phải gõ lại câu hỏi. Nên có nút "Thử lại" trên message lỗi.
* **🟡 Citation footer hiện TẤT CẢ citations** khi `was_refused = true` — nghĩa là bot từ chối trả lời nhưng vẫn liệt kê nguồn tham khảo → mâu thuẫn logic.

**🎯 Gợi ý**: 
- Ưu tiên #1: Thêm SSE streaming cho LLM response.
- Hiển thị confidence badge (🟢/🟡/🔴) cạnh mỗi assistant message.
- Persist chat history vào localStorage.
- Suggestion "Tạo study guide" nên route sang StudioPanel thay vì chat.

---

### 4. 📖 Evidence Panel (Panel Phải)

#### ✅ Điểm tốt — **Killer Feature**
* **Keyword highlighting** tự động: extract keywords từ citation snippet → highlight vàng trong full page blocks → giúp user tìm context xung quanh đoạn trích dẫn rất nhanh.
* **Block type badges** (text/heading/table/image) với màu phân biệt → biết ngay loại content.
* **Confidence bar** cho mỗi block + cho citation tổng → trực quan.
* **Copy button** trên matched snippet → tiện để paste vào tài liệu khác.
* **Collapsible snippets** cho text dài (>6 lines) → không bị tràn screen.
* **Citation navigation** (← 1/5 →) cho phép duyệt qua tất cả citations → rất mượt.
* **Role badge** (Primary/Supporting) → user biết đâu là nguồn chính.

#### ❌ Vấn đề
* **🟡 Không có translated snippet**: Field `snippet_translated` có trong schema và code render nó, nhưng backend không gửi translation → luôn null → user đọc tài liệu tiếng Anh nhưng không thấy bản dịch.
* **🟡 Không có search/filter** trong evidence page blocks. Nếu page có 20+ blocks, khó tìm block cần thiết.
* **🟡 Evidence panel trống khi chưa click citation**: Empty state chỉ có *"Click a citation [N] in the chat"* — OK nhưng có thể thêm recent sources hoặc bookmarks.
* **🟡 Page navigation** chỉ trong 1 page (page number cố định từ citation). Không có nút Previous/Next page để đọc context rộng hơn.

---

### 5. 🧪 Studio Panel — Summary & Study Guide

#### ✅ Điểm tốt
* Artifact card design đẹp: header với icon + collapse/download/delete buttons → gọn gàng.
* Download as `.txt` → tiện cho student muốn lưu offline.

#### ❌ Vấn đề
* **🟡 Study Guide output chất lượng thấp**: Backend dùng regex `[A-Z][A-Za-z0-9\-]{2,}` để extract key concepts → chỉ bắt được capitalized English words. Với tài liệu tiếng Việt (không viết hoa), `_key_concepts()` sẽ trả về fallback `["Core concepts", "Definitions", "Examples"]` → vô nghĩa.
* **🟡 Outline chỉ là `"Review: {concept}"` lặp lại** — không phải outline thật sự theo cấu trúc tài liệu. Prompt gửi cho LLM quá generic, không yêu cầu structured JSON output.
* **🟡 Không có loading skeleton**: Khi generate summary/study guide, panel chỉ hiện spinner trên button. Phần content area trống hoàn toàn → user không biết output sẽ xuất hiện ở đâu.
* **🟡 Summary không hiện citations**: Dù backend trả `citations` nhưng StudioHomeTab không render citation buttons trong summary text.
* **🟡 Chỉ 2 tool (Summarize + Study Guide)**: So với NotebookLM có FAQ, Timeline, Podcast... → Prism Studio cảm giác còn sơ khai.

**🎯 Gợi ý**: 
- Nâng cấp prompt thành structured JSON output.
- Thêm `review_questions` (câu hỏi ôn tập) vào Study Guide.
- Hiện citations inline trong summary text.
- Thêm tools: FAQ Generator, Flashcards, Key Terms Glossary.

---

### 6. ⚖️ Compare Tab

#### ✅ Điểm tốt
* **Dimension tag input** rất cool: gõ → Enter → thêm tag, Backspace xóa tag cuối → UX nhanh.
* **Default dimensions** hợp lý: definition, intuition, example, limitation.
* **ResultCard** có expand/collapse cho text dài → xử lý tốt long content.
* **Confidence pill** màu phân biệt (xanh >70%, vàng 40-70%, đỏ <40%) → trực quan.

#### ❌ Vấn đề
* **🔴 `conflicts` luôn rỗng `[]`**: Backend hardcode `conflicts=[]`, ContradictionDetector đã implement nhưng chưa gọi → Conflicts Detected section KHÔNG BAO GIỜ hiện. UI có code render nhưng không bao giờ trigger.
* **🟡 Mỗi dimension = 1 API call retrieval + rerank riêng**: 4 dimensions = 4 lần retrieval tuần tự → chậm. Nên batch hoặc parallel.
* **🟡 Khi no evidence** cho 1 dimension → hiện "Không tìm thấy evidence cho chiều này." nhưng card vẫn chiếm space → nên collapse hoặc grey out.
* **🟡 Không có export**: Không thể download bảng so sánh ra PDF/CSV/Markdown.

---

### 7. 🕸️ Graph & Mindmap Tabs

#### ✅ Điểm tốt
* Fullscreen mode qua Portal — UX tốt, Esc để đóng.
* Selected node footer hiện label + ID → biết đang chọn node nào.
* Mindmap cho phép customize root topic → linh hoạt.

#### ❌ Vấn đề
* **🟡 Graph không auto-load**: User phải bấm "Generate Graph" mỗi lần mở tab. Nên auto-load khi có collection active.
* **🟡 Khi graph rỗng** (chưa generate), empty state text hơi dài và không actionable: *"Click Generate to visualize..."* → nên thêm button trực tiếp trong empty state.
* **🟡 Node positions không persist**: Khi user kéo node rồi switch tab, quay lại → positions reset. Layout thuật toán cơ bản, nodes có thể overlap.
* **🟡 Không có node detail**: Click node chỉ hiện label + ID ở footer. Không hiện entity properties, evidence refs, hay linked materials.
* **🟡 Mindmap root topic default `"Central Topic"`** quá generic. Nên auto-detect từ collection name hoặc subject.

---

### 8. ⚙️ Settings Modal

#### ✅ Điểm tốt
* "Test API" button rất hữu ích — cho biết backend có online không + số docs indexed.
* Collection selector trong settings sync với Sources panel.

#### ❌ Vấn đề
* **🟡 Chỉ 2 ngôn ngữ**: vi/en. Không hỗ trợ zh, ja, ko... dù backend có `_LANGUAGE_NAMES` cho các ngôn ngữ này.
* **🟡 Settings dùng URL params** (`?settings=open`) thay vì modal state → reload page có thể mở Settings bất ngờ.
* **🟡 Không có dark mode toggle** — trong 2026 thì dark mode gần như là yêu cầu bắt buộc.
* **🟡 API Base URL chỉ hiển thị, không cho chỉnh** → hardcode trong env. User self-host sẽ phải rebuild frontend.

---

### 9. 📱 Mobile & Responsive

#### ✅ Điểm tốt
* Sources panel collapse/slide trên mobile — hamburger menu (Library icon) hoạt động tốt.
* Chat panel responsive, input area fit full width.

#### ❌ Vấn đề
* **🔴 Studio panel (right) hoàn toàn ẩn trên mobile**: CSS `hidden lg:flex` → dưới 1024px, user KHÔNG THỂ truy cập Evidence, Studio, Compare, Graph, Mindmap. Đây là half of the features bị mất trên mobile.
* **🟡 Suggestion chips overflow** trên mobile hẹp — dù có `overflow-x-auto no-scrollbar` nhưng user không biết scroll ngang được.

**🎯 Gợi ý**: Thêm tab bar hoặc bottom sheet để truy cập Studio features trên mobile.

---

### 10. ♿ Accessibility & Polish

* **Không có `aria-label`** trên hầu hết interactive elements → screen reader không mô tả được buttons.
* **Không có keyboard shortcuts**: Không có Ctrl+K (search), Ctrl+/ (help), Ctrl+Enter (submit). Chỉ có Enter để send chat.
* **Favicon chưa set** — browser tab chỉ hiện icon mặc định Vite.
* **Title tag**: `<title>` chỉ là default Vite → nên đổi thành "Prism — Document Intelligence Workspace".
* **Loading states thiếu skeleton**: Mọi loading đều dùng `<Loader2 className="animate-spin" />`. Không có skeleton placeholder nào → layout nhảy khi content load xong.
* **Không có toast/notification system**: Success/error messages dùng inline divs → dễ bị scroll ra ngoài viewport và user không thấy.

---

### 📊 Bảng Tổng Kết UX Theo Tính Năng

| Tính năng | First Impression | Usability | Completeness | Điểm /10 |
|---|---|---|---|---|
| Upload Sources | 🟢 Tốt | 🟡 Thiếu progress | 🟡 Thiếu auto-poll status | 6/10 |
| Chat Q&A | 🟢 Ấn tượng | 🟡 Không streaming | 🟡 Thiếu retry, history | 7/10 |
| Citations/Evidence | 🟢 Xuất sắc | 🟢 Highlight + nav mượt | 🟡 Thiếu page navigation | 8/10 |
| Studio Summary | 🟢 Đẹp | 🟡 Thiếu citations | 🟡 Chỉ plain text | 6/10 |
| Studio Study Guide | 🟡 OK | 🔴 Output generic | 🔴 Regex key concepts thô | 4/10 |
| Compare | 🟢 Card layout đẹp | 🟡 Chậm | 🔴 Conflicts luôn rỗng | 5/10 |
| Graph/Mindmap | 🟢 Trực quan | 🟡 Không auto-load | 🟡 Thiếu node details | 6/10 |
| Settings | 🟡 Cơ bản | 🟢 Test API tiện | 🟡 Thiếu dark mode | 5/10 |
| Mobile | 🟡 Partial | 🔴 Studio ẩn hoàn toàn | 🔴 Mất nửa features | 3/10 |
| **Tổng thể** | | | | **5.6/10** |

> **Nhận xét chung**: Prism có **nền tảng kỹ thuật rất mạnh** (Hybrid RAG + Graph + Reranker + Citation tracing). Citation/Evidence flow là **killer feature** cạnh tranh được với NotebookLM. Tuy nhiên, phần **UX polish, onboarding, mobile, và một số tính năng Studio** cần được nâng cấp đáng kể để đạt mức production-ready.
