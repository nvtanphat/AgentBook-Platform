# 🛠️ Prism — Báo Cáo Gợi Ý Nâng Cấp & Sửa Chữa

> **Vai trò**: AI Engineer  
> **Ngày**: 2026-04-28  
> **Phạm vi**: Toàn bộ tính năng Backend + Frontend + RAG Pipeline

---

## 📊 Tổng Quan Đánh Giá

| Hạng mục | Trạng thái | Mức độ |
|---|---|---|
| Study Guide | ✅ Đã fix — Structured prompt + LLM concept extraction | ~~Critical~~ → Done |
| Chat Q&A | ✅ Đã fix — Smart Refusal 3-tier + friendly messages | ~~Important~~ → Done |
| Compare | ✅ Đã fix — ContradictionDetector tích hợp | ~~Important~~ → Done |
| Graph/Mindmap | 🟢 OK | Minor polish |
| Upload Pipeline | ✅ Đã fix — Progress bar + continue-on-error | ~~Important~~ → Done |
| Guardrails | ✅ Đã fix — ContradictionDetector wired vào Compare | ~~Critical~~ → Done |
| Performance | ✅ Đã fix — Batch fetch $in thay vì N+1 | ~~Important~~ → Done |
| Security | 🔴 Không có Authentication | Critical |

---

## 🔴 TIER 1 — SỬA LỖI NGHIÊM TRỌNG (Làm ngay)

### 1.1 ✅ Graph Retriever: N+1 Sequential DB Queries — ĐÃ FIX

> [!CAUTION]
> Mỗi `EvidenceRef` gọi `await Material.get()` riêng lẻ. 10 refs = 10 MongoDB queries tuần tự!

**File**: [graph_retriever.py](file:///d:/GenAI/DoAn01/backend/src/rag/graph_retriever.py#L92-L120)

**Vấn đề**: `_hydrate_evidence_refs()` gọi `Material.get()` trong vòng lặp — đây là pattern N+1 điển hình.

**Sửa**:
```python
async def _hydrate_evidence_refs(self, refs: list[EvidenceRef]) -> list[EvidenceBlock]:
    if not refs:
        return []
    # Batch fetch all unique materials
    unique_material_ids = list({ref.material_id for ref in refs})
    materials_list = await Material.find({"_id": {"$in": unique_material_ids}}).to_list()
    materials_by_id = {m.id: m for m in materials_list}
    
    evidence: list[EvidenceBlock] = []
    for ref in refs:
        material = materials_by_id.get(ref.material_id)
        if material is None:
            continue
        page = next((p for p in material.pages if p.page_number == ref.page), None)
        if page is None:
            continue
        block = next((b for b in page.blocks if b.block_id == ref.block_id), None)
        if block is None:
            continue
        evidence.append(EvidenceBlock(
            owner_id=material.owner_id,
            collection_id=str(material.collection_id),
            material_id=str(material.id),
            document_name=material.original_name,
            page=page.page_number,
            block_id=block.block_id,
            block_type=block.block_type,
            snippet_original=block.content,
            source_language=block.language,
            bbox=block.bbox,
            confidence=block.ocr_confidence,
            metadata=block.extra,
        ))
    return evidence
```

**Tác động**: Giảm từ N MongoDB queries → 1 query duy nhất. Cải thiện latency 5-10x cho graph retrieval.

---

### 1.2 ✅ Confidence Clamping — Ngăn 500 Error — ĐÃ FIX

**File**: [response_parser.py](file:///d:/GenAI/DoAn01/backend/src/inference/response_parser.py#L44)

**Vấn đề**: `fused_score` từ RRF có thể > 1.0, nhưng `CitationSchema.confidence` có `Field(ge=0.0, le=1.0)` → Pydantic validation error → 500.

**Trạng thái hiện tại**: Đã được fix ở L44 với `min(1.0, max(0.0, ...))` ✅

**Nhưng còn thiếu ở**: [evidence.py](file:///d:/GenAI/DoAn01/backend/src/schemas/evidence.py) — nên thêm `@field_validator` để defensive:

```python
@field_validator("confidence", mode="before")
@classmethod
def clamp_confidence(cls, v: float) -> float:
    return min(1.0, max(0.0, v))
```

---

### 1.3 ✅ Tích Hợp GuardRails (ContradictionDetector vào Compare) — ĐÃ FIX

> [!IMPORTANT]
> Hai module `ClaimVerifier` và `ContradictionDetector` đã được implement hoàn chỉnh nhưng **KHÔNG ĐƯỢC GỌI** ở bất kỳ đâu trong codebase!

**Files**:
- [claim_verifier.py](file:///d:/GenAI/DoAn01/backend/src/guardrails/claim_verifier.py) — Verify claims against evidence
- [contradiction_detector.py](file:///d:/GenAI/DoAn01/backend/src/guardrails/contradiction_detector.py) — Detect numeric contradictions

**Gợi ý tích hợp**:

#### A. Tích hợp ContradictionDetector vào Compare

File: [query_service.py](file:///d:/GenAI/DoAn01/backend/src/services/query_service.py#L60-L89)

Hiện tại `conflicts=[]` luôn trả về rỗng. Sửa:

```python
from src.guardrails.contradiction_detector import ContradictionDetector

async def compare(self, request: CompareRequest) -> CompareResponse:
    # ... existing code ...
    
    # Detect contradictions across all evidence
    all_evidence = []
    for cell_chunks in all_reranked_chunks:
        for chunk in cell_chunks:
            all_evidence.extend(chunk.evidence)
    
    detector = ContradictionDetector()
    contradictions = detector.detect(all_evidence)
    conflict_descriptions = [c.description for c in contradictions]
    
    return CompareResponse(
        topic=request.topic,
        comparison_table=cells,
        citations=list(deduped.values()),
        conflicts=conflict_descriptions,  # Không còn hardcode []
    )
```

#### B. Tích hợp ClaimVerifier vào Q&A (optional post-processing)

File: [inference_engine.py](file:///d:/GenAI/DoAn01/backend/src/inference/inference_engine.py#L128)

```python
from src.guardrails.claim_verifier import ClaimVerifier

# Sau khi LLM trả lời, verify claims trong answer:
verifier = ClaimVerifier()
all_evidence = [e for chunk in reranked for e in chunk.evidence]
verification = verifier.verify(claim=answer, evidence=all_evidence)
# Nếu contradicted → thêm warning vào answer
```

---

### 1.4 OOM Risk — Pagination cho List Endpoints

**Files bị ảnh hưởng**:

| File | Dòng | Vấn đề |
|---|---|---|
| [admin_service.py](file:///d:/GenAI/DoAn01/backend/src/services/admin_service.py) | L35 | `aggregate().to_list()` — OK vì chỉ 1 row |
| [graph_retriever.py](file:///d:/GenAI/DoAn01/backend/src/rag/graph_retriever.py#L52-L55) | L52, L55 | `Entity.find().to_list()`, `Relation.find().to_list()` — **cần limit** |

**Sửa cho Graph Retriever**:
```python
async def _matching_entities(self, *, query, scope) -> list[Entity]:
    # Thêm limit để tránh load quá nhiều entities
    return await Entity.find(
        self._scope_query(scope, {"$or": or_conditions})
    ).limit(50).to_list()

async def _scoped_relations(self, scope) -> list[Relation]:
    return await Relation.find(
        self._scope_query(scope, {"confidence": {"$gte": self.settings.min_graph_confidence}})
    ).limit(200).to_list()
```

---

### 1.5 Authentication — Thêm JWT Middleware

> [!CAUTION]
> Toàn bộ API hiện tại KHÔNG có authentication. `owner_id` truyền từ client → ai cũng có thể truy cập data người khác.

**Gợi ý kiến trúc**:

```
backend/src/core/auth.py          [NEW] — JWT decode + verify
backend/src/api/v1/deps.py        [NEW] — get_current_user dependency
backend/src/models/user.py        [NEW] — User document model
```

**Ví dụ `auth.py`**:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]  # owner_id
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
```

Sau đó inject vào mọi endpoint:
```python
@router.post("/ask")
async def ask(request: QueryRequest, owner_id: str = Depends(get_current_user)):
    request.owner_id = owner_id  # Override client-supplied owner_id
```

---

## 🟡 TIER 2 — NÂNG CẤP TÍNH NĂNG (1-2 tuần)

### 2.1 ✅ Smart Refusal — Trả Lời Một Phần Thay Vì Từ Chối Hoàn Toàn — ĐÃ FIX

**Vấn đề**: Bot quá hay từ chối. User feedback: *"hỏi câu mà chắc chắn tài liệu có nói đến, nhưng bot vẫn từ chối"*

**File**: [confidence_scorer.py](file:///d:/GenAI/DoAn01/backend/src/inference/confidence_scorer.py#L36-L55)

**Gợi ý — 3 mức refusal thay vì 2**:

```python
def should_refuse(self, *, chunks, confidence) -> tuple[bool, str | None]:
    if not chunks:
        return True, "no relevant evidence"
    
    top = chunks[0]
    normalized_top = _sigmoid(top.rerank_score) if top.rerank_score is not None else confidence
    
    # Tier 1: Đủ confident → trả lời bình thường
    if normalized_top >= self.settings.min_evidence_confidence:
        return False, None
    
    # Tier 2: Nửa confident → trả lời kèm cảnh báo (KHÔNG từ chối)
    SOFT_THRESHOLD = self.settings.min_evidence_confidence * 0.6
    if normalized_top >= SOFT_THRESHOLD:
        return False, "partial_confidence"  # Frontend hiển thị warning banner
    
    # Tier 3: Quá thấp → từ chối
    return True, "confidence too low"
```

**Frontend**: Khi `refusal_reason == "partial_confidence"`, hiển thị banner vàng:
> ⚠️ Câu trả lời này dựa trên bằng chứng hạn chế. Vui lòng kiểm tra lại nguồn gốc.

---

### 2.2 Streaming LLM Response

**Vấn đề**: User phải chờ toàn bộ LLM generate xong mới thấy response. Với model 3-4B params, có thể mất 10-30s.

**File**: [inference_engine.py](file:///d:/GenAI/DoAn01/backend/src/inference/inference_engine.py#L104)

**Gợi ý**: Thêm endpoint `/query/ask/stream` dùng SSE:

```python
# backend/src/api/v1/endpoints/query.py
from fastapi.responses import StreamingResponse

@router.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    async def event_generator():
        # 1. Retrieve + Rerank (gửi status event)
        yield f"data: {json.dumps({'type': 'status', 'message': 'Đang tìm kiếm...'})}\n\n"
        
        # 2. Stream LLM tokens
        async for token in llm.generate_stream(prompt=prompt):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        
        # 3. Gửi citations cuối cùng
        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Frontend**: Dùng `EventSource` hoặc `fetch` + `ReadableStream`:
```typescript
const response = await fetch(`${API_V1_BASE_URL}/query/ask/stream`, { method: "POST", body: ... });
const reader = response.body!.getReader();
// Progressive rendering từng token
```

---

### 2.3 ✅ Upload Progress Tracking — ĐÃ FIX

**Vấn đề**: Upload file lớn chỉ có spinner, user không biết tiến độ.

**Gợi ý 2 phần**:

#### A. Frontend: XMLHttpRequest progress event

```typescript
// api/client.ts
export function uploadMaterialWithProgress(
  file: File,
  metadata: MaterialUploadMetadata,
  onProgress: (percent: number) => void,
): Promise<MaterialUploadResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => { /* parse response */ };
    xhr.onerror = () => reject(new Error("Upload failed"));
    
    const formData = new FormData();
    formData.append("metadata", JSON.stringify(metadata));
    formData.append("file", file);
    
    xhr.open("POST", `${API_V1_BASE_URL}/materials/upload`);
    xhr.send(formData);
  });
}
```

#### B. Backend: Pipeline status polling — ĐÃ FIX

Đã thêm endpoint `GET /materials/{id}/status` trả `status`, `stage`, `progress_pct`, `failed_stage`, `error_message`.

Frontend đã dùng endpoint này để refresh session-local material sau upload, đồng thời reload danh sách materials theo `response.collection_id` thật thay vì phụ thuộc vào state `workspace.collectionId` có thể chưa cập nhật.

Mẫu endpoint:
```python
@router.get("/{material_id}/status")
async def material_status(material_id: str, owner_id: str):
    material = await Material.get(PydanticObjectId(material_id))
    job = await PipelineJob.find_one(PipelineJob.material_id == material.id)
    return {
        "status": material.status,
        "stage": job.stage if job else "unknown",
        "progress_pct": _estimate_progress(job),  # UPLOADED=10, PARSING=30, PARSED=50, INDEXING=80, INDEXED=100
    }
```

---

### 2.4 ✅ Cải Thiện Study Guide — Structured Output — ĐÃ FIX

**File**: [study_guide_service.py](file:///d:/GenAI/DoAn01/backend/src/services/study_guide_service.py#L71-L92)

**Vấn đề**: Prompt hiện tại quá generic, kết quả không structured. `_key_concepts()` chỉ regex tìm capitalized words — rất thô.

**Gợi ý nâng cấp prompt**:

```python
STUDY_GUIDE_PROMPT = """
Dựa trên EVIDENCE bên dưới, tạo Study Guide bằng tiếng Việt theo format JSON:

{{
  "overview": "Tóm tắt tổng quan 3-5 câu về nội dung chính",
  "key_concepts": ["Khái niệm 1", "Khái niệm 2", ...],
  "outline": [
    "I. Chương/Phần chính 1",
    "  1.1 Ý chính con",
    "II. Chương/Phần chính 2",
    ...
  ],
  "review_questions": [
    "Câu hỏi ôn tập 1?",
    "Câu hỏi ôn tập 2?"
  ]
}

LUẬT: Chỉ dùng thông tin có trong EVIDENCE. Không suy diễn ngoài.

EVIDENCE:
{evidence}
"""
```

Sau đó parse JSON từ LLM output → populate `StudyGuideResponse`.

**Bonus**: Thêm `review_questions` vào schema:
```python
class StudyGuideResponse(BaseModel):
    overview: str
    key_concepts: list[str]
    outline: list[str]
    review_questions: list[str] = Field(default_factory=list)  # NEW
    citations: list[CitationSchema]
    confidence: float
```

---

### 2.5 Compare Layout — Responsive Table

**File**: [CompareTab.tsx](file:///d:/GenAI/DoAn01/frontend/src/components/workspace/studio/CompareTab.tsx)

**Vấn đề**: Bảng so sánh chật chội khi text dài.

**Gợi ý**: Chuyển từ `<table>` sang card-based layout cho mobile, giữ table cho desktop:

```css
/* Responsive compare */
@media (max-width: 768px) {
  .compare-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .compare-cell {
    border: 1px solid var(--outline);
    border-radius: 8px;
    padding: 12px;
  }
}
```

---

### 2.6 MongoDB Indexes — Tăng Tốc Query

**Vấn đề**: Không có compound indexes → full collection scan cho mọi query.

**Gợi ý** — thêm vào mỗi Beanie Document:

```python
# models/chunk.py
class Chunk(Document):
    class Settings:
        name = "chunks"
        indexes = [
            [("owner_id", 1), ("collection_id", 1)],
            [("material_id", 1)],
            [("collection_id", 1), ("source_pages", 1)],
        ]

# models/material.py
class Material(Document):
    class Settings:
        name = "materials"
        indexes = [
            [("owner_id", 1), ("collection_id", 1), ("status", 1)],
            [("owner_id", 1), ("updated_at", -1)],
        ]

# models/query_log.py
class QueryLog(Document):
    class Settings:
        name = "query_logs"
        indexes = [
            [("owner_id", 1), ("collection_id", 1)],
            [("created_at", -1)],
        ]
```

---

### 2.7 React Error Boundary

**Vấn đề**: Bất kỳ unhandled error nào sẽ blank screen toàn app.

**Gợi ý**: Thêm Error Boundary wrapper:

```tsx
// components/ErrorBoundary.tsx
import { Component, ReactNode } from "react";

export class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  state = { hasError: false, error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="flex flex-col items-center justify-center h-full p-8">
          <h2 className="text-lg font-bold text-red-600">Đã xảy ra lỗi</h2>
          <p className="text-sm text-muted mt-2">{this.state.error?.message}</p>
          <button onClick={() => window.location.reload()} className="mt-4 btn">
            Tải lại trang
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
```

Wrap trong `main.tsx`:
```tsx
<ErrorBoundary>
  <WorkspaceProvider>
    <RouterProvider router={router} />
  </WorkspaceProvider>
</ErrorBoundary>
```

---

### 2.8 Rate Limiting

**File**: [main.py](file:///d:/GenAI/DoAn01/backend/src/main.py)

**Gợi ý**: Dùng `slowapi` (wrapper trên `limits`):

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

def create_app() -> FastAPI:
    app = FastAPI(...)
    app.state.limiter = limiter
    app.add_exception_handler(429, _rate_limit_exceeded_handler)
    
    # LLM endpoints — stricter limit
    @router.post("/ask")
    @limiter.limit("10/minute")
    async def ask(request: Request, body: QueryRequest): ...
```

---

## 🟢 TIER 3 — CẢI TIẾN KIẾN TRÚC (Dài hạn)

### 3.1 Caching Layer

Thêm in-memory cache (hoặc Redis cache) cho:
- **Embedding results**: Cùng query text → cùng embedding → cache 5 phút
- **Material metadata**: Batch-fetched materials → cache trong request scope
- **Collection summaries**: Cache `listCollections` response 30s

```python
from functools import lru_cache
from cachetools import TTLCache

# Embedding cache — tránh re-encode cùng query
embedding_cache = TTLCache(maxsize=100, ttl=300)
```

### 3.2 WebSocket cho Pipeline Status

Thay vì polling `GET /materials/{id}/status`, dùng WebSocket push:

```python
# backend/src/api/v1/endpoints/ws.py
from fastapi import WebSocket

@router.websocket("/ws/pipeline/{job_id}")
async def pipeline_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    while True:
        job = await PipelineJob.find_one(PipelineJob.job_id == job_id)
        await websocket.send_json({"stage": job.stage, "status": job.status})
        if job.status in ("INDEXED", "FAILED"):
            break
        await asyncio.sleep(2)
```

### 3.3 Logging Configuration

**File**: [main.py](file:///d:/GenAI/DoAn01/backend/src/main.py#L16-L21)

Thêm load logging config trong lifespan:

```python
import logging.config
import yaml

async def lifespan(app: FastAPI):
    # Load logging config
    log_config_path = project_root() / "config" / "logging_config.yaml"
    if log_config_path.exists():
        with open(log_config_path) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    
    settings = get_settings()
    await init_database(settings)
    yield
    ...
```

### 3.4 Frontend Testing Setup

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

`vitest.config.ts`:
```typescript
import { defineConfig } from "vitest/config";
export default defineConfig({
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
```

### 3.5 Docker Health Checks

```dockerfile
# backend/Dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## 📋 Ưu Tiên Thực Hiện

| # | Task | Effort | Impact | Priority |
|---|---|---|---|---|
| ~~1~~ | ~~Fix Graph Retriever N+1~~ | ~~30 phút~~ | ~~Performance~~ | ✅ DONE |
| ~~2~~ | ~~Tích hợp ContradictionDetector vào Compare~~ | ~~1 giờ~~ | ~~Feature complete~~ | ✅ DONE |
| ~~3~~ | ~~Smart Refusal (partial confidence)~~ | ~~2 giờ~~ | ~~UX~~ | ✅ DONE |
| ~~4~~ | ~~Upload Progress Tracking~~ | ~~3 giờ~~ | ~~UX~~ | ✅ DONE |
| 5 | MongoDB Indexes | 1 giờ | 🔥🔥 Performance | P1 |
| 6 | Error Boundary | 30 phút | 🔥 Resilience | P1 |
| ~~7~~ | ~~Nâng cấp Study Guide prompt~~ | ~~2 giờ~~ | ~~Feature quality~~ | ✅ DONE |
| 8 | Rate Limiting | 1 giờ | 🔥 Security | P2 |
| 9 | Streaming LLM | 4 giờ | 🔥🔥🔥 UX | P2 |
| 10 | JWT Authentication | 1 ngày | 🔥🔥🔥 Security | P2 |
| 11 | Caching Layer | 3 giờ | 🔥🔥 Performance | P2 |
| 12 | Logging Config | 30 phút | 🔥 Ops | P3 |
| 13 | Compare responsive layout | 2 giờ | 🔥 UX | P3 |
| 14 | Docker Health Checks | 15 phút | 🔥 Ops | P3 |
| 15 | Frontend Testing Setup | 2 giờ | 🔥 Quality | P3 |

---

> [!TIP]
> **Bắt đầu từ đâu?** Items #1, #2, #3 có thể hoàn thành trong **1 buổi chiều** và sẽ cải thiện đáng kể cả performance lẫn UX. Đây là "quick wins" lớn nhất.

---

## 🧑‍💻 APPENDIX: Feedback Từ Góc Nhìn Người Dùng

> Sau khi review toàn bộ UI code từ góc nhìn end-user, dưới đây là **30+ vấn đề UX cụ thể** được phát hiện thêm, bổ sung cho bảng ưu tiên ở trên.

### A. Quick Wins — Sửa Nhanh Trong 1-2 Giờ

| # | Vấn đề | File | Fix | Status |
|---|---|---|---|---|
| QW1 | Welcome message EN nhưng suggestions VI | `ChatPanel.tsx` | `getIntroMessage(language)` + localized suggestions | ✅ DONE |
| QW2 | Refused message hiện raw `refusal_reason` | `ChatPanel.tsx` | `friendlyRefusal()` map thành message thân thiện | ✅ DONE |
| QW3 | Citation footer hiện khi `was_refused=true` | `ChatPanel.tsx` | `&& !message.response.was_refused` guard | ✅ DONE |
| QW4 | Confidence score không hiện | `ChatPanel.tsx` | `ConfidenceBadge` component hiện % | ✅ DONE |
| QW5 | Mindmap root topic default generic | `GraphTab.tsx` | Cần kiểm tra | ⏳ |
| QW6 | Graph không auto-load | `GraphTab.tsx` | Cần kiểm tra | ⏳ |
| QW7 | Favicon + title Vite default | `index.html` | Cần kiểm tra | ⏳ |
| QW8 | Settings language chỉ có vi/en | `SettingsModal.tsx` | Hỗ trợ vi/en/zh/ja/ko | ⏳ |

### B. Vấn Đề UX Quan Trọng Mới Phát Hiện

#### B1. 🔴 Mobile: Studio Panel Hoàn Toàn Ẩn

**File**: [WorkspacePage.tsx](file:///d:/GenAI/DoAn01/frontend/src/pages/WorkspacePage.tsx#L49)

```tsx
// Line 49: CSS `hidden lg:flex` = KHÔNG HIỆN dưới 1024px
<div className="hidden w-[420px] shrink-0 border-l border-outline bg-white lg:flex lg:flex-col">
```

**Impact**: Trên tablet/mobile, user KHÔNG THỂ truy cập: Evidence, Studio, Compare, Graph, Mindmap → mất ~50% features.

**Fix**: Thêm bottom tab bar hoặc slide-over panel cho mobile:
```tsx
{/* Mobile Studio Trigger */}
<div className="lg:hidden fixed bottom-0 left-0 right-0 z-30 flex border-t bg-white">
  {TABS.map(tab => <button onClick={() => openMobileStudio(tab)} .../>)}
</div>
```

#### B2. 🟡 Chat History Mất Khi Reload

**File**: [ChatPanel.tsx](file:///d:/GenAI/DoAn01/frontend/src/components/workspace/ChatPanel.tsx#L161)

Messages chỉ trong `useState`, không persist. User F5 → mất toàn bộ conversation.

**Fix**: Persist vào localStorage:
```tsx
const CHAT_KEY = "prism.chat.v1";
const [messages, setMessages] = useState<ChatMessage[]>(
  () => readStorage(CHAT_KEY, [INTRO_MESSAGE])
);
useEffect(() => {
  localStorage.setItem(CHAT_KEY, JSON.stringify(messages));
}, [messages]);
```

#### B3. 🟡 Upload Batch Break On Error

**File**: [SourcesPanel.tsx](file:///d:/GenAI/DoAn01/frontend/src/components/workspace/SourcesPanel.tsx#L136-L138)

```tsx
} catch (err) {
  setError(`${file.name}: ...`);
  break;  // ← Stop uploading remaining files!
}
```

Upload 5 files, file #2 lỗi → file #3, #4, #5 bị bỏ qua.

**Fix**: `continue` thay vì `break`, và collect errors:
```tsx
const errors: string[] = [];
for (const file of files) {
  try { ... } catch (err) {
    errors.push(`${file.name}: ${err.message}`);
    continue;  // Don't stop
  }
}
if (errors.length) setError(errors.join("\n"));
```

#### B4. 🟡 Study Guide Key Concepts — Regex Thất Bại Với Tiếng Việt

**File**: [study_guide_service.py](file:///d:/GenAI/DoAn01/backend/src/services/study_guide_service.py#L96-L102)

```python
# Chỉ match capitalized English words → Vietnamese text sẽ match 0
candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}...", text)
# Fallback: ["Core concepts", "Definitions", "Examples"] → vô nghĩa
```

**Fix**: Dùng LLM extract thay vì regex:
```python
extract_prompt = (
    f"Từ đoạn văn sau, liệt kê 5-8 khái niệm quan trọng nhất (mỗi khái niệm 1-3 từ).\n"
    f"Trả lời dạng list, mỗi dòng 1 khái niệm.\n\nVăn bản:\n{text[:2000]}"
)
raw = await self.llm.generate(prompt=extract_prompt)
concepts = [line.strip("- •").strip() for line in raw.strip().split("\n") if line.strip()]
```

#### B5. 🟡 Suggestion Chips Không Route Đúng Feature

**File**: [ChatPanel.tsx](file:///d:/GenAI/DoAn01/frontend/src/components/workspace/ChatPanel.tsx#L152-L157)

```tsx
const suggestions = [
  "Tóm tắt các ý chính của collection này",  // → Nên gọi summarizeCollection()
  "So sánh cách các tài liệu giải thích chủ đề này",  // → Nên mở CompareTab
  "Tạo study guide",  // → Nên gọi buildStudyGuide()
  "Vẽ mindmap cho chủ đề..."  // → Nên mở MindmapTab
];
```

Hiện tại: Tất cả chỉ fill text vào chat input → gọi `askQuestion()` → kết quả là chat text thường, KHÔNG phải structured artifacts.

**Fix**: Mỗi suggestion có `action` handler riêng:
```tsx
const suggestions = [
  { label: "Summarize", action: () => onTabChange("studio") },
  { label: "Compare", action: () => onTabChange("compare") },
  { label: "Study Guide", action: () => onTabChange("studio") },
  { label: "Mindmap", action: () => onTabChange("mindmap") },
];
```

---

### C. Bảng Cập Nhật Ưu Tiên (Bổ Sung)

| # | Task (Mới) | Effort | Impact | Priority |
|---|---|---|---|---|
| ~~16~~ | ~~Quick Wins QW1-QW4 (chat UX)~~ | ~~2 giờ~~ | ~~UX Polish~~ | ✅ DONE |
| 17 | Mobile Studio access (B1) | 4 giờ | 🔥🔥🔥 Half features | P1 — NEXT |
| ~~18~~ | ~~Chat history persist (B2)~~ | ~~30 phút~~ | ~~UX~~ | ✅ DONE |
| ~~19~~ | ~~Upload batch continue on error (B3)~~ | ~~15 phút~~ | ~~Reliability~~ | ✅ DONE |
| ~~20~~ | ~~LLM-based key concepts (B4)~~ | ~~1 giờ~~ | ~~Feature quality~~ | ✅ DONE |
| ~~21~~ | ~~Suggestion chips routing (B5)~~ | ~~1 giờ~~ | ~~Feature coherence~~ | ✅ DONE |
| ~~22~~ | ~~Auto-poll material status~~ | ~~1 giờ~~ | ~~🔥🔥 UX~~ | ✅ DONE |
| ~~23~~ | ~~Upload progress bar~~ | ~~2 giờ~~ | ~~UX~~ | ✅ DONE |
| 24 | Skeleton loading states | 2 giờ | 🔥 UX Polish | P3 |
| 25 | Dark mode toggle | 4 giờ | 🔥 Modern feel | P3 |
