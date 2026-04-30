# AgentBook — Kiểm Tra Hệ Thống Evidence (Evidence Trace Audit)

Kiểm tra luồng evidence xuyên suốt từ parse → chunk → index → retrieve → display.

---

## Luồng Evidence trong hệ thống

```
DoclingParser → ParsedBlock (block_id, page, bbox, content)
    ↓
LayoutNormalizer → MaterialBlock (lưu vào Material.pages trong MongoDB)
    ↓
EvidenceMapper → EvidenceBlock (gắn owner_id, collection_id, material_id, document_name)
    ↓
LayoutAwareChunker → TextChunk (giữ evidence: list[EvidenceBlock])
    ↓
QdrantMongoIndexer → Chunk (MongoDB) + Qdrant point (payload có source_block_ids)
    ↓
HybridRetriever → RetrievedChunk (hydrate evidence từ Material.pages)
    ↓
ResponseParser → CitationSchema (trả cho frontend)
    ↓
EvidencePage API → Load lại blocks từ Material.pages
```

---

## 🔴 Vấn đề nghiêm trọng

### 1. Evidence bị MẤT khi hydrate — Retriever tra cứu block_id từ Material nhưng Chunk không lưu evidence

**Luồng lỗi:**

Khi index, `TextChunk.evidence = list[EvidenceBlock]` chứa đầy đủ evidence blocks. Nhưng khi lưu vào MongoDB `Chunk` model:

```python
# models/chunk.py
class Chunk(Document):
    source_block_ids: list[str]   # ← chỉ lưu danh sách block_id
    source_pages: list[int]       # ← chỉ lưu danh sách page
    # KHÔNG CÓ evidence: list[EvidenceBlock]
```

Khi retrieve, `HybridRetriever._chunk_evidence()` phải **tra ngược** từ `Material.pages`:
```python
block_lookup = {
    block.block_id: (page.page_number, block)
    for page in material.pages
    for block in page.blocks
    if block.block_id in chunk.source_block_ids  # ← tìm lại block bằng ID
}
```

**Vấn đề:** Nếu `source_block_ids` có block_id không tồn tại trong `material.pages` (do re-parse, update, hoặc lỗi), evidence block đó bị **silent drop** — không có warning/log.

**Mức độ:** 🟡 Trung bình — hoạt động đúng nếu material không bị re-parse. Nhưng nếu re-index mà material.pages thay đổi → evidence rỗng.

**Cách sửa:**
```python
# Thêm warning khi block_id không tìm thấy
for block_id in chunk.source_block_ids:
    if block_id not in block_lookup:
        logger.warning("Evidence block missing from material", 
            extra={"block_id": block_id, "material_id": str(material.id)})
```

---

### 2. Evidence Page API — Lộ `source_path` (đường dẫn file server) ra frontend

**File:** `evidence.py` endpoint, line 37

```python
result = EvidencePageResponse(
    ...
    source_path=material.storage_path,   # ← trả raw path cho frontend
)
```

`storage_path` dạng `data/raw/user_demo/6745a2b3.../abc123.pdf` — đây là đường dẫn nội bộ của server.

**Frontend hiện thị nó** ở `EvidencePage.tsx` line 79:
```tsx
{result ? <p className="mt-4 text-xs text-muted">{result.source_path}</p> : null}
```

**Mức độ:** 🟡 — Không phải lỗ hổng bảo mật nghiêm trọng (chỉ path tương đối), nhưng không nên expose.

---

### 3. CitationSchema — `confidence` field bị clamp `[0.0, 1.0]` nhưng source data có thể ngoài khoảng

**File:** `evidence.py` line 24

```python
class CitationSchema(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
```

Nhưng `EvidenceBlock.confidence` lấy từ `block.ocr_confidence` — PaddleOCR trả [0,1] nên OK. Vấn đề là ở `response_parser.py` line 41:

```python
confidence=evidence.confidence if evidence.confidence is not None else 1.0
```

Khi OCR confidence = None (từ Docling parser, không phải OCR), **mặc định = 1.0** → gây nhầm lẫn rằng evidence "chắc chắn 100%". Nên dùng giá trị trung tính hơn hoặc field riêng.

---

## 🟡 Vấn đề chất lượng trung bình

### 4. `snippet_translated` luôn = `null` — cross-lingual evidence không có bản dịch

**Luồng:**
- `CitationSchema` có field `snippet_translated: str | None = None`
- `ResponseParser.citations_from_chunks()` **không bao giờ** set `snippet_translated`
- Frontend `EvidencePanel.tsx` kiểm tra `citation.snippet_translated` nhưng luôn null → section "Translated" không bao giờ hiện

**Vấn đề:** Khi user hỏi tiếng Việt mà tài liệu tiếng Anh, citation snippet là tiếng Anh → user phải tự đọc. Không có hỗ trợ dịch snippet.

**Cách sửa:** Thêm translation step vào `response_parser.py` hoặc dùng LLM dịch snippet khi `answer_language != source_language`.

---

### 5. EvidenceBlockSchema thiếu `material_id` và `document_name`

**File:** `evidence.py`

```python
class EvidenceBlockSchema(BaseModel):
    block_id: str
    block_type: str
    page: int
    snippet_original: str
    source_language: str
    bbox: BoundingBoxSchema | None = None
    confidence: float | None = None
    # ← KHÔNG CÓ material_id, document_name
```

Frontend `EvidencePage.tsx` khi click block, phải **tự bổ sung** `doc_id` và `doc_name` từ response:
```tsx
function selectBlock(block: EvidenceBlock) {
    setSelectedCitation({
      doc_id: result?.doc_id ?? docId,    // ← phải tự fill
      doc_name: result?.doc_name ?? "...", // ← phải tự fill
      ...
    });
}
```

**Không phải bug**, nhưng nếu API trả đủ metadata trên mỗi block thì frontend đỡ phải mapping thủ công.

---

### 6. Collections API — N+1 query problem

**File:** `collections.py` line 14-46

```python
for collection in collections:
    materials = await Material.find(...).to_list()   # query 1 per collection
    chunks = await Chunk.find(...).to_list()          # query 1 per collection
```

Nếu có 10 collections → 20 MongoDB queries. Nên dùng aggregate hoặc batch query.

---

### 7. Graph EvidenceRef lưu thiếu — chỉ có `material_id`, `page`, `block_id`

**File:** `knowledge_graph.py`

```python
class EvidenceRef(BaseModel):
    material_id: PydanticObjectId
    page: int | None = None
    block_id: str | None = None
    span: list[int] | None = None
```

**So với claude.md yêu cầu giữ:**
- ✅ `material_id` (doc_id)
- ✅ `page`  
- ✅ `block_id`
- ❌ `owner_id` — thiếu (phải tra ngược từ Entity/Relation.owner_id)
- ❌ `collection_id` — thiếu (phải tra ngược)
- ❌ `document_name` — thiếu
- ❌ `snippet_original` — thiếu
- ❌ `source_language` — thiếu
- ❌ `bbox` — thiếu
- ❌ `confidence` — thiếu

**Hệ quả:** Khi `GraphRetriever._hydrate_evidence_refs()` cần evidence đầy đủ, phải **load lại Material từ MongoDB** rồi scan từng page.blocks → rất chậm và dễ miss nếu block bị thay đổi.

---

### 8. Contradiction Detector — Không được gọi ở đâu cả

**File:** `contradiction_detector.py` — Được import nhưng **không bao giờ gọi** trong:
- `query_service.py` — không import
- `inference_engine.py` — không import
- `parse_index_pipeline.py` — không import

→ Dead code. Tính năng detect contradictions giữa documents **chưa được tích hợp**.

---

## ✅ Những phần Evidence hoạt động tốt

### Evidence Trace xuyên pipeline
- `ParsedBlock` → `EvidenceBlock` → `TextChunk.evidence` → `Chunk.source_block_ids` → Retriever hydrate lại → `CitationSchema` → Frontend
- Mỗi bước đều giữ được `block_id`, `page`, `bbox`

### Scope isolation
- Evidence page API check `owner_id` match ✅
- Evidence page API check `collection_id` match (nếu có) ✅
- Chunk scoped by `owner_id` + `collection_id` ✅

### QueryLog lưu đầy đủ
- Mỗi query được log: citations, confidence, was_refused, latency_ms, retrieval_trace
- Có index trên `(owner_id, collection_id, created_at)` → query nhanh

### EvidencePanel frontend
- Hiển thị snippet_original đúng
- Hiển thị bbox coordinates
- Link tới Evidence page với doc + page params

---

## Bảng tổng hợp

| # | Vấn đề | Mức độ | Ảnh hưởng |
|---|--------|--------|-----------|
| 1 | Evidence silent drop khi block_id missing | 🟡 Medium | Evidence trống không ai biết |
| 2 | source_path lộ ra frontend | 🟡 Medium | Thông tin server path visible |
| 3 | Confidence mặc định 1.0 khi null | 🟡 Medium | Citation confidence misleading |
| 4 | snippet_translated luôn null | 🟡 Medium | Cross-lingual evidence UX kém |
| 5 | EvidenceBlockSchema thiếu metadata | 🟢 Low | Frontend phải self-fill |
| 6 | Collections N+1 query | 🟡 Medium | API chậm khi nhiều collections |
| 7 | EvidenceRef graph quá ít field | 🟡 Medium | Graph hydrate chậm và fragile |
| 8 | ContradictionDetector dead code | 🟢 Low | Feature chưa tích hợp |
