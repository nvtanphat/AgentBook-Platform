# AgentBook — Kiểm Tra Hệ Thống Backend (Indexer, Chunking, RAG Pipeline)

Đánh giá sau khi đọc kỹ toàn bộ ~25 file backend.

---

## Tổng quan Pipeline

```
Upload → DoclingParser/OCR → LayoutNormalizer → EvidenceMapper
       → LayoutAwareChunker → BGEM3Embedder → QdrantMongoIndexer
       → EntityExtractor → EntityResolver → EventExtractor
       → Graph (Entity/Relation/Event) stored in MongoDB
```

---

## 🔴 Vấn đề nghiêm trọng

### 1. Chunking: Token counter quá thô — dùng `len(text.split())` thay vì tokenizer thật

**File:** `chunking.py` line 142-143

```python
@staticmethod
def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))
```

**Vấn đề:**
- BGE-M3 dùng tokenizer riêng (SentencePiece), 1 word tiếng Việt có thể = 3-5 tokens thật
- Với `chunk_target_token_count = 512`, thực tế bạn có thể tạo chunk lên tới **1500-2000 BPE tokens** → vượt quá `embedding_max_length = 1024` tokens của BGE-M3
- Khi chunk bị truncate bởi embedder, phần cuối chunk bị mất → **suy giảm chất lượng embedding nghiêm trọng**
- Tiếng Việt có dấu (ă, ê, ơ, ư...) bị tokenize tốn hơn tiếng Anh rất nhiều

**Cách sửa:**
```python
# Option 1: Dùng tokenizer thật của BGE-M3
from transformers import AutoTokenizer
_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))

# Option 2: Heuristic tốt hơn (nếu không muốn load tokenizer)
def _count_tokens(text: str) -> int:
    # Ước tính 1 word ≈ 1.5 BPE tokens cho VI mixed EN
    return max(1, int(len(text.split()) * 1.5))
```

---

### 2. Chunking: `_split_oversized_block` chia theo word count nhưng so sánh với token target

**File:** `chunking.py` line 117-139

```python
def _split_oversized_block(self, block: EvidenceBlock) -> list[EvidenceBlock]:
    target_tokens = max(1, self.settings.chunk_target_token_count)
    words = block.snippet_original.split()
    if len(words) <= target_tokens:  # ← so sánh word count với token count!
        return [block]
    # ...chia theo target_tokens words mỗi phần
```

**Vấn đề:** 
- `target_tokens = 512` nhưng đang so sánh `len(words) <= 512` → chỉ split khi block > 512 **words**
- Nhưng 512 words ≈ 750-1000 BPE tokens → lại vượt quá embedding max length
- Thực chất hàm này hoạt động gần đúng chỉ khi tỷ lệ word:token ≈ 1:1 (English text)

---

### 3. Indexer: Insert MongoDB từng document một — rất chậm khi batch lớn

**File:** `indexer.py` line 87-110

```python
async def _store_chunks(self, chunks: list[TextChunk]) -> list[Chunk]:
    stored_chunks: list[Chunk] = []
    for chunk in chunks:
        document = Chunk(...)
        await document.insert()  # ← 1 query / chunk
        stored_chunks.append(document)
    return stored_chunks
```

**Vấn đề:**
- Mỗi chunk = 1 MongoDB insert → 100 chunks = 100 round-trip
- Beanie hỗ trợ `Chunk.insert_many()` hoặc batch insert

**Cách sửa:**
```python
async def _store_chunks(self, chunks: list[TextChunk]) -> list[Chunk]:
    documents = [Chunk(...) for chunk in chunks]
    await Chunk.insert_many(documents)
    return documents
```

---

### 4. Indexer: Graph entities cũng insert từng cái một — tương tự

**File:** `indexer.py` line 154-203 — `_store_graph()` dùng loop + `await entity.insert()` cho mỗi entity/event/relation.

---

## 🟡 Vấn đề chất lượng trung bình

### 5. Entity Extractor: Hoàn toàn dựa vào regex — không dùng NER/LLM

**File:** `entity_extractor.py`

**Hiện trạng:**
- Chỉ nhận diện entity bằng 3 cách:
  1. Hardcoded `METHOD_KEYWORDS` (dropout, transformer, attention...) — 9 keywords cố định
  2. Regex `METRIC_PATTERN` (accuracy, f1, loss...) — chỉ cho domain ML
  3. `CAPITALIZED_TERM_PATTERN` — bắt mọi từ viết hoa → **rất nhiều noise** (The, This, That chỉ lọc 3 stopwords)

**Vấn đề:**
- Entity extraction chất lượng rất thấp cho tài liệu ngoài ML domain
- Capitalized pattern sẽ bắt hết tên riêng, viết tắt, header text → graph rác
- Không có NER model (spaCy, underthesea cho tiếng Việt)
- Confidence hardcoded: method=0.72, metric=0.68, concept=0.55 → không phản ánh thực tế

**Cách sửa ngắn hạn:**
- Thêm nhiều stopwords hơn cho capitalized filter
- Tăng min length từ 3 → 4 ký tự
- Lọc bỏ các common English words (I, A, In, The, For, With, etc.)

---

### 6. Entity Resolution: Alias mapping cứng, chỉ có 4 entries

**File:** `entity_resolution.py` line 30-31

```python
aliases = {"dropout regularization": "dropout", "dropout layer": "dropout", "l 1": "l1", "l 2": "l2"}
```

- Chỉ merge được 4 cặp alias cụ thể → không scale
- Cần thêm fuzzy matching hoặc embedding-based similarity

---

### 7. Event Extractor: Confidence cố định, logic đơn giản

**File:** `event_extractor.py`

- Chỉ detect event bằng `EVENT_VERBS` regex (reported, proposed, reduced, đề xuất, giảm...)
- Confidence luôn = 0.58 hoặc 0.5 → không có ý nghĩa phân biệt
- Event name = câu đầu tiên, max 180 ký tự → có thể lấy cả câu không liên quan

---

### 8. Query Processor: "Dịch" VI→EN bằng từ điển 12 entries

**File:** `query_processor.py` line 19-33

```python
TRANSLATIONS = {
    "như thế nào": "how",
    "là gì": "what is",
    "so sánh": "compare",
    # ... 12 entries total
}
```

**Vấn đề:**
- Cross-lingual query chỉ replace từ vựng cơ bản → câu dịch thường vô nghĩa
- Fallback `f"English evidence for: {query}"` → chuỗi query kỳ lạ cho embedding search
- Cần dùng translation model thật (Helsinki-NLP/opus-mt-vi-en hoặc LLM translate)

---

### 9. Claim Verifier: Chỉ so sánh số và term overlap — không dùng NLI/LLM

**File:** `claim_verifier.py`

- "CONTRADICTED" chỉ khi **số liệu** trong claim khác với evidence → bỏ qua contradictions về logic/ý nghĩa
- "SUPPORTED" chỉ cần ≥1-3 important terms trùng → rất dễ false positive
- Không dùng NLI model (cross-encoder nli) hay LLM-based verification

---

### 10. Confidence Scorer: Normalize rerank score bằng `(score + 1) / 2`

**File:** `confidence_scorer.py` line 16

```python
normalized = [max(0.0, min(1.0, (score + 1.0) / 2.0)) for score in rerank_scores]
```

- Giả sử rerank score nằm trong [-1, 1] → nhưng `bge-reranker-v2-m3` trả về logits không bounded
- Score = 5.0 → normalized = 3.0 → bị clamp thành 1.0 → mất phân biệt giữa "rất tốt" và "cực tốt"
- Nên dùng sigmoid: `1 / (1 + exp(-score))`

---

## 🟢 Những phần hoạt động tốt

### ✅ Chunking Logic cơ bản

- `LayoutAwareChunker` chia chunk theo heading boundaries → giữ được ngữ cảnh section
- Overlap mechanism lấy blocks cuối của chunk trước → giữ context liên tục
- Heading-only blocks không bị chia thành chunk riêng → tốt

### ✅ Hybrid Retrieval Pipeline

- Dense + Sparse search trên Qdrant → RRF fusion → rerank → top-k
- RRF formula chuẩn: `1 / (k + rank)` với k=60
- Scope filter đúng: `owner_id` + `collection_id` + `material_ids`
- Dedupe chunks trước khi rerank

### ✅ Indexer Qdrant

- Dual vector index: dense (COSINE) + sparse (IDF modifier) → đúng cho BGE-M3
- Payload metadata đầy đủ: owner_id, collection_id, page_numbers, block_types
- Deterministic point ID dùng uuid5 → upsert an toàn

### ✅ Document Parser Pipeline

- DoclingParser xử lý PDF/DOCX/PPTX + có PyPDF fallback khi Docling fail
- OCR Pipeline: PaddleOCR only; OCR failures are surfaced instead of using a fallback
- LayoutNormalizer: reading order by bbox position → đúng cho multi-column documents
- Evidence trace được giữ xuyên suốt: block_id, page, bbox, confidence

### ✅ Reranker graceful degradation

- Nếu CrossEncoder fail → fallback về fused_score → không crash

### ✅ Graph Retriever

- 1-hop và 2-hop path retrieval → tìm được quan hệ gián tiếp
- Evidence hydration từ MongoDB material pages → đúng

---

## Bảng tổng hợp ưu tiên sửa

| # | Vấn đề | Mức độ | Ảnh hưởng |
|---|--------|--------|-----------|
| 1 | Token counter dùng word split | 🔴 Critical | Chunk vượt max embedding length → embedding bị truncate → retrieval quality sụt |
| 2 | `_split_oversized_block` lẫn word/token | 🔴 Critical | Block quá lớn không được split đúng |
| 3 | MongoDB insert từng document | 🟡 Medium | Chậm khi index tài liệu lớn (50-100 chunks) |
| 4 | Entity Extractor regex-only | 🟡 Medium | Knowledge Graph chất lượng thấp, nhiều noise |
| 5 | Query Processor dịch VI→EN bằng 12 từ | 🟡 Medium | Cross-lingual retrieval kém hiệu quả |
| 6 | Confidence normalize không dùng sigmoid | 🟡 Medium | Score confidence không chính xác |
| 7 | Claim Verifier chỉ so number/term | 🟢 Low (MVP OK) | Claim verification sơ sài nhưng đủ cho demo |
| 8 | Entity Resolution 4 aliases | 🟢 Low | Graph entity merge kém |
| 9 | Event Extractor confidence cố định | 🟢 Low | Events/Relations quality thấp |

---

## Khuyến nghị ưu tiên cao nhất

> **Sửa ngay item #1 và #2:** Token counting sai ảnh hưởng trực tiếp đến chất lượng embedding → retrieval → answer quality. Đây là **lỗi ngầm** mà bạn có thể không nhận ra khi test thủ công (vì hệ thống vẫn trả lời, chỉ là chất lượng kém hơn đáng kể so với tiềm năng).
