# 📊 AgentBook Benchmark — Debug Testing Ladder

> [!IMPORTANT]
> Plan này thiết kế theo kiểu **ladder** — test từng tầng từ thấp lên cao. **Mỗi tầng chạy độc lập**, debug xong mới lên tầng tiếp. Nếu tầng dưới fail thì KHÔNG chạy tầng trên.

---

## Cập nhật trọng tâm — GPT-4o Meta-Dataset & Benchmark SOTA

Plan này không chỉ chạy debug pipeline. Mục tiêu mới là xây một **benchmark dataset chuẩn, thực tế, bao phủ toàn bộ chức năng AgentBook**, trong đó GPT-4o được dùng như **meta-dataset generator + judge có kiểm soát**, còn gold labels vẫn phải neo vào evidence thật từ tài liệu đã parse.

### Nguyên tắc bắt buộc

| Nguyên tắc | Yêu cầu |
|---|---|
| Evidence-first | Mọi câu hỏi gold phải trỏ về `document_name`, `page`, `block_id`, `chunk_id` hoặc `audio_start/end`; không nhận câu hỏi không có evidence anchor. |
| Tiếng Việt là chính | 70-80% query tiếng Việt, 10-15% cross-lingual VN hỏi EN, 5-10% EN hỏi tài liệu VI, phần còn lại adversarial/off-topic. |
| GPT-4o không tự quyết một mình | GPT-4o tạo câu hỏi/rubric/expected answer, nhưng validator phải kiểm tra citation anchor tồn tại, answer không vượt evidence, và sampling thủ công 10-20%. |
| Benchmark không trộn train/test | Chia `smoke`, `dev`, `test_private`; chỉ dùng `smoke/dev` để debug, giữ `test_private` làm bộ chấm cuối. |
| Bao phủ endpoint | Benchmark phải có case cho ask, ask-stream, summarize, study-guide, compare, ask-graph, mindmap, OCR, handwriting, image quality refusal, spreadsheet, audio, off-topic/refusal. |
| Có negative cases | Mỗi nhóm chức năng phải có câu dễ fail: false premise, missing evidence, ambiguous reference, noisy OCR, wrong sheet/table, cross-doc conflict. |

### Dataset artifacts cần tạo

| File | Mục đích | Số lượng mục tiêu |
|---|---|---:|
| `evaluation/datasets/agentbook_meta_dataset.jsonl` | Inventory mọi tài liệu, block, chunk, sheet, slide, audio segment, modality, language, chất lượng parse | 1 record/block hoặc 1 record/chunk |
| `evaluation/datasets/agentbook_retrieval_gold.jsonl` | Query → expected docs/pages/blocks/chunks để tính Recall@K, MRR, nDCG | 150-250 queries |
| `evaluation/datasets/agentbook_e2e_gold.jsonl` | Query → expected answer outline, required facts, forbidden claims, citation anchors | 120-180 queries |
| `evaluation/datasets/agentbook_endpoint_cases.jsonl` | Test theo endpoint và workflow, gồm request payload + expected behavior | 80-120 cases |
| `evaluation/datasets/agentbook_adversarial.jsonl` | Off-topic, false premise, prompt injection, no-evidence, low-quality OCR | 60-100 cases |
| `evaluation/datasets/agentbook_judge_rubric.yaml` | Rubric GPT-4o judge: groundedness, answer relevance, citation correctness, refusal correctness, language quality | 1 rubric |

### GPT-4o generation pipeline

```
Parsed materials
  -> block/chunk inventory
  -> GPT-4o sinh candidate questions + expected facts
  -> rule validator kiểm tra anchors tồn tại
  -> GPT-4o judge kiểm tra answer có nằm trong evidence không
  -> human spot-check 10-20%
  -> freeze smoke/dev/test_private
```

### Prompt contract cho GPT-4o generator

GPT-4o chỉ được sinh JSON theo schema. Không cho sinh markdown tự do.

```json
{
  "case_id": "ab-e2e-0001",
  "task_type": "factual|compare|summarize|study_guide|table|graph|ocr|audio|refusal|cross_lingual|false_premise",
  "query_language": "vi",
  "answer_language": "vi",
  "query": "Câu hỏi người dùng sẽ hỏi",
  "expected_answer_outline": ["ý chính 1", "ý chính 2"],
  "required_facts": ["fact phải xuất hiện"],
  "forbidden_claims": ["claim không được nói nếu không có evidence"],
  "expected_evidence": [
    {
      "document_name": "lecture_notes.pdf",
      "page": 12,
      "block_id": "blk-...",
      "chunk_id": "chunk-...",
      "quote_or_fact": "fact ngắn trích/diễn giải từ evidence"
    }
  ],
  "expected_behavior": "answer|refuse|ask_clarification",
  "difficulty": "easy|medium|hard|adversarial",
  "tags": ["vietnamese", "citation", "pdf"]
}
```

### Coverage matrix chuẩn SOTA cho AgentBook

| Nhóm năng lực | Case cần có | Metric chính |
|---|---|---|
| Parser/layout | PDF VI text, PDF EN, PDF scan, DOCX table, PPTX slide, CSV/XLSX, PNG, handwriting, WAV | parse success, block count, table/figure/audio metadata |
| Retrieval | factual, table lookup, slide lookup, OCR lookup, audio lookup, cross-lingual, multi-hop | Recall@1/3/5, MRR, nDCG@5 |
| Generation | ask, ask-stream, summarize, study-guide | answer relevance, citation coverage, citation validity, language consistency |
| Compare | PDF vs DOCX, two docs same topic, conflicting/overlapping claims | dimension coverage, source balance, contradiction handling |
| GraphRAG | entity relation, dependency/impact query, ask-graph | entity recall, path relevance, graph evidence precision |
| Guardrails | off-topic, false premise, prompt injection, low evidence, low-quality image | refusal precision/recall, false refusal rate |
| Visual/OCR | clear scan, low-quality refusal, handwriting pass/fail, PPTX figures | OCR confidence, quality gate correctness, figure retrieval accuracy |
| Audio | Vietnamese speech WAV, timestamped answer | transcript coverage, timestamp citation correctness |

### Acceptance thresholds

| Metric | Smoke | Dev | Test private |
|---|---:|---:|---:|
| Parser pass rate | ≥95% | ≥98% | ≥98% |
| Retrieval Recall@5 | ≥75% | ≥85% | ≥88% |
| Retrieval MRR | ≥0.55 | ≥0.65 | ≥0.70 |
| Citation validity | ≥95% | ≥98% | ≥98% |
| Faithfulness / groundedness | ≥0.80 | ≥0.88 | ≥0.90 |
| Answer language correctness | ≥95% | ≥98% | ≥98% |
| Refusal precision | ≥85% | ≥92% | ≥95% |
| False refusal rate | ≤10% | ≤6% | ≤4% |
| P95 latency local | ≤120s | ≤90s | ≤60s |
| P95 latency OpenAI provider | ≤15s | ≤8s | ≤5s |

### Scripts cần nâng cấp

| Task | File | Việc cần làm |
|---|---|---|
| GPT-4o meta generator | `backend/scripts/generate_eval_dataset.py` | Thêm `--provider openai --model gpt-4o`, output schema mới, sinh retrieval/e2e/adversarial riêng. |
| Dataset validator | `[NEW] backend/scripts/validate_benchmark_dataset.py` | Kiểm tra JSON schema, anchors tồn tại trong Mongo/Qdrant, không duplicate query, split không rò rỉ. |
| GPT-4o judge | `[NEW] backend/scripts/judge_eval_with_gpt4o.py` | Chấm groundedness/citation/refusal theo rubric, lưu score + rationale ngắn. |
| Retrieval scorer | `backend/scripts/score_retrieval_eval.py` | Thêm Recall@K, MRR, nDCG@K, per-modality breakdown. |
| E2E scorer | `backend/scripts/e2e_eval.py` | Đọc `agentbook_e2e_gold.jsonl`, hỗ trợ endpoint cases, xuất Markdown report. |
| Full runner | `[NEW] backend/scripts/run_full_benchmark.py` | Chạy parser -> retrieval -> e2e -> judge -> report theo ladder. |

---

## Tổng quan — 7 Tầng Test

```
Tầng 6 ── Full Benchmark Suite ────────── cần: tầng 0-5 pass
Tầng 5 ── E2E Single Query ───────────── cần: API + indexed data
Tầng 4 ── Retrieval Only ─────────────── cần: API hoặc direct module
Tầng 3 ── Embedding + Qdrant ──────────── cần: Qdrant running
Tầng 2 ── Chunk Quality ──────────────── không cần API
Tầng 1 ── Parser Unit Test ───────────── không cần API
Tầng 0 ── Health Check ───────────────── kiểm tra infra
```

| Tầng | Mục đích | Cần gì? | Thời gian | Script |
|---|---|---|---|---|
| 0 | Infra sống? | Server, Qdrant, Mongo, Ollama | 10s | `curl` commands |
| 1 | Parse 1 file đúng? | Chỉ Python + libs | 30s-2m | `dry_run_test_data_pipeline.py` |
| 2 | Chunks chất lượng? | Chỉ Python + libs | 30s | `chunk_quality_check.py` |
| 3 | Embedding + vector? | Qdrant running | 30s | `diag_pipeline.py` |
| 4 | Retrieval trả đúng? | API hoặc direct | 1-2m | `quick_eval.py` |
| 5 | 1 câu hỏi E2E? | API + indexed material | 15-30s | `curl` / PowerShell |
| 6 | Full benchmark | Tất cả | 10-30m | `e2e_eval.py` + suite |

---

## Tầng 0 — Health Check (10 giây)

**Mục đích:** Xác nhận tất cả services sống trước khi test bất cứ thứ gì.

### Commands

```powershell
# 0a. Backend API
Invoke-RestMethod http://localhost:8000/health
# Expected: { "status": "ok", "service": "Noelys" }

# 0b. Qdrant
Invoke-RestMethod http://localhost:6333/collections
# Expected: { "result": { "collections": [...] } }

# 0c. MongoDB (thông qua backend)
Invoke-RestMethod http://localhost:8000/api/v1/admin/health
# Expected: db_connected = true

# 0d. Ollama (nếu cần LLM)
Invoke-RestMethod http://localhost:11434/api/tags
# Expected: { "models": [{ "name": "qwen3:4b", ... }] }

# 0e. Redis (Celery broker)
docker exec -it $(docker ps -q -f name=redis) redis-cli PING
# Expected: PONG
```

### Common Bugs

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Backend 500 | MongoDB URI sai | Check `.env` → `MONGODB_URI` |
| Qdrant connection refused | Docker chưa start | `docker start qdrant` |
| Ollama timeout | Model chưa pull | `ollama pull qwen3:4b` |
| Redis PONG fail | Redis container tắt | `docker start redis` |

### Pass Criteria
✅ Tất cả 4 services trả response → **lên tầng 1**

---

## Tầng 1 — Parser Unit Test (không cần API)

**Mục đích:** Test parse **từng file riêng lẻ**, xem blocks/pages/language đúng chưa.

### 1a. Test 1 file cụ thể

```powershell
cd D:\GenAI\DoAn01\backend

# PDF text VI chính
python scripts/dry_run_test_data_pipeline.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_A" `
    --max-files 1
# Expected: lecture_notes.pdf → pages=149, blocks≥50, parser=docling, language≈vi

# PDF text EN phụ trợ cho cross-lingual
python scripts/dry_run_test_data_pipeline.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_A/lecture_notes_en.pdf"
# Expected: lecture_notes_en.pdf → pages=14, blocks≥10, parser=docling, language≈en

# PDF scan VN (OCR)
python scripts/dry_run_test_data_pipeline.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_A" `
    --ocr-only --max-files 4
# Expected: report_vn.pdf → ocr_score > 0, text extracted

# DOCX
python scripts/dry_run_test_data_pipeline.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_A" `
    --max-files 4
# Check: comparison_doc.docx → block_types chứa 'table', ≥2 table blocks

# CSV
python scripts/dry_run_test_data_pipeline.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_B"
# Check: data_table.csv → parser=spreadsheet, blocks có table type
```

### 1b. Test image quality gate

```powershell
cd D:\GenAI\DoAn01\backend

# Chạy Python inline để test quality checker riêng lẻ
python -c "
import sys; sys.path.insert(0, '.')
from src.processing.image_quality_checker import ImageQualityChecker
checker = ImageQualityChecker()

# Should PASS
result = checker.check('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/scan_clear.png')
print(f'scan_clear: pass={result.is_acceptable}, blur={result.blur_variance:.1f}, brightness={result.brightness:.1f}')

# Should FAIL
result = checker.check('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/scan_low_quality.png')
print(f'scan_low_quality: pass={result.is_acceptable}, blur={result.blur_variance:.1f}, brightness={result.brightness:.1f}')

# Should PASS
result = checker.check('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/handwriting_ok.png')
print(f'handwriting_ok: pass={result.is_acceptable}')

# Should FAIL
result = checker.check('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/handwriting_blur.png')
print(f'handwriting_blur: pass={result.is_acceptable}')
"
```

### 1c. Test spreadsheet parser riêng

```powershell
python -c "
import sys; sys.path.insert(0, '.')
from src.processing.spreadsheet_parser import SpreadsheetParser
from pathlib import Path

parser = SpreadsheetParser()

# CSV
result = parser.parse(Path('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/data_table.csv'), display_name='data_table.csv')
print(f'CSV: pages={len(result.pages)}, blocks={sum(len(p.blocks) for p in result.pages)}')

# XLSX multi-sheet
result = parser.parse(Path('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/workbook_multi.xlsx'), display_name='workbook_multi.xlsx')
print(f'XLSX: pages/sheets={len(result.pages)}, blocks={sum(len(p.blocks) for p in result.pages)}')
print(f'Sheet names: {[p.page_label for p in result.pages]}')

# Nếu workbook_multi.xlsx đang bị Windows khóa hoặc chưa có formula,
# dùng bản thực tế hơn:
result = parser.parse(Path('../data/test data/AgentBook_Internet_Test_Pack/Collection_B/workbook_multi_fixed.xlsx'), display_name='workbook_multi_fixed.xlsx')
print(f'XLSX fixed: pages/sheets={len(result.pages)}, blocks={sum(len(p.blocks) for p in result.pages)}')
"
```

### Expected Output per File

| File | Parser | Pages | Blocks min | Key check |
|---|---|---|---|---|
| `lecture_notes.pdf` | docling | 149 | 50 | PDF tiếng Việt chính, text layer tốt |
| `lecture_notes_en.pdf` | docling | 14 | 10 | PDF EN phụ trợ cho cross-lingual |
| `report_vn.pdf` | docling+ocr | 1 | 1 | `ocr_score > 0` |
| `comparison_doc.docx` | docling | ≥1 | 3 | ≥2 table blocks |
| `slides_mixed.pptx` | docling | 64 | 50 | slide bài giảng thật, có text + 9 hình + 9 bảng |
| `data_table.csv` | spreadsheet | 1 | 1 | table block with header |
| `workbook_multi.xlsx` | spreadsheet | 4 | 4 | 4 sheets = 4 pages |
| `workbook_multi_fixed.xlsx` | spreadsheet | 4 | 4 | 4 sheets + formula, nên dùng cho benchmark chính |
| `scan_clear.png` | ocr | 1 | ≥1 | quality pass |
| `scan_low_quality.png` | quality_gate | — | — | quality **FAIL** |
| `handwriting_ok.png` | handwriting | 1 | ≥1 | quality pass |
| `handwriting_blur.png` | quality_gate | — | — | quality **FAIL** |
| `lecture.wav` | audio | 1 | ≥1 | giọng Việt thật 70.66s, `source: audio_whisper`, có timestamps |

### Common Bugs

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Docling ImportError | `docling` chưa install | `pip install docling` |
| OCR trả 0 text | EasyOCR/PaddleOCR chưa install | `pip install easyocr` |
| Spreadsheet 0 pages | openpyxl chưa install | `pip install openpyxl` |
| Audio ImportError | faster-whisper chưa install | `pip install faster-whisper` |
| Image quality check luôn pass | Thresholds sai | Check `guardrails_config.yaml` |

### Pass Criteria
✅ Tất cả 11 file parse đúng expected → **lên tầng 2**

---

## Tầng 2 — Chunk Quality (không cần API)

**Mục đích:** Kiểm tra chunker output — token distribution, không có chunk rỗng/quá lớn.

### Commands

```powershell
cd D:\GenAI\DoAn01\backend

# Test trên Collection A
python scripts/chunk_quality_check.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_A"

# Test trên Collection B
python scripts/chunk_quality_check.py `
    "../data/test data/AgentBook_Internet_Test_Pack/Collection_B"
```

### What to Look For

```
✅ GOOD indicators:
  - Good 150-512: > 60% (chunks đúng target range)
  - Tiny < 50t: < 10% (ít chunk quá nhỏ)
  - Large > 512: < 5% (ít chunk quá lớn)
  - No empty chunks (content length > 0)

❌ BAD indicators:
  - Tiny > 30% → chunker splitting quá aggressive
  - Large > 20% → chunker không split đủ
  - avg token < 100 → chunks quá nhỏ, mất ngữ cảnh
  - max token > 1000 → chunk quá dài, noisy retrieval
```

### Debug cụ thể 1 chunk

```powershell
python -c "
import sys, json; sys.path.insert(0, '.')
from src.core.config import get_settings
from src.processing.chunking import build_chunker
from src.processing.docling_parser import DoclingParser
from src.processing.layout_normalizer import LayoutNormalizer
from src.processing.evidence_mapper import EvidenceMapper
from pathlib import Path

settings = get_settings()
parser = DoclingParser()
normalizer = LayoutNormalizer()
mapper = EvidenceMapper()
chunker = build_chunker(settings, embedder=None)

parsed = parser.parse(Path('../data/test data/AgentBook_Internet_Test_Pack/Collection_A/lecture_notes.pdf'))
normalized = normalizer.normalize(parsed)
emap = mapper.build(parsed=normalized, owner_id='debug', collection_id='debug', material_id='debug', document_name='lecture_notes.pdf')
chunks = chunker.build_chunks(emap)

print(f'Total chunks: {len(chunks)}')
print(f'Token range: {min(c.token_count for c in chunks)} - {max(c.token_count for c in chunks)}')

# In 3 chunks đầu tiên
for i, c in enumerate(chunks[:3]):
    print(f'\n--- Chunk {i+1} ({c.token_count} tokens, pages={c.source_pages}) ---')
    print(c.content[:200])
"
```

### Pass Criteria
✅ Good% > 60%, Tiny% < 10%, no empty chunks → **lên tầng 3**

---

## Tầng 3 — Embedding + Qdrant (cần Qdrant running)

**Mục đích:** Verify embedding quality và Qdrant đã có vectors.

### 3a. Embedding sanity check

```powershell
cd D:\GenAI\DoAn01\backend

python -c "
import sys; sys.path.insert(0, '.')
from src.core.config import get_settings
from src.rag.embedder import BGEM3Embedder
import numpy as np

settings = get_settings()
embedder = BGEM3Embedder(settings)

# 3 câu: VN, EN tương đương, off-topic
texts = [
    'RAG là gì và dùng để làm gì?',
    'What is RAG and what is it used for?',
    'Hôm nay thời tiết đẹp quá'
]
results = embedder.encode(texts)

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

print(f'Dense dim: {len(results[0].dense)}')           # Expected: 1024
print(f'Sparse nnz: {len(results[0].sparse.indices)}')  # Expected: > 0

sim_vn_en = cosine(results[0].dense, results[1].dense)
sim_vn_off = cosine(results[0].dense, results[2].dense)
print(f'Cosine(VN, EN equivalent): {sim_vn_en:.3f}')  # Expected: > 0.70
print(f'Cosine(VN, off-topic):     {sim_vn_off:.3f}')  # Expected: < 0.50

assert len(results[0].dense) == 1024, 'Wrong dense dim!'
assert sim_vn_en > 0.6, f'Cross-lingual similarity too low: {sim_vn_en}'
assert sim_vn_off < sim_vn_en, 'Off-topic should be less similar!'
print('✅ Embedding OK')
"
```

### 3b. Qdrant vector count

```powershell
# Kiểm tra collection tồn tại và có vectors
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(url='http://localhost:6333')
info = client.get_collection('agentbook_chunks')
print(f'Points: {info.points_count}')
print(f'Vectors: {info.vectors_count}')
print(f'Status: {info.status}')
assert info.points_count > 0, 'No vectors in Qdrant! Upload + index material first.'
print('✅ Qdrant has vectors')
"
```

### 3c. Full diagnostic (corpus + embedding + retrieval)

```powershell
cd D:\GenAI\DoAn01\backend
python scripts/diag_pipeline.py
```

Xem output section by section:
- **Section 1 (Corpus Stats):** chunks > 0, reasonable lengths
- **Section 2 (Chunk Quality):** no empty/tiny content
- **Section 3 (Qdrant Stats):** points > 0
- **Section 4 (Embedding Quality):** cross-lingual cosine > 0.7
- **Section 5 (Retrieval + Reranker):** relevant chunks ranked top
- **Section 6 (Keyword Recall):** keywords found in indexed chunks

### Common Bugs

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Dense dim ≠ 1024 | Wrong model loaded | Check `model_config.yaml → embedding_model` |
| Qdrant points = 0 | Chưa upload/index material | Upload + chờ status `indexed` |
| Cross-lingual cosine < 0.5 | BGE-M3 không load đúng | Re-download model |
| Sparse nnz = 0 | Sparse encoding tắt | Check embedder config |

### Pass Criteria
✅ dim=1024, cosine>0.7, Qdrant points>0 → **lên tầng 4**

---

## Tầng 4 — Retrieval Only (cần data indexed)

**Mục đích:** Kiểm tra retrieval trả đúng document không, **chưa qua LLM generation**.

### 4a. Quick retrieval eval (direct module, không qua API)

```powershell
cd D:\GenAI\DoAn01\backend

python scripts/quick_eval.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --output eval_results/retrieval_debug.jsonl
```

### 4b. Test 1 query riêng lẻ (direct module)

```powershell
cd D:\GenAI\DoAn01\backend

python -c "
import asyncio, sys; sys.path.insert(0, '.')

async def test():
    from src.core.config import get_settings
    from src.database import init_database
    from src.rag.embedder import BGEM3Embedder
    from src.rag.retriever import HybridRetriever
    from src.rag.types import RetrievalScope
    from qdrant_client import QdrantClient

    settings = get_settings()
    await init_database(settings)

    qdrant = QdrantClient(url=settings.qdrant_url)
    embedder = BGEM3Embedder(settings)
    retriever = HybridRetriever(settings=settings, qdrant_client=qdrant, embedder=embedder)

    scope = RetrievalScope(owner_id='user_demo', collection_id='<YOUR_COLLECTION_ID>')

    # TEST 1 QUERY
    query = 'RAG pipeline hoat dong nhu the nao?'
    results = await retriever.retrieve(query=query, scope=scope, limit=5)

    print(f'Query: {query}')
    print(f'Results: {len(results)} chunks')
    for i, r in enumerate(results[:3]):
        print(f'  [{i+1}] score={r.fused_score:.4f} doc={r.document_name} pages={r.source_pages}')
        print(f'       content: {(r.content or \"\")[:100]}')

asyncio.run(test())
"
```

### 4c. Auto-score retrieval

```powershell
python scripts/score_retrieval_eval.py `
    --input eval_results/retrieval_debug.jsonl --auto
```

### What to Look For

```
✅ GOOD:
  - Top-1 chunk is from relevant document
  - fused_score > 0.3 for relevant queries
  - Off-topic queries → low scores or irrelevant docs
  - Cross-lingual (VN query) → EN document retrieved

❌ BAD:
  - Top chunk from wrong document → embedding/chunking issue
  - All scores < 0.1 → embedding mismatch or wrong collection
  - Same chunk repeated → dedup issue
  - Off-topic query gets high score → index contaminated
```

### Pass Criteria
✅ ≥80% queries retrieve relevant doc in top-3 → **lên tầng 5**

---

## Tầng 5 — E2E Single Query (cần API running)

**Mục đích:** Test **1 câu hỏi** qua full pipeline (routing → retrieval → rerank → synthesis → guardrails).

### 5a. 1 câu hỏi đơn giản

```powershell
$body = @{
    owner_id      = "user_demo"
    collection_id = "<YOUR_COLLECTION_ID>"
    query         = "Tai lieu nay noi ve gi?"
} | ConvertTo-Json -Depth 3

$resp = Invoke-RestMethod `
    -Uri "http://localhost:8000/api/v1/query/ask" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

# Kiểm tra kết quả
Write-Host "Answer: $($resp.data.answer.Substring(0, [Math]::Min(200, $resp.data.answer.Length)))"
Write-Host "Citations: $($resp.data.citations.Count)"
Write-Host "Confidence: $($resp.data.confidence)"
Write-Host "Was refused: $($resp.data.was_refused)"
Write-Host "Route: $($resp.data.route)"
```

### 5b. Test từng endpoint riêng

```powershell
# --- Summarize ---
$body = @{
    owner_id      = "user_demo"
    collection_id = "<YOUR_COLLECTION_ID>"
    focus         = "Main topics"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/summarize" `
    -Method POST -ContentType "application/json" -Body $body

# --- Study Guide ---
$body = @{
    owner_id      = "user_demo"
    collection_id = "<YOUR_COLLECTION_ID>"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/study-guide" `
    -Method POST -ContentType "application/json" -Body $body

# --- Compare (cần ≥2 materials) ---
$body = @{
    owner_id      = "user_demo"
    collection_id = "<YOUR_COLLECTION_ID>"
    query         = "Hai tai lieu giong va khac nhau o diem nao?"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/compare" `
    -Method POST -ContentType "application/json" -Body $body
```

### 5c. Test off-topic refusal

```powershell
$body = @{
    owner_id      = "user_demo"
    collection_id = "<YOUR_COLLECTION_ID>"
    query         = "Thu do cua nuoc Phap la gi?"
} | ConvertTo-Json

$resp = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query/ask" `
    -Method POST -ContentType "application/json" -Body $body

# Expected: was_refused = true
Write-Host "Was refused: $($resp.data.was_refused)"  # Should be True
```

### 5d. Debug material (xem chunks + vectors)

```powershell
$resp = Invoke-RestMethod `
    "http://localhost:8000/api/v1/materials/<MATERIAL_ID>/debug?owner_id=user_demo"
$resp.data | ConvertTo-Json -Depth 3

# Check:
# - qdrant_vector_count > 0
# - chunks[].token_count in range [100, 512]
# - chunks[].source_pages not empty
```

### Common Bugs

| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| Timeout (>60s) | Ollama model quá chậm | Giảm `num_predict` hoặc dùng model nhỏ hơn |
| `was_refused = true` cho câu hợp lệ | Refusal threshold quá cao | Check `guardrails_config.yaml` |
| Answer không có citation | Prompt template lỗi | Check `prompts/qa_grounded.txt` |
| 500 Internal Server Error | Xem backend log | Check `backend.err.log` |
| Answer bằng EN dù hỏi VN | Language detection sai | Check `answer_language` param |

### Pass Criteria
✅ Answer có citations, was_refused=false cho câu hợp lệ, was_refused=true cho off-topic → **lên tầng 6**

---

## Tầng 6 — Full Benchmark Suite

**Mục đích:** Chạy toàn bộ benchmark, xuất report, kiểm tra thresholds.

### 6a. Tạo benchmark dataset bằng GPT-4o (làm trước E2E)

> Mục tiêu: thay bộ câu hỏi hard-code/mẫu nhỏ bằng bộ benchmark có gold evidence thật, bao phủ toàn bộ chức năng.

```powershell
cd D:\GenAI\DoAn01\backend

# 1) Sinh meta inventory từ collection đã index
python scripts/generate_eval_dataset.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --provider openai `
    --model gpt-4o `
    --mode meta-inventory `
    --output ../evaluation/datasets/agentbook_meta_dataset.jsonl

# 2) Sinh retrieval gold: query -> expected doc/page/block/chunk
python scripts/generate_eval_dataset.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --provider openai `
    --model gpt-4o `
    --mode retrieval-gold `
    --input ../evaluation/datasets/agentbook_meta_dataset.jsonl `
    --output ../evaluation/datasets/agentbook_retrieval_gold.jsonl `
    --target-count 200

# 3) Sinh E2E gold: query -> expected facts + forbidden claims + evidence anchors
python scripts/generate_eval_dataset.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --provider openai `
    --model gpt-4o `
    --mode e2e-gold `
    --input ../evaluation/datasets/agentbook_meta_dataset.jsonl `
    --output ../evaluation/datasets/agentbook_e2e_gold.jsonl `
    --target-count 160

# 4) Sinh adversarial/refusal cases
python scripts/generate_eval_dataset.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --provider openai `
    --model gpt-4o `
    --mode adversarial `
    --input ../evaluation/datasets/agentbook_meta_dataset.jsonl `
    --output ../evaluation/datasets/agentbook_adversarial.jsonl `
    --target-count 80
```

**Lưu ý implementation:** script hiện tại `generate_eval_dataset.py` mới dùng Ollama và query tự do. Cần nâng cấp trước khi chạy các command trên:

- Thêm `--provider openai`, `--model gpt-4o`, `--mode`.
- Output JSONL theo schema trong phần “GPT-4o Meta-Dataset”.
- Không sinh case nếu không có `expected_evidence`.
- Gọi validator sau mỗi phase.

```powershell
python scripts/validate_benchmark_dataset.py `
    --meta ../evaluation/datasets/agentbook_meta_dataset.jsonl `
    --retrieval ../evaluation/datasets/agentbook_retrieval_gold.jsonl `
    --e2e ../evaluation/datasets/agentbook_e2e_gold.jsonl `
    --adversarial ../evaluation/datasets/agentbook_adversarial.jsonl `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID>
```

### 6b. Retrieval Benchmark trên gold set

```powershell
python scripts/quick_eval.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --question-set ../evaluation/datasets/agentbook_retrieval_gold.jsonl `
    --output eval_results/retrieval_gold_run.jsonl

python scripts/score_retrieval_eval.py `
    --input eval_results/retrieval_gold_run.jsonl `
    --gold ../evaluation/datasets/agentbook_retrieval_gold.jsonl `
    --metrics recall@1,recall@3,recall@5,mrr,ndcg@5 `
    --report eval_results/retrieval_gold_report.md
```

### 6c. E2E Eval trên gold set (~30-60 phút local, nhanh hơn nếu OpenAI provider)

```powershell
cd D:\GenAI\DoAn01\backend

python scripts/e2e_eval.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --question-set ../evaluation/datasets/agentbook_e2e_gold.jsonl `
    --output eval_results/e2e_eval_debug.jsonl `
    --timeout 300
```

### 6d. GPT-4o Judge cho E2E outputs

```powershell
python scripts/judge_eval_with_gpt4o.py `
    --input eval_results/e2e_eval_debug.jsonl `
    --gold ../evaluation/datasets/agentbook_e2e_gold.jsonl `
    --rubric ../evaluation/datasets/agentbook_judge_rubric.yaml `
    --model gpt-4o `
    --output eval_results/e2e_gpt4o_judged.jsonl `
    --report eval_results/e2e_gpt4o_report.md
```

GPT-4o judge phải chấm theo 5 trục:

| Trục | Điểm |
|---|---|
| `groundedness` | Từng claim có nằm trong evidence không |
| `answer_relevance` | Trả lời đúng câu hỏi không |
| `citation_correctness` | Citation marker có trỏ đúng evidence không |
| `refusal_correctness` | Từ chối đúng/lố không |
| `vietnamese_quality` | Tiếng Việt tự nhiên, không mất dấu, không lẫn EN vô cớ |

### 6e. Ablation Study (4 configs × 8 queries, ~15-20 phút)

```powershell
# Chạy 1 config trước để test
python scripts/ablation_eval.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID> `
    --configs baseline `
    --timeout 300

# Nếu OK, chạy full
python scripts/ablation_eval.py `
    --owner-id user_demo `
    --collection-id <YOUR_COLLECTION_ID>
```

### 6f. RAGAS LLM-Judge local (chậm, optional)

```powershell
# Chạy trên 3 câu trước (smoke test)
python scripts/ragas_eval.py `
    --input eval_results/e2e_eval_debug.jsonl `
    --output eval_results/ragas_debug.json `
    --judge-model qwen2.5:3b `
    --limit 3

# Nếu OK, chạy full
python scripts/ragas_eval.py `
    --input eval_results/e2e_eval_debug.jsonl `
    --output eval_results/ragas_full.json `
    --judge-model qwen2.5:3b
```

### 6g. Xem kết quả nhanh

```powershell
# Xem worst performers
cd D:\GenAI\DoAn01
python eval_results/summarize.py

# Hoặc inspect failures
python eval_results/inspect_failures.py
```

---

## Nâng Cấp Cần Làm (ưu tiên theo debug value)

### Tier A — Cải thiện debug workflow (làm trước)

| # | Task | File | Mô tả | Effort |
|---|---|---|---|---|
| A1 | **MD report cho e2e_eval** | [e2e_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/e2e_eval.py) | Xuất `E2E_Evaluation_Report.md` với bảng, pass/fail badge | 1d |
| A2 | **Threshold CI guard** | [e2e_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/e2e_eval.py) | `--ci-mode` exit 1 khi faith < 0.85 | 0.5d |
| A3 | **Question-set input** | [e2e_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/e2e_eval.py) | Thêm `--question-set <jsonl>` để đọc `agentbook_e2e_gold.jsonl`, bỏ phụ thuộc list hard-code | 1d |

### Tier B — Parser + Ingestion benchmark (làm khi debug tầng 1-2)

| # | Task | File | Mô tả | Effort |
|---|---|---|---|---|
| B1 | **Parser benchmark script** | [NEW] `parser_benchmark.py` | Parse 11 files + assertions tự động + MD report | 2d |
| B2 | **Ingestion timing** | [parse_index_pipeline.py](file:///D:/GenAI/DoAn01/backend/src/services/parse_index_pipeline.py) | Thêm timing logs cho mỗi stage | 0.5d |

### Tier C — Retrieval benchmark (làm khi debug tầng 4)

| # | Task | File | Mô tả | Effort |
|---|---|---|---|---|
| C1 | **Recall@K, MRR, nDCG** | [quick_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/quick_eval.py) | Thêm proper IR metrics | 1.5d |
| C2 | **Golden retrieval annotations** | [quick_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/quick_eval.py) | Query → expected docs mapping | 1d |
| C3 | **Retrieval MD report** | [score_retrieval_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/score_retrieval_eval.py) | Xuất Markdown thay vì chỉ console | 0.5d |

### Tier C+ — GPT-4o meta-dataset benchmark (bắt buộc để có dataset SOTA)

| # | Task | File | Mô tả | Effort |
|---|---|---|---|---|
| C+1 | **OpenAI provider cho generator** | [generate_eval_dataset.py](file:///D:/GenAI/DoAn01/backend/scripts/generate_eval_dataset.py) | Thêm `--provider openai --model gpt-4o`, structured JSON output, retry/backoff | 1d |
| C+2 | **Meta inventory mode** | [generate_eval_dataset.py](file:///D:/GenAI/DoAn01/backend/scripts/generate_eval_dataset.py) | Xuất `agentbook_meta_dataset.jsonl` từ Mongo chunks/material pages/blocks/audio segments | 1d |
| C+3 | **Gold generation modes** | [generate_eval_dataset.py](file:///D:/GenAI/DoAn01/backend/scripts/generate_eval_dataset.py) | `retrieval-gold`, `e2e-gold`, `endpoint-cases`, `adversarial` | 2d |
| C+4 | **Dataset validator** | [NEW] `validate_benchmark_dataset.py` | Validate schema, anchors tồn tại, split leakage, duplicate/near-duplicate queries | 1d |
| C+5 | **GPT-4o judge** | [NEW] `judge_eval_with_gpt4o.py` | Chấm groundedness/relevance/citation/refusal/Vietnamese quality, xuất JSONL + MD | 1.5d |
| C+6 | **Rubric YAML** | [NEW] `evaluation/datasets/agentbook_judge_rubric.yaml` | Rubric cố định để chấm nhất quán giữa các lần benchmark | 0.5d |
| C+7 | **Private test split** | [NEW] `evaluation/datasets/splits/` | Tạo `smoke/dev/test_private`, không dùng private để tune prompt/config | 0.5d |

### Tier D — Ablation + Full suite (làm khi tầng 0-5 stable)

| # | Task | File | Mô tả | Effort |
|---|---|---|---|---|
| D1 | **Ablation configs mới** | [ablation_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/ablation_eval.py) | Thêm CRAG on/off, adaptive on/off | 1d |
| D2 | **Ablation MD report** | [ablation_eval.py](file:///D:/GenAI/DoAn01/backend/scripts/ablation_eval.py) | Comparison table + delta analysis | 1d |
| D3 | **Master benchmark script** | [NEW] `run_full_benchmark.py` | Chạy tất cả scripts theo thứ tự | 1d |

---

## Quick Reference — Copy-Paste Debug Commands

```powershell
# ═══════════════════════════════════════════════════════════
# AGENTBOOK DEBUG CHEAT SHEET
# ═══════════════════════════════════════════════════════════

# Tầng 0: Infra
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:6333/collections
Invoke-RestMethod http://localhost:11434/api/tags

# Tầng 1: Parse 1 file
cd D:\GenAI\DoAn01\backend
python scripts/dry_run_test_data_pipeline.py "../data/test data/AgentBook_Internet_Test_Pack/Collection_A" --max-files 1

# Tầng 2: Chunk quality
python scripts/chunk_quality_check.py "../data/test data/AgentBook_Internet_Test_Pack/Collection_A"

# Tầng 3: Embedding + Qdrant
python scripts/diag_pipeline.py

# Tầng 4: Retrieval
python scripts/quick_eval.py --owner-id user_demo --collection-id <ID>

# Tầng 5: 1 query E2E
curl -X POST http://localhost:8000/api/v1/query/ask -H "Content-Type: application/json" -d "{\"owner_id\":\"user_demo\",\"collection_id\":\"<ID>\",\"query\":\"Tai lieu nay noi ve gi?\"}"

# Tầng 6a: Sinh benchmark gold bằng GPT-4o (sau khi nâng cấp generate_eval_dataset.py)
python scripts/generate_eval_dataset.py --owner-id user_demo --collection-id <ID> --provider openai --model gpt-4o --mode e2e-gold --output ../evaluation/datasets/agentbook_e2e_gold.jsonl

# Tầng 6b: Validate benchmark dataset
python scripts/validate_benchmark_dataset.py --e2e ../evaluation/datasets/agentbook_e2e_gold.jsonl --owner-id user_demo --collection-id <ID>

# Tầng 6c: Full eval trên gold set
python scripts/e2e_eval.py --owner-id user_demo --collection-id <ID> --question-set ../evaluation/datasets/agentbook_e2e_gold.jsonl --timeout 300

# Tầng 6d: GPT-4o judge
python scripts/judge_eval_with_gpt4o.py --input eval_results/e2e_eval.jsonl --gold ../evaluation/datasets/agentbook_e2e_gold.jsonl --model gpt-4o --report eval_results/e2e_gpt4o_report.md
```
