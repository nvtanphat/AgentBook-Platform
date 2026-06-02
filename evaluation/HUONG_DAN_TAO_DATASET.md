# Hướng dẫn tạo Dataset Benchmark cho AgentBook

Tài liệu này hướng dẫn cách xây **bộ dữ liệu đánh giá (gold set)** để benchmark *toàn bộ* hệ thống AgentBook: từ retrieval, grounding/citation, cross-lingual, graph, đến cổng từ chối (refusal) — phục vụ cả việc đo chất lượng lẫn luận văn về Lightweight Neuro-Fuzzy Confidence Gate.

> **Nguyên tắc số 1:** nhãn phải trỏ tới **ID thật** trong corpus đã index. Metric retrieval so khớp bằng bộ ba `(doc_id, page, block_id)` — không thể bịa. Vì vậy phải **đóng băng corpus trước**, lấy ID thật, rồi mới gán nhãn.

---

## 0. Hạ tầng eval đã có sẵn (đừng làm lại)

| Tầng | Harness | Dataset | Metric |
|---|---|---|---|
| Retrieval / evidence | `evaluation/run_eval.py` + `evaluation/metrics.py` | `evaluation/datasets/gold_qa_pairs.json` (đang là stub 1 mẫu) | Recall@k, Precision@k, MRR@k, nDCG@k, citation_accuracy |
| End-to-end answer | `backend/scripts/e2e_eval.py` | hiện hardcode câu hỏi trong file | faithfulness, citation_coverage/validity, answer_relevance, semantic_faithfulness, grounded_ratio, refusal, false_premise_corrected |
| Sinh câu hỏi nháp | `backend/scripts/generate_eval_dataset.py` | sinh từ chunk thật bằng LLM | — |

Việc cần làm = **xây gold set có nhãn evidence thật + nhãn refuse**, phủ hết năng lực hệ thống.

---

## 1. Hai trục đầu vào của dataset

Mỗi mẫu là một **cặp**: (câu hỏi) chấm trên (tài liệu nào). Cả hai trục phải đa dạng.

### Trục A — Đầu vào TÀI LIỆU (theo allowlist + pipeline)

| Kiểu tài liệu | Kích hoạt nhánh | Tối thiểu |
|---|---|---|
| PDF chữ số (digital) | Docling → chunking → retrieval | 2–3 |
| PDF scan (ảnh chữ **in**) | OCR — `ocr_engine.py` (EasyOCR) | 1 |
| PNG/JPG chữ in | OCR | 1 |
| PNG/JPG **viết tay** rõ | `handwriting_reader.py` | 1 |
| PNG/JPG viết tay **mờ/kém** | `image_quality_checker.py` → **gate từ chối** | 1 |
| DOCX | Docling | 1 |
| PPTX (slide) | Docling | 1 |
| CSV/XLSX | `spreadsheet_parser.py` | 1 |
| Doc có **hình/bảng** | figure_captioner / cross_modal_linker | 1 |
| Doc **tiếng Anh** | dùng cho cross-lingual | 1 |
| Doc của **owner khác** | test cách ly owner/collection (negative) | 1 |

→ Khoảng **8–12 tài liệu** phủ hết các nhánh.

### Trục B — Đầu vào CÂU HỎI (query_type)

| query_type | Ép cái gì | `expect_refused` |
|---|---|---|
| factual | retrieval cơ bản + citation | false |
| summarization | gom nhiều nguồn | false |
| comparison | so sánh ≥2 tài liệu | false |
| graph_relation | đúng cạnh quan hệ trong graph | false |
| cross_lingual (VI↔EN) | query VI trên doc EN, trả lời VI | false |
| anaphora (đa lượt) | "nó", "cái đó" — cần ngữ cảnh hội thoại | false |
| off_topic_should_refuse | phải từ chối | **true** |
| false_premise | sửa tiền đề sai, không bịa | false |
| out_of_scope_in_corpus | đúng miền nhưng corpus không có → từ chối | **true** |
| table/figure question | hỏi vào bảng/hình (cross-modal) | false |

### Giao nhau giữa 2 trục (chỗ tạo độ khó thật)

Không test riêng từng trục mà **kết hợp**:
- factual **trên PDF scan** → đo cả OCR lẫn retrieval.
- cross_lingual **trên doc EN viết tay** → handwriting + dịch + grounding.
- comparison **giữa DOCX và PDF** → compare đa định dạng.
- out_of_scope **hỏi sang collection owner khác** → cách ly + từ chối.

Chính `expected_evidence[].doc_id` quyết định câu hỏi "rơi" vào tài liệu/định dạng nào → bạn chủ động trỏ evidence sang doc scan/handwriting/EN để ép đúng nhánh muốn đo.

---

## 2. Schema mỗi mẫu (1 object JSON)

File gold là **mảng JSON** (`[ {...}, {...} ]`) vì `run_eval.py` lặp `for item in dataset`. Đừng bọc trong object.

```json
{
  "id": "gold-014",
  "query_type": "factual",
  "owner_id": "user_demo",
  "collection_id": "<collection_id THẬT>",
  "query": "Dropout giảm overfitting như thế nào?",
  "query_en": "How does dropout reduce overfitting?",
  "expect_refused": false,
  "expected_behavior": "answer",
  "expected_answer": "Dropout tắt ngẫu nhiên neuron khi train để giảm co-adaptation.",
  "expected_keywords": ["dropout", "overfitting", "regularization"],
  "expected_relations": [
    { "source": "dropout", "target": "overfitting", "relation_type": "reduces" }
  ],
  "source_modality": "digital_text",
  "context": null,
  "expected_evidence": [
    { "doc_id": "<material_id THẬT>", "document_name": "Lecture05.pdf", "page": 14, "block_id": "<block_id THẬT>" }
  ]
}
```

Field nào dùng cho gì:

| Field | Bắt buộc | Ai đọc nó |
|---|---|---|
| `id` | ✅ | run_eval map với predictions |
| `query` | ✅ | chạy qua API |
| `query_type` | ✅ | quyết định slice + metric áp dụng |
| `expect_refused` | ✅ | tính false_accept / false_refusal |
| `expected_evidence[]` | ✅ (trừ hàng refuse) | Recall/Precision/MRR/nDCG/citation_accuracy |
| `expected_answer` | nên có | answer_relevance, đối chiếu người |
| `expected_keywords` | tùy | sanity lexical |
| `expected_relations` | chỉ graph | đối chiếu cạnh graph |
| `query_en` | chỉ cross_lingual | — |
| `context` | chỉ anaphora | lượt hỏi trước |
| `source_modality` | chỉ ocr/handwriting | biết nhánh nào |

**Quy tắc gán evidence:**
- Chỉ đưa block **thật sự chứa** câu trả lời (gold nên *chặt*).
- 1–3 block/câu là hợp lý; summarization/compare thì liệt kê đủ nguồn.
- `page` phải là **số nguyên**, `block_id` là **string** — `EvidenceKey.from_mapping` sẽ lỗi nếu thiếu/sai kiểu.
- Hàng `expect_refused: true` → để `expected_evidence: []` (đừng gán evidence, nếu không false-accept thành "đúng").

---

## 3. Quy trình 6 bước

### Bước 1 — Đóng băng corpus
1. Chọn 8–12 tài liệu phủ đủ Trục A.
2. Upload + chờ index xong **một lần**, rồi **không sửa nữa**.
3. Ghi lại `collection_id`. Quy ước: `gold_v1` ↔ snapshot corpus này. Đổi corpus → bump `gold_v2`.

### Bước 2 — Lấy ID thật (xương sống)

**`collection_id` + `material_id`** — qua Mongo:
```
db.materials.find({ owner_id: "user_demo" }, { _id:1, original_name:1, collection_id:1 })
```

**`block_id` + text từng block** — dùng đúng endpoint UI đang xài:
```
GET /api/v1/evidence/{material_id}/{page}?owner_id=user_demo
```
→ trả `blocks[]` gồm `block_id`, `page`, `snippet_original`. Đọc `snippet_original` để biết block nào chứa câu trả lời, copy `block_id`.

> Mẹo: mở Swagger `http://localhost:8000/docs`, chạy endpoint này từng trang để "duyệt" toàn bộ block kèm text — cách rẻ nhất để gán nhãn.

### Bước 3 — Sinh câu hỏi
- **Tự động (factual/summarization):** `python backend/scripts/generate_eval_dataset.py --owner-id user_demo --collection-id <id> --output eval_results/draft.jsonl` → dùng làm *nháp*, vẫn phải người duyệt.
- **Thủ công (bắt buộc):** slice adversarial (`off_topic`, `false_premise`, `out_of_scope`) + ảnh viết tay mờ. Đây là phần giá trị nhất cho luận văn.

### Bước 4 — Gán nhãn
Với mỗi câu, điền schema ở Mục 2 theo bảng `query_type` ở Trục B. Trỏ `expected_evidence` tới `block_id` thật từ Bước 2.

### Bước 5 — Quy mô & cân bằng
Mục tiêu luận văn: **~100–120 câu / ≥6 doc**, trong đó **25–30% adversarial**.

| query_type | Số câu gợi ý |
|---|---|
| factual | 25 |
| summarization | 12 |
| comparison | 12 |
| graph_relation | 12 |
| cross_lingual | 15 |
| anaphora | 8 |
| off_topic_should_refuse | 12 |
| false_premise | 12 |
| out_of_scope_in_corpus | 8 |
| ocr / handwriting | 4–6 |

### Bước 6 — Sinh predictions & chấm

`run_eval.py` chấm **dataset (gold) ⊕ predictions (hệ chạy ra)**, hai file tách biệt.

1. Chạy từng `query` qua `POST /api/v1/query/ask`, lấy `citations` → map thành `{doc_id, page, block_id}`. (Vòng lặp tham khảo: `e2e_eval.py`; mẫu định dạng: `eval_results/predictions_for_run_eval.json`.)
2. Lưu `predictions.json`:
   ```json
   { "gold-014": { "retrieved_evidence": [ {"doc_id":"...","page":14,"block_id":"..."} ],
                   "citations":          [ {"doc_id":"...","page":14,"block_id":"..."} ] } }
   ```
3. Sửa `dataset:` trong một YAML config (vd copy `evaluation/ablation_configs/full_eval.yaml`) trỏ tới file gold của bạn, rồi:
   ```
   python evaluation/run_eval.py --config evaluation/ablation_configs/full_eval.yaml --predictions eval_results/predictions.json
   ```
   → ra Recall@k, MRR@k, nDCG@k, citation_accuracy.
4. Chất lượng câu trả lời + FAR/FRR: chạy `e2e_eval.py` (cần sửa để đọc câu hỏi từ file gold thay vì hardcode, và log thêm `confidence` để đối chiếu `expect_refused`).

---

## 4. Metric ↔ nhãn cần có

| Metric | Cần nhãn gì |
|---|---|
| Recall@k, Precision@k, MRR@k, nDCG@k | `expected_evidence` (block thật) + predictions `retrieved_evidence` |
| citation_accuracy | `expected_evidence` + predictions `citations` |
| faithfulness / citation_coverage / validity | answer có marker `[N]` (tính tự động từ answer) |
| answer_relevance / semantic_faithfulness | `expected_answer` + embedding |
| **false_accept_rate** | `expect_refused: true` mà hệ vẫn trả lời |
| **false_refusal_rate** | `expect_refused: false` mà hệ từ chối |
| correction_rate | slice `false_premise` + phát hiện ngôn ngữ sửa sai |

Hai dòng in đậm là **cốt lõi cho luận văn cổng tin cậy** — cần `expect_refused` + log `confidence` từng câu để vẽ đường đánh đổi FAR–FRR thay ngưỡng cứng.

---

## 5. Lỗi thường gặp
- **Gán evidence cho câu off_topic/out_of_scope** → sai. Để `[]`.
- **Quên đóng băng corpus trước khi gán block_id** → re-index đổi block_id → nhãn hỏng.
- **`page` để string** hoặc thiếu `block_id` → `run_eval` crash ở `EvidenceKey.from_mapping`.
- **Bọc gold trong object** thay vì mảng → `run_eval` không lặp được.
- **Gold quá rộng** (gán cả block chỉ liên quan mơ hồ) → recall/precision mất ý nghĩa. Gold phải *chặt*.

---

## 6. Checklist trước khi chốt `gold_v1`
- [ ] Corpus đã đóng băng, ghi rõ `collection_id` + danh sách `material_id`.
- [ ] Đủ Trục A (text/scan/ảnh in/viết tay rõ+mờ/docx/pptx/csv/EN/owner khác).
- [ ] Đủ Trục B với phân bổ ~100–120 câu, 25–30% adversarial.
- [ ] Mọi `expected_evidence.block_id` là ID thật, lấy từ endpoint evidence.
- [ ] Hàng refuse có `expect_refused: true` + `expected_evidence: []`.
- [ ] File là mảng JSON hợp lệ; `page` là int.
- [ ] Có `predictions.json` sinh từ API; chạy `run_eval.py` ra số.
- [ ] Đặt tên + version: `evaluation/datasets/gold_v1.json`.
