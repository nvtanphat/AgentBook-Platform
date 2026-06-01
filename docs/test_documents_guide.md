# Tài liệu đầu vào cần có để test toàn bộ hệ thống AgentBook

> [!IMPORTANT]
> Chuẩn bị **ít nhất 2 collection khác nhau** và đặt tài liệu vào đúng collection theo mục đích test.

---

## Tổng quan — Bảng mapping tính năng → file

| File | Parser kích hoạt | Tính năng được test |
|------|----------------|-------------------|
| PDF text-based (EN) | Docling | RAG cơ bản, Query/Ask, Summarize, Study-guide |
| PDF scan/image (VN) | Docling + EasyOCR | OCR pipeline, Cross-lingual query |
| DOCX có bảng | Docling + python-docx | Table retrieval, Compare |
| PPTX slide nhiều hình | Docling + FigureCaptioner | Figure block, mixed modality |
| PNG/JPG rõ nét (scan) | EasyOCR / PaddleOCR | OCR text từ ảnh |
| PNG chữ viết tay | image_quality_checker + handwriting_reader | Handwriting pipeline, refuse khi kém |
| PNG chất lượng thấp | image_quality_checker | Refusal test (mờ/tối quá) |
| CSV dữ liệu bảng | SpreadsheetParser | Table chunking, Ask trên data |
| XLSX nhiều sheet | SpreadsheetParser | Multi-sheet, Compare |
| MP3 / WAV | AudioParser (faster-whisper) | Audio transcription |
| 2 PDF cùng chủ đề | Pipeline bình thường | Compare, GraphRAG ask-graph |

---

## Chi tiết từng file

### 1. 📄 PDF Text-based (tiếng Anh) — **BẮT BUỘC**

**Mục đích:** Test pipeline cơ bản, tất cả query endpoint

**Yêu cầu nội dung:**
- Có tiêu đề, đề mục, đoạn văn (heading / paragraph)
- Có ít nhất **1 bảng** và **1 hình vẽ/biểu đồ**
- Dài ≥ 5 trang
- Ngôn ngữ: **tiếng Anh** (để test cross-lingual VN→EN)

**Ví dụ dùng được:**
- Bất kỳ paper học thuật (arxiv.org)
- Slide bài giảng export PDF
- Tài liệu kỹ thuật bất kỳ

**Tính năng test được:**
- `POST /query/ask` — hỏi bằng **tiếng Việt**, nhận trả lời tiếng Việt từ tài liệu EN
- `POST /query/summarize`
- `POST /query/study-guide`
- `GET /materials/{id}/debug` — kiểm tra chunks và `qdrant_vector_count > 0`

---

### 2. 📄 PDF Scan / chỉ có ảnh (tiếng Việt) — **Quan trọng**

**Mục đích:** Test EasyOCR cho trang scan, cross-lingual pipeline

**Yêu cầu nội dung:**
- File PDF nhưng **nội dung là ảnh scan** (không extract được text bằng text layer)
- Văn bản tiếng Việt rõ ràng
- Chất lượng scan tốt (sáng, nét, không nghiêng quá 12°)

**Cách tạo nếu chưa có:**
```
Chụp ảnh trang sách/tài liệu tiếng Việt → ghép thành PDF bằng Adobe/Word
Hoặc: In tài liệu → scan lại thành PDF
```

**Tính năng test được:**
- OCR pipeline (EasyOCR → PaddleOCR)
- `ocr_confidence` trong debug response
- Query tiếng Việt trên tài liệu tiếng Việt

---

### 3. 📝 DOCX có bảng và đa định dạng — **Quan trọng**

**Mục đích:** Test parser bảng DOCX, so sánh tài liệu

**Yêu cầu nội dung:**
- Có ít nhất **2–3 bảng** (python-docx table augmentation)
- Có mục lục, tiêu đề H1/H2
- Cùng chủ đề với PDF #1 (để test `compare`)

**Tính năng test được:**
- `block_type: "table"` xuất hiện trong debug
- `POST /query/compare` giữa DOCX và PDF cùng chủ đề
- Table retrieval khi hỏi câu liên quan đến bảng

---

### 4. 🎞️ PPTX Slide — **Bổ sung**

**Mục đích:** Test figure captioner, mixed modality

**Yêu cầu nội dung:**
- Có nhiều slide với **hình ảnh, biểu đồ, sơ đồ**
- Có text giải thích
- ≥ 10 slides

**Tính năng test được:**
- `block_type: "figure"` trong debug
- `FigureCaptioner` tạo caption tự động qua Ollama
- Modality `mixed` trong material record

---

### 5. 🖼️ PNG/JPG Scan rõ nét — **Cho test ảnh đơn**

**Mục đích:** Test OCR trực tiếp trên ảnh, modality image

**Yêu cầu:**
- Ảnh chụp tài liệu in, **rõ nét, sáng, không mờ**
- Blur variance > 80, brightness 45–230, contrast > 18
- Độ nghiêng < 12°

> [!NOTE]
> Kiểm tra quality bằng `GET /materials/{id}/debug` — xem `ocr_confidence`

**Tính năng test được:**
- Upload ảnh đơn → index như material
- `POST /query/ask-image` — upload ảnh làm câu hỏi (nếu visual embedding enabled)

---

### 6. ✍️ PNG Chữ viết tay — **Để test refusal**

**Mục đích:** Test handwriting pipeline và guardrail chất lượng

**Cần 2 file:**

| File | Chất lượng | Kết quả mong đợi |
|------|-----------|-----------------|
| `handwriting_good.png` | Chữ viết tay **rõ ràng**, đủ sáng | `status: indexed` |
| `handwriting_bad.png` | **Mờ, tối, cong vênh** nhiều | `status: failed`, error về quality |

**Thông số fail:** `min_handwriting_quality_score = 0.72`, `min_handwriting_confidence = 0.8`

---

### 7. 🖼️ PNG Chất lượng thấp — **Để test OCR quality gate**

**Mục đích:** Test từ chối xử lý khi ảnh không đạt chuẩn

**Yêu cầu:**
- Ảnh rất mờ (blur variance < 80) **HOẶC**
- Quá tối (brightness < 45) **HOẶC**  
- Quá sáng/trắng (brightness > 230)

**Kết quả mong đợi:** `status: failed`, `error_message` chứa thông tin chất lượng

---

### 8. 📊 CSV dữ liệu có cấu trúc — **Bổ sung**

**Mục đích:** Test SpreadsheetParser, table verbalization

**Yêu cầu:**
- Có header row rõ ràng
- Ít nhất **3 cột, 20+ dòng dữ liệu**
- Có cả cột text và cột số

**Ví dụ nội dung:**
```csv
Tên sản phẩm,Danh mục,Giá,Số lượng,Đánh giá
Laptop Dell XPS,Máy tính,25000000,50,4.5
iPhone 15 Pro,Điện thoại,29000000,120,4.8
...
```

**Tính năng test được:**
- `block_type: "table"` + `block_kind: "table_block"` trong debug
- Câu hỏi tổng hợp: "Sản phẩm nào có giá cao nhất?"
- Câu hỏi lọc: "Liệt kê sản phẩm có đánh giá trên 4.5"

---

### 9. 📊 XLSX Nhiều sheet — **Bổ sung**

**Mục đích:** Test xử lý workbook đa sheet

**Yêu cầu:**
- File Excel có **2–4 sheets** tên khác nhau
- Mỗi sheet chứa bảng dữ liệu khác nhau
- Sheet cuối có dạng document (không phải table)

**Tính năng test được:**
- Mỗi sheet → 1 page trong material
- `sheet_count` trong debug extra
- Query có thể truy xuất từ sheet cụ thể

---

### 10. 🎵 Audio MP3/WAV — **Nếu muốn test audio**

**Mục đích:** Test AudioParser (faster-whisper)

> [!WARNING]
> Cần cài `faster-whisper`: `pip install faster-whisper`  
> Và model whisper `small` (~460 MB) sẽ tự download lần đầu

**Yêu cầu:**
- File MP3 hoặc WAV nói rõ ràng
- Thời lượng: 1–5 phút (để test nhanh)
- Ngôn ngữ: tiếng Việt (để test `_apply_vn_corrections`)

**Kết quả mong đợi:**
- Blocks có `source: "audio_whisper"`, `start_seconds`, `end_seconds`
- Query hỏi về nội dung bài nói

---

## Bộ tài liệu tối thiểu để test "full pipeline"

```
Collection "Test_Collection_A":
├── lecture_notes.pdf          ← PDF text EN (5+ trang, có bảng + hình)
├── report_vn.pdf              ← PDF scan VN (OCR test)
├── comparison_doc.docx        ← DOCX cùng chủ đề với lecture_notes.pdf
└── slides_mixed.pptx          ← PPTX có nhiều hình

Collection "Test_Collection_B":
├── data_table.csv             ← CSV dữ liệu bảng
├── workbook_multi.xlsx        ← XLSX nhiều sheet
├── scan_clear.png             ← Ảnh scan rõ
├── handwriting_ok.png         ← Chữ viết tay đọc được
└── handwriting_blur.png       ← Chữ viết tay mờ (test refusal)
```

---

## Kịch bản test theo tính năng

| Tính năng | Cần file | Collection |
|-----------|---------|-----------|
| `POST /query/ask` (VN→EN) | `lecture_notes.pdf` | A |
| `POST /query/ask-stream` | Bất kỳ indexed material | A |
| `POST /query/compare` | `lecture_notes.pdf` + `comparison_doc.docx` | A |
| `POST /query/summarize` | Bất kỳ PDF | A |
| `POST /query/study-guide` | Bất kỳ PDF | A |
| `POST /query/ask-graph` | Bất kỳ (sau khi entity extraction chạy) | A |
| OCR quality pass | `scan_clear.png` | B |
| OCR quality fail | `handwriting_blur.png` | B |
| Handwriting OK | `handwriting_ok.png` | B |
| Table retrieval | `data_table.csv` hoặc `workbook_multi.xlsx` | B |
| Audio transcription | `lecture.mp3` | B |
| Cross-lingual VN→EN | `lecture_notes.pdf` (EN) + câu hỏi VN | A |

---

## Nguồn tài liệu mẫu gợi ý

| Loại | Nguồn |
|------|-------|
| PDF học thuật (EN) | https://arxiv.org (tải trực tiếp) |
| PDF tiếng Việt | Giáo trình đại học, sách điện tử |
| PDF scan | Chụp ảnh + ghép PDF (Adobe Acrobat / MS Word) |
| PPTX slide | Export từ Google Slides / PowerPoint |
| CSV | Tạo bằng Excel/Python pandas |
| XLSX | Tạo bằng Excel với nhiều sheet |
| Audio | Ghi âm bằng điện thoại, hoặc text-to-speech |

> [!TIP]
> Để test nhanh nhất: dùng một paper PDF từ arxiv.org (EN, có bảng + hình) và một file CSV đơn giản. Hai file này đã cover ~80% tính năng cốt lõi.

