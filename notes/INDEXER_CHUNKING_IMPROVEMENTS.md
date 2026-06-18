# Gợi ý cải thiện indexer và chunking cho corpus nhiều slide

## Bối cảnh

Corpus hiện tại không chỉ gồm tài liệu dài như PDF/DOCX, mà có nhiều tài liệu dạng slide, infographic, bảng tính và ảnh OCR. Vì vậy không nên đánh giá chunking chỉ bằng tiêu chí "chunk càng dài càng tốt". Với slide, mỗi block thường là heading, bullet, caption hoặc một cụm ý ngắn. Chunk ngắn có thể hợp lý nếu nó giữ trọn một ý, một slide, hoặc một cụm bullet có quan hệ chặt chẽ.

Kết quả kiểm tra trên 6 file trong `data/test data`:

- 6/6 tài liệu đã `indexed`.
- MongoDB có 129 chunks.
- Qdrant có 129 points tương ứng.
- Không có chunk rỗng.
- Không có chunk vượt `target_token_count=512`.
- Chunker thật đang dùng `semantic_breakpoint_v1`, version `layout_v2_table_aware`.

Điểm cần cải thiện không phải là indexer fail, mà là cần quality gate tinh hơn cho loại tài liệu nhiều slide.

## Số liệu hiện tại

| Tài liệu | Chunks | Tiny `<50` | Small `50-99` | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| `ML_Metrics_CheatSheet.png` | 1 | 0 | 0 | Tốt, OCR infographic được gom đủ ngữ cảnh |
| `ML_Roadmap_Infographic.png` | 1 | 0 | 0 | Tốt, một ảnh thành một chunk hợp lý |
| `ML_Starter_Pack_Slides.pptx` | 53 | 9 | 39 | Nhiều chunk nhỏ, nhưng có thể chấp nhận một phần do slide/bullet ngắn |
| `ML_Study_Workbook.xlsx` | 17 | 0 | 1 | Tốt, bảng được giữ khá đầy đủ |
| `ML_Tai_lieu_hoc_20_trang.docx` | 50 | 1 | 25 | Có hơi vụn, cần kiểm tra theo section |
| `rag_mau_hoc_tap.pdf` | 7 | 0 | 6 | Tài liệu ngắn, chunk nhỏ có thể hợp lý nếu citation tốt |

## Nguyên tắc đánh giá mới

### Không xem mọi chunk nhỏ là lỗi

Chunk `<100` token vẫn có thể hợp lệ khi:

- Nó là một slide ngắn nhưng đủ heading và bullet chính.
- Nó là caption hoặc nội dung OCR ngắn.
- Nó là một hàng/bảng nhỏ cần giữ nguyên cấu trúc.
- Nó chứa khái niệm tự đủ nghĩa, ví dụ định nghĩa metric hoặc checklist ngắn.
- Retrieval trả về chunk đó cùng citation đúng trang/block.

Chunk nhỏ chỉ nên bị xem là vấn đề khi:

- Nội dung bị cụt, mất heading hoặc mất ngữ cảnh.
- Nhiều chunk liên tiếp cùng slide/section bị tách rời quá mức.
- Chunk chỉ chứa footer, số trang, tiêu đề lặp, hoặc bullet rời không đủ nghĩa.
- Retrieval lấy chunk đúng keyword nhưng answer vẫn thiếu bằng chứng.

## Cải thiện ưu tiên

### 0. Làm rõ khi nào dùng OCR, khi nào dùng VLM

Pipeline nên dùng OCR và VLM theo vai trò khác nhau, không thay thế hoàn toàn cho nhau.

OCR phù hợp khi mục tiêu chính là trích xuất chữ chính xác:

- Ảnh scan tài liệu in.
- Trang PDF scan có nhiều đoạn văn.
- Screenshot có nhiều text UI, bảng, checklist.
- Hình chứa công thức, số liệu, nhãn ngắn cần giữ nguyên.
- Trường hợp cần bbox, confidence và citation theo vùng chữ.

VLM phù hợp khi mục tiêu chính là hiểu bố cục hoặc mô tả hình:

- Infographic nhiều cột, nhiều mũi tên, flow/process.
- Chart/diagram cần diễn giải quan hệ giữa các thành phần.
- Figure trong DOCX/PPTX/PDF không có text extraction tốt.
- Ảnh mà OCR đọc được chữ nhưng sai reading order.
- Slide hoặc hình có nội dung trực quan quan trọng ngoài chữ.

Không nên dùng VLM làm parser chính cho tài liệu text dài, vì VLM dễ diễn giải thêm, thiếu bbox, khó đảm bảo citation chính xác từng dòng. Không nên chỉ dùng OCR cho infographic phức tạp, vì OCR thường đọc sai thứ tự cột/nhánh và làm mất quan hệ layout.

Rule đề xuất:

| Loại input | Parser chính | Fallback/phụ trợ | Lý do |
| --- | --- | --- | --- |
| DOCX/PPTX text | Docling | VLM/OCR cho figure thiếu caption | Text/layout có cấu trúc sẵn |
| PDF text/layout | Docling | OCR cho trang scan/missing text | Giữ page/block/citation tốt |
| PDF scan | OCR | VLM cho figure/diagram phức tạp | Cần text chính xác và bbox |
| PNG/JPG scan văn bản | OCR | VLM nếu OCR quality thấp hoặc reading order rối | OCR tốt cho chữ, VLM hỗ trợ layout |
| PNG/JPG infographic | VLM-first | OCR để lấy text raw và kiểm chứng | VLM hiểu bố cục tốt hơn |
| Chart/diagram/figure | VLM-first | OCR fallback lấy nhãn/text | Cần mô tả quan hệ trực quan |
| XLSX/CSV | Spreadsheet parser | Không cần OCR/VLM | Dữ liệu có cấu trúc |

Với mỗi block nên lưu rõ metadata:

- `parse_method`: `docling`, `spreadsheet`, `ocr`, `vlm`, hoặc `hybrid`.
- `caption_source`: `vlm` hoặc `ocr` cho figure.
- `ocr_quality` nếu có OCR.
- `vlm_model` nếu có VLM.
- `fallback_reason` khi chuyển từ VLM sang OCR hoặc ngược lại.

Điều này giúp debug rõ một chunk đến từ đâu và vì sao chất lượng retrieval thay đổi.

### 1. Thêm slide-aware chunking

Với PPTX, nên coi slide là đơn vị layout quan trọng hơn paragraph đơn lẻ.

Đề xuất:

- Gom heading + subtitle + bullet cùng slide trước khi semantic split.
- Không tách một slide thành nhiều chunk nếu tổng token của slide <= 250.
- Nếu một slide dài, split theo cụm bullet hoặc vùng layout, không split từng bullet nhỏ.
- Giữ metadata `slide_number`, `source_pages`, `block_ids`, và heading của slide trong chunk.
- Loại hoặc hạ trọng số footer như `ML Starter Pack • 10`.

Mục tiêu không phải giảm số chunk bằng mọi giá, mà là mỗi chunk slide phải đại diện cho một ý trình chiếu hoàn chỉnh.

### 2. Merge theo slide/section thay vì merge mù theo token

Pass merge sau semantic split nên ưu tiên quan hệ layout:

- Merge chunk `<50` với neighbor cùng slide.
- Merge chunk `50-99` nếu neighbor cùng heading hoặc cùng section.
- Không merge qua slide nếu hai slide có chủ đề khác nhau.
- Không merge table với paragraph nếu table đang là chunk riêng có cấu trúc tốt.
- Cho phép vượt nhẹ `max_blocks_per_chunk=8` với PPTX khi các block là bullet rất ngắn, nhưng vẫn giữ `target_token_count=512`.

Điều này phù hợp hơn với slide vì nhiều block ngắn không đồng nghĩa với nhiễu.

### 3. Dùng ngưỡng quality riêng theo loại tài liệu

Không nên dùng cùng ngưỡng cho PPTX, DOCX, PDF, XLSX và ảnh.

Ngưỡng đề xuất:

| Loại file | Tiny `<50` | Small `50-99` | Ghi chú |
| --- | ---: | ---: | --- |
| PPTX | Cảnh báo nếu >20% | Cảnh báo nếu >80% | Chấp nhận nhiều chunk nhỏ nếu mỗi chunk đủ heading/slide |
| DOCX/PDF dài | Cảnh báo nếu >5% | Cảnh báo nếu >50% | Nên merge theo heading/section |
| XLSX/CSV | Không áp cứng | Không áp cứng | Ưu tiên giữ table/row semantic |
| PNG/JPG OCR | Không áp cứng | Cảnh báo nếu OCR bị rời rạc | Ưu tiên chất lượng OCR và bounding box |

Với dữ liệu hiện tại, PPTX có nhiều chunk nhỏ nhưng không nhất thiết là lỗi. Cần sample nội dung chunk để phân biệt slide chunk hợp lệ và chunk nhiễu.

### 4. Thêm chỉ số quality có ý nghĩa hơn token count

Ngoài token count, nên đo thêm:

- Tỷ lệ chunk có heading hoặc slide title.
- Tỷ lệ chunk chỉ chứa footer/page number.
- Số chunk trung bình trên mỗi slide.
- Tỷ lệ chunk có `source_pages` hợp lệ.
- Tỷ lệ chunk có evidence block đầy đủ.
- Retrieval hit rate theo câu hỏi kiểm thử.
- Citation validity và false refusal rate trong e2e eval.

Với corpus slide, `chunks_per_slide` và `has_slide_title` quan trọng hơn `small_50_99`.

### 5. Cập nhật diagnostic script

Script `backend/scripts/check_indexed_test_data.py` nên được mở rộng:

- In thêm extension/type của tài liệu.
- Với PPTX: báo `chunks_per_slide`, số chunk thiếu heading, số chunk chỉ footer.
- Với DOCX/PDF: báo chunk nhỏ liên tiếp trong cùng section.
- Với XLSX: báo modality `table`/`paragraph`.
- Với image OCR: báo OCR quality nếu có metadata.

Lệnh chạy:

```powershell
cd backend
$env:PYTHONIOENCODING='utf-8'
python scripts\check_indexed_test_data.py
```

### 6. Cập nhật integration test theo dataset hiện tại

`backend/tests/integration/test_sample_corpus_smoke.py` đang assert các file mẫu cũ, trong khi `data/test data` hiện là bộ `ML_*` và `rag_mau_hoc_tap.pdf`.

Nên cập nhật test để:

- Không hard-code tên file cũ đã bị xóa.
- Có expected behavior riêng cho từng loại file hiện tại.
- Với PPTX, không fail chỉ vì nhiều chunk nhỏ.
- Fail nếu có chunk rỗng, Qdrant mismatch, hoặc chunk slide chỉ chứa footer/page number.

## Tiêu chí đạt hợp lý

Một pipeline tốt cho corpus nhiều slide nên đạt:

- 100% tài liệu index thành công.
- `mongo_chunks == qdrant_points`.
- `empty=0`.
- `over_512=0`.
- PPTX có chunk gắn với slide title hoặc cụm bullet rõ nghĩa.
- DOCX/PDF không bị tách section thành nhiều mảnh thiếu ngữ cảnh.
- XLSX giữ được bảng dưới dạng table chunk khi cần.
- Ảnh OCR không bị mất nội dung chính.
- Retrieval/e2e eval giảm false refusal và citation vẫn đúng.

## Thứ tự triển khai đề xuất

1. Cập nhật diagnostic để phân loại theo file type, đặc biệt PPTX.
2. Sample 10 chunk nhỏ nhất của PPTX/DOCX để xác định chunk nhỏ nào là hợp lệ.
3. Thêm slide-aware merge trong semantic chunker.
4. Cập nhật integration test theo dataset `ML_*`.
5. Reindex 6 tài liệu test data.
6. So sánh trước/sau bằng diagnostic và chạy retrieval/e2e eval.

## Kết luận

Indexer hiện đã đúng về mặt coverage. Việc cần làm tiếp theo là cải thiện chunk quality theo đặc thù tài liệu. Với corpus nhiều slide, mục tiêu không phải xóa hết chunk nhỏ, mà là đảm bảo chunk nhỏ vẫn có đủ ngữ cảnh, đủ evidence, và hữu ích cho retrieval.
