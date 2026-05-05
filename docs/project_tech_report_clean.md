# Báo Cáo Kỹ Thuật Tổng Hợp

> Phiên bản này trình bày theo hướng kỹ thuật và sản phẩm, không nhắc tên file mã nguồn trong phần nội dung chính.  
> Mục tiêu là giúp người đọc hiểu hệ thống đang dùng kỹ thuật gì, nó giải quyết vấn đề gì, và hiện đang ở mức nào.

## 1) Tổng Quan

Hệ thống là một nền tảng **Graph RAG chotài liệu học tập** với các mục tiêu chính:

- ingest nhiều định dạng tài liệu,
- giữ được layout và evidence trace,
- chunk theo cấu trúc thay vì cắt cứng,
- retrieval hybrid dense + sparse,
- có graph reasoning cho câu hỏi đa tài liệu,
- có refusal/guardrail,
- có memory hội thoại ngắn hạn,
- và có giao diện workspace để xem nguồn chứng cứ.

Đánh giá tổng quát:

- Các kỹ thuật lõi cho ingestion, chunking, retrieval, rerank, citation và memory đều đã có.
- Chất lượng scan/OCR là điểm yếu rõ nhất.
- Mức độ sản phẩm hóa còn cần siết ở auth, observability, eval và quality gating.

## 2) Kiến Trúc Hệ Thống

Luồng xử lý chính:

1. Người dùng upload tài liệu vào workspace.
2. Hệ thống kiểm tra định dạng, dung lượng, magic bytes và duplicate.
3. Tài liệu đi vào pipeline parse theo loại đầu vào.
4. Block được normalize để ổn định layout, language và reading order.
5. Evidence được sinh ở mức block.
6. Chunk được tạo theo layout và ngưỡng token.
7. Graph metadata được trích xuất song song.
8. Embedding dense và sparse được lưu vào vector store.
9. Khi hỏi, hệ thống truy xuất scoped retrieval, rerank, kiểm tra confidence, rồi sinh câu trả lời có citation.
10. Memory hội thoại được bơm vào prompt nếu có ngữ cảnh trước đó.

Sơ đồ kiến trúc rút gọn:

```text
User Upload
   |
   v
File Validation & Deduplication
   |
   v
Parser / OCR / Spreadsheet Parser
   |
   v
Layout Normalization
   |
   v
Evidence Block Generation
   |
   v
Layout-aware Chunking
   |
   v
Dense + Sparse Embedding
   |
   v
Vector Store + Metadata Store
   |
   v
Hybrid Retrieval
   |
   v
Graph Expansion
   |
   v
Reranking
   |
   v
Confidence Check / Refusal
   |
   v
LLM Answer + Citation
   |
   v
Workspace UI
```

### Bảng hiện trạng hệ thống

| Thành phần | Hiện trạng | Đánh giá | Cần cải thiện |
|---|---|---|---|
| Ingestion | Có validate file, checksum, job stage | Tốt cho MVP | Cần auth/audit log mạnh hơn |
| Parsing | Có layout-aware parser, OCR, spreadsheet parser | Khá tốt | Scan tiếng Việt còn yếu |
| Chunking | Chunk theo layout + semantic breakpoint | Tốt | Cần benchmark thêm |
| Retrieval | Hybrid dense + sparse + rerank | Tốt | Cần đo Recall@K/MRR |
| Citation | Có evidence trace theo block/page/bbox | Rất quan trọng | Cần kiểm tra citation accuracy |
| Guardrail | Confidence check, refusal, verifier | Khá tốt | Cần tuning threshold |
| Frontend | Workspace 3 cột, evidence panel, graph panel | Tốt cho demo | Cần loading/error state |

### Stack kỹ thuật

| Lớp hệ thống | Công nghệ đang dùng | Vai trò |
|---|---|---|
| Backend API | FastAPI | Upload, query, inference, admin |
| Metadata store | MongoDB + Beanie | Document, chunk, graph, memory, logs |
| Vector store | Qdrant | Dense/sparse vectors và payload metadata |
| Async jobs | Celery + Redis | Parse/index nền |
| Embedding model | BGE-M3 | Dense + sparse text embeddings |
| OCR | PaddleOCR | Đọc ảnh scan/chữ in |
| Parser | Docling + spreadsheet parser | Parse tài liệu có cấu trúc |
| Frontend | React + Vite + React Router | Workspace, chat, evidence, graph |
| LLM | Local/OpenAI-compatible LLM | Sinh câu trả lời grounded |

### Tech stack đầy đủ theo lớp

| Lớp | Công nghệ | Vai trò trong hệ thống | Ghi chú |
|---|---|---|---|
| Runtime ngôn ngữ | Python | Toàn bộ backend, xử lý tài liệu, AI pipeline | Phù hợp hệ sinh thái NLP/RAG |
| Backend web | FastAPI | API upload, query, job status, inference | Hợp async và schema type-safe |
| ASGI server | Uvicorn | Chạy API backend | Dùng cho môi trường dev/prod nhẹ |
| Data model | Pydantic | Validate request/response và schema nội bộ | Giảm lỗi kiểu dữ liệu |
| ODM metadata | Beanie | Mapping document MongoDB | Hợp dữ liệu bán cấu trúc |
| Metadata DB | MongoDB | Lưu document, chunk, graph, memory, logs | Linh hoạt với schema thay đổi |
| Vector DB | Qdrant | Lưu dense/sparse vector và payload | Phục vụ hybrid retrieval |
| Job queue | Celery | Parse/index async | Tách xử lý nặng khỏi request |
| Broker/cache | Redis | Queue backend và cache | Hỗ trợ job pipeline |
| File/storage | Local filesystem | Lưu raw file, page image, artifact | Dễ quản lý trong MVP |
| Document parser | Docling | Parse PDF/DOCX/PPTX giữ layout, figures, tables | Nền cho evidence trace |
| Spreadsheet parser | Custom spreadsheet parser | Parse Excel/CSV theo sheet/table | Hữu ích cho dữ liệu bảng |
| OCR engine | PaddleOCR | Đọc scan/chữ in | Nhánh OCR chính cho ảnh scan |
| Image quality gate | Blur/skew/brightness checks | Chặn ảnh mờ hoặc quá nhiễu | Dùng trước OCR/handwriting |
| Handwriting pipeline | Handwriting reader / VLM fallback | Xử lý ảnh viết tay rõ nét | Chỉ dùng khi confidence đạt ngưỡng |
| Embedding model | BGE-M3 | Dense + sparse embeddings | Phù hợp hybrid multilingual retrieval |
| Retriever | Hybrid dense + sparse retriever | Lấy evidence ứng viên | Tối ưu recall |
| Reranker | Cross-encoder reranker | Xếp lại evidence | Tối ưu precision |
| Graph layer | Evidence graph / entity extractor | Liên kết khái niệm, block, page, relation | Hỗ trợ multi-hop reasoning |
| Guardrail / verifier | Confidence check, refusal, verifier | Chặn trả lời thiếu evidence | Tăng groundedness |
| LLM layer | Local hoặc OpenAI-compatible LLM | Sinh câu trả lời, tóm tắt, rewrite | Không làm parse/OCR core |
| Frontend framework | React | Xây UI workspace | Tách component rõ ràng |
| Frontend build tool | Vite | Dev server và build nhanh | Phù hợp app React hiện đại |
| Frontend router | React Router | Điều hướng workspace | Hợp app nhiều màn hình |
| UI layout | Workspace 3 cột | Chat, evidence, graph cùng lúc | Mạnh cho demo và kiểm chứng |
| Testing | pytest | Unit/integration/regression | Khóa hành vi pipeline |
| Containerization | Docker / Docker Compose | Chạy services cục bộ | Dễ tái lập môi trường |
| Experiment corpus | Sample document corpus | Smoke test pipeline | Dùng làm regression baseline |

### Vì sao chọn công nghệ này thay vì phương án khác

| Quyết định | Chọn gì | Không chọn gì | Lý do |
|---|---|---|---|
| Backend API | FastAPI | Flask/Django | FastAPI hợp async, type-safe schema và phù hợp pipeline nhiều I/O như upload, retrieval, OCR, job status. |
| Metadata store | MongoDB + Beanie | PostgreSQL/SQLAlchemy | Dữ liệu tài liệu là bán cấu trúc, cần pages, blocks, evidence, graph và memory lồng nhau; document model tự nhiên hơn relational schema cứng. |
| Vector store | Qdrant | Milvus/Weaviate/FAISS thuần | Qdrant đơn giản, dễ chạy local, hỗ trợ dense + sparse payload tốt cho hybrid retrieval của dự án. |
| Embedding | BGE-M3 | Dense-only embedding | Dự án cần đa ngôn ngữ, đa granularity, và lexical + semantic trong cùng một model để hỗ trợ tài liệu học tập song ngữ. |
| Parsing tài liệu | Docling | Unstructured thuần text | Docling mạnh ở layout, table, reading order và structured output, phù hợp tài liệu học tập hơn parser chỉ trả text phẳng. |
| OCR | PaddleOCR | Tesseract/EasyOCR | PaddleOCR có pipeline hiện đại hơn cho tài liệu scan, hỗ trợ layout/document parsing tốt và thường ổn hơn trong bài toán tiếng Việt. |
| Chunking | Layout-aware + semantic breakpoint | Fixed-size token chunking | Cắt cứng token dễ phá heading/table; chunk theo layout giữ ngữ cảnh và citation tốt hơn. |
| Retrieval | Hybrid dense + sparse + rerank | Vector-only retrieval | Vector-only dễ miss thuật ngữ chính xác; hybrid retrieval tăng recall và rerank tăng precision. |
| Graph reasoning | Evidence graph nội bộ | Không dùng graph | Câu hỏi so sánh, liên hệ nhiều file, nhiều khái niệm cần lớp graph để mở rộng câu trả lời. |
| Async jobs | Celery + Redis | Thread nền trong request | Upload/index nặng không nên chạy inline; Celery cho phép tách job và theo dõi stage rõ. |
| Frontend | Workspace 3 cột | Chat UI đơn giản | Đồ án cần xem nguồn, evidence, compare và graph cùng lúc; workspace giúp chứng minh năng lực hệ thống. |

### Ghi chú lập luận

- Điểm quan trọng của stack này không phải mỗi công nghệ đều “mạnh nhất”, mà là **phù hợp nhất cho bài toán tài liệu học tập có citation và graph reasoning**.
- Các lựa chọn ưu tiên:
  - dễ chạy local,
  - dễ debug,
  - có thể chứng minh bằng evidence,
  - và có đường nâng cấp lên production.

### Vì sao không dùng một LLM làm tất cả

Hệ thống này không chọn kiến trúc “một LLM xử lý mọi bước” vì bài toán document intelligence cần cả độ chính xác lẫn độ ổn định:

- **Các tác vụ xác định không nên giao hết cho LLM**: parse layout, tách block, trích bbox, đọc bảng, OCR, dedup và scope filter là những bước cần kết quả ổn định và kiểm thử được.
- **Chi phí và latency sẽ tăng mạnh** nếu mọi bước đều gọi LLM, đặc biệt với tài liệu dài hoặc corpus lớn.
- **Parser/OCR chuyên dụng thường tốt hơn** cho việc đọc ảnh scan, bảng biểu và layout phức tạp so với việc để LLM tự suy đoán từ raw input.
- **Evidence trace sẽ yếu đi** nếu không có block/page/bbox rõ ràng; trong khi dự án cần citation kiểm chứng được.
- **Khó kiểm thử và khó debug** nếu mọi thứ nằm trong một black box duy nhất.

Vì vậy kiến trúc hiện tại tách vai trò rõ ràng:

- **LLM** dùng cho các bước cần suy luận và diễn đạt:
  - query rewriting
  - answer synthesis
  - summary
  - study guide
- **Module chuyên dụng** dùng cho các bước cần chính xác và ổn định:
  - parsing
  - OCR
  - chunking
  - embedding
  - retrieval
  - reranking
  - citation

Kết luận kỹ thuật:

> Không dùng một LLM làm tất cả vì hệ thống cần vừa có khả năng suy luận ngôn ngữ, vừa có độ ổn định, chi phí thấp và evidence trace rõ ràng. Kiến trúc lai giúp giữ được tính xác thực của dữ liệu, đồng thời vẫn tận dụng LLM ở những chỗ thật sự cần sinh ngôn ngữ và điều phối.

## 3) Ingestion Và An Toàn Dữ Liệu

### Kỹ thuật

- Validate định dạng bằng magic bytes và allowlist extension.
- Stream upload ra tệp tạm thay vì đọc toàn bộ vào RAM.
- Dùng checksum để chặn file trùng trong cùng scope.
- Lưu raw file và parsed artifact tách biệt.
- Quản lý trạng thái job theo stage rõ ràng.

### Nó giải quyết gì

- Tránh upload sai kiểu file.
- Giảm nguy cơ path traversal và lỗi bộ nhớ.
- Cho phép debug pipeline theo từng stage.

### Đánh giá

**Điểm mạnh**

- Có kiểm soát đầu vào.
- Có trạng thái job để theo dõi.
- Có khả năng tái hiện lỗi theo từng bước.

**Điểm yếu**

- Cơ chế xác thực người dùng chưa đủ mạnh ở mức production.
- Quan sát lỗi vẫn còn thiên về log hơn là telemetry.

**Mức sản phẩm**

- Tốt cho MVP
- Cần hardening thêm để chạy production thật

## 4) Parsing Tài Liệu

### 4.1 Parse layout-aware cho tài liệu văn bản

Hệ thống dùng parser layout-aware để giữ cấu trúc của tài liệu học tập như heading, bảng, đoạn văn và slide.

**Điểm mạnh**

- Giữ layout tốt hơn parser thuần text.
- Tạo đầu vào sạch cho chunking và citation.

**Điểm yếu**

- Phụ thuộc cache/model runtime.
- Scan chất lượng thấp vẫn cần hỗ trợ thêm.

### 4.2 Parse bảng tính

Tài liệu bảng tính được chuyển thành:

- summary của sheet,
- bảng markdown,
- và row-level verbalization.

**Điểm mạnh**

- Retrieval tốt hơn cho câu hỏi vào dữ liệu bảng.
- Có cả mô tả ngắn và nội dung chi tiết.

### 4.3 OCR cho ảnh scan

Ảnh scan/chữ in được xử lý bằng OCR CPU-friendly, có routing theo ngôn ngữ và có thể thử biến thể grayscale.

**Điểm mạnh**

- Hỗ trợ ingest ảnh scan, không chỉ tài liệu text.
- Có bbox và confidence cho citation.

**Điểm yếu**

- Với scan tiếng Việt, output có thể nhiễu.
- Runtime khá nặng và phụ thuộc môi trường model.

### 4.4 Xử lý ảnh viết tay

Ảnh viết tay đi qua quality gate trước khi nhận làm evidence.

Chất lượng ảnh được chấm theo:

- blur,
- brightness,
- contrast,
- skew.

**Điểm mạnh**

- Chặn rác trước khi index.
- Không coi mọi ảnh viết tay là evidence.

**Mức sản phẩm**

- Tốt về guardrail
- Phụ thuộc chất lượng input

### 4.5 Trích xuất ảnh nhúng trong DOCX/PDF/PPTX

Phần này cần phân biệt rõ giữa hai bài toán:

- **OCR ảnh scan**: đọc chữ trong ảnh chụp hoặc scan.
- **Trích xuất ảnh nhúng / figure / illustration**: lấy chính đối tượng ảnh, caption, vị trí trang và metadata từ tài liệu gốc.

Trong kế hoạch hiện tại, **không có một repo chuyên biệt nào được ghi rõ chỉ để trích xuất ảnh nhúng** từ DOCX/PDF/PPTX. Hướng phù hợp nhất là kết hợp các công cụ sau:

- **Docling**: phù hợp nhất cho PDF/DOCX/PPTX vì giữ layout, reading order, tables, figures và metadata. Đây là nguồn gần nhất cho bài toán trích xuất hình ảnh kèm ngữ cảnh.
- **Unstructured**: hữu ích cho ETL và preprocessing đa định dạng, đặc biệt khi cần chuẩn hóa tài liệu trước khi đi vào pipeline.
- **RAG-Anything**: tham khảo cho hướng multimodal RAG, khi muốn xử lý text + image + table + equation trong cùng một kiến trúc.
- **PaddleOCR**: không phải công cụ lấy figure nhúng, nhưng hữu ích khi ảnh scan hoặc slide scan chứa chữ in cần đọc lại phần text đi kèm hình.

Áp dụng trong báo cáo này:

- Với **DOCX/PPTX/PDF có ảnh nhúng**, hệ thống ưu tiên parse layout trước, rồi lưu lại:
  - `page`
  - `block_id`
  - `figure/caption`
  - `bbox` nếu parser cung cấp
  - đường dẫn ảnh đã render hoặc ảnh trích xuất nếu có
- Với **ảnh scan**, hệ thống đi theo nhánh OCR thay vì coi đó là embedded image extraction.
- Với **multimodal retrieval đầy đủ**, việc index ảnh bằng image embedding như `jina-clip-v2` hoặc ColPali được xem là **stretch goal**, không phải core MVP.

Kết luận kỹ thuật:

- Nếu mục tiêu là **lấy ảnh nhúng ra kèm ngữ cảnh**, chọn **Docling** làm nguồn tham chiếu chính.
- Nếu mục tiêu là **multimodal document understanding** rộng hơn, dùng thêm **RAG-Anything** và chỉ mở rộng sang image embedding ở giai đoạn sau.

## 5) Chuẩn Hóa Layout Và Evidence

### Kỹ thuật

- Sort block theo reading order.
- Merge các dòng OCR rời rạc khi phù hợp.
- Merge fragment text ngắn.
- Chuẩn hóa block type như heading, list, table, paragraph.
- Detect language ở mức block và document.
- Sinh evidence block theo từng block nguồn.

### Nó giải quyết gì

- Làm cho đầu vào ổn định hơn trước chunking.
- Giữ được trace từ câu trả lời về đúng nguồn gốc.

### Đánh giá

**Điểm mạnh**

- Đây là lớp nền rất quan trọng của pipeline.
- Tăng chất lượng citation và retrieval.

**Mức sản phẩm**

- Đang ở trạng thái đúng vai trò và đáng tin cậy

## 6) Chunking

### 6.1 Chunk theo layout

Chunking mặc định tôn trọng:

- heading boundary,
- token budget,
- overlap,
- split block dài,
- table boundary,
- và metadata trace.

### 6.2 Chunk theo semantic breakpoint

Khi bật chế độ semantic, hệ thống có thể dùng embedding của block để tìm điểm ngắt theo ngữ nghĩa, nhưng vẫn tôn trọng hard break của layout.

### Đánh giá

**Điểm mạnh**

- Phù hợp tài liệu học thuật và slide.
- Không cắt cứng theo token một cách thô.

**Điểm yếu**

- Rule-based vẫn có thể split chưa tối ưu trên tài liệu lộn xộn.
- Semantic mode phụ thuộc embedder.

**Mức sản phẩm**

- Tốt cho MVP
- Semantic mode là nâng cao, nên dùng có kiểm soát

## 7) Indexing Và Vector Store

### Kỹ thuật

- Lưu dense vector và sparse vector trên cùng point.
- Metadata/chunk được lưu vào document database.
- Có payload để filter theo owner, collection, material, language, modality.
- Có versioning cho parse, chunk, embedding và index.

### Nó giải quyết gì

- Cho phép hybrid retrieval.
- Cho phép re-index khi đổi model hoặc chunk strategy.

### Đánh giá

**Điểm mạnh**

- Có cấu trúc version rõ.
- Hybrid retrieval đúng hướng cho tài liệu học tập.

**Mức sản phẩm**

- Tốt cho MVP

## 8) Retrieval Core

### 8.1 Hybrid retrieval

Truy xuất kết hợp:

- dense semantic search,
- sparse lexical search,
- metadata scope filter,
- và evidence hydration.

### 8.2 Graph retrieval

Hệ thống có thêm lớp graph retrieval để mở rộng câu hỏi đa bước, đa tài liệu, và câu hỏi cần liên hệ khái niệm.

### 8.3 Reranking

Candidate chunks sau retrieval được rerank lại để tăng precision trước khi đưa vào prompt.

### 8.4 Query rewriting và routing

Query được xử lý thêm qua:

- intent classification,
- language normalization,
- multi-query routing,
- graph routing,
- và refusal routing.

### Đánh giá

**Điểm mạnh**

- Hybrid retrieval là một lựa chọn rất hợp lý cho tài liệu học tập.
- Graph retrieval giúp cross-document reasoning tốt hơn.
- Rerank làm giảm nhiễu rõ rệt.

**Điểm yếu**

- Có nhiều lớp nên runtime nặng hơn retrieval đơn giản.
- Phụ thuộc model và config để đạt chất lượng tốt.

**Mức sản phẩm**

- Tốt
- Đã vượt mức prototype

## 9) Guardrails, Refusal, Và Citation

### Kỹ thuật

- Score confidence từ retrieval/rerank.
- Từ chối nếu evidence quá yếu.
- Verifier kiểm tra claim so với evidence.
- Detector kiểm tra mâu thuẫn trong luồng compare.
- Citation luôn bám về block/page/bbox khi có.

### Nó giải quyết gì

- Giảm hallucination.
- Làm cho kết quả có thể kiểm chứng.

### Đánh giá

**Điểm mạnh**

- Có lớp bảo vệ hậu kiểm.
- Citation trace đủ mạnh để demo và review.

**Điểm yếu**

- Threshold cần tuning để tránh từ chối quá gắt.
- Không phải mọi luồng đều đã được wiring đầy đủ như nhau.

**Mức sản phẩm**

- Tốt nhưng vẫn cần tinh chỉnh

## 10) Window Context Memory

### Kỹ thuật

- Lấy các lượt gần nhất trong cùng owner, collection và conversation.
- Có summary memory cho các lượt cũ hơn.
- Bơm vào prompt như một phần riêng.
- Giới hạn độ dài để tránh prompt phình quá mức.

### Nó giải quyết gì

- Giúp câu hỏi nối tiếp hiểu đúng ngữ cảnh.
- Hợp với kiểu hỏi “cái đó”, “nó”, “tiếp theo”.

### Đánh giá

**Điểm mạnh**

- Scope rõ.
- Không trộn hội thoại giữa các workspace.

**Điểm yếu**

- Fallback khá im lặng nếu memory store lỗi.
- Còn dựa nhiều vào giới hạn ký tự hơn là token-aware budget.

**Mức sản phẩm**

- Ổn cho MVP
- Nên thêm test và observability

## 11) Inference Flow

### Kỹ thuật

- Intent classification.
- Off-topic refusal.
- Retrieval theo scope.
- Optional graph expansion.
- Rerank.
- Confidence scoring.
- Prompt synthesis.
- LLM generation.
- Citation injection.
- Claim verification.

### Nó giải quyết gì

- Biến retrieval thành câu trả lời grounded.
- Đảm bảo có cơ chế từ chối khi evidence yếu.

### Đánh giá

**Điểm mạnh**

- Flow đủ lớp để kiểm soát chất lượng.

**Mức sản phẩm**

- Tốt

## 12) Async Pipeline

### Kỹ thuật

- Job parse/index chạy nền.
- Có eager mode cho môi trường local/test.
- Mỗi job có status/stage và error trace.

### Nó giải quyết gì

- Tránh blocking upload request.
- Cho phép xử lý file nặng theo nền.

### Đánh giá

**Điểm mạnh**

- Dễ vận hành cho corpus lớn hơn.

**Mức sản phẩm**

- Tốt cho hệ thống quy mô nhỏ và vừa

## 13) Frontend Workspace

### Kỹ thuật

- Giao diện workspace ba cột.
- Có panel riêng cho chat, nguồn, evidence, graph, mindmap, compare.
- Có state persist cho workspace và materials.
- Có mobile slide-over và bottom tabs.

### Nó giải quyết gì

- Làm cho hệ thống có cảm giác như một product chứ không phải demo script.

### Đánh giá

**Điểm mạnh**

- Luồng thao tác rõ.
- Dễ trình bày và demo.

**Điểm yếu**

- local state vẫn có thể stale.
- Cần thêm skeleton/loading/error boundary ở vài chỗ nếu muốn polish hơn nữa.

**Mức sản phẩm**

- Khá tốt

## 14) Dữ Liệu Và Mô Hình

### Kỹ thuật

- Dùng ODM kiểu document để lưu metadata và graph.
- Lưu version trên tài liệu và chunk.
- Lưu query log và chat summary riêng.

### Nó giải quyết gì

- Phù hợp dữ liệu bán cấu trúc của tài liệu học tập.
- Cho phép trace đầy đủ từ query đến evidence.

### Đánh giá

**Điểm mạnh**

- Mô hình dữ liệu linh hoạt.

**Điểm yếu**

- Cần chú ý index và query tối ưu nếu dữ liệu lớn hơn.

**Mức sản phẩm**

- Khá tốt

## 15) Test Và Đánh Giá

### Kỹ thuật

- Unit test cho parser/chunking/indexer/retriever/inference.
- Integration test với MongoDB và vector store thật.
- Corpus smoke test cho bộ tài liệu mẫu.

### Nó giải quyết gì

- Khóa hành vi pipeline.
- Tránh regression khi đổi parser/chunker/retrieval.

### Đánh giá

**Điểm mạnh**

- Có test cho các lớp lõi.
- Có corpus smoke để làm baseline thực tế.

**Điểm yếu**

- Cần thêm benchmark retrieval end-to-end nhiều câu hỏi thật hơn.

### Metric đánh giá đề xuất

| Nhóm đánh giá | Metric | Ý nghĩa |
|---|---|---|
| Retrieval | Recall@5, Recall@10 | Chunk đúng có nằm trong top kết quả không |
| Reranking | MRR, nDCG@10 | Kết quả đúng có được xếp cao không |
| Citation | Citation Accuracy | Câu trả lời có trích đúng nguồn không |
| Answer Quality | Groundedness Score | Câu trả lời có bám evidence không |
| OCR | OCR Confidence | Độ tin cậy khi đọc scan |
| Refusal | False Refusal Rate | Có từ chối nhầm không |
| Performance | Latency/query | Mỗi câu hỏi mất bao lâu |

Diễn giải ngắn:

- Recall@K: trong K kết quả đầu có chứa đáp án đúng hay không.
- MRR: đáp án đúng càng đứng cao thì điểm càng tốt.
- Groundedness: mức độ câu trả lời bám vào evidence thay vì suy đoán.

### Ví dụ query thực tế

| Câu hỏi người dùng | Kỹ thuật được kích hoạt | Kết quả mong muốn |
|---|---|---|
| Tóm tắt chương 2 tài liệu này | Scoped retrieval + summarization | Tóm tắt có citation |
| So sánh khái niệm A và B trong hai file | Multi-document retrieval + graph reasoning | Bảng so sánh có nguồn |
| Công thức này nằm ở trang nào? | Sparse + dense retrieval | Trả về block/page chính xác |
| Nội dung này có trong tài liệu không? | Evidence check + refusal | Có/không, kèm nguồn |

### Rủi ro kỹ thuật

| Rủi ro | Ảnh hưởng | Hướng xử lý |
|---|---|---|
| OCR sai với scan mờ | Retrieval sai, citation sai | Quality gate + cảnh báo người dùng |
| Chunking sai ngữ cảnh | Câu trả lời thiếu ý | Layout-aware + semantic breakpoint |
| Retrieval lấy nhầm nguồn | LLM trả lời sai | Rerank + verifier |
| Threshold refusal chưa tốt | Từ chối quá nhiều hoặc trả lời khi thiếu nguồn | Eval tập câu hỏi chuẩn |
| Dữ liệu lớn làm chậm query | UX kém | Cache, batch indexing, async job |
| Scope user chưa chặt | Lộ dữ liệu workspace khác | Auth, RBAC, metadata filter bắt buộc |

### Roadmap phát triển

| Giai đoạn | Mục tiêu |
|---|---|
| Phase 1: MVP | Upload tài liệu, parse, chunk, search, chat có citation |
| Phase 2: Quality | Rerank, refusal, verifier, eval benchmark |
| Phase 3: Graph RAG | Entity graph, cross-document reasoning, compare mode |
| Phase 4: Production | Auth, logging, monitoring, quota, scaling |
| Phase 5: Learning Assistant | Flashcard, mindmap, quiz generation, study plan |

**Mức sản phẩm**

- Tốt cho nền tảng kỹ thuật
- Cần thêm eval sản phẩm

## 16) Kết Luận

Tổng thể, hệ thống không chỉ là một RAG chatbot thông thường mà là một nền tảng document intelligence cho tài liệu học tập. Điểm nổi bật nằm ở việc kết hợp layout-aware parsing, evidence-level citation, hybrid dense-sparse retrieval, reranking, graph reasoning và workspace UI để giúp người dùng không chỉ nhận câu trả lời mà còn kiểm chứng được nguồn gốc thông tin.

Ở trạng thái hiện tại, hệ thống phù hợp với mức MVP nâng cao, đủ để trình diễn các chức năng cốt lõi. Để tiến tới production-grade, các hướng cần ưu tiên gồm cải thiện OCR tiếng Việt, chuẩn hóa benchmark đánh giá retrieval, tăng cường authentication/authorization, bổ sung observability và tối ưu latency cho tập tài liệu lớn.

## 17) Tài Liệu Tham Khảo

### 17.1 Paper và preprint nên trích dẫn

Các nguồn dưới đây khớp nhất với những kỹ thuật đang dùng trong hệ thống:

1. Patrick Lewis et al., 2020. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**.  
   Dùng làm nguồn gốc cho tư duy RAG, retrieve-then-generate và grounded QA.  
   Link: https://nlp.cs.ucl.ac.uk/publications/2020-05-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks/

2. Vladimir Karpukhin et al., 2020. **Dense Passage Retrieval for Open-Domain Question Answering**.  
   Dùng cho dense retriever / semantic retrieval.  
   Link: https://nlp.cs.ucl.ac.uk/publications/2020-05-dense-passage-retrieval-for-open-domain-question-answering/

3. Jianlv Chen et al., 2024. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**.  
   Dùng cho hybrid dense + sparse embedding, đa ngôn ngữ, đa granularity.  
   Link: https://huggingface.co/BAAI/bge-m3

4. Nikolaos Livathinos et al., 2025. **Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion**.  
   Dùng cho document conversion, layout-aware parsing, table extraction và structured document representation.  
   Link: https://research.ibm.com/publications/docling-an-efficient-open-source-toolkit-for-ai-driven-document-conversion

5. Cheng Cui et al., 2025. **PaddleOCR 3.0 Technical Report**.  
   Dùng cho OCR pipeline, printed text recognition và document parsing trên ảnh scan.  
   Link: https://huggingface.co/papers/2507.05595

6. Darren Edge et al., 2024. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**.  
   Dùng làm nền tham khảo cho graph-based retrieval, entity graph và query-focused summarization.  
   Link: https://www.microsoft.com/en-us/research/project/graphrag/publications/

7. Zhengbao Jiang et al., 2023. **Active Retrieval Augmented Generation**.  
   Dùng để tham khảo retrieval/generation synergy và ý tưởng điều khiển retrieval động.  
   Link: https://aclanthology.org/2023.emnlp-main.495/

8. Zhihong Shao et al., 2023. **Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy**.  
   Dùng để tham khảo iterative retrieval, multi-step retrieval và cải thiện khó hỏi.  
   Link: https://aclanthology.org/2023.findings-emnlp.620/

9. Yuntong Hu et al., 2025. **GRAG: Graph Retrieval-Augmented Generation**.  
   Dùng để tham khảo graph retrieval ở mức subgraph retrieval và graph-context-aware generation.  
   Link: https://aclanthology.org/2025.findings-naacl.232/

10. Zengyi Gao et al., 2025. **FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs**.  
    Dùng để tham khảo modular graph RAG và cách tách các module graph/retrieval/generation.  
    Link: https://aclanthology.org/2025.findings-acl.321/

### 17.2 Paper gợi ý theo từng kỹ thuật

- RAG core, grounded QA: Lewis et al. 2020.
- Dense retrieval: Karpukhin et al. 2020.
- Hybrid embedding / multilingual retrieval: BGE-M3.
- Layout-aware parsing và structured conversion: Docling.
- OCR và document parsing: PaddleOCR 3.0 technical report.
- Graph retrieval / cross-document reasoning: GraphRAG, GRAG, FRAG.
- Iterative retrieval / retrieval-generation synergy: Active Retrieval Augmented Generation, Iter-RetGen.

### 17.3 Nguồn đối chiếu trực tiếp trong repo

- Tổng quan dự án trong `README.md`
- Kiến trúc hệ thống trong `docs/architecture.md`
- Kế hoạch triển khai trong `AgentBook_Implementation_Plan.md`
- Gợi ý kỹ thuật trong `docs/engineering_recommendations.md`
- Hướng dẫn kiểm thử trong `docs/testing_guide.md`
- Báo cáo audit tổng dự án trong `docs/full_project_audit.md`
- Code backend, frontend, tests và corpus mẫu hiện tại trong repo

## 18) Phụ Lục Chi Tiết Technical

Phần này bổ sung chi tiết triển khai có thể đối chiếu trực tiếp với cấu hình, API và module hiện tại của hệ thống. Mục tiêu là làm rõ hệ thống đang chạy như thế nào ở mức engineering, không chỉ mô tả theo hướng sản phẩm.

### 18.1 Runtime service và lifecycle backend

Backend được đóng gói quanh một ứng dụng FastAPI có lifecycle rõ:

1. Khi service khởi động, hệ thống nạp cấu hình logging từ YAML nếu có.
2. Settings được tổng hợp từ file cấu hình, biến môi trường và giá trị mặc định.
3. Backend khởi tạo kết nối metadata database.
4. Backend kiểm tra và tạo collection vector nếu chưa tồn tại.
5. Backend tạo payload index trong Qdrant cho các trường lọc quan trọng.
6. Khi service shutdown, hệ thống đóng query service, Qdrant client và kết nối database.

Các payload index quan trọng trong vector store:

| Payload field | Vai trò |
|---|---|
| `owner_id` | Bắt buộc để cô lập dữ liệu theo người dùng/workspace |
| `collection_id` | Scope retrieval theo collection |
| `material_id` | Scope retrieval theo tài liệu cụ thể |
| `language` | Lọc theo ngôn ngữ block/chunk |
| `modality` | Phân biệt text, table, figure, OCR, handwriting |
| `content_text` | Text index đa ngôn ngữ phục vụ lexical matching |

Qdrant collection dùng hai vector channel:

| Vector channel | Kiểu | Mục đích |
|---|---|---|
| `dense` | Dense vector 1024 chiều | Semantic retrieval |
| `bge_m3_sparse` | Sparse vector | Lexical/sparse retrieval |

Thiết kế này giúp retrieval không phụ thuộc hoàn toàn vào một tín hiệu. Dense vector bắt quan hệ ngữ nghĩa, sparse vector giữ tốt thuật ngữ chính xác, tên riêng, công thức, mã môn học và cụm từ trong slide.

### 18.2 API surface

API chính nằm dưới prefix `/api/v1`. Các nhóm endpoint hiện tại:

| Nhóm | Endpoint tiêu biểu | Chức năng |
|---|---|---|
| Collections | `/collections` | Tạo, liệt kê và quản lý workspace/collection |
| Materials | `/materials`, `/materials/upload`, `/materials/batch_upload` | Upload, batch upload, list tài liệu |
| Material status | `/materials/{material_id}/status` | Theo dõi stage parse/index |
| Material debug | `/materials/{material_id}/debug` | Xem page, block, bbox, chunk phục vụ debug |
| Query | `/query/ask` | Hỏi đáp grounded theo scope |
| Compare | `/query/compare` | So sánh nhiều tài liệu/chủ đề |
| Summary | `/query/summarize` | Tóm tắt có citation |
| Study guide | `/query/study-guide` | Sinh hướng dẫn học tập |
| Evidence | `/evidence/...` | Lấy evidence theo trang/block |
| Graph | `/graph`, `/graph/mindmap` | Lấy graph/mindmap từ entity và relation |
| Admin | `/admin/metrics`, `/admin/feedback` | Metrics và feedback |

Các endpoint query có rate limit ở những luồng tốn tài nguyên:

| Endpoint | Rate limit |
|---|---|
| `/query/ask` | 15 request/phút |
| `/query/compare` | 10 request/phút |

Điểm quan trọng về bảo mật dữ liệu là mọi request theo scope đều kiểm tra `owner_id`. Khi bật auth production, API key được dùng để tránh expose service trực tiếp ra internet khi chưa có lớp xác thực hoàn chỉnh.

### 18.3 Upload, validation và lưu trữ file

Luồng upload không đọc toàn bộ file vào RAM. File được stream vào vùng tạm, đồng thời tính checksum và đọc phần đầu file để validate. Sau đó service mới chuyển file sang storage chính.

Các cơ chế kiểm soát:

| Cơ chế | Chi tiết |
|---|---|
| Extension allowlist | `pdf`, `docx`, `pptx`, `png`, `jpg`, `jpeg`, `csv`, `xlsx`, `xls` |
| Giới hạn dung lượng | Mặc định 20 MB, lấy từ guardrail config |
| Magic bytes/head check | Giảm rủi ro file giả đuôi |
| Checksum SHA-256 | Phát hiện trùng file trong cùng scope |
| Scoped path | File được lưu theo vùng dữ liệu của hệ thống, tránh path traversal |
| Job record | Mỗi upload tạo pipeline job để theo dõi stage |

Trạng thái job được ánh xạ thành progress:

| Stage | Progress |
|---|---:|
| `uploaded` | 10% |
| `parsing` | 30% |
| `parsed` | 55% |
| `indexing` | 80% |
| `indexed` | 100% |
| `failed` | 100% |

### 18.4 Parse/index pipeline chi tiết

Pipeline parse/index chạy theo thứ tự:

1. Kiểm tra material và job còn tồn tại.
2. Chuyển status sang `parsing`.
3. Route parser theo loại file.
4. Caption figure nếu block cần caption và có vision model phù hợp.
5. Normalize layout, reading order, block type và metadata.
6. Detect language ở mức block/document.
7. Ghi page/block đã parse vào metadata store.
8. Ghi artifact đã xử lý để debug.
9. Tạo evidence map theo block.
10. Chunk theo layout hoặc semantic strategy.
11. Sinh QA nội bộ cho chunk nếu cần.
12. Contextual enrichment cho chunk nếu bật.
13. Trích xuất entity, resolve alias, trích xuất event/relation.
14. Chuyển status sang `indexing`.
15. Index chunk vào MongoDB và vector vào Qdrant.
16. Ghi entity/event/relation vào graph store.
17. Chuyển status sang `indexed`; nếu lỗi thì ghi `failed_stage` và `error_message`.

Routing parser:

| Loại input | Parser chính | Ghi chú |
|---|---|---|
| PDF/DOCX/PPTX | Docling parser | Ưu tiên layout, table, figure, reading order |
| CSV/XLS/XLSX | Spreadsheet parser | Sinh summary, markdown table và row-level verbalization |
| PNG/JPG/JPEG chữ in | OCR engine | Dùng OCR quality score và bbox |
| PNG/JPG/JPEG viết tay | Handwriting reader | Chỉ nhận evidence khi vượt quality/confidence gate |

Pipeline có cơ chế kiểm tra material giữa các stage để tránh index tiếp một tài liệu đã bị xóa trong lúc job đang chạy.

### 18.5 Cấu hình model và retrieval

Cấu hình hiện tại ưu tiên chạy được local trên CPU nhưng vẫn giữ đường nâng cấp sang GPU/API.

| Thành phần | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| LLM provider mặc định | `local` | Ưu tiên model local |
| Local model | `qwen2.5:3b` trong config, Docker override `qwen3:4b` | Sinh answer, summary, rewrite |
| OpenAI-compatible fallback | `gpt-4o-mini` | Dự phòng khi dùng API |
| Temperature | `0.1` | Giảm sáng tạo, tăng tính ổn định |
| Max output tokens | `1024` | Giới hạn độ dài trả lời |
| Embedding | `BAAI/bge-m3` | Dense + sparse, đa ngôn ngữ |
| Dense size | `1024` | Khớp Qdrant vector config |
| Embedding batch size | `8` | Phù hợp CPU/local |
| Reranker | `BAAI/bge-reranker-v2-m3` | Xếp lại candidate trước prompt |

Thông số retrieval chính:

| Tham số | Giá trị | Vai trò |
|---|---:|---|
| `dense_top_k` | 20 | Số candidate từ dense search |
| `sparse_top_k` | 20 | Số candidate từ sparse search |
| `graph_top_k` | 10 | Số candidate từ graph expansion |
| `final_top_k` | 5 | Evidence cuối đưa vào synthesis |
| `rrf_k` | 60 | Hằng số Reciprocal Rank Fusion |
| `rerank_input_k` | 15 | Số candidate đưa vào reranker |
| `graph_max_hops` | 2 | Giới hạn mở rộng graph |
| `query_rewriter_enabled` | true | Bật rewrite/multi-query |

Thông số chunking:

| Tham số | Giá trị | Vai trò |
|---|---:|---|
| `target_token_count` | 512 | Kích thước chunk mục tiêu |
| `overlap_token_count` | 50 | Giữ ngữ cảnh giữa chunk |
| `max_blocks_per_chunk` | 8 | Chặn chunk gom quá nhiều block |
| `breakpoint_percentile` | 95 | Ngưỡng semantic breakpoint nếu dùng semantic chunking |

### 18.6 Data model metadata

Metadata store dùng document model vì dữ liệu tài liệu có cấu trúc linh hoạt: page, block, bbox, graph, evidence, memory và query log có thể khác nhau theo loại file.

Các collection lõi:

| Collection | Nội dung chính |
|---|---|
| `materials` | Metadata tài liệu, checksum, status, version, storage path |
| `material_pages` | Page/block đã parse, bbox, reading order, OCR confidence |
| `chunks` | Chunk text, source block/page, model version, chunk strategy |
| `entities` | Entity canonical, alias, type, mention evidence |
| `events` | Event, participant, thời gian, evidence |
| `relations` | Edge giữa entity/event/concept, confidence, conflict flag |
| `pipeline_jobs` | Trạng thái parse/index background |
| `query_logs` | Log câu hỏi, retrieval, answer, confidence |
| `feedback` | Rating/comment của người dùng |
| `chat_memory` | Memory hội thoại theo conversation/workspace |

Trường trace quan trọng:

| Trường | Nơi dùng | Ý nghĩa |
|---|---|---|
| `owner_id` | Tất cả dữ liệu scoped | Cô lập người dùng |
| `collection_id` | Material/chunk/graph/query | Cô lập workspace |
| `material_id` | Page/chunk/evidence | Trỏ về tài liệu gốc |
| `page_number` / `page` | Page/evidence | Trỏ về trang |
| `block_id` | Block/evidence/chunk | Trỏ về block nguồn |
| `bbox` | Material block/evidence | Hiển thị vùng nguồn trên UI |
| `source_block_ids` | Chunk | Biết chunk được ghép từ block nào |
| `source_pages` | Chunk | Biết chunk trải trên trang nào |

### 18.7 Evidence, citation và confidence

Citation không được sinh tự do bởi LLM. Citation được nối từ retrieval result và evidence metadata. Một evidence tốt cần có:

- tài liệu nguồn,
- trang,
- block id,
- snippet gốc,
- bbox nếu parser/OCR cung cấp,
- confidence hoặc score từ retrieval/rerank.

Luồng confidence gồm nhiều tín hiệu:

| Tín hiệu | Vai trò |
|---|---|
| Retrieval score | Candidate có liên quan không |
| Reranker score | Evidence có thật sự khớp query không |
| OCR quality | Text scan có đáng tin để index không |
| Graph confidence | Edge/entity có đủ tin cậy không |
| Claim verifier | Claim trong answer có bám evidence không |
| Contradiction detector | Phát hiện mâu thuẫn trong compare/multi-doc |

Ngưỡng mặc định đáng chú ý:

| Ngưỡng | Giá trị | Ý nghĩa |
|---|---:|---|
| `min_reranker_score` | 0.35 | Dưới ngưỡng này evidence yếu |
| `min_evidence_confidence` | 0.35 | Confidence tối thiểu cho answer grounded |
| `min_graph_confidence` | 0.45 | Confidence tối thiểu cho graph evidence |
| `min_ocr_text_quality` | 0.35 | Dưới ngưỡng này OCR quá yếu |
| `warn_ocr_text_quality` | 0.55 | Cảnh báo chất lượng OCR |
| `min_handwriting_quality_score` | 0.72 | Chặn ảnh viết tay chất lượng thấp |
| `min_handwriting_confidence` | 0.8 | Chỉ nhận handwriting khi đủ tin cậy |

### 18.8 Graph layer

Graph layer không thay thế vector retrieval mà bổ sung khả năng liên hệ khái niệm. Hệ thống trích xuất và lưu:

| Đối tượng | Ý nghĩa |
|---|---|
| Entity | Khái niệm, thuật ngữ, người, tổ chức, chủ đề |
| Alias | Các cách gọi khác nhau của cùng entity |
| Event | Sự kiện hoặc mốc có participant/thời gian |
| Relation | Quan hệ giữa entity/entity hoặc entity/event |
| Evidence ref | Dẫn ngược về material/page/block/span |

Graph hữu ích nhất trong các tình huống:

- câu hỏi so sánh hai tài liệu,
- câu hỏi cần nối nhiều khái niệm,
- câu hỏi hỏi “liên quan đến gì” hoặc “vì sao”,
- mindmap và topic exploration,
- phát hiện relation có mâu thuẫn giữa hai nguồn.

Giới hạn kỹ thuật hiện tại là graph vẫn nhẹ, chủ yếu dựa vào extraction rule/model đơn giản và evidence refs. Đây là thiết kế hợp lý cho MVP vì không làm hệ thống phụ thuộc vào một graph database riêng, nhưng nếu corpus lớn hơn thì nên cân nhắc graph index hoặc graph database chuyên dụng.

### 18.8.1 Flow Graph RAG trong codebase

Flow Graph RAG của hệ thống có thể chia thành ba phase: build graph khi upload, dùng graph khi query, và visualize graph trên UI.

#### Phase 1: Build graph khi upload tài liệu

```text
Tài liệu đầu vào
PDF / DOCX / PPTX / PNG / JPG / XLSX / CSV
        |
        v
DoclingParser / SpreadsheetParser / EasyOCR hoặc OCR branch
        |
        v
ParsedDocument
pages + blocks + block_id + page + bbox + ocr_confidence
        |
        v
LayoutNormalizer
        |
        v
EvidenceMapper
        |
        v
EvidenceMap
block -> EvidenceRef(material_id, page, block_id, bbox, snippet_original)
        |
        v
EntityExtractor
        |
        v
EntityResolver
        |
        v
EventExtractor / structural relation extraction
        |
        v
QdrantMongoIndexer
        |
        +--> MongoDB: Entity
        +--> MongoDB: Event
        +--> MongoDB: Relation
        +--> MongoDB: Chunk
        +--> Qdrant: dense + sparse vectors
```

Entity extraction hiện theo hướng rule-based:

| Nguồn tín hiệu | Entity type | Ví dụ |
|---|---|---|
| Regex/keyword nhóm phương pháp | `method` | Dropout, Transformer, RAG |
| Metric pattern | `metric` | accuracy, F1, loss, recall |
| CamelCase/ALLCAPS/technical terms | `concept` | Deep Learning, Graph RAG |
| NER tùy chọn | `per`, `org`, `loc` | Người, tổ chức, địa điểm nếu runtime có NER |

Entity sau khi extract được resolve/deduplicate:

```text
list[ExtractedEntity]
        |
        v
EntityResolver
        |
        v
MongoDB Entity {
  canonical_name,
  aliases,
  entity_type,
  confidence,
  mention_refs: [
    { material_id, page, block_id }
  ]
}
```

Relation hiện được lưu theo schema:

```text
MongoDB Relation {
  source_id,
  target_id,
  relation_type,
  confidence,
  evidence_refs: [
    { material_id, page, block_id }
  ],
  is_conflicting
}
```

Điểm cần phân biệt:

- Code đã có đường lưu `Relation` vào MongoDB nếu extractor trả về relation đủ confidence.
- Relation hiện thiên về **event/structural relations** như `mentioned_in_event`, `mentioned_in_block`, `section_contains`, `has_caption`, `caption_of`.
- Semantic relation kiểu `dropout affects overfitting` chưa phải lớp extraction mạnh/chuyên biệt; vì vậy graph retrieval có thể không tìm được quan hệ ngữ nghĩa như ví dụ nếu collection `relations` thiếu edge tương ứng.

#### Phase 2: Query Graph RAG khi người dùng hỏi

Ví dụ query:

```text
"dropout ảnh hưởng thế nào đến overfitting?"
```

Query flow:

```text
User query
        |
        v
QueryRouter.route()
RouteType.GRAPH_RELATION nếu query có tín hiệu như:
"ảnh hưởng", "tác động", "liên quan", "so sánh", "khác nhau"
        |
        +-----------------------------+
        |                             |
        v                             v
HybridRetriever                  GraphRetriever
dense + sparse + RRF             retrieve_paths()
        |                             |
        |                             v
        |                      Extract terms từ query
        |                      ví dụ: dropout, ảnh, hưởng, overfitting
        |                             |
        |                             v
        |                      Entity.find()
        |                      regex match canonical_name/aliases
        |                             |
        |                             v
        |                      seed_ids:
        |                      entity:dropout
        |                      entity:overfitting
        |                             |
        |                             v
        |                      Relation.find()
        |                      source_id hoặc target_id nằm trong seed_ids
        |                             |
        |                             v
        |                      2-hop expansion nếu bật
        |                      graph_max_hops <= 2
        |                             |
        |                             v
        |                      Hydrate EvidenceRef
        |                      Material + page + block + snippet_original
        |                             |
        |                             v
        |                      GraphPath(
        |                        path=[source, relation, target],
        |                        confidence,
        |                        evidence_refs=[EvidenceBlock...]
        |                      )
        |
        +-------------+---------------+
                      |
                      v
graph_chunks + retrieved_chunks
        |
        v
dedupe_retrieved_chunks()
        |
        v
rerank_multilingual()
BGE-reranker-v2-m3
        |
        v
confidence_scorer.should_refuse()
        |
        v
LLM synthesis
local qwen model hoặc OpenAI-compatible fallback
        |
        v
ClaimVerifier / contradiction checks
        |
        v
QueryResponse
answer + citations(page, block_id, bbox)
```

Trong GraphRetriever, các bước chính là:

| Bước | Cách làm |
|---|---|
| Tách term | Regex lấy các token đủ dài từ query |
| Tìm seed entity | Match `canonical_name` hoặc `aliases` bằng regex không phân biệt hoa thường |
| Tìm relation hop 1 | Query `Relation` có `source_id` hoặc `target_id` thuộc seed ids |
| Mở rộng hop 2 | Lấy frontier entity từ hop 1 rồi query tiếp relation liên quan |
| Hydrate evidence | Dùng `EvidenceRef` để lookup material/page/block và tạo `EvidenceBlock` |
| Sort path | Sắp theo confidence giảm dần |
| Limit | Trả tối đa `graph_top_k` path |

Ví dụ lý tưởng nếu relation semantic đã tồn tại:

```text
Entity("Dropout")
Entity("Overfitting")
Relation(
  source_id="entity:dropout",
  target_id="entity:overfitting",
  relation_type="affects",
  confidence=0.81
)
```

Khi đó GraphRetriever có thể tạo path:

```text
GraphPath(
  path=[
    "entity:dropout",
    "relation:affects",
    "entity:overfitting"
  ],
  confidence=0.81,
  evidence_refs=[...]
)
```

Nhưng với hiện trạng extractor còn nhẹ, quan hệ dạng này không nên được xem là luôn có sẵn.

#### Phase 3: Visualization Graph UI

Graph visualization flow:

```text
GET/POST graph API
        |
        v
Entity.find()
        |
        v
GraphNode(type = entity_type)

Event.find()
        |
        v
GraphNode(type = "event")

Relation.find()
        |
        v
GraphEdge(source, target, relation_type, confidence)
        |
        v
Frontend React Flow render graph/mindmap
```

Nếu không có relation đủ tốt, graph UI có thể dùng fallback co-occurrence:

```text
_entity_cooccurrence_edges()
2 entity cùng xuất hiện trong một block
        |
        v
edge relation_type = "co_occurs_in_block"
```

Fallback này giúp UI vẫn có graph để xem, nhưng cần hiểu đúng rằng `co_occurs_in_block` chỉ nói hai entity cùng xuất hiện trong cùng block, không chứng minh quan hệ nhân quả hay tác động.

#### Hiện trạng Graph RAG cần biết

| Thành phần | Trạng thái |
|---|---|
| Entity extraction | Có, chủ yếu rule-based bằng regex, keyword và pattern |
| Underthesea NER | Optional, không nên coi là dependency luôn có sẵn |
| EntityResolver | Có dedup và alias merge |
| Event extraction | Có extractor nhẹ dựa trên event verbs/date pattern |
| Relation extraction | Có structural/event relations, nhưng semantic relation chuyên sâu còn yếu |
| Relation persistence | Có code lưu `Relation` nếu extractor sinh relation và vượt ngưỡng confidence |
| Graph retrieval | Code đúng hướng, phụ thuộc vào `Entity` và `Relation` đã có trong MongoDB |
| Graph UI | Có thể render entity/event/relation và fallback co-occurrence nếu relation thiếu |
| Điểm nghẽn lớn nhất | Thiếu semantic relation extractor mạnh cho quan hệ kiểu `affects`, `causes`, `prevents`, `improves`, `reduces` |

Kết luận kỹ thuật cho Graph RAG:

> Graph RAG trong hệ thống đã có khung đầy đủ từ entity store, relation schema, graph retrieval, evidence hydration đến visualization. Tuy nhiên chất lượng graph reasoning hiện bị giới hạn bởi lớp relation extraction. Nếu muốn câu hỏi như “Dropout ảnh hưởng thế nào đến overfitting?” hoạt động ổn định bằng graph path, cần bổ sung extractor cho semantic relation và đảm bảo các edge đó được lưu vào `relations` collection với evidence refs rõ ràng.

### 18.9 Frontend technical

Frontend dùng React + Vite + TypeScript. Giao diện chính là workspace nhiều panel, phù hợp thao tác học tập và kiểm chứng evidence.

Thành phần frontend chính:

| Thành phần | Vai trò |
|---|---|
| App shell | Khung điều hướng và layout |
| Workspace page | Màn hình thao tác chính |
| Chat panel | Nhập câu hỏi và xem câu trả lời |
| Sources panel | Quản lý tài liệu trong workspace |
| Evidence panel | Xem citation/snippet/bbox |
| Graph canvas | Hiển thị graph/mindmap bằng React Flow |
| Studio panel | Các tab nâng cao như graph, compare, study guide |
| Markdown renderer | Render answer có định dạng |
| Error boundary | Chặn crash UI theo component |

Các dependency UI đáng chú ý:

| Package | Vai trò |
|---|---|
| `react` / `react-dom` | Runtime UI |
| `react-router-dom` | Routing |
| `reactflow` | Graph và mindmap |
| `@dagrejs/dagre` | Layout graph |
| `lucide-react` | Icon |
| `tailwindcss` | Styling utility |
| `vitest` | Test frontend |

### 18.10 Container và vận hành local

Docker Compose hiện có các service:

| Service | Vai trò |
|---|---|
| `api` | FastAPI backend |
| `worker` | Celery worker chạy parse/index |
| `qdrant` | Vector database |
| `redis` | Broker/result backend cho Celery |

Mapping local:

| Port | Service |
|---:|---|
| 8000 | FastAPI API |
| 6333 | Qdrant HTTP |
| 6334 | Qdrant gRPC |
| 6379 | Redis |

Volume quan trọng:

| Volume/path | Vai trò |
|---|---|
| `./data:/app/data` | Raw file, processed artifact, vector local data |
| `./config:/app/config:ro` | Config YAML read-only trong container |
| `redis-data:/data` | Redis append-only persistence |

Điểm cần chú ý khi vận hành:

- Nếu dùng model local, Ollama chạy trên host và container gọi qua `host.docker.internal`.
- Qdrant và Redis chỉ bind `127.0.0.1`, phù hợp môi trường local.
- Production cần bổ sung auth thật, secret management, backup database/vector store và centralized logging.

### 18.11 Test coverage và regression

Test hiện được chia theo lớp:

| Nhóm test | Nội dung |
|---|---|
| API | Upload material, graph endpoint, admin, connections |
| Processing | Docling parser, OCR gate, spreadsheet parser, chunking, layout normalizer |
| RAG | Embed/index/retrieve/rerank/query router/graph retriever |
| Inference | Query endpoint, summary, response parser, intent classifier, verifier |
| Integration | Retrieval end-to-end và sample corpus smoke test |
| Evaluation | Model adaptation và benchmark scripts |

Chiến lược regression nên giữ:

1. Unit test cho rule và schema.
2. Integration test cho MongoDB/Qdrant path.
3. Corpus smoke test cho tài liệu mẫu.
4. Evaluation benchmark cho Recall@K, MRR, nDCG và citation accuracy.
5. Ablation suite để chứng minh từng lớp như rerank, graph, layout-aware chunking có đóng góp thật.

### 18.12 Các điểm cần hardening trước production

| Hạng mục | Hiện trạng | Việc cần làm |
|---|---|---|
| Auth | Có API key gate, chưa phải auth đầy đủ | Thêm user/session/RBAC hoặc tích hợp IdP |
| Authorization | Scope filter theo `owner_id`/`collection_id` | Audit toàn bộ endpoint và background job |
| Observability | Có logging và admin metrics | Thêm tracing, structured logs, latency histogram |
| Secrets | Dựa vào `.env` | Secret manager cho production |
| Data retention | Chưa nêu chính sách rõ | Thêm retention/delete/export policy |
| OCR quality | Có gate nhưng scan Việt vẫn yếu | Thêm benchmark OCR và feedback loop |
| Evaluation | Có framework và scripts | Chuẩn hóa bộ câu hỏi vàng thực tế |
| Scaling | Phù hợp nhỏ/vừa | Tách worker pool, batch embedding, cache retrieval |
| Backup | Chưa là phần chính của báo cáo | Backup MongoDB, Qdrant và raw artifacts |
| Frontend robustness | Có ErrorBoundary, còn thiếu polish | Thêm skeleton, retry state, stale data handling |

### 18.13 Tech stack toàn bộ

Bảng dưới đây gom toàn bộ stack kỹ thuật đang dùng hoặc đã được wiring trong repo, chia theo lớp để dễ trình bày khi bảo vệ đồ án.

#### 18.13.1 Nền tảng runtime và ngôn ngữ

| Lớp | Công nghệ | Phiên bản/nguồn | Vai trò | Ghi chú kỹ thuật |
|---|---|---|---|---|
| Backend language | Python | Docker dùng `python:3.12-slim`; README ghi Python 3.11+ | Runtime chính cho API, RAG, parsing, indexing | Nên thống nhất tài liệu triển khai là Python 3.12 trong Docker, 3.11+ cho local |
| Frontend language | TypeScript | `typescript ^5.7.2` | Type-safe frontend | Build bằng `tsc` trước Vite |
| Frontend runtime | Node.js / npm | Theo `package-lock.json` và npm scripts | Cài dependency, dev server, build | Cần Node tương thích Vite 6 |
| Container runtime | Docker / Docker Compose | `docker-compose.yml` | Chạy API, worker, Qdrant, Redis | Phù hợp local/dev reproducibility |
| OS base image | Debian slim | `python:3.12-slim` | Base backend container | Cài thêm `libgl1`, `libglib2.0-0` cho OpenCV/OCR |

#### 18.13.2 Backend API stack

| Thành phần | Công nghệ | Phiên bản | Vai trò | Lý do dùng |
|---|---|---:|---|---|
| Web framework | FastAPI | `0.115.5` | REST API cho upload, query, graph, admin | Async tốt, schema rõ, hợp Pydantic |
| ASGI server | Uvicorn | `0.32.1` | Chạy FastAPI | Nhẹ, phổ biến trong FastAPI |
| Request/response schema | Pydantic | `2.9.2` | Validate schema API và nội bộ | Giảm lỗi dữ liệu sai kiểu |
| Settings management | pydantic-settings | `2.6.1` | Nạp `.env`, env var và config | Tập trung cấu hình runtime |
| Multipart upload | python-multipart | `0.0.17` | Nhận file upload | Bắt buộc cho FastAPI multipart |
| HTTP client | httpx | `0.27.2` | Gọi LLM/Ollama/API nội bộ | Hỗ trợ async và timeout |
| YAML config | PyYAML | `6.0.2` | Đọc model/retrieval/guardrail/logging config | Tách config khỏi code |
| Rate limiting | slowapi | `0.1.9` | Giới hạn request query/compare | Bảo vệ endpoint tốn tài nguyên |
| Logging | Python logging + YAML dictConfig | `config/logging_config.yaml` | Log console structured cơ bản | Dễ chuyển sang centralized logging |
| CORS | FastAPI CORSMiddleware | Built-in | Cho frontend local gọi API | Origin lấy từ settings |

#### 18.13.3 Persistence và data infrastructure

| Thành phần | Công nghệ | Phiên bản | Vai trò | Dữ liệu lưu |
|---|---|---:|---|---|
| Metadata database | MongoDB | Atlas/local, version không cố định trong repo | Lưu metadata bán cấu trúc | materials, pages, chunks, graph, memory, logs |
| Python Mongo driver | Motor | `3.6.0` | Async MongoDB driver | Dùng dưới Beanie |
| ODM | Beanie | `1.26.0` | Mapping Pydantic document sang MongoDB | Index model, query document |
| Vector database | Qdrant | Docker `qdrant/qdrant:v1.12.4` | Lưu dense/sparse vector và payload | Collection `agentbook_chunks` |
| Qdrant client | qdrant-client | `1.12.1` | Tạo collection, upsert/search vector | Khớp Qdrant 1.12.x |
| Queue broker | Redis | Docker `redis:7.4-alpine` | Celery broker | Queue parse/index |
| Result backend/cache | Redis | `redis 5.2.0` client | Celery result backend và cache nhẹ | DB 0 broker, DB 1 result |
| File storage | Local filesystem | `./data` mount vào container | Raw files, processed artifacts, cache | MVP/local-first |
| Vector storage | Qdrant local volume | `./data/vectordb` | Qdrant persistent storage | Mount vào `/qdrant/storage` |

#### 18.13.4 Async job và pipeline orchestration

| Thành phần | Công nghệ | Phiên bản | Vai trò | Ghi chú |
|---|---|---:|---|---|
| Job queue | Celery | `5.4.0` | Chạy parse/index ngoài request | Worker command dùng pool `solo` |
| Broker | Redis | `7.4-alpine` service | Nhận task từ API | Docker service `redis` |
| Result backend | Redis | DB 1 | Lưu kết quả/trạng thái task | Tách khỏi broker DB 0 |
| Pipeline state | MongoDB `pipeline_jobs` | Beanie document | Theo dõi stage, failed stage, error | UI đọc status/progress |
| Eager mode | Celery eager config | Env `AGENTBOOK_CELERY_TASK_ALWAYS_EAGER` | Chạy đồng bộ khi local/test | Hữu ích cho smoke test |

#### 18.13.5 Document processing stack

| Bài toán | Công nghệ/module | Dependency | Vai trò | Ghi chú |
|---|---|---|---|---|
| PDF/DOCX/PPTX parsing | Docling parser | `docling` | Layout-aware parse, table, figure, reading order | Parser chính cho tài liệu văn bản |
| PDF text fallback | pypdf | `pypdf` | Đọc text PDF khi cần fallback | Giảm phụ thuộc OCR khi PDF có text layer |
| Spreadsheet parsing | Custom parser + openpyxl/xlrd | `openpyxl`, `xlrd` | Parse XLSX/XLS/CSV thành summary/table/row text | Phù hợp retrieval trên dữ liệu bảng |
| OCR chính cho tiếng Việt | EasyOCR engine | `easyocr` | OCR ảnh tiếng Việt với dấu | Code hiện mô tả EasyOCR là primary cho Vietnamese images |
| OCR/Paddle branch | PaddleOCR-compatible engine | Cần `paddleocr`, `paddlepaddle` nếu bật | OCR printed image/PDF scan theo PP-OCR config | Config có model PP-OCRv5; dependency chưa pin trong `requirements.txt` |
| Image preprocessing | OpenCV + NumPy | `opencv-python`, `numpy` | Enhance, binarize, grayscale, blur/contrast/skew checks | Cải thiện input trước OCR |
| Language detection | Custom language detector | Nội bộ | Detect `vi`, `en`, `unknown` theo block/document | Dùng cho routing và metadata |
| Layout normalization | Custom normalizer | Nội bộ | Sort reading order, merge fragments, chuẩn hóa block | Nền cho evidence và chunking |
| Evidence mapping | Custom mapper | Nội bộ | Tạo evidence refs từ page/block/bbox | Bảo toàn citation |
| Figure captioning | VLM qua Ollama nếu có | HTTP/Ollama | Caption figure block cần mô tả | Fallback im lặng nếu không có vision model |
| Handwriting gate | Custom quality + confidence gate | OpenCV/LLM tùy runtime | Chặn ảnh viết tay chất lượng thấp | Không index mọi ảnh viết tay một cách mù quáng |

#### 18.13.6 AI, RAG và inference stack

| Thành phần | Công nghệ/model | Cấu hình hiện tại | Vai trò | Ghi chú |
|---|---|---|---|---|
| Embedding model | BAAI BGE-M3 | `BAAI/bge-m3`, dense size 1024 | Sinh dense và sparse embeddings | Hợp multilingual/hybrid retrieval |
| Embedding runtime | FlagEmbedding | `FlagEmbedding >=1.4.0` | Chạy BGE-M3 | Có thể dùng CPU/GPU tùy config |
| Sentence embedding utils | sentence-transformers | `>=2.7.0` | Hỗ trợ embedding/reranking ecosystem | Dependency phụ trợ |
| Transformer runtime | Hugging Face Transformers | `>=4.49,<4.53` | Model tokenizer/transformer backend | Pin range để tránh breaking change |
| Vector retrieval | Qdrant dense search | `dense_top_k=20` | Semantic candidate retrieval | Cosine distance |
| Sparse retrieval | BGE-M3 sparse + Qdrant sparse | `sparse_top_k=20` | Lexical candidate retrieval | Tốt cho thuật ngữ/tên riêng |
| Fusion | Reciprocal Rank Fusion | `rrf_k=60` | Hợp nhất dense/sparse/graph candidates | Giảm lệch do một retriever |
| Reranker | BGE reranker v2 M3 | `BAAI/bge-reranker-v2-m3` | Xếp lại candidate trước prompt | Tăng precision |
| Query rewriting | LLM-based rewriter | `query_rewriter_enabled=true` | Multi-query/RAG-Fusion | Thay dictionary VI-EN cứng |
| Graph retrieval | Custom graph retriever | `graph_top_k=10`, `graph_max_hops=2` | Mở rộng evidence theo entity/relation | Hỗ trợ multi-hop/cross-doc |
| LLM local | Ollama-compatible local model | Config `qwen2.5:3b`, Docker override `qwen3:4b` | Answer synthesis, summary, rewrite | Local-first, ít phụ thuộc API |
| LLM fallback | OpenAI-compatible API | `gpt-4o-mini`, base URL configurable | Dự phòng hoặc nâng chất lượng | Cần API key nếu bật |
| Prompting | Text prompt templates | `qa_grounded`, `summarization`, `off_topic`, `chitchat` | Kiểm soát answer grounded | Tách prompt khỏi code |
| Guardrails | Confidence scorer, claim verifier, contradiction detector | Ngưỡng trong guardrails config | Refusal và hậu kiểm | Giảm hallucination |
| Memory | Window context + summary memory | MongoDB-backed | Giữ ngữ cảnh hội thoại | Scope theo owner/collection/conversation |

#### 18.13.7 Frontend stack

| Thành phần | Công nghệ | Phiên bản | Vai trò | Ghi chú |
|---|---|---:|---|---|
| UI framework | React | `18.3.1` | Xây giao diện workspace | Component-based |
| DOM renderer | React DOM | `18.3.1` | Render React app | Đồng bộ React 18 |
| Build tool/dev server | Vite | `6.0.1` | Dev server và production build | Build nhanh |
| Vite React plugin | @vitejs/plugin-react | `4.3.4` | React Fast Refresh/build support | Nằm trong dependencies |
| Routing | react-router-dom | `6.28.0` | Điều hướng page/workspace | Client-side routing |
| Graph visualization | React Flow | `11.11.4` | Graph canvas, mindmap | Trực quan hóa node/edge |
| Graph layout | @dagrejs/dagre | `3.0.0` | Auto-layout graph | Hỗ trợ mindmap/graph readable |
| Icons | lucide-react | `0.468.0` | Icon UI | Nhẹ, consistent |
| CSS framework | Tailwind CSS | `3.4.15` | Utility styling | Kèm PostCSS/autoprefixer |
| CSS tooling | PostCSS | `8.4.49` | CSS processing | Dùng với Tailwind |
| Browser compatibility | autoprefixer | `10.4.20` | Tự thêm vendor prefixes | Build CSS |
| Markdown render | Custom renderer | Nội bộ | Render answer formatted | Có thể gắn citation/snippet |
| Error boundary | Custom component | Nội bộ | Chặn lỗi UI component | Tăng robustness |

#### 18.13.8 Testing, evaluation và tooling

| Nhóm | Công nghệ | Phiên bản/nguồn | Vai trò |
|---|---|---|---|
| Backend unit/integration test | pytest | `8.3.4` | Test API, processing, RAG, inference |
| Async test | pytest-asyncio | `0.24.0` | Test coroutine/service async |
| Frontend test runner | Vitest | `4.1.5` | Test React/TS |
| DOM test env | jsdom | `29.1.0` | Browser-like test environment |
| React testing | Testing Library | `@testing-library/*` | Test component/user event |
| Evaluation scripts | Custom Python evaluation | `evaluation/run_eval.py` | Recall@K, MRR, nDCG, ablation |
| Ablation configs | YAML configs | `evaluation/ablation_configs` | So sánh hybrid/vector, rerank/no-rerank, graph/no-graph |
| Smoke corpus | Sample documents | `data/test data` | Test pipeline thực tế với PDF/PPTX/DOCX/XLSX/ảnh |

#### 18.13.9 Configuration stack

| File cấu hình | Nội dung | Vai trò |
|---|---|---|
| `config/model_config.yaml` | LLM, embedding, reranker, OCR, PDF render | Điều khiển model/runtime |
| `config/retrieval_config.yaml` | Qdrant collection, top-k, RRF, chunking | Điều khiển retrieval/indexing |
| `config/guardrails_config.yaml` | Upload allowlist, refusal threshold, image quality | Điều khiển safety/gate |
| `config/logging_config.yaml` | Format, handler, log level | Điều khiển logging |
| `backend/.env` | MongoDB URI, API key, runtime env | Secret và environment-specific config |
| `docker-compose.yml` | API, worker, Qdrant, Redis | Local orchestration |
| `frontend/package.json` | Dependency và scripts frontend | Build/dev/test frontend |
| `backend/requirements.txt` | Dependency backend | Reproducible Python install |

#### 18.13.10 Deployment và vận hành

| Thành phần | Công nghệ | Cấu hình hiện tại | Vai trò |
|---|---|---|---|
| API container | Docker build từ `backend/Dockerfile` | Expose `127.0.0.1:8000` | Chạy FastAPI |
| Worker container | Cùng image backend | Celery worker `--pool=solo` | Parse/index nền |
| Vector DB container | Qdrant | Expose `127.0.0.1:6333/6334` | Vector search |
| Broker container | Redis | Expose `127.0.0.1:6379`, appendonly yes | Queue/cache |
| Healthcheck | Python urllib call `/health` | 30s interval, 5s timeout | Kiểm tra API alive |
| Data mount | `./data:/app/data` | Shared API/worker | Raw + processed data |
| Config mount | `./config:/app/config:ro` | Read-only config | Tránh sửa config trong container |
| Local LLM bridge | `host.docker.internal:11434` | Ollama trên host | Cho container gọi model local |

#### 18.13.11 Stack theo luồng xử lý

| Luồng | Stack tham gia | Kết quả |
|---|---|---|
| Upload tài liệu | FastAPI, python-multipart, security validator, MongoDB, Celery, Redis | File hợp lệ, material record, pipeline job |
| Parse PDF/DOCX/PPTX | Celery, Docling, pypdf fallback, layout normalizer, MongoDB | Page/block/evidence metadata |
| Parse ảnh scan | OpenCV, EasyOCR/PaddleOCR branch, OCR quality gate, language detector | OCR blocks có bbox/confidence |
| Parse bảng tính | Spreadsheet parser, openpyxl/xlrd, Markdown/table verbalization | Sheet/table chunks dễ retrieve |
| Chunk và enrich | Layout-aware/semantic chunker, BGE-M3 embedder, contextual enricher | Chunk giàu ngữ cảnh |
| Index | Beanie/MongoDB, Qdrant client, BGE-M3 dense/sparse | Chunk metadata + vector points |
| Hỏi đáp | FastAPI, query router, hybrid retriever, graph retriever, reranker, LLM, verifier | Answer grounded có citation |
| Graph/mindmap | Entity/event/relation extractor, MongoDB graph store, React Flow | Knowledge graph trực quan |
| Frontend workspace | React, Vite, Router, React Flow, Tailwind, API client | UI chat/source/evidence/graph |

Kết luận technical của phụ lục: hệ thống đã có kiến trúc module đủ rõ để bảo trì và mở rộng. Rủi ro lớn nhất không nằm ở thiếu thành phần RAG lõi, mà nằm ở hardening production: auth, observability, eval chuẩn hóa, backup và tối ưu vận hành cho corpus lớn.
