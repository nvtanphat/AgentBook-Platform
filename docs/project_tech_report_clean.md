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
