# Báo Cáo Kỹ Thuật Toàn Dự Án

> Phạm vi: quét toàn bộ repo hiện tại của AgentBook/Prism và mô tả từng kỹ thuật đang dùng, trạng thái triển khai, điểm mạnh, điểm yếu, và mức độ sẵn sàng cho sản phẩm.
>
> Nguồn đọc chính: code backend/frontend, config, test suite, và các tài liệu kiến trúc hiện có trong `docs/`.

## 1) Tóm tắt nhanh

AgentBook là một hệ thống **Graph RAG cho tài liệu học tập**. Mục tiêu không chỉ là hỏi đáp văn bản, mà còn là:

- parse đa định dạng tài liệu,
- giữ layout và evidence trace,
- chunk theo cấu trúc,
- index hybrid dense + sparse,
- truy xuất có scope,
- rerank và refusal có kiểm soát,
- lưu memory hội thoại ngắn hạn,
- và hiển thị citation/evidence trong UI.

Kết luận ngắn:

- **Nhóm kỹ thuật lõi retrieval/ingestion đã có và chạy được**.
- **Nhóm kỹ thuật sản phẩm hóa** như observability, quality gate, auth, regression corpus, và UX polish vẫn còn cần siết thêm.
- **OCR scan là điểm yếu lớn nhất** trong corpus mẫu hiện tại.

## 2) Bản đồ công nghệ

| Lớp | Kỹ thuật chính | Vai trò |
|---|---|---|
| API | FastAPI + Pydantic + DI | Expose upload/query/graph/admin endpoints |
| Lưu trữ metadata | MongoDB + Beanie | Lưu material, pages, chunks, graph, query logs, memory |
| Vector store | Qdrant | Lưu dense + sparse vectors cho retrieval hybrid |
| Async jobs | Celery + Redis | Chạy parse/index nền |
| Parser | Docling, SpreadsheetParser | Parse PDF/DOCX/PPTX và CSV/XLS/XLSX |
| OCR | PaddleOCR | OCR ảnh scan/chữ in |
| Handwriting | HandwritingReader + quality gate | Chỉ nhận ảnh viết tay đủ chất lượng |
| Normalization | LayoutNormalizer | Chuẩn hóa block/layout/language |
| Chunking | LayoutAwareChunker, SemanticChunker | Chunk theo layout/token budget/semantic breakpoints |
| Retrieval | HybridRetriever, graph retrieval, reranker | Truy xuất dense+sparse+graph rồi rerank |
| Guardrails | ConfidenceScorer, ClaimVerifier, refusal | Chặn trả lời yếu hoặc mâu thuẫn |
| Memory | QueryLog + ChatSummaryMemory | Nhớ hội thoại gần đây theo scope |
| Frontend | React + Vite + React Router + React Flow | Workspace, chat, evidence, compare, graph, mindmap |
| Testing | Pytest + corpus smoke + integration tests | Khóa hành vi ingest/retrieve/index |

## 3) Kiến trúc tổng thể

Luồng end-to-end hiện tại:

1. Upload file qua API `materials/upload`.
2. `MaterialService` validate file, checksum, duplicate, và tạo `Material` + `PipelineJob`.
3. `ParseIndexPipeline` parse tài liệu theo loại file.
4. `LayoutNormalizer` chuẩn hóa block, language, reading order, merge fragment.
5. `EvidenceMapper` chuyển parsed blocks thành evidence blocks.
6. `LayoutAwareChunker` hoặc `SemanticChunker` tạo chunk có metadata đầy đủ.
7. `EntityExtractor`, `EventExtractor`, `EntityResolver` tạo graph metadata.
8. `QdrantMongoIndexer` lưu `Chunk` vào MongoDB và upsert vector vào Qdrant.
9. Query đi qua `QueryService` và `InferenceEngine`.
10. Retrieval hybrid + graph + rerank + refusal + citations.
11. `MemoryService` nhét memory hội thoại vào prompt nếu có.

Các thành phần này khớp với mô tả kiến trúc trong [docs/architecture.md](./architecture.md) và các rule vận hành trong [CLAUDE.md](../CLAUDE.md).

## 4) Upload, an toàn dữ liệu, và vòng đời tài liệu

### 4.1 Kỹ thuật đang dùng

- Magic-byte validation và allowlist extension ở `materials/upload`.
- Stream upload ra temp file, tránh đọc toàn bộ file lớn vào memory một lần.
- Hash SHA-256 để chống trùng file trong cùng collection.
- Storage path được scope theo `owner_id` và `collection_id`.
- Material lifecycle có status/stage: `uploaded -> parsing -> parsed -> indexing -> indexed -> failed`.

Tham chiếu:

- [backend/src/api/v1/endpoints/materials.py](../backend/src/api/v1/endpoints/materials.py)
- [backend/src/services/material_service.py](../backend/src/services/material_service.py)
- [backend/src/core/security.py](../backend/src/core/security.py)
- [backend/src/models/material.py](../backend/src/models/material.py)
- [backend/src/models/pipeline_job.py](../backend/src/models/pipeline_job.py)

### 4.2 Nó giải quyết gì

- Ngăn upload file sai định dạng.
- Giảm rủi ro path traversal.
- Cho phép job pipeline có trace trạng thái rõ.
- Giữ raw file và parsed artifact tách bạch.

### 4.3 Đánh giá

**Mạnh**

- Có scope rõ.
- Có duplicate detection bằng checksum.
- Có job status để UI/ops debug.

**Yếu**

- Chưa có auth thật sự ở mức user identity, hiện vẫn phụ thuộc `owner_id` từ client và một cơ chế verify đơn giản.
- Quan sát lỗi vẫn thiên về log hơn là dashboard/telemetry.

**Mức sản phẩm**

- **Khá tốt cho MVP**
- **Chưa đủ production-grade về auth/ops**

## 5) Parsing tài liệu

### 5.1 Docling parser

Kỹ thuật chính cho PDF/DOCX/PPTX là `DoclingParser`.

Tham chiếu:

- [backend/src/processing/docling_parser.py](../backend/src/processing/docling_parser.py)

Điểm đáng chú ý:

- Parser ưu tiên layout-aware export.
- PDF có fallback sang `pypdf` text extraction nếu Docling fail.
- Có thêm OCR fallback cho PDF ở các trang thiếu text.
- Output được map thành `ParsedDocument`, `ParsedPage`, `ParsedBlock`.

**Giá trị**

- Giữ được heading/table/layout tốt hơn parser thuần text.
- Tạo đầu vào sạch hơn cho chunking và evidence trace.

**Rủi ro**

- Phụ thuộc môi trường và cache model.
- PDF scan nặng vẫn có thể xấu nếu OCR nền yếu.

**Mức sản phẩm**

- **Tốt**

### 5.2 Spreadsheet parser

`SpreadsheetParser` xử lý CSV/XLS/XLSX.

Tham chiếu:

- [backend/src/processing/spreadsheet_parser.py](../backend/src/processing/spreadsheet_parser.py)

Kỹ thuật:

- Đọc workbook theo sheet.
- Truncate số cột và số hàng để tránh nổ bộ nhớ.
- Sinh cả:
  - block summary của sheet,
  - block table markdown,
  - verbalized row blocks cho retrieval text.

**Giá trị**

- Tăng khả năng retrieval cho bảng biểu.
- Giúp câu hỏi ngôn ngữ tự nhiên bám được vào dòng dữ liệu cụ thể.

**Mức sản phẩm**

- **Khá tốt**

### 5.3 OCR printed scan

`PaddleOCREngine` là OCR chính cho ảnh scan/chữ in.

Tham chiếu:

- [backend/src/processing/ocr_engine.py](../backend/src/processing/ocr_engine.py)

Kỹ thuật:

- Chạy PaddleOCR trên CPU.
- Có grayscale variant cho chất lượng thấp.
- Có routing theo ngôn ngữ `vi/en`.
- Có confidence và bbox cho từng block OCR.

**Giá trị**

- Cho phép ingest ảnh scan, không chỉ PDF text.
- Giữ bbox để cite evidence tốt hơn.

**Rủi ro**

- Output OCR có thể nhiễu, nhất là scan tiếng Việt.
- Model load/runtime khá nặng.

**Mức sản phẩm**

- **MVP tốt**
- **Chưa ổn hoàn toàn cho scan bẩn**

### 5.4 Handwriting reader

`HandwritingReader` không dùng OCR blind cho mọi ảnh viết tay. Nó đi qua quality gate trước.

Tham chiếu:

- [backend/src/processing/handwriting_reader.py](../backend/src/processing/handwriting_reader.py)
- [backend/src/processing/image_quality_checker.py](../backend/src/processing/image_quality_checker.py)

Kỹ thuật:

- Kiểm tra blur, brightness, contrast, skew.
- Chỉ cho phép ảnh đủ ngưỡng đi tiếp sang OCR.
- Nếu confidence thấp thì không nhận làm evidence.

**Giá trị**

- Tránh index rác từ ảnh viết tay quá xấu.

**Mức sản phẩm**

- **Tốt về guardrail**
- **Phụ thuộc chất lượng input**

### 5.5 Layout normalization

`LayoutNormalizer` là lớp rất quan trọng vì nó biến output parser/OCR thành đầu vào ổn định cho chunker.

Tham chiếu:

- [backend/src/processing/layout_normalizer.py](../backend/src/processing/layout_normalizer.py)

Kỹ thuật:

- Sort block theo reading order.
- Merge line OCR khi nhiều dòng rời rạc.
- Merge fragment text ngắn.
- Normalize block type:
  - `table`
  - `list`
  - `heading`
  - `paragraph`
- Detect language ở block/document level.

**Giá trị**

- Làm sạch đầu vào trước chunking.
- Tăng chất lượng evidence trace.

**Mức sản phẩm**

- **Rất quan trọng**
- **Đang làm đúng vai trò**

### 5.6 Evidence mapping

`EvidenceMapper` chuyển `ParsedDocument` thành `EvidenceMap`.

Tham chiếu:

- [backend/src/processing/evidence_mapper.py](../backend/src/processing/evidence_mapper.py)

Kỹ thuật:

- Mỗi block trở thành evidence block có:
  - `owner_id`
  - `collection_id`
  - `material_id`
  - `page`
  - `block_id`
  - `block_type`
  - `snippet_original`
  - `source_language`
  - `bbox`
  - `confidence`
  - `metadata`

**Giá trị**

- Đây là “xương sống” của citation trace.
- Sau này retrieve/chunk/citation đều bám vào cùng một nguồn evidence.

**Mức sản phẩm**

- **Tốt**

## 6) Chunking

### 6.1 Layout-aware chunking

`LayoutAwareChunker` là chunker mặc định, và đây là kỹ thuật đúng hướng cho tài liệu học tập.

Tham chiếu:

- [backend/src/processing/chunking.py](../backend/src/processing/chunking.py)

Kỹ thuật:

- Chunk theo heading boundary.
- Có overlap token count.
- Split block quá dài theo sentence/table row.
- Giữ metadata:
  - page list
  - block ids
  - bbox list
  - language
  - modality
  - parser/chunker/embedding/index version

**Giá trị**

- Không cắt cứng token như cách naive.
- Giữ được ngữ cảnh của layout, bảng và heading.

**Điểm mạnh**

- Phù hợp tài liệu học thuật, slide, bảng.
- Evidence trace rõ.

**Điểm yếu**

- Rule-based nên vẫn có thể split chưa tối ưu trên tài liệu rất lộn xộn.

**Mức sản phẩm**

- **Tốt**

### 6.2 Semantic chunking

`SemanticChunker` là hướng nâng cao hơn.

Kỹ thuật:

- Lấy embedding từng block.
- Tính cosine distance giữa block kề nhau.
- Split ở percentile threshold của document.
- Vẫn tôn trọng hard break của layout.

**Giá trị**

- Có thể bắt các điểm chuyển nghĩa tốt hơn layout-only.

**Rủi ro**

- Phụ thuộc embedder.
- Tốn compute hơn.
- Nếu embedding không ổn thì lợi ích giảm.

**Mức sản phẩm**

- **Nâng cao / tùy cấu hình**

### 6.3 Đánh giá chunking hiện tại

Từ corpus mẫu đã test:

- XLSX sinh chunk `paragraph` + `table`.
- DOCX/PDF/PPTX giữ layout tương đối tốt.
- OCR ảnh scan vẫn sinh được chunk, nhưng text còn nhiễu.

Kết luận:

- Chunking core ổn.
- Quality phụ thuộc parser/OCR đầu vào.
- Cần corpus regression để khóa hành vi khi đổi parser/chunker.

## 7) Indexing và vector store

### 7.1 QdrantMongoIndexer

Tham chiếu:

- [backend/src/rag/indexer.py](../backend/src/rag/indexer.py)

Kỹ thuật:

- Upsert dense vector và sparse vector vào cùng point.
- Store chunk metadata vào MongoDB `Chunk`.
- Tạo payload indexes cho các field filter quan trọng.
- Lưu payload:
  - content text
  - language
  - modality
  - page numbers
  - block types
  - source block ids
  - parser/chunker/embedding/index versions

**Giá trị**

- Hybrid search thật, không phải vector-only.
- Có versioning nên dễ re-index.

**Mức sản phẩm**

- **Tốt cho MVP**

### 7.2 Dense + sparse retrieval

Hệ embedding dùng BGE-M3.

Tham chiếu:

- [backend/src/rag/embedder.py](../backend/src/rag/embedder.py)
- [backend/src/rag/vector_store.py](../backend/src/rag/vector_store.py)

Kỹ thuật:

- Dense vector cho semantic similarity.
- Sparse vector cho lexical match.
- Qdrant lưu cả hai branch.

**Giá trị**

- Tăng recall cho câu hỏi có từ khóa chính xác.
- Giảm miss trên thuật ngữ chuyên môn.

**Điểm mạnh**

- Phù hợp tài liệu học tập đa dạng.

**Điểm yếu**

- Cần embedder/model load tốt.
- Local inference nặng hơn vector-only stack.

**Mức sản phẩm**

- **Tốt**

### 7.3 Versioning index

Material và chunk đều lưu version:

- parse_version
- chunk_version
- embedding_version
- index_version

**Giá trị**

- Khi đổi model/chunk strategy, có thể re-index có kiểm soát.

**Mức sản phẩm**

- **Đúng hướng sản phẩm**

## 8) Retrieval core

### 8.1 Hybrid retriever

`HybridRetriever` là core retrieval.

Tham chiếu:

- [backend/src/rag/retriever.py](../backend/src/rag/retriever.py)

Kỹ thuật thường đi qua:

- dense search
- sparse search
- fusion
- metadata scope filter
- hydration evidence từ MongoDB

**Giá trị**

- Tổng hợp semantic + lexical.
- Trả về chunk có evidence trace thay vì raw string.

**Mức sản phẩm**

- **Tốt**

### 8.2 Graph retrieval

Graph retrieval dùng evidence graph trong MongoDB.

Tham chiếu:

- [backend/src/rag/graph_retriever.py](../backend/src/rag/graph_retriever.py)

Kỹ thuật:

- Lấy entity/event/relation liên quan.
- Path expansion theo evidence refs.
- Hỗ trợ câu hỏi nhiều bước, cross-document reasoning.

**Giá trị**

- Dùng tốt cho compare, relation, concept tracing.

**Điểm yếu**

- Chưa phải graph DB chuyên dụng.
- Query graph sâu có thể nặng nếu không batch hợp lý.

**Mức sản phẩm**

- **Khá tốt cho đồ án / MVP**

### 8.3 Reranker

`CrossEncoderReranker` dùng để sắp xếp lại candidate chunks sau retrieval.

Tham chiếu:

- [backend/src/rag/reranker.py](../backend/src/rag/reranker.py)

Kỹ thuật:

- Rerank candidate theo query/chunk pair.
- Hỗ trợ limit và multilingual route.

**Giá trị**

- Chọn evidence tốt hơn top-k vector raw.

**Mức sản phẩm**

- **Tốt**

### 8.4 Query processing và rewriting

Tham chiếu:

- [backend/src/rag/query_processor.py](../backend/src/rag/query_processor.py)
- [backend/src/rag/query_rewriter.py](../backend/src/rag/query_rewriter.py)
- [backend/src/rag/query_router.py](../backend/src/rag/query_router.py)

Kỹ thuật:

- Detect intent.
- Normalize query language.
- Optional query rewriting.
- Route multi-query, graph, mmr theo loại câu hỏi.

**Giá trị**

- Giảm retrieval sai hình thức.
- Hỗ trợ câu hỏi tiếng Việt/English lẫn nhau.

**Mức sản phẩm**

- **Tốt nhưng phụ thuộc model runtime**

## 9) Guardrails, refusal, và citation

### 9.1 Confidence scorer

Tham chiếu:

- [backend/src/inference/confidence_scorer.py](../backend/src/inference/confidence_scorer.py)

Kỹ thuật:

- Score theo retrieval/rerank confidence.
- Quyết định có trả lời hay từ chối.

**Giá trị**

- Giảm hallucination trong câu hỏi thiếu evidence.

**Điểm yếu**

- Có thể từ chối hơi gắt nếu threshold cao.

### 9.2 Claim verifier và contradiction detector

Tham chiếu:

- [backend/src/guardrails/claim_verifier.py](../backend/src/guardrails/claim_verifier.py)
- [backend/src/guardrails/contradiction_detector.py](../backend/src/guardrails/contradiction_detector.py)

Kỹ thuật:

- Verify câu trả lời với evidence.
- Detect contradiction trong compare flow.

**Giá trị**

- Có lớp bảo vệ hậu kiểm, không chỉ dựa retrieval score.

**Mức sản phẩm**

- **Tốt, nhưng cần wiring đầy đủ ở mọi luồng**

### 9.3 Citation parser

Tham chiếu:

- [backend/src/inference/response_parser.py](../backend/src/inference/response_parser.py)
- [backend/src/schemas/evidence.py](../backend/src/schemas/evidence.py)

Kỹ thuật:

- Trích citation từ chunk/evidence.
- Inject citation vào answer.
- Bảo toàn page/block/bbox khi có.

**Giá trị**

- Đây là điểm bắt buộc để grounded QA có thể kiểm chứng.

**Mức sản phẩm**

- **Tốt**

## 10) Window context memory

### 10.1 Nó làm gì

Memory ở đây là **memory của hội thoại**, không phải memory của tài liệu.

Tham chiếu:

- [backend/src/services/memory_service.py](../backend/src/services/memory_service.py)
- [backend/src/models/chat_memory.py](../backend/src/models/chat_memory.py)
- [backend/src/models/query_log.py](../backend/src/models/query_log.py)
- [backend/src/services/query_service.py](../backend/src/services/query_service.py)
- [backend/src/inference/inference_engine.py](../backend/src/inference/inference_engine.py)

### 10.2 Kỹ thuật

- Lấy `QueryLog` gần nhất trong cùng:
  - `owner_id`
  - `collection_id`
  - `conversation_id`
- Ghép:
  - short-term memory: vài lượt gần nhất
  - summary memory: tóm tắt các lượt cũ hơn
- Cắt tổng memory theo giới hạn ký tự trước khi nhét vào prompt.

### 10.3 Giá trị

- Giữ được ngữ cảnh cho câu hỏi nối tiếp:
  - “nó là gì?”
  - “so sánh với cái trước”
  - “tiếp tục giải thích”
- Giảm rủi ro câu hỏi thứ hai bị hiểu như câu hỏi độc lập.

### 10.4 Đánh giá

**Mạnh**

- Scope rõ.
- Không trộn chat giữa user/collection.
- Không phụ thuộc token memory phức tạp.

**Yếu**

- Fallback im lặng nếu DB memory lỗi.
- Giới hạn theo ký tự, chưa phải token-aware.
- Chưa có test trực tiếp khóa behavior.

**Mức sản phẩm**

- **Ổn cho MVP**
- **Nên thêm regression test và observability**

## 11) Inference engine và query flow

Tham chiếu:

- [backend/src/inference/inference_engine.py](../backend/src/inference/inference_engine.py)
- [backend/src/services/query_service.py](../backend/src/services/query_service.py)
- [backend/src/api/v1/endpoints/query.py](../backend/src/api/v1/endpoints/query.py)

Kỹ thuật:

- Intent classification.
- Off-topic refusal.
- Retrieval + graph retrieval + rerank.
- Build prompt với evidence + memory context.
- LLM generate.
- Inject citations và verify claims.

**Giá trị**

- Đây là lớp orchestration cuối cùng biến retrieval thành câu trả lời có dẫn chứng.

**Mức sản phẩm**

- **Tốt**

## 12) Async pipeline và Celery

### 12.1 Kỹ thuật

Tham chiếu:

- [backend/src/tasks/celery_tasks.py](../backend/src/tasks/celery_tasks.py)
- [backend/src/services/material_service.py](../backend/src/services/material_service.py)
- [backend/src/services/parse_index_pipeline.py](../backend/src/services/parse_index_pipeline.py)

Kỹ thuật:

- Upload xong tạo job.
- Nếu eager mode thì chạy synchronous trong request.
- Nếu không thì enqueue Celery.
- Pipeline tự mark failed stage nếu lỗi xảy ra.

**Giá trị**

- Giảm blocking request.
- Dễ mở rộng sang background processing thật.

**Mức sản phẩm**

- **Tốt cho hệ thống vừa và nhỏ**

## 13) Frontend

### 13.1 Workspace UI

Tham chiếu:

- [frontend/src/main.tsx](../frontend/src/main.tsx)
- [frontend/src/pages/WorkspacePage.tsx](../frontend/src/pages/WorkspacePage.tsx)
- [frontend/src/state/workspace.tsx](../frontend/src/state/workspace.tsx)

Kỹ thuật:

- React Router với route redirect sang workspace.
- 3-pane workspace:
  - Sources
  - Chat
  - Studio / Evidence / Graph / Mindmap / Compare
- Mobile bottom tabs và slide-over panel.

**Giá trị**

- UI phù hợp demo sản phẩm tài liệu.
- Có center chat và right-side evidence panel.

**Mức sản phẩm**

- **Khá tốt**

### 13.2 State management

Kỹ thuật:

- `WorkspaceProvider` giữ workspace state trong React context.
- Persist workspace/materials lên localStorage.
- Tự suy ra `scopedMaterialIds` từ materials đã indexed.

**Giá trị**

- Giảm phụ thuộc server cho UI state.
- Tạo trải nghiệm “workspace” nhất quán.

**Điểm yếu**

- localStorage có thể stale nếu session thay đổi.
- Cần đồng bộ kỹ hơn khi có nhiều tab hoặc nhiều collection.

### 13.3 Frontend schema alignment

Frontend gọi API theo model trong `frontend/src/api/client.ts` và các endpoint query/materials.

Điểm đáng chú ý:

- Upload, query, compare, evidence, graph đều được map thành panel riêng.
- Evidence redirect chuyển tới workspace panel evidence.

**Mức sản phẩm**

- **Tốt về layout**
- **Cần tiếp tục siết schema/UX edge cases**

## 14) Data model

Tham chiếu chính:

- [backend/src/models/material.py](../backend/src/models/material.py)
- [backend/src/models/chunk.py](../backend/src/models/chunk.py)
- [backend/src/models/knowledge_graph.py](../backend/src/models/knowledge_graph.py)
- [backend/src/models/query_log.py](../backend/src/models/query_log.py)
- [backend/src/models/chat_memory.py](../backend/src/models/chat_memory.py)

Kỹ thuật:

- Beanie ODM với Pydantic models.
- Material có versioning parse/chunk/embed/index.
- Chunk lưu evidence trace và payload metadata.
- QueryLog lưu citations + retrieval trace.
- ChatSummaryMemory lưu summary hội thoại.

**Giá trị**

- Schema đủ linh hoạt cho document intelligence.
- Truy xuất và debug có trace tốt.

**Điểm yếu**

- Một số luồng còn phụ thuộc query filtering mà chưa thấy index tối ưu đầy đủ trong code.

**Mức sản phẩm**

- **Khá tốt**

## 15) Testing và evaluation

### 15.1 Unit tests

- Parser tests
- Chunking tests
- OCR/handwriting tests
- Retriever/reranker/indexer tests
- Query/inference tests

Tham chiếu:

- [backend/tests/test_processing/](../backend/tests/test_processing/)
- [backend/tests/test_rag/](../backend/tests/test_rag/)
- [backend/tests/test_inference/](../backend/tests/test_inference/)

### 15.2 Integration tests

- Real MongoDB + Qdrant integration retrieval test
- Corpus smoke test cho bộ tài liệu mẫu

Tham chiếu:

- [backend/tests/integration/test_retrieval_e2e.py](../backend/tests/integration/test_retrieval_e2e.py)
- [backend/tests/integration/test_sample_corpus_smoke.py](../backend/tests/integration/test_sample_corpus_smoke.py)

### 15.3 Đánh giá hiện tại

**Mạnh**

- Có test cho các lớp lõi.
- Có corpus smoke để khóa ingest/index behavior.

**Yếu**

- Chưa thấy benchmark retrieval end-to-end nhiều câu hỏi thật ở mức báo cáo sản phẩm.
- OCR corpus tiếng Việt scan vẫn cần quality gate chặt hơn.

**Mức sản phẩm**

- **Tốt cho nền tảng kỹ thuật**
- **Cần thêm eval thật cho phần QA**

## 16) Chất lượng kỹ thuật theo mức sản phẩm

| Kỹ thuật | Trạng thái | Nhận xét ngắn |
|---|---|---|
| Docling parsing | Tốt | Là parser chính hợp lý cho tài liệu học tập |
| Spreadsheet parsing | Tốt | Đã có summary + row verbalization |
| OCR printed scan | MVP tốt | Có chạy, nhưng scan bẩn còn nhiễu |
| Handwriting gate | Tốt | Có quality gate trước khi nhận evidence |
| Layout normalization | Rất quan trọng và đang ổn | Là lớp nền cho chunking/evidence |
| Layout-aware chunking | Tốt | Phù hợp tài liệu có heading/table |
| Semantic chunking | Nâng cao | Phụ thuộc embedder và cấu hình |
| Hybrid retrieval dense+sparse | Tốt | Đúng hướng cho search học thuật |
| Graph retrieval | Khá tốt | Hợp cross-document reasoning |
| Reranker | Tốt | Tăng precision sau retrieval |
| Refusal/confidence | Tốt nhưng cần tuning | Có guardrail nhưng threshold cần cân |
| Window memory | Ổn | Đúng scope, cần test và observability |
| Async pipeline | Tốt | Đã có Celery/job lifecycle |
| Frontend workspace | Khá tốt | Cấu trúc rõ, còn cần polish |
| Regression corpus smoke | Tốt | Bước quan trọng để product hóa |

## 17) Kết luận

Về mặt kỹ thuật, dự án hiện tại đã có một backbone đúng:

- ingestion có kiểm soát,
- parse/layout/evidence trace đầy đủ,
- chunking theo cấu trúc,
- retrieval hybrid,
- rerank,
- refusal,
- memory hội thoại,
- và workspace UI đủ để demo sản phẩm.

Điểm còn cần nâng để thật sự “product-grade” là:

- chất lượng OCR scan,
- observability cho pipeline và memory,
- auth/scoping cứng hơn,
- retrieval evaluation theo corpus thật,
- và test regression rộng hơn cho các tài liệu đầu vào xấu.

Nếu nhìn như một nền tảng document intelligence học thuật, codebase này đã đi qua giai đoạn prototype và đang ở mức **MVP khá hoàn chỉnh**. Nếu nhìn như sản phẩm triển khai cho người dùng thật, phần còn thiếu chủ yếu nằm ở **quality gating, security hardening, và eval/telemetry**.
