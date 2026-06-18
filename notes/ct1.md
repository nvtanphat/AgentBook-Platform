# Prism / AgentBook - Solution Architecture & Security Audit

Tài liệu này tổng hợp phân tích dựa trên codebase hiện tại và các file guide/config đã cung cấp.
Mục tiêu của bản này là đánh giá hệ thống theo góc nhìn Solution Architect và Security Auditor, đồng thời chỉ ra các cải tiến cần có để nâng hệ thống từ mức đồ án lên mức product.

## 1) Kiến Trúc & Data Flow

### Luồng từ upload PDF đến Knowledge Graph

Luồng hiện tại đi qua các module chính sau:

1. `frontend/src/components/workspace/SourcesPanel.tsx`
2. `frontend/src/api/client.ts`
3. `backend/src/api/v1/endpoints/materials.py`
4. `backend/src/services/material_service.py`
5. `backend/src/tasks/celery_tasks.py`
6. `backend/src/services/parse_index_pipeline.py`
7. `backend/src/processing/docling_parser.py`
8. `backend/src/processing/layout_normalizer.py`
9. `backend/src/processing/evidence_mapper.py`
10. `backend/src/processing/chunking.py`
11. `backend/src/processing/entity_extractor.py`
12. `backend/src/processing/entity_resolution.py`
13. `backend/src/processing/event_extractor.py`
14. `backend/src/rag/indexer.py`
15. `backend/src/models/chunk.py`
16. `backend/src/models/knowledge_graph.py`

### Đánh giá kiến trúc

- Pipeline là đúng hướng cho Graph RAG.
- Trace từ document đến evidence được giữ khá tốt qua `page`, `block_id`, `bbox`, `snippet_original`, `source_block_ids` và `EvidenceRef`.
- Graph không chỉ là metadata phụ vì entity, event, relation được lưu riêng trong MongoDB và gắn evidence refs.
- Kiến trúc tổng thể đã vượt mức “chat + vector search”, nhưng vẫn chưa đạt chuẩn product-grade nếu xét về vận hành, hardening và đánh giá chất lượng.

### Bottleneck async

Các điểm nghẽn hiện tại:

- `docker-compose.yml` dùng Celery worker với `--pool=solo`, nên worker xử lý tuần tự.
- `backend/src/services/material_service.py` có semaphore pipeline để tránh OOM, nhưng đổi lại throughput thấp.
- `backend/src/services/parse_index_pipeline.py` giữ nhiều stage nặng trong cùng một task.
- `backend/src/services/query_service.py` phần `compare()` chạy tuần tự theo từng dimension.
- `backend/src/rag/indexer.py` index theo batch nhưng vẫn là chuỗi xử lý tuần tự giữa embed, store và upsert.

Điểm tốt:

- `backend/src/inference/inference_engine.py` đã parallel hóa multi-query retrieval và graph retrieval bằng `asyncio.gather`.

Kết luận:

- Kiến trúc đúng.
- Nhưng throughput và khả năng scale đồng thời chưa đủ cho production.

## 2) Consistency Check

### Backend schema vs Frontend interface

Các schema chính hiện nhìn chung đã khớp:

- `backend/src/schemas/query.py` khớp với `frontend/src/api/client.ts`
- `backend/src/schemas/material.py` khớp với `frontend/src/api/client.ts`
- `backend/src/schemas/evidence.py` khớp với `frontend/src/api/client.ts` và `frontend/src/components/EvidencePanel.tsx`
- `backend/src/schemas/graph.py`, `backend/src/schemas/mindmap.py` khớp với `frontend/src/api/client.ts` và `frontend/src/components/workspace/studio/GraphTab.tsx`
- `backend/src/schemas/collection.py` khớp với `frontend/src/api/client.ts`

Kết luận:

- Không thấy mismatch schema nghiêm trọng trong tree hiện tại.
- Những mismatch từng xuất hiện trong audit cũ có vẻ đã được sửa.

### Evidence Trace metadata

MongoDB hiện lưu đủ metadata để truy vết:

- `backend/src/models/material.py`
- `backend/src/models/chunk.py`
- `backend/src/models/knowledge_graph.py`
- `backend/src/models/query_log.py`

Điểm quan trọng:

- `MaterialPageDocument` giữ page, block, bbox, ocr_confidence, reading_order.
- `Chunk` giữ `source_block_ids`, `source_pages` và versioning.
- `Entity`, `Event`, `Relation` đều giữ `EvidenceRef`.
- `QueryLog` lưu citations, confidence, refusal reason và latency.
- `backend/src/schemas/evidence.py` expose `evidence_blocks` để UI render trace đầy đủ.

Kết luận:

- Metadata đủ tốt cho Evidence Trace ở mức hiện tại.
- Nếu muốn lên product, cần bổ sung lineage, version history và audit trail rõ hơn cho từng lần re-index / re-parse.

## 3) Lỗ Hổng Hệ Thống

### Rủi ro bảo mật

#### High: Auth mặc định tắt, owner_id do client tự khai báo

Module liên quan:

- `backend/src/core/config.py`
- `backend/src/dependencies.py`
- `backend/src/api/v1/endpoints/*`
- `frontend/src/state/workspace.tsx`

Nhận định:

- `verify_owner_access()` chỉ chặn khi `api_auth_enabled = True`.
- Frontend default `ownerId = "user_demo"`.
- Nếu deploy mà không bật auth, bất kỳ ai biết `owner_id` đều có thể truy cập tài nguyên của owner đó.

Đây là lỗ hổng tenant isolation nghiêm trọng.

#### Medium: Secrets phụ thuộc `.env`

Module liên quan:

- `docker-compose.yml`
- `backend/.env`
- `backend/.env.example`
- `.gitignore`

Nhận định:

- `.gitignore` đã chặn `backend/.env`.
- Nhưng compose vẫn load trực tiếp `backend/.env`.
- Nếu file này bị lộ hoặc commit nhầm, secrets sẽ lộ ngay.

#### Medium: Redis/Qdrant chỉ an toàn nhờ bind loopback

Module liên quan:

- `docker-compose.yml`

Nhận định:

- Ports đều bind `127.0.0.1`, nên ổn cho local.
- Chưa có auth riêng cho Redis/Qdrant.
- Nếu chuyển compose sang server khác mà giữ nguyên kiểu triển khai này, bảo mật phụ thuộc hoàn toàn vào network boundary.

### File/path safety

Điểm tốt:

- `backend/src/core/security.py`
- `backend/src/api/v1/endpoints/materials.py`
- `backend/src/services/material_service.py`

Có:

- magic-byte validation
- MIME allowlist
- checksum filename
- path traversal guard
- raw file serving có `relative_to(data_root)`

### Resilience

Khả năng chịu lỗi hiện ở mức trung bình:

- Query side có fallback/refusal khi retrieval hoặc LLM fail.
- Ingest side chưa đủ bền nếu enqueue task fail.
- `backend/src/main.py` phụ thuộc Qdrant lúc startup.
- `docker-compose.yml` chưa có healthcheck.

Kết luận:

- Hệ thống chưa đạt mức production hardening.
- Điểm yếu lớn nhất là auth/tenant isolation và readiness/failover của các service nền.

## 4) Tối Ưu Hóa RAG

### Chunking

Module:

- `backend/src/processing/chunking.py`
- `backend/src/core/tokenizer.py`
- `backend/src/processing/layout_normalizer.py`

Đánh giá:

- Layout-aware chunking là đúng hướng.
- Có giữ heading/table/figure/context tốt hơn fixed-size chunking.
- Semantic chunking giúp tốt hơn RAG vector thuần.

### Graph Retrieval

Module:

- `backend/src/rag/graph_retriever.py`
- `backend/src/rag/query_router.py`
- `backend/src/inference/inference_engine.py`
- `backend/src/api/v1/endpoints/graph.py`

Đánh giá:

- Hệ thống có graph reasoning thật: entity/event/relation, 1-hop và 2-hop paths, evidence refs.
- Nhưng graph retrieval chỉ được bật mạnh ở một số route nhất định.
- `compare()` và `summarize()` vẫn thiên về hybrid retrieval + rerank, chưa graph-first.

### Chất lượng RAG hiện tại

Đây là hybrid RAG + graph augmentation, chưa phải cross-document reasoning đầy đủ.

Điểm mạnh:

- Có citation.
- Có evidence blocks.
- Có rerank.
- Có claim verification ở một số luồng.
- Có hỗ trợ tiếng Việt và rewrite query.

Điểm yếu:

- Graph chưa tham gia rộng vào mọi dạng truy vấn.
- Chưa có evaluation loop chuẩn production.
- Chưa có metric tracking liên tục cho recall, citation accuracy, faithfulness, contradiction rate.
- Chưa có caching chiến lược cho retrieval / rerank / synthesis.

Kết luận:

- Dùng được cho đồ án và nội bộ.
- Chưa đủ ổn định để gọi là product-grade RAG.

### Cải thiện để lên product

Nếu muốn đạt chuẩn sản phẩm, cần bổ sung các lớp sau:

1. **Retrieval evaluation layer**
   - Benchmark Recall@k, MRR, citation precision, faithfulness, refusal accuracy.
   - Lưu test set chuẩn theo domain.
   - Tự động chạy regression khi đổi prompt, chunking, reranker hoặc graph logic.

2. **Graph-first routing**
   - Không chỉ dùng graph cho route relation.
   - Cho graph tham gia vào compare, claim-check, synthesis và multi-hop QA.
   - Ưu tiên paths có evidence score cao thay vì chỉ dựa vào top-k vector chunks.

3. **Evidence ranking và confidence calibration**
   - Calibrate confidence score bằng dữ liệu thực nghiệm.
   - Tách rõ confidence của retrieval, rerank và answer synthesis.
   - Khi evidence yếu, hệ thống phải refuse hoặc yêu cầu làm rõ thay vì trả lời đoán.

4. **Caching và latency control**
   - Cache query embedding, query rewrite và graph neighborhood.
   - Có timeout và fallback theo từng stage.
   - Đo p95 latency theo route.

5. **Versioned ingestion**
   - Mỗi lần re-parse/re-index phải có version và lineage rõ ràng.
   - Không ghi đè mù lên artifact cũ.
   - Có khả năng rollback index nếu pipeline lỗi.

6. **LLM contract hardening**
   - Answer phải sinh theo schema cố định.
   - Response parser cần reject output lệch format.
   - Có guardrail để chặn hallucination khi evidence không đủ.

## 5) Roadmap

### 3 cải tiến quan trọng nhất để thành product

#### 1. Bật auth và tenant isolation thật sự

Modules:

- `backend/src/dependencies.py`
- `backend/src/core/config.py`
- `frontend/src/state/workspace.tsx`
- toàn bộ `backend/src/api/v1/endpoints/*`

Việc cần làm:

- Bắt auth mặc định.
- Không tin `owner_id` do client gửi.
- Dùng JWT/RBAC hoặc IdP.
- Gắn scope theo owner/collection/material ở server.
- Tách public API và internal API rõ ràng.

#### 2. Hardening async và vận hành

Modules:

- `docker-compose.yml`
- `backend/src/tasks/celery_tasks.py`
- `backend/src/services/material_service.py`
- `backend/src/rag/vector_store.py`

Việc cần làm:

- Thêm healthcheck và readiness probe.
- Tách queue theo job type.
- Retry/backoff cho Qdrant, Redis và LLM.
- Tăng worker concurrency.
- Bổ sung idempotency cho task.
- Có dead-letter / failure tracking cho job lỗi.

#### 3. Productize RAG contract và graph reasoning

Modules:

- `backend/src/rag/*`
- `backend/src/guardrails/*`
- `backend/src/api/v1/endpoints/query.py`
- `frontend/src/api/client.ts`

Việc cần làm:

- Sinh shared contract từ OpenAPI.
- Thêm contract tests frontend/backend.
- Đưa graph retrieval vào nhiều route hơn.
- Cải thiện entity resolution và evidence scoring.
- Thêm evaluation set và regression benchmark.
- Theo dõi Recall@k, citation accuracy, false refusal rate, latency p95.

### Checklist ngắn để gọi là product-ready

- Auth bật mặc định.
- Không có tài nguyên đa tenant nào dựa trên `owner_id` do client tự khai báo.
- Có healthcheck cho toàn bộ service nền.
- Có retry/backoff và quan sát lỗi.
- Có evaluation suite cho RAG.
- Có versioning cho ingest và index.
- Có confidence calibration và refusal hợp lý.
- Có benchmark latency và throughput.
- Có contract test giữa backend và frontend.

## 6) Kết Luận

Hệ thống hiện tại đã có nền tảng Graph RAG thật và tốt hơn mức demo thông thường. Tuy nhiên, để lên product, trọng tâm không còn là “có chạy được hay không” mà là:

- Có an toàn không
- Có ổn định không
- Có đo được chất lượng không
- Có chịu tải được không
- Có truy vết và rollback được không

Hiện tại, phần mạnh nhất là trace/evidence và kiến trúc retrieval. Phần cần nâng cấp mạnh nhất là auth, resilience, observability và evaluation loop cho RAG.
