# RAG Pipeline Review - 2026-06-14

## Ket luan ngan

Pipeline RAG hien tai dung huong va co nhieu thanh phan can thiet: ingestion da dinh dang, evidence schema, semantic/layout chunking, BGE-M3 dense+sparse, Qdrant RRF, cross-encoder rerank, GraphRAG, multimodal retrieval, refusal policy, SLEC va citation aligner.

Tuy nhien, chua nen coi la "chuan production" vi cac diem yeu lon nam o retrieval finalization va citation grounding. Eval hien tai cung xac nhan dieu nay: `backend/eval_results/e2e_judge_report.md` ghi groundedness 0.462, citation_correctness 0.394, answer_relevance 0.668, trong khi refusal_correctness dat 0.908.

## Bang chung da kiem

- Inventory repo: 629 text/config/source files, trong do `backend/src` 162 file, `backend/tests` 59 file, `config` 7 file, `evaluation` 23 file, `frontend/src` 29 file.
- Targeted tests lien quan RAG/guardrails/citation:
  - `backend/tests/test_rag/test_retriever.py`
  - `backend/tests/test_guardrails/test_evidence_validator.py`
  - `backend/tests/test_guardrails/test_citation_aligner.py`
  - `backend/tests/test_guardrails/test_quality_gate.py`
  - Ket qua: 36 passed.
- Eval reports:
  - `backend/eval_results/e2e_judge_report.md`: citation/groundedness khong dat nguong smoke.
  - `eval_results/e2e_eval_v22_phaseB_only.jsonl`: v22 smoke tot hon nhung semantic_faithfulness trung binh chi khoang 0.58 va answer_relevance khoang 0.716.

## Diem manh

1. Ingestion co evidence trace day du: page, block, bbox, audio timestamp, table metadata.
2. Retrieval co hybrid dense+sparse, RRF, lexical fallback va modality-aware extra pass.
3. Guardrails co nhieu lop: pre-generation evidence validation, refusal policy, claim verifier, SLEC, citation aligner, quality gate.
4. Query pipeline da co route-specific hooks cho factual, summarization, comparison, claim_check va graph_relation.
5. Co debug tools va eval dataset de dieu tra retrieval/chunking/reranking.

## Van de uu tien

### P0 - Agentic path co the bo qua reranker sai

`backend/src/agentic/service.py` co nhan dinh `HybridTextSearchTool already reranked internally`, nhung `backend/src/agentic/tools/hybrid_text_search.py` chi goi `HybridRetriever.retrieve()`. Retriever tra ve RRF/hybrid chunks, khong set `rerank_score`.

He qua: voi `single_pass`, agentic path co the dua chunk chua qua cross-encoder vao context cuoi.

De xuat:
- Bo nhanh skip rerank trong agentic single-pass; hoac
- Dua rerank that vao `HybridTextSearchTool` va chi skip khi chunk da co `rerank_score`.

### P0 - Adaptive fast-path dang khong dung ten va khong co score rieng

`HybridRetriever.retrieve_fast()` thuc chat goi `retrieve()`, tuc la hybrid dense+sparse/RRF, khong phai dense-only. `RetrievedChunk.dense_score` ton tai trong schema nhung khong duoc set khi hydrate, trong khi `InferenceEngine` log/calibrate bang `dense_score`.

He qua: fast-path skip reranker/graph/multi-query dua tren `fused_score`, nhung comment va log noi dense confidence. Day la vung de gay regression kho debug.

De xuat:
- Tam tat `adaptive_retrieval.enabled` neu uu tien chat luong.
- Neu can giu latency win: luu dense/sparse/fused score rieng, them test cho fast-path, va chi skip reranker khi co evaluation gate ro rang.

### P0 - Citation van chu yeu chunk-level, chua sentence/block-level

`ResponseParser.citations_from_chunks()` tao 1 citation cho 1 chunk. Chunk co the gom nhieu block, nen citation co the tro sang block gan dung nhung khong phai block ho tro cau tra loi.

He qua: citation validity/range co the pass, nhung citation correctness thap.

De xuat:
- Sau SLEC, dung `supporting_block_ids` de map moi cau ve block ho tro.
- Neu LLM citation `[N]` tro chunk, response parser nen chon citation snippet theo block co support score cao nhat cho cau do.
- Them test "answer sentence cites wrong block inside same chunk" de bat loi hien tai.

### P1 - Can block-level evidence compression sau rerank

Chunk 512 tokens va toi da 8 block tot cho recall, nhung qua rong cho generation/citation.

De xuat:
- Sau rerank chunk, rerank lai evidence blocks trong top chunks theo query.
- Prompt chi dua block/snippet top-N thay vi toan bo chunk khi route la factual/claim_check/table lookup.
- Giu full chunk trong metadata de UI trace, nhung context cho LLM nen gon hon.

### P1 - QualityGate chi ghi trace, chua la gate that

`QualityGate.should_refuse` chi true khi tat ca stages fail. Citation fail rieng le chi sua marker invalid, khong ngan cau tra loi co citation dung range nhung sai block.

De xuat:
- Neu citation stage FAIL hoac SLEC FAIL thi route factual/claim_check nen refuse/repair.
- Cho summarization/comparison co the downgrade thanh partial warning thay vi hard refuse.

### P1 - Eval can duoc dua thanh quality gate bat buoc

Targeted unit tests xanh, nhung judge report fail. Nghia la unit tests phu dung module nho, chua phu end-to-end correctness.

De xuat nguong truoc khi goi pipeline "chuan":
- groundedness >= 0.70
- citation_correctness >= 0.85
- answer_relevance >= 0.70
- refusal_correctness >= 0.85
- forbidden_claims_violated = 0

## Thu tu lam tiep

1. Sua agentic rerank skip.
2. Tat hoac sua adaptive fast-path.
3. Them block-level citation mapping tu SLEC.
4. Them block-level evidence compression.
5. Chay lai judge 50 cau va v22 smoke.
6. Chi khi eval dat nguong moi coi pipeline da chuan.

## Trang thai

Chua chuan production. Nen coi hien tai la pipeline RAG nang cap tot ve architecture, nhung can sua cac diem P0/P1 tren de dam bao grounding va citation correctness.
