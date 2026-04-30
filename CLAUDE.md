# AGENTBOOK - AI AGENT WORKING RULES

This file tells an AI coding agent how to behave in this project.

It is not the product plan.
The product plan is:

`AgentBook_Implementation_Plan.md`

The agent must read the plan when it needs scope, API details, architecture, use cases, roadmap, or acceptance criteria.

---

## 1. Agent Mission

You are working on AgentBook, an educational document assistant grounded in uploaded learning materials.

Your job is to help build the system safely and coherently:

- Respect the plan.
- Keep evidence trace intact.
- Avoid hallucinated behavior in code and AI outputs.
- Prefer simple MVP implementation over over-engineered research features.
- Finish the requested task end to end when possible.

A correct refusal is better than a fluent hallucination.

---

## 2. Source Of Truth

Use `AgentBook_Implementation_Plan.md` for:

- product scope
- top 5 demo use cases
- API endpoints
- architecture
- folder structure
- database schema
- retrieval flow
- security policy
- roadmap
- evaluation requirements

Use this file for:

- agent behavior
- coding discipline
- implementation safety
- review habits
- testing expectations

If this file and the plan conflict, follow the plan and mention the conflict.

---

## 3. Before You Code

Before writing or changing code, identify the work in plain terms:

- What layer is touched?
- What data comes in?
- What data goes out?
- What must remain traceable?
- What can fail?
- How will this be tested?

Do not write code first and reason later.

For small edits, keep this reasoning brief.
For larger tasks, state a short plan before editing.

---

## 4. Work Like An Agent

Do not stop at suggestions if the user asked you to fix or implement something.

Default behavior:

1. Inspect the relevant files.
2. Compare with the plan.
3. Make the smallest correct change.
4. Run relevant checks if possible.
5. Report what changed and what remains.

Ask the user only when:

- the requested behavior conflicts with the plan
- the collection/material scope is ambiguous and unsafe to assume
- a schema change may break evidence trace
- the task requires a stretch feature not in MVP
- external credentials or network access are required

---

## 5. MVP Discipline

Do not add features just because they are interesting.

MVP priorities:

- upload and manage learning documents
- parse/OCR/index documents
- ask grounded questions with citations
- cross-lingual query VI over EN sources
- mindmap/graph demo from collection
- cross-document comparison
- claim verification/refusal
- summary/study guide as supporting features

Avoid unless explicitly requested:

- Neo4j
- full agents/ReAct planners
- visual retrieval with CLIP/ColPali
- audio podcast
- QLoRA/fine-tuning
- complex Excel/CSV handling
- complete quiz/flashcard product

---

## 6. Stack Rules

Use the stack from the plan:

- FastAPI
- Beanie + MongoDB Atlas
- Qdrant
- Celery + Redis
- Docling
- PaddleOCR for printed scans
- separate handwriting reader for clear handwritten images
- BGE-M3 dense + sparse retrieval
- BGE reranker
- Qwen3 4B via Ollama when feasible, API fallback when needed

Do not introduce a new framework or database without explicit user approval.

---

## 7. Folder And Layer Rules

Follow the project structure in the plan.

General rules:

- `api/v1/endpoints/` stays thin.
- `services/` owns business workflows.
- `processing/` owns parsing, OCR, handwriting, layout normalization.
- `rag/` owns embedding, Qdrant, retrieval, reranking, graph retrieval.
- `guardrails/` owns refusal, confidence policy, claim verification.
- `schemas/` owns Pydantic request/response schemas.
- `models/` owns Beanie documents.
- `tasks/` owns Celery jobs.

Do not put heavy logic inside route handlers.
Do not create circular imports.

---

## 8. Evidence Trace Is Sacred

Every chunk, citation, and evidence object must preserve:

- `owner_id`
- `collection_id`
- `material_id` or `doc_id`
- `document_name`
- `page` or `page_numbers`
- `block_id` or `source_block_ids`
- `bbox` when available
- `snippet_original`
- `source_language`
- confidence score

Never drop these fields during:

- chunking
- embedding
- retrieval
- reranking
- answer synthesis
- summary/study guide generation
- graph/mindmap generation

If bbox is unavailable, keep block-level citation and snippet.
Never invent bbox.

---

## 9. Retrieval Rules

All retrieval must be scoped.

Every Qdrant search and MongoDB query over user data must filter by:

- `owner_id`
- `collection_id`
- or explicit `material_ids`

Global retrieval is forbidden in MVP code.

MVP retrieval uses:

- Qdrant dense vector
- BGE-M3 sparse vector
- RRF fusion when multiple branches exist
- bounded reranking

Return scored chunks or structured retrieval results.
Do not return raw strings from retrieval services.

---

## 10. Cross-Lingual Rules

Never translate the full source document before indexing.

For Vietnamese query over English documents:

1. Detect query language.
2. Keep original Vietnamese query.
3. Create translated English query when useful.
4. Retrieve using both original and translated query.
5. Fuse results with RRF.
6. Rerank.
7. Generate answer in Vietnamese.
8. Keep citation snippet in original English.

`snippet_translated` is optional.
`snippet_original` is mandatory.

---

## 11. OCR And Handwriting Rules

Do not treat PaddleOCR as the handwriting solution.

Printed scans:

- use `ocr_engine.py`
- suitable for slide scans, textbooks, printed exercises, captions, tables

Handwriting:

- first run `image_quality_checker.py`
- then run `handwriting_reader.py`
- may use VLM/API fallback or a handwriting OCR pipeline
- must pass confidence gate before becoming primary evidence

If the image is blurry, skewed, dark, shadowed, or low confidence, warn/refuse instead of pretending the answer is grounded.

---

## 12. Refusal Rules

The system must not answer beyond evidence.

Refuse when:

- no relevant evidence is found
- reranker score is below configured threshold
- source scope is missing or unsafe
- handwriting quality/confidence is too low
- graph relation has no evidence path
- user asks outside uploaded materials

If the user gives a false premise, correct it using evidence.
Do not answer as if the false premise is true.

Thresholds must come from config, not hardcoded values.

---

## 13. Upload Safety

Every upload flow must enforce:

- allowlist: `pdf`, `docx`, `pptx`, `png`, `jpg`, `jpeg`, `csv`, `xlsx`
- MVP size limit: 20 MB/file unless config overrides it
- MIME/magic-byte validation
- UUID/checksum safe filename
- path traversal guard with resolved absolute path
- storage under configured media/data directory
- rate limit for upload and query endpoints
- `owner_id` and `collection_id` isolation

Never store raw file bytes in MongoDB.

---

## 14. Secrets And Config

Never hardcode:

- API keys
- database URLs
- model credentials
- Qdrant tokens
- thresholds
- top_k values
- prompts
- file size limits

Use `.env`, `.env.example`, and YAML config.

---

## 15. Coding Style

Use clear typed code.

Rules:

- Pydantic v2 style.
- Type hints on public functions.
- Structured schemas for request/response/domain objects.
- No `print()` in backend code.
- No vague TODOs.
- No placeholder business logic.
- No unscoped DB/vector queries.
- No swallowed exceptions.

Use structured logging with enough context:

- `owner_id`
- `collection_id`
- `material_id`
- `job_id`
- stage/service name

---

## 16. Async And Celery

FastAPI routes should be `async def`.
Beanie queries must be awaited.

Heavy work goes through Celery:

- parsing
- OCR
- handwriting reading
- embedding
- indexing

Celery job records should preserve:

- `job_id`
- `material_id`
- `stage`
- `status`
- `retry_count`
- `failed_stage`
- `last_error`
- `started_at`
- `finished_at`

---

## 17. LLM Output Is Untrusted

Always validate LLM outputs.

For JSON:

```python
try:
    parsed = ResponseSchema.model_validate_json(raw_output)
except ValidationError:
    ...
```

Retry once with a stricter prompt if appropriate.
Otherwise return a controlled failure.

Do not store malformed extraction results.
Do not treat LLM-created graph facts as truth unless they have evidence refs and confidence.

---

## 18. Testing Expectations

When changing code, add or update focused tests when practical.

Important test areas:

- upload validation
- path traversal guard
- owner/collection isolation
- parser output normalization
- OCR/handwriting confidence gates
- evidence trace preservation
- dual-query cross-lingual retrieval
- refusal behavior
- citation formatting
- claim verification
- summary/study guide grounding

Retrieval should expose enough scores/traces to evaluate:

- Recall@k
- MRR@k
- nDCG@k
- citation accuracy
- false accept rate
- false refusal rate

---

## 19. Self-Review Before Final Response

Before finishing a coding task, check:

- Did I follow the plan?
- Did I keep owner/collection scope?
- Did I preserve evidence trace?
- Did I avoid hardcoded thresholds?
- Did I handle empty results?
- Did I handle low confidence/refusal paths?
- Did I avoid introducing stretch features accidentally?
- Did I run a relevant check or explain why not?

Then respond briefly with:

- what changed
- what was verified
- any remaining risk

---

## 20. Communication Style

Match the user's language.

Be direct.
Do not over-explain simple changes.
Do not pretend uncertainty is certainty.
Do not repeat the whole plan unless asked.

When the user asks for a code change, implement it.
When the user asks for review, lead with issues.
When the user asks for status, summarize current state and continue if work remains.
