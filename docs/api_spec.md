# AgentBook API Spec

Base path: `/api/v1`

## Materials

- `POST /materials/upload`
  - Multipart fields: `metadata` JSON and `file`.
  - Returns `material_id`, `collection_id`, `job_id`, `status`, checksum, and storage path.

## Query

- `POST /query/ask`
  - Body: `owner_id`, `collection_id` or `material_ids`, `query`, optional `top_k`.
  - Returns answer, language metadata, citations, confidence, and refusal state.

- `POST /query/compare`
  - Body: `owner_id`, scope, `topic`, `dimensions`.
  - Returns comparison table cells with citations.

- `POST /query/summarize`
  - Body: `owner_id`, `collection_id` or `material_id`, `scope`.
  - Returns grounded summary and citations.

- `POST /query/study-guide`
  - Body: `owner_id`, `collection_id` or `material_id`, `scope`, `format`.
  - Returns overview, key concepts, outline, citations.

## Evidence

- `GET /evidence/{doc_id}/{page}?owner_id=...&collection_id=...`
  - Returns page blocks, snippets, bbox refs, confidence, and source path.

## Graph

- `POST /graph`
  - Body: `owner_id`, `collection_id` or `material_ids`, optional `root_topic`.
  - Returns graph nodes, relation edges, confidence scores, and evidence refs.

- `POST /graph/mindmap`
  - Body: `owner_id`, `collection_id` or `material_ids`, optional `root_topic`.
  - Returns topic nodes and evidence citations.

## Admin

- `GET /admin/metrics`
  - Returns total docs, failed jobs, indexed docs, query stats, retrieval stats, and feedback count.

- `POST /admin/feedback`
  - Body: `owner_id`, `query_log_id`, `rating`, optional `comment`.
  - Persists feedback and embeds a copy in the query log.
