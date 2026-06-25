export const API_BASE_URL = (import.meta.env.VITE_AGENTBOOK_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
export const API_V1_BASE_URL = `${API_BASE_URL}/api/v1`;
const AUTH_TOKEN_STORAGE_KEY = "prism.auth.access";  // unified with state/auth.tsx
const LEGACY_TOKEN_KEY = "prism.auth.token";

function authHeaders(): HeadersInit {
  const token =
    localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) ||
    localStorage.getItem(LEGACY_TOKEN_KEY) ||
    import.meta.env.VITE_AGENTBOOK_AUTH_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Broadcast 401s so AuthProvider can clear session and bounce to login.
function notifyUnauthorized() {
  try {
    window.dispatchEvent(new CustomEvent("prism:auth:unauthorized"));
  } catch {}
}

type ApiEnvelope<T> = {
  success: boolean;
  message: string;
  data: T | null;
  error: string | null;
};

export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Citation = {
  doc_id: string;
  doc_name: string;
  page: number | null;
  pages: number[];
  block_id: string | null;
  block_type: string | null;
  snippet_original: string;
  snippet_translated: string | null;
  cited_span: string | null;
  bbox: BoundingBox | null;
  role: string;
  source_language: string;
  confidence: number;
  figure_image_url?: string | null;
  evidence_blocks?: EvidenceBlock[];
};

export type MaterialUploadMetadata = {
  owner_id: string;
  collection_id?: string | null;
  collection_name?: string | null;
  collection_description?: string | null;
  subject?: string | null;
  topic?: string | null;
  language: string;
  modality: string;
  source_type?: string | null;
  version: string;
  extra_metadata?: Record<string, unknown>;
};

export type MaterialUploadResponse = {
  material_id: string;
  doc_id: string;
  collection_id: string;
  job_id: string;
  status: string;
  stage: string;
  filename: string;
  original_name: string;
  checksum_sha256: string;
  file_size_bytes: number;
  storage_path: string;
};

export type MaterialBatchUploadItem = {
  filename: string;
  success: boolean;
  data: MaterialUploadResponse | null;
  error: string | null;
};

export type MaterialBatchUploadResponse = {
  results: MaterialBatchUploadItem[];
};

export type MaterialInfo = {
  material_id: string;
  collection_id: string;
  owner_id: string;
  filename: string;
  original_name: string;
  file_type: string;
  status: string;
  subject: string | null;
  topic: string | null;
  page_count: number | null;
  version: string;
};

export type DebugBlock = {
  block_id: string;
  block_index: number;
  block_type: string;
  content: string;
  language: string;
  bbox: BoundingBox | null;
  ocr_confidence: number | null;
  reading_order: number;
};

export type DebugPage = {
  page_number: number;
  width: number | null;
  height: number | null;
  ocr_confidence: number | null;
  blocks: DebugBlock[];
};

export type DebugChunk = {
  chunk_id: string;
  content: string;
  language: string;
  modality: string;
  token_count: number | null;
  source_block_ids: string[];
  source_pages: number[];
  chunk_strategy: string;
  embedding_model: string;
};

export type MaterialDebugResponse = {
  material_id: string;
  collection_id: string;
  owner_id: string;
  original_name: string;
  file_type: string;
  status: string;
  modality: string;
  language: string;
  page_count: number;
  pages: DebugPage[];
  chunks: DebugChunk[];
  qdrant_vector_count: number;
  raw_image_url: string | null;
};

export type MaterialStatusResponse = {
  material_id: string;
  collection_id: string;
  status: string;
  stage: string;
  progress_pct: number;
  failed_stage: string | null;
  error_message: string | null;
};

export type QueryRequest = {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  conversation_id?: string;
  query: string;
  top_k?: number | null;
  answer_language?: string | null;
  rag_flags?: { agentic_rag_enabled?: boolean; reranker_enabled?: boolean };
};

export type ReasoningStep = {
  step_type: 'retrieve' | 'traverse' | 'synthesize';
  entities: string[];
  relations: string[];
  confidence: number;
  description: string;
};

export type AgentTraceStep = {
  name: string;
  status: "pending" | "running" | "completed" | "skipped" | "failed";
  query?: string | null;
  tool?: string | null;
  duration_ms?: number | null;
  sources_requested?: number | null;
  sources_covered?: number | null;
  evidence_count?: number | null;
  warning?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type AgentVerification = {
  verdict: string;
  confidence: number;
  warning?: string | null;
  unsupported_sentence_count?: number | null;
  invalid_citation_count?: number | null;
  repair_attempted?: boolean;
};

export type AgentTrace = {
  plan_type: string;
  steps: AgentTraceStep[];
  repair_attempted: boolean;
  verification?: AgentVerification | null;
};

export type SentenceSupport = {
  index: number;
  text: string;
  status: "supported" | "partial" | "unsupported";
  score: number;
  supporting_block_ids: string[];
  citation_refs: number[];
};

export type SentenceCoverageReport = {
  enabled: boolean;
  total_sentences: number;
  supported_count: number;
  partial_count: number;
  unsupported_count: number;
  dropped_count: number;
  coverage_ratio: number;
  refused: boolean;
  sentences: SentenceSupport[];
};

export type QueryResponse = {
  answer: string;
  answer_language: string;
  query_language: string;
  translated_query: string | null;
  source_languages: string[];
  citations: Citation[];
  confidence: number;
  was_refused: boolean;
  refusal_reason: string | null;
  reasoning_path: ReasoningStep[];
  coverage?: CoverageReport | null;
  agent_trace?: AgentTrace | null;
  sentence_coverage?: SentenceCoverageReport | null;
  used_entity_ids?: string[];
  used_relation_ids?: string[];
};

export type CompareRequest = {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  topic: string;
  dimensions: string[];
  top_k?: number | null;
};

export type ComparisonCell = {
  dimension: string;
  value: string;
  source: string;
  citation: Citation | null;
  confidence: number;
  source_id?: string | null;
  citation_ids?: string[];
  missing_evidence?: boolean;
};

export type CoverageSource = {
  material_id: string;
  name: string;
  covered: boolean;
};

export type CoverageReport = {
  requested_count: number;
  covered_count: number;
  sources: CoverageSource[];
};

export type CompareSource = {
  source_id: string;
  name: string;
};

export type CompareMatrixCell = {
  value: string;
  confidence: number;
  citation_ids: string[];
  missing_evidence: boolean;
};

export type DimensionCoverage = {
  dimension: string;
  requested_count: number;
  covered_count: number;
  missing_source_ids: string[];
};

export type CompareResponse = {
  topic: string;
  comparison_table: ComparisonCell[];
  conflicts: string[];
  citations: Citation[];
  coverage?: CoverageReport | null;
  sources?: CompareSource[];
  matrix?: Record<string, Record<string, CompareMatrixCell>>;
  cell_citations?: Record<string, string[]>;
  dimension_coverage?: DimensionCoverage[];
};

export type EvidenceBlock = {
  block_id: string;
  block_type: string;
  page: number;
  snippet_original: string;
  source_language: string;
  bbox: BoundingBox | null;
  confidence: number | null;
  material_id: string | null;
  doc_name: string | null;
  // Audio-only — present when block came from audio transcription
  audio_start_seconds?: number | null;
  audio_end_seconds?: number | null;
  audio_file?: string | null;
  // Figure-only — API URL to cropped/embedded figure image
  figure_image_url?: string | null;
};

export type EvidencePageResponse = {
  doc_id: string;
  doc_name: string;
  page: number;
  blocks: EvidenceBlock[];
  source_filename: string;
  raw_image_url: string | null;
  file_type: string | null;
};

export type MindmapNode = {
  id: string;
  label: string;
  entity_type?: string;  // NEW: Explicit type for better clustering
  summary: string | null;
  children: MindmapNode[];
  citations: Array<Record<string, string | number>>;
  collapsed?: boolean;  // NEW: For collapsible branches
};

export type MindmapResponse = {
  root_topic: string;
  nodes: MindmapNode[];
};

export type GraphNode = {
  id: string;
  label: string;
  type: string;
  confidence: number | null;
  mention_count?: number;
  degree?: number;
  importance?: number;          // Phase 2 — PageRank score [0, 1]
  community?: number;           // Phase 2 — Louvain community id
  community_label?: string | null;
  is_hub?: boolean;             // Phase 2 — top 10% by importance
  is_focused?: boolean;         // Focus mode — primary entity from citations
  source_docs?: string[];
  evidence_refs?: Array<Record<string, string | number>>;
  evidence_text?: string | null;   // server-verified passage of the node's primary mention block
};

export type GraphEdge = {
  source: string;
  target: string;
  relation_type: string;
  source_label?: string | null;
  target_label?: string | null;
  confidence: number | null;
  evidence_count?: number;
  evidence_refs: Array<Record<string, string | number>>;
  evidence_text_chunk?: string | null;
};

export type GraphResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type AdminMetricsResponse = {
  total_docs: number;
  failed_jobs: number;
  indexed_docs: number;
  query_stats: {
    total_queries: number;
    refused_queries: number;
    average_confidence: number;
    average_latency_ms: number;
  };
  retrieval_stats: {
    average_top_k: number;
    average_sources_used: number;
    average_retrieval_time_ms: number;
  };
  feedback_count: number;
};

export type HealthResponse = {
  status: string;
  service: string;
};

export type CollectionSummary = {
  collection_id: string;
  name: string;
  owner_id: string;
  subject: string | null;
  description: string | null;
  material_count: number;
  indexed_material_count: number;
  retrievable_chunk_count: number;
  latest_material_name: string | null;
  created_at: string;
  updated_at: string;
};

export type CollectionDashboard = CollectionSummary & {
  entity_count: number;
  status_counts: Record<string, number>;
  language_counts: Record<string, number>;
};

async function parseError(response: Response) {
  const fallback = `Noelys API request failed: ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: unknown; message?: string; error?: string };
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg ?? JSON.stringify(item)).join("; ");
    return payload.error ?? payload.message ?? fallback;
  } catch {
    return fallback;
  }
}

async function request<T>(path: string, init?: RequestInit, useApiV1 = true): Promise<T> {
  const response = await fetch(`${useApiV1 ? API_V1_BASE_URL : API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  if (response.status === 401 && !path.startsWith("/auth/")) {
    notifyUnauthorized();
  }
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  const payload = (await response.json()) as ApiEnvelope<T> | T;
  if (payload && typeof payload === "object" && "success" in payload) {
    const envelope = payload as ApiEnvelope<T>;
    if (!envelope.success || envelope.data === null) {
      throw new Error(envelope.error ?? envelope.message);
    }
    return envelope.data;
  }
  return payload;
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function apiPost<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export async function checkHealth() {
  return request<HealthResponse>("/health", undefined, false);
}

export function getAdminMetrics() {
  return apiGet<AdminMetricsResponse>("/admin/metrics");
}

export function listCollections(ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return apiGet<CollectionSummary[]>(`/collections?${params.toString()}`);
}

export function getCollectionDashboard(ownerId: string, collectionId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return apiGet<CollectionDashboard>(`/collections/${encodeURIComponent(collectionId)}/dashboard?${params.toString()}`);
}

export function createCollection(payload: {
  owner_id: string;
  name: string;
  subject?: string | null;
  description?: string | null;
}) {
  return apiPost<CollectionSummary>("/collections", payload);
}

export function updateCollection(collectionId: string, payload: {
  owner_id: string;
  name?: string | null;
  subject?: string | null;
  description?: string | null;
}) {
  return request<CollectionSummary>(`/collections/${encodeURIComponent(collectionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function deleteCollection(collectionId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return request<Record<string, number>>(`/collections/${encodeURIComponent(collectionId)}?${params.toString()}`, { method: "DELETE" });
}

export function deleteMaterial(materialId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return request<Record<string, number>>(`/materials/${encodeURIComponent(materialId)}?${params.toString()}`, { method: "DELETE" });
}

export function retryMaterial(materialId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return request<{ material_id: string; job_id: string; status: string }>(
    `/materials/${encodeURIComponent(materialId)}/retry?${params.toString()}`,
    { method: "POST" },
  );
}

export function listMaterials(ownerId: string, collectionId?: string | null) {
  const params = new URLSearchParams({ owner_id: ownerId });
  if (collectionId) params.set("collection_id", collectionId);
  return apiGet<MaterialInfo[]>(`/materials?${params.toString()}`);
}

export function getMaterialStatus(materialId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return apiGet<MaterialStatusResponse>(`/materials/${encodeURIComponent(materialId)}/status?${params.toString()}`);
}

export function getMaterialDebug(materialId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return apiGet<MaterialDebugResponse>(`/materials/${encodeURIComponent(materialId)}/debug?${params.toString()}`);
}

export function getMaterialRawUrl(materialId: string, ownerId: string) {
  const params = new URLSearchParams({ owner_id: ownerId });
  return `${API_V1_BASE_URL}/materials/${encodeURIComponent(materialId)}/raw?${params.toString()}`;
}

export function uploadMaterial(file: File, metadata: MaterialUploadMetadata) {
  const formData = new FormData();
  formData.append("metadata", JSON.stringify(metadata));
  formData.append("file", file);
  return request<MaterialUploadResponse>("/materials/upload", {
    method: "POST",
    body: formData
  });
}

export function uploadMaterialWithProgress(
  file: File,
  metadata: MaterialUploadMetadata,
  onProgress: (pct: number) => void,
): Promise<MaterialUploadResponse> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("metadata", JSON.stringify(metadata));
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const payload = JSON.parse(xhr.responseText) as ApiEnvelope<MaterialUploadResponse>;
          if (!payload.success || payload.data === null) {
            reject(new Error(payload.error ?? payload.message ?? "Upload failed"));
          } else {
            resolve(payload.data);
          }
        } catch {
          reject(new Error("Invalid response from upload"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText) as { detail?: string; error?: string };
          reject(new Error(err.detail ?? err.error ?? `Upload failed: ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.open("POST", `${API_V1_BASE_URL}/materials/upload`);
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || import.meta.env.VITE_AGENTBOOK_AUTH_TOKEN;
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(formData);
  });
}

export function uploadMaterialsBatchWithProgress(
  files: File[],
  metadata: MaterialUploadMetadata[],
  onProgress: (pct: number) => void,
): Promise<MaterialBatchUploadResponse> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("metadata", JSON.stringify(metadata));
    files.forEach((file) => formData.append("files", file));

    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const payload = JSON.parse(xhr.responseText) as ApiEnvelope<MaterialBatchUploadResponse>;
          if (payload.data === null) {
            reject(new Error(payload.error ?? payload.message ?? "Upload failed"));
          } else {
            resolve(payload.data);
          }
        } catch {
          reject(new Error("Invalid response from upload"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText) as { detail?: string; error?: string };
          reject(new Error(err.detail ?? err.error ?? `Upload failed: ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.open("POST", `${API_V1_BASE_URL}/materials/batch_upload`);
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || import.meta.env.VITE_AGENTBOOK_AUTH_TOKEN;
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(formData);
  });
}

export function askQuestion(payload: QueryRequest) {
  return apiPost<QueryResponse>("/query/ask", payload);
}

export type ImageQueryPayload = {
  ownerId: string;
  collectionId?: string | null;
  materialIds?: string[];
  conversationId?: string;
  queryText?: string;
  topK?: number | null;
  answerLanguage?: string | null;
  image: File | Blob;
  imageFilename?: string;
};

export async function askQuestionWithImage(payload: ImageQueryPayload): Promise<QueryResponse> {
  const form = new FormData();
  const filename = payload.imageFilename || (payload.image instanceof File ? payload.image.name : "upload.png");
  form.append("image", payload.image, filename);
  form.append("owner_id", payload.ownerId);
  if (payload.collectionId) form.append("collection_id", payload.collectionId);
  if (payload.materialIds && payload.materialIds.length > 0) {
    form.append("material_ids", JSON.stringify(payload.materialIds));
  }
  form.append("conversation_id", payload.conversationId ?? "default");
  if (payload.queryText) form.append("query_text", payload.queryText);
  if (payload.topK != null) form.append("top_k", String(payload.topK));
  if (payload.answerLanguage) form.append("answer_language", payload.answerLanguage);

  const response = await fetch(`${API_V1_BASE_URL}/query/ask-image`, {
    method: "POST",
    headers: { ...authHeaders() },
    body: form,
  });
  if (response.status === 401) notifyUnauthorized();
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string; message?: string };
      detail = body.detail || body.message || detail;
    } catch {}
    throw new Error(detail);
  }
  const envelope = (await response.json()) as ApiEnvelope<QueryResponse>;
  if (!envelope.success || !envelope.data) {
    throw new Error(envelope.error || envelope.message || "Image query failed");
  }
  return envelope.data;
}

// ── GraphRAG (G2 + G4) ──────────────────────────────────────────────────────

export type GraphQueryPayload = {
  ownerId: string;
  collectionId?: string | null;
  materialIds?: string[];
  conversationId?: string;
  query: string;
  entityIds?: string[];   // slug-form ids (e.g. "entity:dropout")
  relationIds?: string[]; // Mongo _id strings
  hops?: 1 | 2;
  topK?: number | null;
  answerLanguage?: string | null;
};

export async function askWithGraphAnchor(payload: GraphQueryPayload): Promise<QueryResponse> {
  const body = {
    owner_id: payload.ownerId,
    collection_id: payload.collectionId ?? null,
    material_ids: payload.materialIds ?? [],
    conversation_id: payload.conversationId ?? "default",
    query: payload.query,
    entity_ids: payload.entityIds ?? [],
    relation_ids: payload.relationIds ?? [],
    hops: payload.hops ?? 2,
    top_k: payload.topK ?? null,
    answer_language: payload.answerLanguage ?? null,
  };
  const response = await fetch(`${API_V1_BASE_URL}/query/ask-graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (response.status === 401) notifyUnauthorized();
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const j = (await response.json()) as { detail?: string; message?: string };
      detail = j.detail || j.message || detail;
    } catch {}
    throw new Error(detail);
  }
  const envelope = (await response.json()) as ApiEnvelope<QueryResponse>;
  if (!envelope.success || !envelope.data) {
    throw new Error(envelope.error || envelope.message || "Graph query failed");
  }
  return envelope.data;
}

export async function fetchEntitySubgraph(
  entityId: string,
  opts: { ownerId: string; collectionId: string; hops?: 1 | 2 },
): Promise<{ nodes: unknown[]; edges: unknown[] }> {
  const url = `${API_V1_BASE_URL}/graph/entity/${encodeURIComponent(entityId)}/subgraph?owner_id=${encodeURIComponent(opts.ownerId)}&collection_id=${encodeURIComponent(opts.collectionId)}&hops=${opts.hops ?? 2}`;
  const response = await fetch(url, { headers: { ...authHeaders() } });
  if (response.status === 401) notifyUnauthorized();
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const envelope = (await response.json()) as ApiEnvelope<{ nodes: unknown[]; edges: unknown[] }>;
  if (!envelope.success || !envelope.data) {
    throw new Error(envelope.error || "Subgraph fetch failed");
  }
  return envelope.data;
}

export async function askQuestionStream(
  payload: QueryRequest,
  callbacks: {
    onToken: (token: string) => void;
    onDone: (response: QueryResponse) => void;
    onError: (message: string) => void;
    onAgentStep?: (step: AgentTraceStep) => void;
    onVerifying?: () => void;
  },
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_V1_BASE_URL}/query/ask-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    // AbortError = user cancelled intentionally, not an error
    if (err instanceof Error && err.name === "AbortError") return;
    callbacks.onError("Không thể kết nối đến máy chủ.");
    return;
  }
  if (!response.ok || !response.body) {
    callbacks.onError(`HTTP ${response.status}`);
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let dataStr = "";

  const dispatchEvent = () => {
    if (!eventType || !dataStr) return;
    try {
      if (eventType === "token") {
        const parsed = JSON.parse(dataStr) as { token: string };
        callbacks.onToken(parsed.token);
      } else if (eventType === "agent_step") {
        callbacks.onAgentStep?.(JSON.parse(dataStr) as AgentTraceStep);
      } else if (eventType === "verifying") {
        callbacks.onVerifying?.();
      } else if (eventType === "done") {
        callbacks.onDone(JSON.parse(dataStr) as QueryResponse);
      } else if (eventType === "error") {
        const parsed = JSON.parse(dataStr) as { message: string };
        callbacks.onError(parsed.message);
      }
    } catch {
      // malformed event — ignore
    }
    eventType = "";
    dataStr = "";
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          dataStr = line.slice(6).trim();
        } else if (line === "") {
          dispatchEvent();
        }
      }
    }
    // flush remaining
    if (buffer) {
      if (buffer.startsWith("data: ")) dataStr = buffer.slice(6).trim();
      dispatchEvent();
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") return;
    callbacks.onError("Lỗi kết nối trong khi nhận dữ liệu.");
  }
}

export function compareDocuments(payload: CompareRequest) {
  return apiPost<CompareResponse>("/query/compare", payload);
}

export function loadEvidencePage(docId: string, page: number, ownerId: string, collectionId?: string | null) {
  const params = new URLSearchParams({ owner_id: ownerId });
  if (collectionId) params.set("collection_id", collectionId);
  return apiGet<EvidencePageResponse>(`/evidence/${docId}/${page}?${params.toString()}`);
}

export function loadMindmap(payload: {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  root_topic?: string | null;
  detail_level?: "brief" | "overview" | "detailed";
  use_llm?: boolean;
}) {
  return apiPost<MindmapResponse>("/graph/mindmap", payload);
}

export function loadGraph(payload: {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  root_topic?: string | null;
  focus_block_ids?: string[];
  focus_material_ids?: string[];
  focus_pages?: string[];
  focus_query_text?: string;
  focus_answer_text?: string;
}) {
  return apiPost<GraphResponse>("/graph", payload);
}

// Structure-adaptive visualization: backend picks viz_mode from measured
// document structure (hierarchy tree for legal/manuals, concept graph for
// papers). For hierarchy/citation_network it returns a nested `tree`.
export type VizSignals = {
  hierarchy: number;
  reference: number;
  semantic: number;
  temporal: number;
  counts: Record<string, number>;
};

export type AutoVizResponse = {
  viz_mode: "hierarchy" | "citation_network" | "concept_graph" | "timeline" | string;
  signals: VizSignals;
  tree: MindmapNode[];
  graph: GraphResponse | null;
};

export function loadAutoViz(payload: {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  root_topic?: string | null;
  focus_block_ids?: string[];
  focus_material_ids?: string[];
  focus_pages?: string[];
  focus_query_text?: string;
  focus_answer_text?: string;
  // "verify" = show only cited Điều (precise, for Kiểm chứng)
  // "explore" = cited Điều + query-text match (broader)
  // "auto"    = backend decides based on whether focus_block_ids provided
  graph_mode?: "verify" | "explore" | "auto";
}) {
  return apiPost<AutoVizResponse>("/graph/auto", payload);
}

export type SummaryRequest = {
  owner_id: string;
  collection_id?: string | null;
  material_id?: string | null;
  material_ids?: string[];
  scope?: string;
  top_k?: number | null;
};

export type SummaryResponse = {
  summary: string;
  citations: Citation[];
  confidence: number;
  was_refused?: boolean;
  refusal_reason?: string | null;
  coverage?: CoverageReport | null;
};

export type StudyGuideRequest = {
  owner_id: string;
  collection_id?: string | null;
  material_id?: string | null;
  scope?: string;
  format?: string;
  top_k?: number | null;
};

export type StudyGuideResponse = {
  overview: string;
  key_concepts: string[];
  outline: string[];
  citations: Citation[];
  confidence: number;
};

export function summarizeCollection(payload: SummaryRequest) {
  return apiPost<SummaryResponse>("/query/summarize", payload);
}

export function buildStudyGuide(payload: StudyGuideRequest) {
  return apiPost<StudyGuideResponse>("/query/study-guide", payload);
}
