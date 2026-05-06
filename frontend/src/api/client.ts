export const API_BASE_URL = (import.meta.env.VITE_AGENTBOOK_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const API_V1_BASE_URL = `${API_BASE_URL}/api/v1`;
const AUTH_TOKEN_STORAGE_KEY = "prism.auth.token";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || import.meta.env.VITE_AGENTBOOK_AUTH_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
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
  bbox: BoundingBox | null;
  role: string;
  source_language: string;
  confidence: number;
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
  sources_requested?: number | null;
  sources_covered?: number | null;
  evidence_count?: number | null;
  warning?: string | null;
};

export type AgentVerification = {
  verdict: string;
  confidence: number;
  warning?: string | null;
};

export type AgentTrace = {
  plan_type: string;
  steps: AgentTraceStep[];
  repair_attempted: boolean;
  verification?: AgentVerification | null;
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

export type CompareResponse = {
  topic: string;
  comparison_table: ComparisonCell[];
  conflicts: string[];
  citations: Citation[];
  coverage?: CoverageReport | null;
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
  source_docs?: string[];
  evidence_refs?: Array<Record<string, string | number>>;
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

export async function askQuestionStream(
  payload: QueryRequest,
  callbacks: {
    onToken: (token: string) => void;
    onDone: (response: QueryResponse) => void;
    onError: (message: string) => void;
    onAgentStep?: (step: AgentTraceStep) => void;
  },
): Promise<void> {
  const response = await fetch(`${API_V1_BASE_URL}/query/ask-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
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
  detail_level?: "overview" | "detailed";
  use_llm?: boolean;
}) {
  return apiPost<MindmapResponse>("/graph/mindmap", payload);
}

export function loadGraph(payload: {
  owner_id: string;
  collection_id?: string | null;
  material_ids?: string[];
  root_topic?: string | null;
}) {
  return apiPost<GraphResponse>("/graph", payload);
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
