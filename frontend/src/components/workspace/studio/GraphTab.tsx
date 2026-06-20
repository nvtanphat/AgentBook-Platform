import { useState, useEffect, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import { AlertCircle, BookOpen, FileText, Link2, Loader2, Maximize2, MessageCircleQuestion, Network, RefreshCw, Send, ShieldCheck, Sparkles, Target, X } from "lucide-react";
import { EvidenceBlock, GraphResponse, MindmapResponse, QueryResponse, askWithGraphAnchor, loadAutoViz, loadEvidencePage, loadGraph, loadMindmap } from "../../../api/client";
import GraphCanvas, { CanvasEdge, CanvasNode } from "../../GraphCanvas";
import MarkdownRenderer from "../../MarkdownRenderer";
import { useWorkspace } from "../../../state/workspace";

// ─── Data transforms ──────────────────────────────────────────────────────────

const MAX_GRAPH_NODES = 80;
const MAX_GRAPH_EDGES = 120;
const MAX_MINDMAP_GROUPS = 8;
const MAX_MINDMAP_ITEMS_PER_GROUP = 9;
const MINDMAP_BRANCH_COLORS = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0891b2", "#65a30d", "#4f46e5"];

// NOTE: Frontend KHÔNG có stoplist semantic nào (no `NOISY_GRAPH_LABELS`,
// `FORMAT_GRAPH_WORDS`, etc.). Backend's `entity_type` allowlist
// (model/algorithm/concept/...) đã filter ra noise — trust nó.
// Frontend chỉ làm visual safety: mojibake repair + length + invalid char.

type GraphFocus = {
  labels: Set<string>;
  materialIds: Set<string>;
  pages: Set<string>;
};

function normalizeGraphText(value: string) {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/\u0111/g, "d")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function repairMojibakeText(value: string) {
  if (!value || [...value].some((char) => char.charCodeAt(0) > 255)) return value;
  try {
    const bytes = Uint8Array.from([...value], (char) => char.charCodeAt(0));
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return value;
  }
}

/**
 * Frontend label cleanup — VISUAL SAFETY ONLY.
 *
 * Semantic filtering (drop authors, format tokens, OCR garbage) is done
 * server-side via entity_type allowlist + `_clean_entity_label` in
 * `backend/src/api/v1/endpoints/graph.py`. Frontend only:
 *   1. Repairs mojibake (utf-8 misread as latin1)
 *   2. Trims whitespace / surrounding punctuation
 *   3. Rejects obviously broken display strings (replacement char, dup separator)
 *   4. Caps word count for layout sanity
 *
 * If something noisy still slips through, fix it upstream (entity extractor
 * prompt or `entity_resolution.py`) — do NOT add another denylist here.
 */
function cleanGraphLabel(value: string) {
  const label = repairMojibakeText(value)
    .replace(/\s*[,;:/]\s*/g, " ")
    .replace(/^[^\wÀ-ỹ]+/u, "")
    .replace(/[|()[\]{}'"]+/g, " ")
    .replace(/\.+$/g, "")
    .replace(/\s+/g, " ")
    .trim();

  // Length sanity for visual layout
  if (label.length < 3) return null;

  // Replacement char (�) or invalid mojibake remnant — undisplayable
  if (/[�]/.test(label)) return null;

  // Word count cap: > 6 words doesn't fit a mindmap node visually
  const words = label.split(/\s+/);
  if (words.length > 6) return null;

  return label;
}

function focusScore(node: GraphResponse["nodes"][number], focus: GraphFocus | null) {
  if (!focus) return 0;

  const nodeLabel = normalizeGraphText(node.label);
  const labelHit = [...focus.labels].some((label) => label && (nodeLabel.includes(label) || label.includes(nodeLabel)));
  const evidenceRefs = ((node as any).evidence_refs ?? []) as Array<Record<string, string | number>>;
  const materialHit = evidenceRefs.some((ref) => {
    const materialId = String(ref.material_id ?? ref.doc_id ?? "");
    const page = String(ref.page ?? "");
    return focus.materialIds.has(materialId) || focus.pages.has(`${materialId}:${page}`);
  });

  return (labelHit ? 100 : 0) + (materialHit ? 55 : 0);
}

function nodeWeight(node: GraphResponse["nodes"][number], degree = 0) {
  const mentions = (node as any).mention_count ?? 0;
  const confidence = node.confidence ?? 0;
  return degree * 4 + mentions * 1.5 + confidence;
}

function toGraph(response: GraphResponse | null, focus: GraphFocus | null): { nodes?: CanvasNode[]; edges?: CanvasEdge[] } {
  if (!response || !response.nodes.length) return { nodes: [], edges: [] };

  const degree: Record<string, number> = {};
  response.nodes.forEach((n) => { degree[n.id] = 0; });
  response.edges.forEach((e) => {
    degree[e.source] = (degree[e.source] ?? 0) + 1;
    degree[e.target] = (degree[e.target] ?? 0) + 1;
  });

  const entityNodes = response.nodes
    .filter((n) => !n.id.startsWith("block:"))
    .map((n) => ({ ...n, label: cleanGraphLabel(n.label) ?? "" }))
    .filter((n) => n.label);
  const focusIds = new Set(entityNodes.filter((n) => focusScore(n, focus) > 0).map((n) => n.id));
  const neighborIds = new Set<string>();
  response.edges.forEach((edge) => {
    if (focusIds.has(edge.source)) neighborIds.add(edge.target);
    if (focusIds.has(edge.target)) neighborIds.add(edge.source);
  });

  const visibleNodes = entityNodes
    .sort((a, b) => {
      const aFocused = focusIds.has(a.id) ? 1 : 0;
      const bFocused = focusIds.has(b.id) ? 1 : 0;
      const aNeighbor = neighborIds.has(a.id) ? 1 : 0;
      const bNeighbor = neighborIds.has(b.id) ? 1 : 0;
      return (
        bFocused - aFocused ||
        bNeighbor - aNeighbor ||
        focusScore(b, focus) - focusScore(a, focus) ||
        nodeWeight(b, degree[b.id] ?? 0) - nodeWeight(a, degree[a.id] ?? 0)
      );
    })
    .slice(0, MAX_GRAPH_NODES);

  const allowedIds = new Set(visibleNodes.map((n) => n.id));
  const visibleEdges = response.edges.filter(
    (e) => allowedIds.has(e.source) && allowedIds.has(e.target) && ((e as any).evidence_refs?.length ?? 0) > 0
  ).sort((a, b) => {
    const aFocused = Number(focusIds.has(a.source) || focusIds.has(a.target));
    const bFocused = Number(focusIds.has(b.source) || focusIds.has(b.target));
    const aSemantic = Number(!a.relation_type.startsWith("co_occurs"));
    const bSemantic = Number(!b.relation_type.startsWith("co_occurs"));
    return bFocused - aFocused || bSemantic - aSemantic;
  })
    .slice(0, MAX_GRAPH_EDGES);

  const nodes: CanvasNode[] = visibleNodes.map((n) => ({
    id: n.id,
    label: n.label,
    type: n.type,
    confidence: n.confidence,
    degree: n.degree ?? degree[n.id] ?? 0,
    mention_count: (n as any).mention_count ?? 0,
    importance: (n as any).importance ?? 0,
    community: (n as any).community ?? 0,
    is_hub: (n as any).is_hub ?? false,
    source_docs: (n as any).source_docs ?? [],
    evidence_refs: (n as any).evidence_refs ?? [],
    // Backend `is_focused` (primary entity from citations) → frontend `focused` flag
    focused: (n as any).is_focused ?? focusIds.has(n.id),
  }));
  const edges: CanvasEdge[] = visibleEdges.map((e, i) => ({
    id: `${e.source}-${e.target}-${i}`,
    source: e.source,
    target: e.target,
    label: e.relation_type,
    confidence: e.confidence,
    evidence_count: (e as any).evidence_count ?? ((e as any).evidence_refs?.length ?? 0),
    evidence_refs: (e as any).evidence_refs ?? [],
    evidence_text_chunk: e.evidence_text_chunk ?? null,
    source_label: (e as any).source_label ?? visibleNodes.find((node) => node.id === e.source)?.label ?? null,
    target_label: (e as any).target_label ?? visibleNodes.find((node) => node.id === e.target)?.label ?? null,
    focused: focusIds.has(e.source) || focusIds.has(e.target),
  }));
  return { nodes, edges };
}
// Build the short "source attribution" badge shown in hover preview.
//  e.g.  "DeAn.docx · p.12"  or  "3 nguồn" when many citations.
function buildSourceLabel(citations: Array<Record<string, unknown>>, materialNameMap: Map<string, string>): string | null {
  if (!citations || citations.length === 0) return null;
  const first = citations[0] ?? {};
  const materialId = String(first.material_id ?? first.doc_id ?? "");
  const page = first.page;
  const docName = materialNameMap.get(materialId);
  const pageSuffix = page !== undefined && page !== null && page !== "" ? ` · p.${page}` : "";
  if (citations.length === 1) {
    return docName ? `${docName}${pageSuffix}` : `Trang ${page ?? "?"}`;
  }
  return docName
    ? `${docName}${pageSuffix} +${citations.length - 1} nguồn`
    : `${citations.length} nguồn`;
}

function toMindmap(response: MindmapResponse | null, materialNameMap: Map<string, string> = new Map()): { nodes?: CanvasNode[]; edges?: CanvasEdge[] } {
  if (!response || !response.nodes.length) return { nodes: [], edges: [] };

  const rootId = "root-topic";
  const nodes: CanvasNode[] = [{ id: rootId, label: response.root_topic, type: "root" }];
  const edges: CanvasEdge[] = [];

  if (response.nodes.some((node) => node.children?.length)) {
    const walk = (items: MindmapResponse["nodes"], parentId: string, depth = 1, branchColor = MINDMAP_BRANCH_COLORS[0]) => {
      for (const [index, item] of items.entries()) {
        const childCount = item.children?.length ?? 0;
        const type = item.entity_type || (childCount ? "topic" : "concept");
        const color = depth === 1 ? MINDMAP_BRANCH_COLORS[index % MINDMAP_BRANCH_COLORS.length] : branchColor;
        const citations = ((item as any).citations ?? []) as Array<Record<string, unknown>>;
        nodes.push({
          id: item.id,
          label: cleanGraphLabel(item.label) ?? item.label,
          type,
          degree: childCount,
          confidence: null,
          evidence_refs: citations as Array<Record<string, string | number>>,
          branchColor: color,
          depth,
          summary: (item as any).summary ?? null,
          source_label: buildSourceLabel(citations, materialNameMap),
        });
        edges.push({ id: `${parentId}-${item.id}`, source: parentId, target: item.id, label: "", branchColor: color });
        if (childCount) {
          walk(item.children, item.id, depth + 1, color);
        }
      }
    };
    walk(response.nodes, rootId);
    return { nodes, edges };
  }

  const groups = new Map<string, typeof response.nodes>();
  const seenLabels = new Set<string>();
  for (const node of response.nodes) {
    const label = cleanGraphLabel(node.label);
    if (!label) continue;
    const labelKey = label.toLowerCase();
    if (seenLabels.has(labelKey)) continue;
    seenLabels.add(labelKey);
    const cleanNode = { ...node, label };
    // Use explicit entity_type field instead of parsing summary
    const entityType = (node as any).entity_type || "concept";
    if (!groups.has(entityType)) groups.set(entityType, []);
    groups.get(entityType)!.push(cleanNode);
  }

  const rankedGroups = [...groups.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, MAX_MINDMAP_GROUPS);

  for (const [typeName, members] of rankedGroups) {
    const clusterId = `cluster-${typeName}`;
    const visibleMembers = members.slice(0, MAX_MINDMAP_ITEMS_PER_GROUP);
    nodes.push({
      id: clusterId,
      label: displayMindmapType(typeName),
      type: "cluster",
      degree: members.length,
    });
    edges.push({ id: `${rootId}-${clusterId}`, source: rootId, target: clusterId, label: "" });
    for (const m of visibleMembers) {
      const citations = ((m as any).citations ?? []) as Array<Record<string, unknown>>;
      nodes.push({
        id: m.id,
        label: m.label,
        type: typeName,
        confidence: null,
        evidence_refs: citations as Array<Record<string, string | number>>,
        summary: (m as any).summary ?? null,
        source_label: buildSourceLabel(citations, materialNameMap),
      });
      edges.push({ id: `${clusterId}-${m.id}`, source: clusterId, target: m.id, label: "" });
    }
  }

  return { nodes, edges };
}

function buildGraphFocus(activeQueryContext: ReturnType<typeof useWorkspace>["activeQueryContext"]): GraphFocus | null {
  if (!activeQueryContext || activeQueryContext.response.was_refused) {
    return null;
  }

  const labels = new Set<string>();
  const materialIds = new Set<string>();
  const pages = new Set<string>();

  activeQueryContext.response.reasoning_path.forEach((step) => {
    step.entities.forEach((entity) => {
      const normalized = normalizeGraphText(entity);
      if (normalized) labels.add(normalized);
    });
  });

  activeQueryContext.response.citations.forEach((citation) => {
    if (citation.doc_id) {
      materialIds.add(citation.doc_id);
      if (citation.page) pages.add(`${citation.doc_id}:${citation.page}`);
    }
  });

  return labels.size || materialIds.size || pages.size ? { labels, materialIds, pages } : null;
}

// ─── Selected node type ───────────────────────────────────────────────────────

type SelectedNode = {
  id: string;
  label: string;
  type: string;
  confidence: number | null;
  degree: number;
  mention_count: number;
  source_docs: string[];
  evidenceRefs: Array<Record<string, string | number>>;
  connections: { label: string; relation: string }[];
};

type SelectedRelation = {
  source: string;
  target: string;
  sourceLabel: string;
  targetLabel: string;
  relation: string;
  confidence: number | null;
  evidenceCount: number;
  evidenceRefs: Array<Record<string, string | number>>;
  evidenceTextChunk?: string | null;
};

// ─── Color map (mirrors GraphCanvas) ─────────────────────────────────────────

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  // Core entity types (from new LLM extractor)
  model:        { bg: "#e0f2fe", border: "#0284c7", text: "#0369a1" },
  algorithm:    { bg: "#f0fdf4", border: "#22c55e", text: "#15803d" },
  metric:       { bg: "#fef9c3", border: "#ca8a04", text: "#92400e" },
  dataset:      { bg: "#fdf4ff", border: "#a855f7", text: "#7e22ce" },
  framework:    { bg: "#cffafe", border: "#06b6d4", text: "#155e75" },
  author:       { bg: "#ede9fe", border: "#8b5cf6", text: "#6d28d9" },
  field:        { bg: "#fff1f2", border: "#f43f5e", text: "#9f1239" },
  // Legacy / general types
  concept:      { bg: "#dbeafe", border: "#3b82f6", text: "#1d4ed8" },
  person:       { bg: "#ede9fe", border: "#8b5cf6", text: "#6d28d9" },
  event:        { bg: "#d1fae5", border: "#10b981", text: "#065f46" },
  location:     { bg: "#fef3c7", border: "#f59e0b", text: "#92400e" },
  date:         { bg: "#fce7f3", border: "#ec4899", text: "#9d174d" },
  technology:   { bg: "#cffafe", border: "#06b6d4", text: "#155e75" },
  method:       { bg: "#f0fdf4", border: "#22c55e", text: "#15803d" },
  organization: { bg: "#fee2e2", border: "#ef4444", text: "#991b1b" },
  org:          { bg: "#fee2e2", border: "#ef4444", text: "#991b1b" },
  entity:       { bg: "#fef9c3", border: "#ca8a04", text: "#92400e" },
};

const MINDMAP_TYPE_LABELS: Record<string, string> = {
  model:        "Mô hình AI/ML",
  algorithm:    "Thuật toán",
  metric:       "Độ đo",
  dataset:      "Bộ dữ liệu",
  framework:    "Framework",
  author:       "Tác giả",
  field:        "Lĩnh vực",
  concept:      "Khái niệm",
  technology:   "Công nghệ",
  method:       "Phương pháp",
  person:       "Nhân vật",
  organization: "Tổ chức",
  org:          "Tổ chức",
  location:     "Địa điểm",
  event:        "Sự kiện",
  date:         "Mốc thời gian",
  entity:       "Thực thể",
};

function displayMindmapType(typeName: string) {
  return MINDMAP_TYPE_LABELS[typeName.toLowerCase()] ?? typeName.charAt(0).toUpperCase() + typeName.slice(1);
}

function typeColor(type: string) {
  const key = type.toLowerCase().split(/[_\s]/)[0];
  return TYPE_COLORS[key] ?? { bg: "#f1f5f9", border: "#94a3b8", text: "#475569" };
}

const LEGEND_TYPES = ["model", "algorithm", "metric", "dataset", "framework", "concept", "organization", "author"] as const;

// ─── Node info card ───────────────────────────────────────────────────────────

// Quick-read: lazy-load the source block text for the clicked node so the user
// can read the grounding passage inline, without leaving the graph for the
// evidence panel. Reuses the existing /evidence/{doc}/{page} endpoint and
// preserves the evidence trace (doc · page · block_id stay attached).
function NodeQuickRead({
  node,
  ownerId,
  collectionId,
  onOpenEvidence,
}: {
  node: SelectedNode;
  ownerId: string;
  collectionId?: string | null;
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const [src, setSrc] = useState<{ docId: string; page: number; blockId: string | null; docName: string | null } | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const ref = node.evidenceRefs.find((r) => {
      const id = r.doc_id ?? r.material_id;
      const page = Number(r.page);
      return typeof id === "string" && id.length > 0 && Number.isFinite(page) && page > 0;
    });
    if (!ref) {
      setText(null);
      setSrc(null);
      setFailed(false);
      setLoading(false);
      return;
    }
    const docId = String(ref.doc_id ?? ref.material_id);
    const page = Number(ref.page);
    const blockId = typeof ref.block_id === "string" && ref.block_id ? ref.block_id : null;
    setLoading(true);
    setFailed(false);
    setText(null);
    loadEvidencePage(docId, page, ownerId, collectionId)
      .then((res) => {
        if (cancelled) return;
        const blocks: EvidenceBlock[] = res.blocks ?? [];
        // Show the exact cited block. This is the precise grounding passage the
        // node points to and is correct for any document type (no language- or
        // format-specific reconstruction). Reading the surrounding section is one
        // click away via "Mở toàn văn". Fallbacks stay generic:
        //   1. exact block_id match
        //   2. the block whose text contains the node label
        //   3. the first block on the page
        let chosen = blockId ? blocks.filter((b) => b.block_id === blockId) : [];
        if (!chosen.length) {
          const needle = node.label.trim().toLowerCase();
          const match = needle
            ? blocks.find((b) => (b.snippet_original ?? "").toLowerCase().includes(needle))
            : undefined;
          chosen = match ? [match] : blocks.slice(0, 1);
        }
        const joined = chosen.map((b) => b.snippet_original).filter(Boolean).join("\n\n").trim();
        setText(joined || null);
        setSrc({ docId, page, blockId, docName: res.doc_name ?? null });
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node.id, ownerId, collectionId]);

  if (!node.evidenceRefs.length) return null;

  return (
    <div className="px-4 pb-3">
      <div className="mb-1 flex items-center justify-between">
        <p className="text-[9px] font-semibold uppercase tracking-wider text-muted flex items-center gap-1">
          <BookOpen size={9} /> Nội dung nguồn
        </p>
        {src && (
          <span className="text-[9px] text-muted truncate max-w-[150px]" title={`${src.docName ?? ""} · trang ${src.page}`}>
            {src.docName ? `${src.docName} · ` : ""}tr.{src.page}
          </span>
        )}
      </div>
      {loading ? (
        <div className="flex items-center gap-2 rounded-lg border border-outline/40 bg-slate-50 px-3 py-3 text-[11px] text-muted">
          <Loader2 size={12} className="animate-spin" /> Đang tải đoạn văn nguồn…
        </div>
      ) : text ? (
        <blockquote className="max-h-[160px] overflow-y-auto whitespace-pre-wrap rounded-lg border-l-2 border-primary/40 bg-slate-50 px-3 py-2 text-[11px] leading-relaxed text-text/85">
          {text}
        </blockquote>
      ) : (
        <p className="rounded-lg border border-outline/40 bg-slate-50 px-3 py-2 text-[11px] text-muted">
          {failed ? "Không tải được đoạn văn nguồn." : "Node này chưa gắn đoạn văn text."}
        </p>
      )}
      {onOpenEvidence && src && (
        <button
          type="button"
          onClick={() => onOpenEvidence({ docId: src.docId, page: src.page, blockId: src.blockId })}
          className="mt-1.5 text-[10px] font-semibold text-primary hover:underline"
        >
          Mở toàn văn trong tài liệu →
        </button>
      )}
    </div>
  );
}

function NodeInfoCard({
  node,
  ownerId,
  collectionId,
  onClose,
  onAskAboutNode,
  onOpenEvidence,
}: {
  node: SelectedNode;
  ownerId: string;
  collectionId?: string | null;
  onClose: () => void;
  onAskAboutNode?: (entityId: string, label: string) => void;
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
}) {
  const color = typeColor(node.type);
  const pct   = node.confidence != null ? Math.round(node.confidence * 100) : null;
  const barColor = pct == null ? "" : pct >= 70 ? "bg-emerald-400" : pct >= 40 ? "bg-yellow-400" : "bg-red-400";

  return (
    <div className="shrink-0 border-t border-outline bg-white shadow-[0_-1px_8px_rgba(0,0,0,.06)]">
      {/* Header row */}
      <div className="flex items-start gap-3 px-4 pb-2 pt-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide"
              style={{ background: color.bg, color: color.text, border: `1px solid ${color.border}` }}
            >
              {node.type}
            </span>
            {pct != null && (
              <div className="flex items-center gap-1">
                <div className="h-1 w-12 overflow-hidden rounded-full bg-slate-200">
                  <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
                </div>
                <span className="text-[10px] font-semibold text-muted">{pct}%</span>
              </div>
            )}
          </div>
          <p className="font-semibold text-sm text-text leading-snug" title={node.label}>
            {node.label}
          </p>
        </div>

        {/* Stats */}
        <div className="flex shrink-0 gap-3 text-center">
          <div>
            <p className="text-base font-bold text-primary leading-none">{node.degree}</p>
            <p className="text-[9px] uppercase tracking-wide text-muted mt-0.5">liên kết</p>
          </div>
          <div>
            <p className="text-base font-bold text-secondary leading-none">{node.mention_count}</p>
            <p className="text-[9px] uppercase tracking-wide text-muted mt-0.5">đề cập</p>
          </div>
        </div>

        <button onClick={onClose} className="shrink-0 text-muted hover:text-text transition mt-0.5">
          <X size={13} />
        </button>
      </div>

      {/* Quick-read source passage (inline, no panel switch) */}
      <NodeQuickRead node={node} ownerId={ownerId} collectionId={collectionId} onOpenEvidence={onOpenEvidence} />

      {/* Source documents */}
      {node.source_docs.length > 0 && (
        <div className="px-4 pb-2">
          <p className="text-[9px] font-semibold uppercase tracking-wider text-muted mb-1 flex items-center gap-1">
            <FileText size={9} /> Xuất hiện trong
          </p>
          <div className="flex flex-wrap gap-1">
            {node.source_docs.map((doc, i) => (
              <span key={i} className="rounded bg-blue-50 border border-blue-100 px-1.5 py-0.5 text-[10px] text-blue-700 max-w-[160px] truncate" title={doc}>
                {doc}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Connected concepts */}
      {node.connections.length > 0 && (
        <div className="px-4 pb-3">
          <p className="text-[9px] font-semibold uppercase tracking-wider text-muted mb-1 flex items-center gap-1">
            <Link2 size={9} /> Liên kết với
          </p>
          <div className="flex flex-wrap gap-1">
            {node.connections.slice(0, 8).map((c, i) => (
              <span key={i} className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-text" title={c.relation}>
                {c.label}
                {c.relation && !c.relation.startsWith("co_") && (
                  <span className="ml-1 text-muted">· {c.relation.replace(/_/g, " ")}</span>
                )}
              </span>
            ))}
            {node.connections.length > 8 && (
              <span className="text-[10px] text-muted">+{node.connections.length - 8}</span>
            )}
          </div>
        </div>
      )}

      {/* GraphRAG: ask anchored question */}
      {onAskAboutNode && (
        <div className="border-t border-outline/40 px-4 py-2.5 bg-gradient-to-r from-primary/5 to-transparent">
          <button
            type="button"
            onClick={() => onAskAboutNode(node.id, node.label)}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-[11px] font-semibold text-white hover:bg-primary/90 transition shadow-sm"
          >
            <Sparkles size={12} />
            Hỏi về node này
          </button>
          <p className="mt-1 text-[10px] text-muted">
            Tìm bằng chứng quanh "{node.label}" + neighbour 2-hop trên knowledge graph.
          </p>
        </div>
      )}
    </div>
  );
}

// ─── GraphRAG ask-about-node modal ─────────────────────────────────────────

function AskAboutNodeModal({
  anchorId,
  anchorLabel,
  ownerId,
  collectionId,
  conversationId,
  onClose,
  onAnswered,
}: {
  anchorId: string;
  anchorLabel: string;
  ownerId: string;
  collectionId: string;
  conversationId: string;
  onClose: () => void;
  onAnswered: (response: QueryResponse) => void;
}) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setTimeout(() => taRef.current?.focus(), 50);
  }, []);

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true); setError(null); setResponse(null);
    try {
      const r = await askWithGraphAnchor({
        ownerId, collectionId, conversationId,
        query: q,
        entityIds: [anchorId],
        hops: 2,
        answerLanguage: "vi",
      });
      setResponse(r);
      onAnswered(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi truy vấn");
    } finally {
      setLoading(false);
    }
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="w-[600px] max-w-[92vw] max-h-[85vh] overflow-hidden rounded-xl bg-white shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-outline/40 px-4 py-2.5 bg-gradient-to-r from-primary/8 to-transparent">
          <div className="flex items-center gap-2">
            <MessageCircleQuestion size={16} className="text-primary" />
            <p className="text-sm font-semibold text-text">
              Hỏi về <span className="text-primary">{anchorLabel}</span>
            </p>
          </div>
          <button onClick={onClose} className="text-muted hover:text-text"><X size={14} /></button>
        </div>

        <div className="px-4 py-3 flex flex-col gap-2 overflow-y-auto flex-1">
          <textarea
            ref={taRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submit(); }}
            placeholder={`VD: "${anchorLabel}" tác động đến điều gì? Hoặc hỏi tự do — graph sẽ truy vết quanh node này.`}
            className="w-full resize-none rounded-lg border border-outline/40 bg-white px-3 py-2 text-sm leading-relaxed outline-none focus:border-primary/60 min-h-[68px]"
            rows={3}
            disabled={loading}
          />
          {error && (
            <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-[12px] text-red-700">
              <AlertCircle size={14} className="shrink-0 mt-0.5" /> {error}
            </div>
          )}

          {response && (
            <div className="mt-1 space-y-2">
              <div className="rounded-lg bg-surface-low/60 border border-outline/30 px-3 py-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted mb-1.5">Trả lời</p>
                {response.was_refused ? (
                  <p className="text-sm text-muted italic">{response.answer}</p>
                ) : (
                  <MarkdownRenderer text={response.answer} />
                )}
              </div>
              {!response.was_refused && (
                <div className="flex flex-wrap gap-1.5 text-[10px]">
                  {response.used_entity_ids && response.used_entity_ids.length > 0 && (
                    <span className="rounded-full border border-primary/30 bg-primary/5 px-2 py-0.5 font-semibold text-primary">
                      {response.used_entity_ids.length} node được dùng (đã highlight trên graph)
                    </span>
                  )}
                  {response.sentence_coverage && (
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-700">
                      Bằng chứng phủ {Math.round((response.sentence_coverage.coverage_ratio || 0) * 100)}%
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-outline/40 px-4 py-2 flex items-center justify-end gap-2 bg-surface-low/40">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-3 py-1.5 text-[12px] font-semibold text-muted hover:bg-surface-low transition"
          >
            Đóng
          </button>
          <button
            type="button"
            onClick={submit}
            disabled={loading || !question.trim()}
            className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-[12px] font-semibold text-white disabled:opacity-40 hover:bg-primary/90 transition"
          >
            {loading ? <Loader2 size={12} className="animate-spin" /> : <Send size={12} />}
            {loading ? "Đang tìm..." : "Gửi"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function openEvidenceRef(refs: Array<Record<string, string | number>>, onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void) {
  if (!onOpenEvidence) return false;
  const ref = refs.find((item) => typeof (item.doc_id ?? item.material_id) === "string");
  if (!ref) return false;
  const docId = String(ref.doc_id ?? ref.material_id ?? "");
  const page = Number(ref.page ?? 0);
  if (!docId || !Number.isFinite(page) || page <= 0) return false;
  onOpenEvidence({ docId, page, blockId: typeof ref.block_id === "string" ? ref.block_id : null });
  return true;
}

function RelationInfoCard({
  relation,
  onClose,
  onOpenEvidence,
}: {
  relation: SelectedRelation;
  onClose: () => void;
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
}) {
  const pct = relation.confidence != null ? Math.round(relation.confidence * 100) : null;
  const confidenceClass = pct == null
    ? "bg-slate-100 text-muted"
    : pct >= 70
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : pct >= 40
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-red-50 text-red-700 border-red-200";
  return (
    <div className="shrink-0 border-t border-outline bg-white shadow-[0_-1px_8px_rgba(0,0,0,.06)]">
      <div className="flex items-start gap-3 px-4 py-3">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/8 text-primary">
          <Link2 size={15} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[9px] font-bold uppercase tracking-wider text-muted">Quan hệ dùng để kiểm chứng</p>
          <div className="mt-1 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-1.5 text-xs font-semibold text-text">
            <span className="truncate rounded bg-slate-100 px-1.5 py-1" title={relation.sourceLabel}>{relation.sourceLabel}</span>
            <span className="rounded bg-primary/10 px-1.5 py-1 text-[10px] font-bold text-primary" title={relation.relation}>
              {relation.relation.replace(/_/g, " ")}
            </span>
            <span className="truncate rounded bg-slate-100 px-1.5 py-1" title={relation.targetLabel}>{relation.targetLabel}</span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] font-semibold text-muted">
            {pct != null && <span className={`rounded-full border px-2 py-0.5 ${confidenceClass}`}>confidence {pct}%</span>}
            <span className="rounded-full border border-outline bg-slate-50 px-2 py-0.5">{relation.evidenceCount || relation.evidenceRefs.length} bằng chứng</span>
            <button
              type="button"
              onClick={() => openEvidenceRef(relation.evidenceRefs, onOpenEvidence)}
              className="rounded-full border border-primary/30 bg-primary/5 px-2 py-0.5 text-primary hover:border-primary/50"
            >
              Mở evidence
            </button>
          </div>
          {relation.evidenceTextChunk && (
            <blockquote className="mt-2 rounded border-l-2 border-primary/30 bg-slate-50 px-3 py-2 text-[10px] leading-relaxed text-text/80 italic line-clamp-4">
              {relation.evidenceTextChunk}
            </blockquote>
          )}
        </div>
        <button onClick={onClose} className="shrink-0 text-muted hover:text-text transition mt-0.5">
          <X size={13} />
        </button>
      </div>
    </div>
  );
}

// ─── Legend ───────────────────────────────────────────────────────────────────

function GraphLegend() {
  return (
    <div className="absolute bottom-3 left-3 z-10 rounded-lg border border-outline bg-white/92 backdrop-blur-sm px-2.5 py-2 shadow-sm">
      <p className="mb-1.5 text-[9px] font-bold uppercase tracking-wider text-muted">Loại node</p>
      <div className="space-y-1">
        {LEGEND_TYPES.map((t) => {
          const c = typeColor(t);
          return (
            <div key={t} className="flex items-center gap-1.5">
              <div className="h-3 w-3 rounded-full border flex-shrink-0"
                   style={{ background: c.bg, borderColor: c.border }} />
              <span className="text-[10px] capitalize text-text">{t}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TraceHeader({
  question,
  nodeCount,
  edgeCount,
  evidenceCount,
  sourceCount,
}: {
  question: string | null;
  nodeCount: number;
  edgeCount: number;
  evidenceCount: number;
  sourceCount: number;
}) {
  return (
    <div className="rounded-lg border border-primary/15 bg-primary/5 px-3 py-2">
      <div className="flex items-start gap-2">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary text-white">
          <ShieldCheck size={14} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-bold uppercase tracking-wider text-primary">Answer trace graph</p>
          <p className="mt-0.5 truncate text-xs font-semibold text-text" title={question ?? undefined}>
            {question ? question : "Graph quan hệ từ collection hiện tại"}
          </p>
          <p className="mt-1 text-[11px] leading-relaxed text-muted">
            Truy vet quan he de kiem chung cau tra loi; khong thay the Mindmap hoc tap.
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-semibold text-muted">
            <span className="rounded border border-outline bg-white px-2 py-0.5">{nodeCount} nodes</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">{edgeCount} relations</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">{evidenceCount} evidence</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">{sourceCount} sources</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Verification dashboard (verify mode) ───────────────────────────────────
// Thresholds — easy to tweak, no magic numbers inline.
const VERIFY_STRONG = 0.7;
const VERIFY_PARTIAL_T = 0.4;
const HUB_DEGREE_MIN = 3;  // node with degree ≥ this + no evidence = important gap

function nodeVerifyStatus(n: CanvasNode): "verified" | "partial" | "weak" | "unverified" {
  const ev = (n.evidence_refs ?? []).length;
  const focused = Boolean(n.focused);
  if (focused) return (n.confidence ?? VERIFY_STRONG) >= VERIFY_PARTIAL_T ? "verified" : "partial";
  if (ev > 0) {
    const c = n.confidence ?? VERIFY_PARTIAL_T;
    if (c >= VERIFY_STRONG) return "verified";
    if (c >= VERIFY_PARTIAL_T) return "partial";
    return "weak";
  }
  return "unverified";
}

function VerificationDashboard({
  question,
  coverage,
  nodes,
  edges,
  sourceCount,
}: {
  question: string | null;
  coverage: { coverage_ratio: number; total_sentences?: number; supported?: number; partial?: number; unsupported?: number } | null;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  sourceCount: number;
}) {
  // Overall score: prefer answer-level sentence coverage; else fall back to the
  // fraction of graph nodes that are verified.
  const statuses = nodes.map(nodeVerifyStatus);
  const verifiedCount = statuses.filter((s) => s === "verified").length;
  const partialCount = statuses.filter((s) => s === "partial").length;
  const weakCount = statuses.filter((s) => s === "weak").length;
  const unverifiedCount = statuses.filter((s) => s === "unverified").length;
  const total = nodes.length || 1;

  const score = coverage
    ? Math.round((coverage.coverage_ratio || 0) * 100)
    : Math.round((verifiedCount / total) * 100);
  const scoreColor = score >= 70 ? "#16a34a" : score >= 40 ? "#d97706" : "#dc2626";
  const scoreLabel = score >= 70 ? "Đáng tin cậy" : score >= 40 ? "Cần xem xét" : "Yếu";

  const edgesWithEvidence = edges.filter((e) => (e.evidence_count ?? (e.evidence_refs?.length ?? 0)) > 0).length;

  // Coverage bar segments — sentence-level if available, else node-level.
  const seg = coverage && coverage.total_sentences
    ? [
        { n: coverage.supported ?? 0, color: "#22c55e", label: "Mạnh" },
        { n: coverage.partial ?? 0, color: "#f59e0b", label: "Một phần" },
        { n: coverage.unsupported ?? 0, color: "#ef4444", label: "Yếu" },
      ]
    : [
        { n: verifiedCount, color: "#22c55e", label: "Đã xác minh" },
        { n: partialCount, color: "#f59e0b", label: "Một phần" },
        { n: weakCount, color: "#ef4444", label: "Yếu" },
        { n: unverifiedCount, color: "#94a3b8", label: "Chưa có" },
      ];
  const segTotal = seg.reduce((s, x) => s + x.n, 0) || 1;

  // Gap warning: high-degree (hub) nodes with no evidence.
  const gapNodes = nodes.filter((n) => (n.degree ?? 0) >= HUB_DEGREE_MIN && (n.evidence_refs?.length ?? 0) === 0);

  return (
    <div className="rounded-lg border border-primary/15 bg-gradient-to-br from-primary/5 to-transparent px-3 py-2.5">
      {/* Header + overall score */}
      <div className="flex items-center gap-3">
        <div
          className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full"
          style={{ background: `conic-gradient(${scoreColor} ${score * 3.6}deg, #e2e8f0 0deg)` }}
          title="Điểm kiểm chứng tổng thể"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-white">
            <span className="text-[13px] font-extrabold" style={{ color: scoreColor }}>{score}</span>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <ShieldCheck size={13} style={{ color: scoreColor }} />
            <p className="text-[11px] font-bold uppercase tracking-wider" style={{ color: scoreColor }}>
              Kiểm chứng · {scoreLabel}
            </p>
          </div>
          <p className="mt-0.5 truncate text-xs font-semibold text-text" title={question ?? undefined}>
            {question || "Câu trả lời hiện tại"}
          </p>
        </div>
      </div>

      {/* Stratified coverage bar */}
      <div className="mt-2.5">
        <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
          {seg.map((s, i) => s.n > 0 && (
            <div key={i} style={{ width: `${(s.n / segTotal) * 100}%`, background: s.color }} title={`${s.label}: ${s.n}`} />
          ))}
        </div>
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[9.5px] font-semibold text-muted">
          {seg.map((s, i) => s.n > 0 && (
            <span key={i} className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.label} {s.n}
            </span>
          ))}
        </div>
      </div>

      {/* Quick stat cards */}
      <div className="mt-2.5 grid grid-cols-4 gap-1.5">
        {[
          { icon: <ShieldCheck size={12} />, big: `${verifiedCount}/${nodes.length}`, label: "node verified" },
          { icon: <Network size={12} />, big: `${edgesWithEvidence}`, label: "edge có BC" },
          { icon: <FileText size={12} />, big: `${sourceCount}`, label: "nguồn" },
          { icon: <Target size={12} />, big: `${score}%`, label: "tổng thể" },
        ].map((c, i) => (
          <div key={i} className="rounded-md border border-outline bg-white px-1.5 py-1 text-center">
            <div className="flex items-center justify-center text-primary">{c.icon}</div>
            <div className="text-[13px] font-extrabold leading-tight text-text">{c.big}</div>
            <div className="text-[8.5px] leading-tight text-muted">{c.label}</div>
          </div>
        ))}
      </div>

      {/* Gap warning */}
      {gapNodes.length > 0 && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md border border-amber-300 bg-amber-50 px-2 py-1.5 text-[10.5px] text-amber-800">
          <AlertCircle size={12} className="mt-0.5 shrink-0" />
          <span>Phát hiện <b>{gapNodes.length}</b> thực thể quan trọng chưa có bằng chứng — cần kiểm tra thêm.</span>
        </div>
      )}
    </div>
  );
}

function GraphPurposeCard({ hasTrace }: { hasTrace: boolean }) {
  return (
    <div className="rounded-lg border border-outline bg-slate-50 px-3 py-2">
      <div className="flex items-start gap-2">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-white text-primary ring-1 ring-outline">
          <Link2 size={14} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted">Graph role</p>
          <p className="mt-0.5 text-xs leading-relaxed text-text">
            Knowledge Graph dung de truy vet quan he va evidence; Mindmap dung de hoc va to chuc y.
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-semibold text-muted">
            <span className="rounded border border-outline bg-white px-2 py-0.5">edge = relation</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">click edge = evidence</span>
            <span className="rounded border border-outline bg-white px-2 py-0.5">{hasTrace ? "scoped to answer" : "scoped to sources"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Fullscreen overlay ───────────────────────────────────────────────────────

function FullscreenOverlay({
  mode, verifyMode, canvas, selectedNode, selectedRelation, ownerId, collectionId, onSelect, onEdgeSelect, onClose, onOpenEvidence, onDraftQuestion, onFindRelated,
}: {
  mode: "graph" | "mindmap";
  verifyMode?: boolean;
  canvas: { nodes?: CanvasNode[]; edges?: CanvasEdge[] };
  selectedNode: SelectedNode | null;
  selectedRelation: SelectedRelation | null;
  ownerId: string;
  collectionId?: string | null;
  onSelect: (id: string, label: string) => void;
  onEdgeSelect?: (edge: CanvasEdge) => void;
  onClose: () => void;
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
  onDraftQuestion?: (draft: string) => void;
  onFindRelated?: (draft: string) => void;
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="fixed inset-0 z-50 flex flex-col bg-white">
      <div className="flex shrink-0 items-center justify-between border-b border-outline bg-white px-5 py-3 shadow-sm">
        <span className="text-sm font-semibold text-text">
          {mode === "graph" ? "Knowledge Graph" : "Mindmap"} — full view
        </span>
        <div className="flex items-center gap-3">
          {selectedNode && (
            <span className="text-xs font-semibold text-primary truncate max-w-[280px]">
              {selectedNode.label}
            </span>
          )}
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 rounded-md border border-outline px-3 py-1.5 text-xs font-semibold text-muted hover:border-primary/40 hover:text-primary transition"
          >
            <X size={13} /> Close <span className="ml-0.5 opacity-50 text-[10px]">Esc</span>
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-hidden relative">
        <GraphCanvas
          mode={mode}
          verifyMode={verifyMode}
          canvasNodes={canvas.nodes ?? []}
          canvasEdges={canvas.edges ?? []}
          onSelect={onSelect}
          onEdgeSelect={onEdgeSelect}
          onOpenEvidence={onOpenEvidence}
          onDraftQuestion={onDraftQuestion}
          onFindRelated={onFindRelated}
        />
        {mode === "graph" && <GraphLegend />}
      </div>
      {selectedRelation && (
        <RelationInfoCard relation={selectedRelation} onClose={() => {}} onOpenEvidence={onOpenEvidence} />
      )}
      {!selectedRelation && selectedNode && (
        <NodeInfoCard node={selectedNode} ownerId={ownerId} collectionId={collectionId} onClose={() => {}} onOpenEvidence={onOpenEvidence} />
      )}
    </div>,
    document.body
  );
}

// ─── Main tab ─────────────────────────────────────────────────────────────────

export default function GraphTab({
  mode,
  onOpenEvidence,
}: {
  mode: "graph" | "mindmap";
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
}) {
  const { workspace, scopedMaterialIds, sourceScopeMode, activeQueryContext, setChatDraft, graphFocusOnAnswer, setGraphFocusOnAnswer, materials } = useWorkspace();
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [selectedRelation, setSelectedRelation] = useState<SelectedRelation | null>(null);
  const [rootTopic, setRootTopic] = useState("");
  const [graphResult, setGraphResult] = useState<GraphResponse | null>(null);
  // viz_mode chosen by /graph/auto (citation_network / concept_graph …) — shown
  // as a small badge so users know why the graph looks the way it does.
  const [autoMode, setAutoMode] = useState<string | null>(null);
  const [mindmapResult, setMindmapResult] = useState<MindmapResponse | null>(null);
  const [mindmapDetail, setMindmapDetail] = useState<"brief" | "overview" | "detailed">("overview");
  const [mindmapUseLlm, setMindmapUseLlm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [graphSearch, setGraphSearch] = useState("");
  // G4 — anchored ask + highlight from last graph answer
  const [askAnchor, setAskAnchor] = useState<{ id: string; label: string } | null>(null);
  const [graphAnswerHighlights, setGraphAnswerHighlights] = useState<string[]>([]);
  const lastAutoLoadKey = useRef<string | null>(null);

  const graphFocus = buildGraphFocus(activeQueryContext);
  // Material id → display name map, used to build hover-preview source labels.
  const materialNameMap = new Map<string, string>(
    materials.map((m) => [m.materialId, m.originalName || m.filename || m.materialId]),
  );
  // Knowledge Graph tab always renders a node-edge graph. For legal docs the
  // backend supplies a citation graph (Điều + dẫn-chiếu); for concept docs the
  // usual entity graph. Hierarchy trees live in the Mindmap tab, not here.
  const canvasRenderMode: "graph" | "mindmap" = mode;
  const canvas = mode === "graph"
    ? toGraph(graphResult, graphFocus)
    : toMindmap(mindmapResult, materialNameMap);
  const answerTraceMaterialIds = mode === "graph" && activeQueryContext && !activeQueryContext.response.was_refused
    ? Array.from(new Set(activeQueryContext.response.citations.map((citation) => citation.doc_id).filter(Boolean)))
    : [];
  const graphMaterialIds = answerTraceMaterialIds.length ? answerTraceMaterialIds : scopedMaterialIds;

  const handleSelect = useCallback((id: string, label: string) => {
    setSelectedRelation(null);
    const node = canvas.nodes?.find((n) => n.id === id);
    const edges = canvas.edges ?? [];
    const nodeMap = new Map((canvas.nodes ?? []).map((n) => [n.id, n]));

    const connections = edges
      .filter((e) => e.source === id || e.target === id)
      .map((e) => {
        const otherId = e.source === id ? e.target : e.source;
        const other = nodeMap.get(otherId);
        return other ? { label: other.label, relation: e.label ?? "" } : null;
      })
      .filter(Boolean) as { label: string; relation: string }[];

    setSelectedNode({
      id,
      label,
      type: node?.type ?? "concept",
      confidence: node?.confidence ?? null,
      degree: node?.degree ?? connections.length,
      mention_count: (node as any)?.mention_count ?? 0,
      source_docs: (node as any)?.source_docs ?? [],
      evidenceRefs: (node as any)?.evidence_refs ?? [],
      connections,
    });
  }, [canvas]);

  const handleEdgeSelect = useCallback((edge: CanvasEdge) => {
    const nodeMap = new Map((canvas.nodes ?? []).map((node) => [node.id, node]));
    setSelectedNode(null);
    setSelectedRelation({
      source: edge.source,
      target: edge.target,
      sourceLabel: edge.source_label || nodeMap.get(edge.source)?.label || edge.source,
      targetLabel: edge.target_label || nodeMap.get(edge.target)?.label || edge.target,
      relation: edge.label,
      confidence: edge.confidence ?? null,
      evidenceCount: edge.evidence_count ?? edge.evidence_refs?.length ?? 0,
      evidenceRefs: edge.evidence_refs ?? [],
      evidenceTextChunk: edge.evidence_text_chunk ?? null,
    });
  }, [canvas.nodes]);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      if (mode === "graph") {
        // Verify-mode: extract block_ids + material_ids + pages from last answer's
        // citations so backend filters graph to entities backing the answer.
        const focusBlockIds: string[] = [];
        const focusMaterialIds: string[] = [];
        const focusPages: string[] = [];
        if (graphFocusOnAnswer && activeQueryContext && !activeQueryContext.response.was_refused) {
          for (const citation of activeQueryContext.response.citations) {
            if (citation.doc_id) {
              focusMaterialIds.push(citation.doc_id);
              const pages = citation.pages ?? (citation.page ? [citation.page] : []);
              for (const p of pages) {
                focusPages.push(`${citation.doc_id}:${p}`);
              }
            }
            for (const ev of citation.evidence_blocks ?? []) {
              if (ev.block_id) focusBlockIds.push(ev.block_id);
            }
            if (citation.block_id) focusBlockIds.push(citation.block_id);
          }
        }
        const focusPayload = {
          owner_id: workspace.ownerId,
          collection_id: workspace.collectionId || null,
          material_ids: graphMaterialIds,
          root_topic: activeQueryContext?.question || rootTopic || "Knowledge Graph",
          focus_block_ids: Array.from(new Set(focusBlockIds)),
          focus_material_ids: Array.from(new Set(focusMaterialIds)),
          focus_pages: Array.from(new Set(focusPages)),
          focus_query_text: graphFocusOnAnswer && activeQueryContext ? activeQueryContext.question : undefined,
          focus_answer_text: graphFocusOnAnswer && activeQueryContext && !activeQueryContext.response.was_refused
            ? activeQueryContext.response.answer
            : undefined,
        };
        // Structure-adaptive: backend returns a citation graph (Điều + dẫn-chiếu
        // edges) for legal/hierarchical docs, or null for concept docs. Either
        // way the Knowledge Graph tab renders a node-edge graph (never a tree).
        const auto = await loadAutoViz({
          ...focusPayload,
          // verify = show only Điều directly cited in answer (precise).
          // explore = broader view with query-text matching.
          graph_mode: graphFocusOnAnswer && focusBlockIds.length > 0 ? "verify" : "explore",
        });
        setAutoMode(auto.viz_mode);
        if (auto.graph && auto.graph.nodes.length > 0) {
          setGraphResult(auto.graph);
        } else {
          const response = await loadGraph(focusPayload);
          setGraphResult(response);
        }
        setSelectedNode(null);
        setSelectedRelation(null);
      } else {
        const response = await loadMindmap({
          owner_id: workspace.ownerId,
          collection_id: workspace.collectionId || null,
          material_ids: scopedMaterialIds,
          root_topic: rootTopic || workspace.collectionName || workspace.subject || "Central Topic",
          detail_level: mindmapDetail,
          use_llm: mindmapUseLlm,
        });
        setMindmapResult(response);
        setSelectedNode(null);
        setSelectedRelation(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load.");
    } finally {
      setLoading(false);
    }
  }

  const hasScope = Boolean(workspace.collectionId ? sourceScopeMode === "all" || scopedMaterialIds.length : scopedMaterialIds.length);

  useEffect(() => {
    setError(null);
    setSelectedNode(null);
    setSelectedRelation(null);
    setAutoMode(null);
  }, [mode]);

  useEffect(() => {
    if (selectedNode && (!canvas.nodes || !canvas.nodes.some((node) => node.id === selectedNode.id))) {
      setSelectedNode(null);
    }
  }, [canvas.nodes, selectedNode]);

  useEffect(() => {
    if (hasScope) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // Re-fetch when verify-mode toggles so user sees focused / full graph immediately
  useEffect(() => {
    if (hasScope && mode === "graph") refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphFocusOnAnswer]);

  useEffect(() => {
    if (!hasScope) {
      lastAutoLoadKey.current = null;
      return;
    }

    const traceKey = mode === "graph" ? (activeQueryContext?.createdAt ?? "no-trace") : "";
    const scopedIds = mode === "graph" ? graphMaterialIds : scopedMaterialIds;
    const scopeKey = `${mode}|${workspace.collectionId || "materials"}|${scopedIds.join(",")}|${mindmapDetail}|${mindmapUseLlm}|${traceKey}`;
    if (lastAutoLoadKey.current === scopeKey) {
      return;
    }

    lastAutoLoadKey.current = scopeKey;
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasScope, mode, scopedMaterialIds, workspace.collectionId, mindmapDetail, mindmapUseLlm]);

  const hasCanvas = Boolean(canvas.nodes && canvas.nodes.length > 0);
  const graphStats = mode === "graph" && hasCanvas
    ? {
        nodes: canvas.nodes?.length ?? 0,
        edges: canvas.edges?.length ?? 0,
        evidence: (canvas.edges ?? []).reduce((sum, edge) => sum + (edge.evidence_count ?? edge.evidence_refs?.length ?? 0), 0),
        sources: answerTraceMaterialIds.length || scopedMaterialIds.length || (workspace.collectionId ? 1 : 0),
      }
    : null;
  const mindmapStats = mode === "mindmap" && hasCanvas
    ? {
        groups: (canvas.nodes ?? []).filter((node) => node.type === "topic").length,
        concepts: (canvas.nodes ?? []).filter((node) => node.type !== "root" && node.type !== "topic").length,
        sources: scopedMaterialIds.length || (workspace.collectionId ? 1 : 0),
      }
    : null;

  return (
    <>
      <div className="flex h-full flex-col bg-slate-50">
        {/* Toolbar */}
        <div className="shrink-0 px-4 pt-4 pb-3 border-b border-outline bg-white flex flex-col gap-3">
          {mode === "graph" && graphStats && graphFocusOnAnswer ? (
            <VerificationDashboard
              question={activeQueryContext?.question ?? null}
              coverage={activeQueryContext?.response?.sentence_coverage ?? null}
              nodes={canvas.nodes ?? []}
              edges={canvas.edges ?? []}
              sourceCount={graphStats.sources}
            />
          ) : mode === "graph" && graphStats && (
            <TraceHeader
              question={activeQueryContext?.question ?? null}
              nodeCount={graphStats.nodes}
              edgeCount={graphStats.edges}
              evidenceCount={graphStats.evidence}
              sourceCount={graphStats.sources}
            />
          )}
          {mode === "graph" && !graphStats && (
            <GraphPurposeCard hasTrace={Boolean(activeQueryContext)} />
          )}
          {mode === "mindmap" && (
            <div className="flex flex-col gap-2">
              <input
                className="w-full rounded-md border border-outline px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition"
                value={rootTopic}
                onChange={(e) => setRootTopic(e.target.value)}
                placeholder="Chủ đề gốc của mindmap…"
              />
              <div className="flex flex-wrap items-center gap-2">
                <div className="inline-flex overflow-hidden rounded-md border border-outline bg-slate-50 p-0.5">
                  <button
                    type="button"
                    onClick={() => setMindmapDetail("brief")}
                    title="Brief — 4 nhóm × 3 khái niệm, gọn nhất"
                    className={`px-2.5 py-1 text-[11px] font-semibold transition ${
                      mindmapDetail === "brief" ? "rounded bg-white text-primary shadow-sm" : "text-muted hover:text-text"
                    }`}
                  >
                    Brief
                  </button>
                  <button
                    type="button"
                    onClick={() => setMindmapDetail("overview")}
                    title="Overview — 6 nhóm × 5 khái niệm, cân bằng"
                    className={`px-2.5 py-1 text-[11px] font-semibold transition ${
                      mindmapDetail === "overview" ? "rounded bg-white text-primary shadow-sm" : "text-muted hover:text-text"
                    }`}
                  >
                    Overview
                  </button>
                  <button
                    type="button"
                    onClick={() => setMindmapDetail("detailed")}
                    title="Detailed — 8 nhóm × 8 khái niệm, đầy đủ"
                    className={`px-2.5 py-1 text-[11px] font-semibold transition ${
                      mindmapDetail === "detailed" ? "rounded bg-white text-primary shadow-sm" : "text-muted hover:text-text"
                    }`}
                  >
                    Detailed
                  </button>
                </div>
                <label className="inline-flex items-center gap-1.5 rounded-md border border-outline bg-white px-2.5 py-1 text-[11px] font-semibold text-muted">
                  <input
                    type="checkbox"
                    checked={mindmapUseLlm}
                    onChange={(event) => setMindmapUseLlm(event.target.checked)}
                  />
                  LLM refine
                </label>
                <span className="text-[10px] text-muted">
                  Overview ưu tiên map gọn; Detailed mở rộng thêm concept.
                </span>
              </div>
            </div>
          )}
          <div className="flex gap-2">
            <button
              className="flex flex-1 items-center justify-center gap-2 rounded-md bg-primary py-2 text-sm font-semibold text-white disabled:opacity-50 transition hover:bg-primary/90"
              onClick={refresh}
              disabled={loading || !hasScope}
            >
              {loading ? <Loader2 className="animate-spin" size={14} /> : mode === "graph" ? <Network size={14} /> : <RefreshCw size={14} />}
              {mode === "graph" ? (activeQueryContext ? "Refresh relation trace" : "Trace relations") : "Tạo Mindmap"}
            </button>
            {hasCanvas && (
              <button
                title="Toàn màn hình"
                onClick={() => setFullscreen(true)}
                className="flex items-center justify-center gap-1.5 rounded-md border border-outline px-3 py-2 text-xs font-semibold text-muted hover:border-primary/40 hover:text-primary transition"
              >
                <Maximize2 size={14} />
              </button>
            )}
          </div>

          {/* Graph search + stats */}
          {hasCanvas && mode === "graph" && (
            <div className="flex flex-col gap-2">
              <div className="relative">
                <input
                  className="w-full rounded-md border border-outline bg-slate-50 py-1.5 pl-7 pr-3 text-xs focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition"
                  placeholder="Tìm khái niệm trong graph…"
                  value={graphSearch}
                  onChange={(e) => setGraphSearch(e.target.value)}
                />
                <svg className="absolute left-2 top-1/2 -translate-y-1/2 text-muted" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                </svg>
                {graphSearch && (
                  <button
                    type="button"
                    onClick={() => setGraphSearch("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-text"
                    aria-label="Xóa tìm kiếm"
                  >
                    ×
                  </button>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-[10px] text-muted">
                {graphFocusOnAnswer && activeQueryContext && (
                  <span className="flex items-center gap-1 rounded bg-amber-100 px-2 py-0.5 text-amber-800 font-semibold border border-amber-300">
                    <Target size={9} />
                    Kiểm chứng câu trả lời
                    <button
                      type="button"
                      onClick={() => {
                        setGraphFocusOnAnswer(false);
                        setTimeout(() => void refresh(), 50);
                      }}
                      className="ml-1 rounded px-1 text-amber-700 hover:bg-amber-200 hover:text-amber-900"
                      title="Hiện toàn bộ graph"
                    >
                      ×
                    </button>
                  </span>
                )}
                {graphFocus && activeQueryContext && !graphFocusOnAnswer && (
                  <span className="flex items-center gap-1 rounded bg-primary/8 px-1.5 py-0.5 text-primary font-medium">
                    <Target size={9} />
                    <span className="max-w-[180px] truncate" title={activeQueryContext.question}>
                      Theo câu hỏi: {activeQueryContext.question}
                    </span>
                  </span>
                )}
                {autoMode && (
                  <span className="flex items-center gap-1 rounded bg-primary/8 px-1.5 py-0.5 text-primary font-medium" title="Kiểu graph tự chọn theo cấu trúc tài liệu">
                    {autoMode === "citation_network"
                      ? "Mạng dẫn chiếu điều luật"
                      : autoMode === "hierarchy"
                        ? "Graph theo Điều/Mục"
                        : autoMode === "timeline"
                          ? "Dòng thời gian"
                          : "Đồ thị khái niệm"}
                  </span>
                )}
                <span className="ml-auto">Click node · Chuột phải để hành động</span>
              </div>
            </div>
          )}
          {mindmapStats && (
            <div className="flex flex-wrap items-center gap-2 text-[10px] font-semibold text-muted">
              <span className="rounded border border-outline bg-slate-50 px-2 py-1">
                {mindmapStats.groups} nhóm
              </span>
              <span className="rounded border border-outline bg-slate-50 px-2 py-1">
                {mindmapStats.concepts} khái niệm
              </span>
              <span className="rounded border border-outline bg-slate-50 px-2 py-1">
                {mindmapStats.sources} nguồn
              </span>
              <span className="ml-auto text-[10px] font-medium text-muted">
                Click để xem chi tiết, chuột phải để mở tác vụ.
              </span>
            </div>
          )}
        </div>

        {/* Canvas */}
        <div className="flex-1 overflow-hidden relative">
          {error && (
            <div className="absolute top-4 left-4 right-4 z-10 flex items-start gap-2 rounded-lg border border-red-200 bg-white/90 backdrop-blur p-3 text-xs text-red-700 shadow-sm">
              <AlertCircle size={14} className="shrink-0 mt-0.5" /> {error}
            </div>
          )}
          {!hasCanvas && !loading && !error && (
            <div className="h-full flex items-center justify-center text-xs text-muted p-6 text-center">
              {hasScope
                ? mode === "graph"
                  ? "Nhấn Trace relations để xem các quan hệ có evidence trong tài liệu."
                  : "Nhấn Tạo Mindmap để tổ chức ý chính từ tài liệu."
                : "Chọn hoặc tải tài liệu trước khi tạo visualization."}
            </div>
          )}
          {hasCanvas && (
            <div className="h-full bg-white">
              <GraphCanvas
                mode={canvasRenderMode}
                verifyMode={graphFocusOnAnswer && canvasRenderMode === "graph"}
                onSelect={handleSelect}
                onEdgeSelect={handleEdgeSelect}
                canvasNodes={canvas.nodes!}
                canvasEdges={canvas.edges ?? []}
                onOpenEvidence={onOpenEvidence}
                onDraftQuestion={(draft) => setChatDraft(draft)}
                onFindRelated={(draft) => setChatDraft(draft)}
                searchQuery={graphSearch}
                answerEntityIds={graphAnswerHighlights}
              />
              {canvasRenderMode === "graph" && <GraphLegend />}
            </div>
          )}
        </div>

        {/* Node info card */}
        {selectedRelation && (
          <RelationInfoCard relation={selectedRelation} onClose={() => setSelectedRelation(null)} onOpenEvidence={onOpenEvidence} />
        )}
        {!selectedRelation && selectedNode && (
          <NodeInfoCard
            node={selectedNode}
            ownerId={workspace.ownerId}
            collectionId={workspace.collectionId}
            onClose={() => setSelectedNode(null)}
            onOpenEvidence={onOpenEvidence}
            onAskAboutNode={
              workspace.collectionId
                ? (id, label) => setAskAnchor({ id, label })
                : undefined
            }
          />
        )}
      </div>

      {askAnchor && workspace.collectionId && (
        <AskAboutNodeModal
          anchorId={askAnchor.id}
          anchorLabel={askAnchor.label}
          ownerId={workspace.ownerId}
          collectionId={workspace.collectionId}
          conversationId={`graph-ask:${workspace.collectionId}:${askAnchor.id}`}
          onClose={() => setAskAnchor(null)}
          onAnswered={(r) => {
            if (r.used_entity_ids && r.used_entity_ids.length > 0) {
              setGraphAnswerHighlights(r.used_entity_ids);
            }
          }}
        />
      )}

      {fullscreen && hasCanvas && (
        <FullscreenOverlay
          mode={canvasRenderMode}
          verifyMode={graphFocusOnAnswer && canvasRenderMode === "graph"}
          canvas={canvas}
          selectedNode={selectedNode}
          selectedRelation={selectedRelation}
          ownerId={workspace.ownerId}
          collectionId={workspace.collectionId}
          onSelect={handleSelect}
          onEdgeSelect={handleEdgeSelect}
          onClose={() => setFullscreen(false)}
          onOpenEvidence={onOpenEvidence}
          onDraftQuestion={(draft) => setChatDraft(draft)}
          onFindRelated={(draft) => setChatDraft(draft)}
        />
      )}
    </>
  );
}
