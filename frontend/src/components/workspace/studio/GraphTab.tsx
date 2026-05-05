import { useState, useEffect, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import { AlertCircle, BookOpen, FileText, Link2, Loader2, Maximize2, RefreshCw, Target, X } from "lucide-react";
import { GraphResponse, MindmapResponse, loadGraph, loadMindmap } from "../../../api/client";
import GraphCanvas, { CanvasEdge, CanvasNode } from "../../GraphCanvas";
import { useWorkspace } from "../../../state/workspace";

// ─── Data transforms ──────────────────────────────────────────────────────────

const MAX_GRAPH_NODES = 34;
const MAX_GRAPH_EDGES = 44;
const MAX_MINDMAP_GROUPS = 8;
const MAX_MINDMAP_ITEMS_PER_GROUP = 9;
const MINDMAP_BRANCH_COLORS = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0891b2", "#65a30d", "#4f46e5"];
const NOISY_GRAPH_LABELS = new Set([
  "caption",
  "chart",
  "converted",
  "docx",
  "file word",
  "jpg",
  "jpeg",
  "llm",
  "ocr",
  "ocr engine",
  "ocr engine png",
  "parser",
  "pass",
  "pdf",
  "png",
  "png ocr",
  "pptx",
  "randomly",
  "section",
  "slide",
  "test",
  "text",
  "vlm",
  "word",
  "xlsx",
]);
const NOISY_GRAPH_WORDS = new Set(["adds", "description", "file", "source", "stabilizes", "stops", "technique"]);
const FORMAT_GRAPH_WORDS = new Set(["docx", "jpg", "jpeg", "llm", "ocr", "pdf", "png", "pptx", "text", "vlm", "xlsx"]);

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

function cleanGraphLabel(value: string) {
  let label = repairMojibakeText(value)
    .replace(/\s*[,;:/]\s*/g, " ")
    .replace(/^[^\wÀ-ỹ]+/u, "")
    .replace(/[|()[\]{}'"]+/g, " ")
    .replace(/\.+$/g, "")
    .replace(/^(?:\d+(?:[.,]\d+)?\s+){1,3}/, "")
    .replace(/\s+/g, " ")
    .trim();
  if (label.length < 3 || label.includes("|")) return null;

  const lower = label.toLowerCase();
  const words = label.split(/\s+/);
  if (NOISY_GRAPH_LABELS.has(lower) || NOISY_GRAPH_WORDS.has(lower)) return null;
  if (NOISY_GRAPH_WORDS.has(words[0]?.toLowerCase() ?? "")) return null;
  if (NOISY_GRAPH_WORDS.has(words[words.length - 1]?.toLowerCase() ?? "")) return null;
  const formatWordCount = words.filter((word) => FORMAT_GRAPH_WORDS.has(word.toLowerCase())).length;
  if (formatWordCount > 1 || formatWordCount === words.length) return null;
  if (FORMAT_GRAPH_WORDS.has(words[words.length - 1]?.toLowerCase() ?? "") && words.length <= 3) return null;
  if (words.length > 4) return null;
  if (new Set(words.map((word) => word.toLowerCase())).size < words.length) return null;
  if (words.length >= 3 && words.every((word) => /^[A-Z0-9]{2,}$/.test(word))) return null;
  if (/\.(png|jpe?g|pdf|docx|pptx|xlsx)$/i.test(label)) return null;
  if ((label.match(/\d+/g) ?? []).length >= 3) return null;

  const chars = label.replace(/\s/g, "");
  const digits = (chars.match(/\d/g) ?? []).length;
  const letters = (chars.match(/\p{L}/gu) ?? []).length;
  const symbols = Math.max(0, chars.length - digits - letters);
  if (digits >= 3 && digits >= letters) return null;
  if (symbols > letters && symbols > 1) return null;
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
    (e) => allowedIds.has(e.source) && allowedIds.has(e.target)
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
    degree: degree[n.id] ?? 0,
    mention_count: (n as any).mention_count ?? 0,
    source_docs: (n as any).source_docs ?? [],
    evidence_refs: (n as any).evidence_refs ?? [],
    focused: focusIds.has(n.id),
  }));
  const edges: CanvasEdge[] = visibleEdges.map((e, i) => ({
    id: `${e.source}-${e.target}-${i}`,
    source: e.source,
    target: e.target,
    label: e.relation_type,
    focused: focusIds.has(e.source) || focusIds.has(e.target),
  }));
  return { nodes, edges };
}
function toMindmap(response: MindmapResponse | null): { nodes?: CanvasNode[]; edges?: CanvasEdge[] } {
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
        nodes.push({
          id: item.id,
          label: cleanGraphLabel(item.label) ?? item.label,
          type,
          degree: childCount,
          confidence: null,
          evidence_refs: (item as any).citations ?? [],
          branchColor: color,
          depth,
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
      nodes.push({
        id: m.id,
        label: m.label,
        type: typeName,
        confidence: null,
        evidence_refs: (m as any).citations ?? [],
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

// ─── Color map (mirrors GraphCanvas) ─────────────────────────────────────────

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
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
  concept: "Khái niệm",
  technology: "Công nghệ",
  method: "Phương pháp",
  person: "Nhân vật",
  organization: "Tổ chức",
  org: "Tổ chức",
  location: "Địa điểm",
  event: "Sự kiện",
  date: "Mốc thời gian",
  entity: "Thực thể",
};

function displayMindmapType(typeName: string) {
  return MINDMAP_TYPE_LABELS[typeName.toLowerCase()] ?? typeName.charAt(0).toUpperCase() + typeName.slice(1);
}

function typeColor(type: string) {
  const key = type.toLowerCase().split(/[_\s]/)[0];
  return TYPE_COLORS[key] ?? { bg: "#f1f5f9", border: "#94a3b8", text: "#475569" };
}

const LEGEND_TYPES = ["concept", "technology", "method", "person", "organization", "location"] as const;

// ─── Node info card ───────────────────────────────────────────────────────────

function NodeInfoCard({ node, onClose }: { node: SelectedNode; onClose: () => void }) {
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

// ─── Fullscreen overlay ───────────────────────────────────────────────────────

function FullscreenOverlay({
  mode, canvas, selectedNode, onSelect, onClose, onOpenEvidence, onDraftQuestion, onFindRelated,
}: {
  mode: "graph" | "mindmap";
  canvas: { nodes?: CanvasNode[]; edges?: CanvasEdge[] };
  selectedNode: SelectedNode | null;
  onSelect: (id: string, label: string) => void;
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
          canvasNodes={canvas.nodes ?? []}
          canvasEdges={canvas.edges ?? []}
          onSelect={onSelect}
          onOpenEvidence={onOpenEvidence}
          onDraftQuestion={onDraftQuestion}
          onFindRelated={onFindRelated}
        />
        {mode === "graph" && <GraphLegend />}
      </div>
      {selectedNode && (
        <NodeInfoCard node={selectedNode} onClose={() => {}} />
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
  const { workspace, scopedMaterialIds, activeQueryContext, setChatDraft } = useWorkspace();
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [rootTopic, setRootTopic] = useState("");
  const [graphResult, setGraphResult] = useState<GraphResponse | null>(null);
  const [mindmapResult, setMindmapResult] = useState<MindmapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const lastAutoLoadKey = useRef<string | null>(null);

  const graphFocus = buildGraphFocus(activeQueryContext);
  const canvas = mode === "graph" ? toGraph(graphResult, graphFocus) : toMindmap(mindmapResult);

  const handleSelect = useCallback((id: string, label: string) => {
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

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      if (mode === "graph") {
        const response = await loadGraph({
          owner_id: workspace.ownerId,
          collection_id: workspace.collectionId || null,
          material_ids: workspace.collectionId ? [] : scopedMaterialIds,
          root_topic: "Knowledge Graph",
        });
        setGraphResult(response);
        setSelectedNode(null);
      } else {
        const response = await loadMindmap({
          owner_id: workspace.ownerId,
          collection_id: workspace.collectionId || null,
          material_ids: workspace.collectionId ? [] : scopedMaterialIds,
          root_topic: rootTopic || workspace.collectionName || workspace.subject || "Central Topic",
        });
        setMindmapResult(response);
        setSelectedNode(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load.");
    } finally {
      setLoading(false);
    }
  }

  const hasScope = Boolean(workspace.collectionId || scopedMaterialIds.length);

  useEffect(() => {
    setError(null);
    setSelectedNode(null);
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

  useEffect(() => {
    if (!hasScope) {
      lastAutoLoadKey.current = null;
      return;
    }

    const scopeKey = `${mode}|${workspace.collectionId || "materials"}|${scopedMaterialIds.join(",")}`;
    if (lastAutoLoadKey.current === scopeKey) {
      return;
    }

    lastAutoLoadKey.current = scopeKey;
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasScope, mode, scopedMaterialIds, workspace.collectionId]);

  const hasCanvas = Boolean(canvas.nodes && canvas.nodes.length > 0);
  const mindmapStats = mode === "mindmap" && hasCanvas
    ? {
        groups: (canvas.nodes ?? []).filter((node) => node.type === "cluster").length,
        concepts: (canvas.nodes ?? []).filter((node) => node.type !== "root" && node.type !== "cluster").length,
        sources: scopedMaterialIds.length || (workspace.collectionId ? 1 : 0),
      }
    : null;

  return (
    <>
      <div className="flex h-full flex-col bg-slate-50">
        {/* Toolbar */}
        <div className="shrink-0 px-4 pt-4 pb-3 border-b border-outline bg-white flex flex-col gap-3">
          {mode === "mindmap" && (
            <input
              className="w-full rounded-md border border-outline px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition"
              value={rootTopic}
              onChange={(e) => setRootTopic(e.target.value)}
              placeholder="Chủ đề gốc của mindmap…"
            />
          )}
          <div className="flex gap-2">
            <button
              className="flex flex-1 items-center justify-center gap-2 rounded-md bg-primary py-2 text-sm font-semibold text-white disabled:opacity-50 transition hover:bg-primary/90"
              onClick={refresh}
              disabled={loading || !hasScope}
            >
              {loading ? <Loader2 className="animate-spin" size={14} /> : <RefreshCw size={14} />}
              {mode === "graph" ? "Tạo Knowledge Graph" : "Tạo Mindmap"}
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
          {hasCanvas && mode === "graph" && graphFocus && activeQueryContext && (
            <div className="flex items-start gap-2 text-[10px] text-muted">
              <Target size={11} className="mt-0.5 shrink-0 text-primary" />
              <span className="min-w-0 truncate">
                Dang uu tien graph theo cau hoi: {activeQueryContext.question}
              </span>
            </div>
          )}
          {hasCanvas && mode === "graph" && (
            <p className="text-[10px] text-muted">
              <BookOpen size={9} className="inline mr-1" />
              Click vào node để xem chi tiết khái niệm và tài liệu nguồn.
            </p>
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
                ? `Nhấn "Tạo ${mode === "graph" ? "Knowledge Graph" : "Mindmap"}" để trực quan hóa tri thức từ tài liệu.`
                : "Chọn hoặc tải tài liệu trước khi tạo visualization."}
            </div>
          )}
          {hasCanvas && (
            <div className="h-full bg-white">
              <GraphCanvas
                mode={mode}
                onSelect={handleSelect}
                canvasNodes={canvas.nodes!}
                canvasEdges={canvas.edges ?? []}
                onOpenEvidence={onOpenEvidence}
                onDraftQuestion={(draft) => setChatDraft(draft)}
                onFindRelated={(draft) => setChatDraft(draft)}
              />
              {mode === "graph" && <GraphLegend />}
            </div>
          )}
        </div>

        {/* Node info card */}
        {selectedNode && (
          <NodeInfoCard node={selectedNode} onClose={() => setSelectedNode(null)} />
        )}
      </div>

      {fullscreen && hasCanvas && (
        <FullscreenOverlay
          mode={mode}
          canvas={canvas}
          selectedNode={selectedNode}
          onSelect={handleSelect}
          onClose={() => setFullscreen(false)}
          onOpenEvidence={onOpenEvidence}
          onDraftQuestion={(draft) => setChatDraft(draft)}
          onFindRelated={(draft) => setChatDraft(draft)}
        />
      )}
    </>
  );
}


