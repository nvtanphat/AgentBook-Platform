import { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { AlertCircle, Loader2, Maximize2, RefreshCw, X } from "lucide-react";
import { GraphResponse, MindmapResponse, loadGraph, loadMindmap } from "../../../api/client";
import GraphCanvas, { CanvasEdge, CanvasNode } from "../../GraphCanvas";
import { useWorkspace } from "../../../state/workspace";

function toGraph(response: GraphResponse | null): { nodes?: CanvasNode[]; edges?: CanvasEdge[] } {
  if (!response || !response.nodes.length) return { nodes: [], edges: [] };
  const nodes = response.nodes.map((node, index) => ({
    id: node.id, label: node.label, type: node.type,
    position: index === 0 ? { x: 200, y: 150 } : undefined,
    confidence: node.confidence,
  }));
  const edges = response.edges.map((edge, index) => ({
    id: `${edge.source}-${edge.target}-${index}`,
    source: edge.source, target: edge.target, label: edge.relation_type,
  }));
  return { nodes, edges };
}

function toMindmap(response: MindmapResponse | null): { nodes?: CanvasNode[]; edges?: CanvasEdge[] } {
  if (!response || !response.nodes.length) return { nodes: [], edges: [] };
  const rootId = "root-topic";
  return {
    nodes: [
      { id: rootId, label: response.root_topic, type: "root", position: { x: 200, y: 150 } },
      ...response.nodes.map((node) => ({
        id: node.id, label: node.label, type: node.summary ?? "entity", position: undefined, confidence: null,
      })),
    ],
    edges: response.nodes.map((node, index) => ({
      id: `${rootId}-${node.id}`, source: rootId, target: node.id, label: index < 8 ? "related" : "context",
    })),
  };
}

// ─── Fullscreen overlay ───────────────────────────────────────────────────────

function FullscreenOverlay({
  mode, canvas, selectedNode, onSelect, onClose,
}: {
  mode: "graph" | "mindmap";
  canvas: { nodes?: CanvasNode[]; edges?: CanvasEdge[] };
  selectedNode: { id: string; label: string } | null;
  onSelect: (id: string, label: string) => void;
  onClose: () => void;
}) {
  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div className="fixed inset-0 z-50 flex flex-col bg-white">
      {/* Header bar */}
      <div className="flex shrink-0 items-center justify-between border-b border-outline bg-white px-5 py-3 shadow-sm">
        <span className="text-sm font-semibold text-text capitalize">
          {mode === "graph" ? "Knowledge Graph" : "Mindmap"} — full view
        </span>
        <div className="flex items-center gap-3">
          {selectedNode && (
            <span className="text-xs text-muted truncate max-w-[320px]">
              <span className="font-semibold text-primary">{selectedNode.label}</span>
              <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-muted">{selectedNode.id}</span>
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

      {/* Canvas */}
      <div className="flex-1 overflow-hidden">
        <GraphCanvas
          mode={mode}
          canvasNodes={canvas.nodes ?? []}
          canvasEdges={canvas.edges ?? []}
          onSelect={onSelect}
        />
      </div>
    </div>,
    document.body
  );
}

// ─── Main tab ─────────────────────────────────────────────────────────────────

export default function GraphTab({ mode }: { mode: "graph" | "mindmap" }) {
  const { workspace, scopedMaterialIds } = useWorkspace();
  const [selectedNode, setSelectedNode] = useState<{ id: string; label: string } | null>(null);
  const [rootTopic, setRootTopic] = useState("");
  const [graphResult, setGraphResult] = useState<GraphResponse | null>(null);
  const [mindmapResult, setMindmapResult] = useState<MindmapResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  const canvas = mode === "graph" ? toGraph(graphResult) : toMindmap(mindmapResult);

  const handleSelect = useCallback((id: string, label: string) => {
    setSelectedNode({ id, label });
  }, []);

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
        const first = response.nodes[0];
        setSelectedNode(first ? { id: first.id, label: first.label } : null);
      } else {
        const response = await loadMindmap({
          owner_id: workspace.ownerId,
          collection_id: workspace.collectionId || null,
          material_ids: workspace.collectionId ? [] : scopedMaterialIds,
          root_topic: rootTopic || workspace.collectionName || workspace.subject || "Central Topic",
        });
        setMindmapResult(response);
        const first = response.nodes[0];
        setSelectedNode(first ? { id: first.id, label: first.label } : null);
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

  // Auto-load when tab first opens if scope is available
  useEffect(() => {
    if (hasScope) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const hasCanvas = Boolean(canvas.nodes && canvas.nodes.length > 0);

  return (
    <>
      <div className="flex h-full flex-col bg-slate-50">
        {/* Toolbar */}
        <div className="shrink-0 p-4 border-b border-outline bg-white flex flex-col gap-3">
          {mode === "mindmap" && (
            <input
              className="w-full rounded-md border border-outline px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition"
              value={rootTopic}
              onChange={(e) => setRootTopic(e.target.value)}
              placeholder="Mindmap root topic…"
            />
          )}
          <div className="flex gap-2">
            <button
              className="flex flex-1 items-center justify-center gap-2 rounded-md bg-primary py-2 text-sm font-semibold text-white disabled:opacity-50 transition hover:bg-primary/90"
              onClick={refresh}
              disabled={loading || !hasScope}
            >
              {loading ? <Loader2 className="animate-spin" size={14} /> : <RefreshCw size={14} />}
              Generate {mode === "graph" ? "Graph" : "Mindmap"}
            </button>
            {hasCanvas && (
              <button
                title="Open fullscreen"
                aria-label="Mở chế độ toàn màn hình"
                onClick={() => setFullscreen(true)}
                className="flex items-center justify-center gap-1.5 rounded-md border border-outline px-3 py-2 text-xs font-semibold text-muted hover:border-primary/40 hover:text-primary transition"
              >
                <Maximize2 size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Canvas area */}
        <div className="flex-1 overflow-hidden relative">
          {error && (
            <div className="absolute top-4 left-4 right-4 z-10 flex items-start gap-2 rounded-lg border border-red-200 bg-white/90 backdrop-blur p-3 text-xs text-red-700 shadow-sm">
              <AlertCircle size={14} className="shrink-0 mt-0.5" /> {error}
            </div>
          )}

          {!hasCanvas && !loading && !error && (
            <div className="h-full flex items-center justify-center text-xs text-muted p-6 text-center">
              {hasScope
                ? `Click Generate to visualize the ${mode === "graph" ? "knowledge graph" : "mindmap"} for your active sources.`
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
              />
            </div>
          )}
        </div>

        {/* Selected node footer */}
        {selectedNode && (
          <div className="shrink-0 px-4 py-3 border-t border-outline bg-white flex items-center gap-3">
            <div className="min-w-0 flex-1">
              <p className="label-caps">Selected Node</p>
              <p className="mt-0.5 font-heading text-sm font-semibold text-primary truncate" title={selectedNode.label}>
                {selectedNode.label}
              </p>
            </div>
            <span
              className="shrink-0 rounded bg-slate-100 px-2 py-0.5 text-[10px] font-mono text-muted truncate max-w-[120px]"
              title={selectedNode.id}
            >
              {selectedNode.id}
            </span>
          </div>
        )}
      </div>

      {/* Fullscreen overlay — rendered in portal */}
      {fullscreen && hasCanvas && (
        <FullscreenOverlay
          mode={mode}
          canvas={canvas}
          selectedNode={selectedNode}
          onSelect={handleSelect}
          onClose={() => setFullscreen(false)}
        />
      )}
    </>
  );
}
