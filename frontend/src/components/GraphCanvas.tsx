import dagre from "@dagrejs/dagre";
import { useCallback, useEffect, useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";

export type CanvasNode = {
  id: string;
  label: string;
  type: string;
  position?: { x: number; y: number };
  confidence?: number | null;
};

export type CanvasEdge = {
  id?: string;
  source: string;
  target: string;
  label: string;
};

// ─── Entity-type color palette ────────────────────────────────────────────────

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  concept:    { bg: "#dbeafe", border: "#3b82f6", text: "#1d4ed8" },
  person:     { bg: "#ede9fe", border: "#8b5cf6", text: "#6d28d9" },
  event:      { bg: "#d1fae5", border: "#10b981", text: "#065f46" },
  location:   { bg: "#fef3c7", border: "#f59e0b", text: "#92400e" },
  date:       { bg: "#fce7f3", border: "#ec4899", text: "#9d174d" },
  technology: { bg: "#cffafe", border: "#06b6d4", text: "#155e75" },
  method:     { bg: "#f0fdf4", border: "#22c55e", text: "#15803d" },
  root:       { bg: "#dce9ff", border: "#006591", text: "#003a5c" },
};

function typeColor(type: string) {
  const key = type.toLowerCase().split(/[_\s]/)[0];
  return TYPE_COLORS[key] ?? { bg: "#f8fafc", border: "#bec8d2", text: "#334155" };
}

// ─── Custom node ──────────────────────────────────────────────────────────────

type NodeData = { label: string; entityType: string; confidence: number | null };

function EntityNode({ data, selected }: NodeProps<NodeData>) {
  const color = typeColor(data.entityType);
  const pct = data.confidence != null ? Math.round(data.confidence * 100) : null;

  return (
    <div
      style={{
        background: selected ? color.border : color.bg,
        border: `2px solid ${color.border}`,
        borderRadius: 10,
        minWidth: 130,
        maxWidth: 180,
        padding: "6px 10px",
        boxShadow: selected ? `0 0 0 3px ${color.border}44` : "0 1px 4px rgba(0,0,0,.08)",
        transition: "all .15s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: color.border, width: 8, height: 8 }} />
      <p
        style={{
          fontSize: 11,
          fontWeight: 700,
          color: selected ? "#fff" : color.text,
          wordBreak: "break-word",
          lineHeight: 1.3,
          marginBottom: 3,
        }}
      >
        {data.label}
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span
          style={{
            fontSize: 9,
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: ".04em",
            color: selected ? "#ffffffaa" : color.border,
            background: selected ? "#ffffff22" : `${color.border}18`,
            borderRadius: 4,
            padding: "1px 5px",
          }}
        >
          {data.entityType}
        </span>
        {pct != null && (
          <span style={{ fontSize: 9, color: selected ? "#ffffffaa" : "#94a3b8", marginLeft: "auto" }}>
            {pct}%
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: color.border, width: 8, height: 8 }} />
    </div>
  );
}

const NODE_TYPES = { entity: EntityNode };

// ─── Dagre auto-layout ────────────────────────────────────────────────────────

const NODE_W = 170;
const NODE_H = 58;

function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
  direction: "LR" | "TB" = "LR"
): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, ranksep: 90, nodesep: 50, edgesep: 20 });

  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => { try { g.setEdge(e.source, e.target); } catch { /* skip invalid */ } });

  dagre.layout(g);

  return nodes.map((n) => {
    const pos = g.node(n.id);
    return pos ? { ...n, position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 } } : n;
  });
}

// ─── Main canvas component ────────────────────────────────────────────────────

type GraphCanvasProps = {
  onSelect: (nodeId: string, label: string) => void;
  mode: "graph" | "mindmap";
  canvasNodes?: CanvasNode[];
  canvasEdges?: CanvasEdge[];
};

function FlowInner({ onSelect, mode, canvasNodes = [], canvasEdges = [] }: GraphCanvasProps) {
  const direction = mode === "mindmap" ? "TB" : "LR";

  const rfNodes = useMemo<Node[]>(() =>
    canvasNodes.map((n) => ({
      id: n.id,
      type: "entity",
      position: n.position ?? { x: 0, y: 0 },
      data: { label: n.label, entityType: n.type || "concept", confidence: n.confidence ?? null },
    })),
    [canvasNodes]
  );

  const rfEdges = useMemo<Edge[]>(() =>
    canvasEdges.map((e, i) => ({
      id: e.id ?? `${e.source}__${e.target}__${i}`,
      source: e.source,
      target: e.target,
      label: e.label.length > 22 ? e.label.slice(0, 20) + "…" : e.label,
      animated: false,
      style: { stroke: "#94a3b8", strokeWidth: 1.5 },
      labelStyle: { fill: "#64748b", fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: "#f8fafc", fillOpacity: 0.9 },
      labelBgPadding: [3, 4] as [number, number],
    })),
    [canvasEdges]
  );

  const layoutedNodes = useMemo(
    () => applyDagreLayout(rfNodes, rfEdges, direction),
    [rfNodes, rfEdges, direction]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rfEdges);

  useEffect(() => { setNodes(layoutedNodes); }, [layoutedNodes, setNodes]);
  useEffect(() => { setEdges(rfEdges); }, [rfEdges, setEdges]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => onSelect(node.id, (node.data as NodeData).label),
    [onSelect]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.2}
      maxZoom={2}
      attributionPosition="bottom-left"
    >
      <MiniMap
        nodeColor={(n) => typeColor((n.data as NodeData)?.entityType ?? "").bg}
        pannable
        zoomable
        style={{ background: "#f8fafc" }}
      />
      <Controls showInteractive={false} />
      <Background gap={20} color="#e2e8f0" />
    </ReactFlow>
  );
}

export default function GraphCanvas(props: GraphCanvasProps) {
  const { canvasNodes = [] } = props;

  if (!canvasNodes.length) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted">
        Generate the graph to visualize knowledge nodes from your sources.
      </div>
    );
  }

  return (
    <div className="h-full">
      <ReactFlowProvider>
        <FlowInner {...props} />
      </ReactFlowProvider>
    </div>
  );
}
