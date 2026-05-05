import dagre from "@dagrejs/dagre";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  Panel,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";
import CollapsibleMindmapNode from "./CollapsibleMindmapNode";
import MindmapContextMenu from "./MindmapContextMenu";

export type CanvasNode = {
  id: string;
  label: string;
  type: string;
  position?: { x: number; y: number };
  confidence?: number | null;
  degree?: number;
  mention_count?: number;
  source_docs?: string[];
  evidence_refs?: Array<Record<string, string | number>>;
  focused?: boolean;
  branchColor?: string;
  depth?: number;
};

export type CanvasEdge = {
  id?: string;
  source: string;
  target: string;
  label: string;
  focused?: boolean;
  branchColor?: string;
};

type MindmapNodeData = {
  label: string;
  entityType: string;
  confidence: number | null;
  degree: number;
  evidenceRefs: Array<Record<string, string | number>>;
  hasChildren?: boolean;
  collapsed?: boolean;
  onToggle?: (nodeId: string) => void;
  branchColor?: string;
  depth?: number;
};

type ContextMenuState = {
  nodeId: string;
  nodeLabel: string;
  position: { x: number; y: number };
  evidenceRefs: Array<Record<string, string | number>>;
} | null;

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  concept: { bg: "#dbeafe", border: "#3b82f6", text: "#1d4ed8" },
  person: { bg: "#ede9fe", border: "#8b5cf6", text: "#6d28d9" },
  event: { bg: "#d1fae5", border: "#10b981", text: "#065f46" },
  location: { bg: "#fef3c7", border: "#f59e0b", text: "#92400e" },
  date: { bg: "#fce7f3", border: "#ec4899", text: "#9d174d" },
  technology: { bg: "#cffafe", border: "#06b6d4", text: "#155e75" },
  method: { bg: "#f0fdf4", border: "#22c55e", text: "#15803d" },
  organization: { bg: "#fee2e2", border: "#ef4444", text: "#991b1b" },
  org: { bg: "#fee2e2", border: "#ef4444", text: "#991b1b" },
  entity: { bg: "#fef9c3", border: "#ca8a04", text: "#92400e" },
  root: { bg: "#1e3a8a", border: "#1e3a8a", text: "#ffffff" },
  cluster: { bg: "#f1f5f9", border: "#64748b", text: "#1e293b" },
};

function typeColor(type: string) {
  const key = type.toLowerCase().split(/[_\s]/)[0];
  return TYPE_COLORS[key] ?? { bg: "#f8fafc", border: "#94a3b8", text: "#475569" };
}

type CircleNodeData = { label: string; entityType: string; confidence: number | null; degree: number; focused?: boolean };

function CircleNode({ data, selected }: NodeProps<CircleNodeData>) {
  const color = typeColor(data.entityType);
  const deg = Math.min(data.degree || 1, 12);
  const isFocused = Boolean(data.focused);
  const size = 76 + Math.min(deg, 8) * 4;

  const centerHandle: React.CSSProperties = {
    left: "50%",
    top: "50%",
    right: "auto",
    bottom: "auto",
    transform: "translate(-50%, -50%)",
    opacity: 0,
    width: 2,
    height: 2,
    minWidth: 0,
    minHeight: 0,
    pointerEvents: "none",
  };

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: selected ? color.border : color.bg,
        border: `${selected || isFocused ? 3 : 2}px solid ${color.border}`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 4,
        padding: 8,
        boxShadow: selected
          ? `0 0 0 3px ${color.border}24, 0 8px 18px rgba(15, 23, 42, .18)`
          : isFocused
            ? `0 0 0 3px ${color.border}18, 0 5px 14px rgba(15, 23, 42, .12)`
            : "0 2px 8px rgba(15, 23, 42, .08)",
        cursor: "pointer",
        transition: "all .12s",
      }}
    >
      <Handle type="target" position={Position.Left} style={centerHandle} />
      <p
        style={{
          fontSize: 11,
          fontWeight: 800,
          color: selected ? "#fff" : color.text,
          textAlign: "center",
          lineHeight: 1.15,
          wordBreak: "break-word",
          maxWidth: size - 20,
          userSelect: "none",
          zIndex: 1,
          position: "relative",
        }}
      >
        {data.label.length > 24 ? `${data.label.slice(0, 22)}...` : data.label}
      </p>
      <span
        style={{
          alignItems: "center",
          background: selected ? "rgba(255,255,255,.18)" : color.bg,
          borderRadius: 999,
          color: selected ? "#fff" : color.text,
          display: "inline-flex",
          fontSize: 10,
          fontWeight: 800,
          height: 18,
          justifyContent: "center",
          minWidth: 18,
          padding: "0 5px",
        }}
      >
        {deg}
      </span>
      <Handle type="source" position={Position.Right} style={centerHandle} />
    </div>
  );
}

type PillNodeData = { label: string; entityType: string; branchColor?: string; depth?: number };

function PillNode({ data, selected }: NodeProps<PillNodeData>) {
  const isRoot = data.entityType === "root";
  const isCluster = data.entityType === "cluster";
  const color = typeColor(data.entityType);
  const accent = data.branchColor || color.border;

  return (
    <div
      style={{
        alignItems: "center",
        background: isRoot
          ? `linear-gradient(135deg, ${color.border || "#1e3a8a"}, #0f172a)`
          : isCluster
            ? "#f8fafc"
            : selected
              ? accent
              : "#ffffff",
        border: `${isRoot ? 0 : 1}px solid ${
          isRoot ? "transparent" : isCluster ? "#cbd5e1" : selected ? accent : "#dbe5f0"
        }`,
        borderRadius: 999,
        padding: isRoot ? "10px 22px" : isCluster ? "8px 16px" : "6px 10px",
        minWidth: isRoot ? 120 : isCluster ? 90 : 128,
        maxWidth: isRoot ? 240 : 205,
        boxShadow: isRoot
          ? `0 10px 25px -5px ${color.border || "#1e3a8a"}60, 0 8px 10px -6px ${color.border || "#1e3a8a"}40`
          : isCluster
            ? "0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025)"
            : selected
              ? `0 0 0 3px ${accent}24, 0 4px 12px rgba(0,0,0,0.1)`
              : "0 2px 5px rgba(0,0,0,0.04)",
        cursor: "pointer",
        display: "flex",
        gap: 8,
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        transform: selected && !isRoot ? "translateY(-1px)" : "none",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {isRoot && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "linear-gradient(to bottom, rgba(255,255,255,0.15), transparent)",
            pointerEvents: "none",
          }}
        />
      )}
      <Handle type="target" position={Position.Left} style={{ opacity: 0, width: 0, height: 0 }} />
      {!isRoot && !isCluster && (
        <span
          style={{
            background: accent,
            borderRadius: 999,
            boxShadow: `0 0 0 3px ${accent}14`,
            flexShrink: 0,
            height: 8,
            width: 8,
          }}
        />
      )}
      <p
        style={{
          flex: 1,
          fontSize: isRoot ? 14 : isCluster ? 12 : 10.5,
          fontWeight: isRoot ? 700 : isCluster ? 600 : 500,
          color: isRoot ? "#ffffff" : isCluster ? "#334155" : selected ? "#ffffff" : "#334155",
          textAlign: isRoot ? "center" : "left",
          wordBreak: "break-word",
          lineHeight: 1.4,
          userSelect: "none",
          position: "relative",
          zIndex: 1,
          letterSpacing: isRoot ? "0.02em" : "0",
        }}
      >
        {data.label}
      </p>
      <Handle type="source" position={Position.Right} style={{ opacity: 0, width: 0, height: 0 }} />
    </div>
  );
}

const NODE_TYPES = {
  circle: CircleNode,
  pill: PillNode,
  collapsible: CollapsibleMindmapNode,
};

function computeForceLayout(
  nodes: Node[],
  edges: Edge[],
  width = 1200,
  height = 900
): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>();
  const n = nodes.length;
  if (n === 0) {
    return pos;
  }

  nodes.forEach((node, i) => {
    const angle = (i / n) * 2 * Math.PI;
    const r = Math.min(width, height) * 0.31;
    const jitter = seededJitter(node.id);
    pos.set(node.id, {
      x: width / 2 + r * Math.cos(angle) + jitter.x,
      y: height / 2 + r * Math.sin(angle) + jitter.y,
    });
  });

  const REPEL = 36000;
  const IDEAL = 170;
  const SPRING = 0.035;
  const GRAVITY = 0.018;
  const ITERS = 160;

  for (let iter = 0; iter < ITERS; iter++) {
    const cool = Math.pow(0.978, iter);
    const forces = new Map(nodes.map((node) => [node.id, { x: 0, y: 0 }]));

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = pos.get(nodes[i].id)!;
        const b = pos.get(nodes[j].id)!;
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d2 = dx * dx + dy * dy || 0.01;
        const d = Math.sqrt(d2);
        const f = REPEL / d2;
        const fa = forces.get(nodes[i].id)!;
        const fb = forces.get(nodes[j].id)!;
        fa.x += (f * dx) / d;
        fa.y += (f * dy) / d;
        fb.x -= (f * dx) / d;
        fb.y -= (f * dy) / d;
      }
    }

    for (const edge of edges) {
      const a = pos.get(edge.source);
      const b = pos.get(edge.target);
      if (!a || !b) {
        continue;
      }
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - IDEAL) * SPRING;
      const fa = forces.get(edge.source);
      const fb = forces.get(edge.target);
      if (fa) {
        fa.x += (f * dx) / d;
        fa.y += (f * dy) / d;
      }
      if (fb) {
        fb.x -= (f * dx) / d;
        fb.y -= (f * dy) / d;
      }
    }

    for (const node of nodes) {
      const p = pos.get(node.id)!;
      const f = forces.get(node.id)!;
      f.x += (width / 2 - p.x) * GRAVITY;
      f.y += (height / 2 - p.y) * GRAVITY;
      p.x = Math.max(90, Math.min(width - 180, p.x + f.x * cool));
      p.y = Math.max(70, Math.min(height - 90, p.y + f.y * cool));
    }
  }

  return pos;
}

function seededJitter(id: string) {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = (hash * 31 + id.charCodeAt(i)) >>> 0;
  }
  return {
    x: ((hash % 41) - 20) * 0.8,
    y: (((hash >> 8) % 41) - 20) * 0.8,
  };
}

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", ranksep: 92, nodesep: 46, edgesep: 18, marginx: 44, marginy: 34 });
  nodes.forEach((node) => {
    const labelLen = (node.data as { label?: string } | undefined)?.label?.length || 10;
    const entityType = (node.data as { entityType?: string } | undefined)?.entityType;
    const isRoot = entityType === "root";
    const isTopic = entityType === "topic";
    const isBranch = entityType === "branch";
    const isCluster = entityType === "cluster";
    const width = isRoot
      ? Math.min(360, labelLen * 8 + 110)
      : isTopic
        ? Math.min(310, labelLen * 7 + 92)
        : isBranch || isCluster
          ? Math.min(270, labelLen * 6 + 78)
          : Math.min(230, labelLen * 6 + 54);
    g.setNode(node.id, { width, height: isRoot ? 62 : isTopic ? 52 : isBranch || isCluster ? 42 : 34 });
  });
  edges.forEach((edge) => {
    try {
      g.setEdge(edge.source, edge.target);
    } catch {
      // Ignore layout edge errors for malformed or hidden edges.
    }
  });
  dagre.layout(g);
  return nodes.map((node) => {
    const p = g.node(node.id);
    return p ? { ...node, position: { x: p.x - p.width / 2, y: p.y - p.height / 2 } } : node;
  });
}

function applyMindmapTreeLayout(nodes: Node[], edges: Edge[]): Node[] {
  const children = new Map<string, string[]>();
  edges.forEach((edge) => {
    children.set(edge.source, [...(children.get(edge.source) ?? []), edge.target]);
  });

  const root = nodes.find((node) => (node.data as MindmapNodeData).entityType === "root") ?? nodes[0];
  if (!root) return nodes;

  const leafHeight = 46;
  const branchGap = 18;
  const levelX = [40, 340, 610, 850, 1080];
  const subtreeHeight = new Map<string, number>();

  const measure = (nodeId: string): number => {
    const childIds = children.get(nodeId) ?? [];
    if (!childIds.length) {
      subtreeHeight.set(nodeId, leafHeight);
      return leafHeight;
    }
    const total = childIds.reduce((sum, childId) => sum + measure(childId), 0) + Math.max(0, childIds.length - 1) * branchGap;
    const height = Math.max(60, total);
    subtreeHeight.set(nodeId, height);
    return height;
  };

  const totalHeight = measure(root.id);
  const positions = new Map<string, { x: number; y: number }>();
  const rootY = Math.max(80, totalHeight / 2);
  positions.set(root.id, { x: levelX[0], y: rootY });

  const placeChildren = (nodeId: string, depth: number, top: number) => {
    const childIds = children.get(nodeId) ?? [];
    let cursor = top;
    for (const childId of childIds) {
      const height = subtreeHeight.get(childId) ?? leafHeight;
      const childY = cursor + height / 2;
      positions.set(childId, { x: levelX[Math.min(depth, levelX.length - 1)], y: childY });
      placeChildren(childId, depth + 1, cursor);
      cursor += height + branchGap;
    }
  };

  placeChildren(root.id, 1, Math.max(20, rootY - totalHeight / 2));

  return nodes.map((node) => {
    const position = positions.get(node.id);
    return position ? { ...node, position } : node;
  });
}

export type GraphCanvasProps = {
  onSelect: (nodeId: string, label: string) => void;
  mode: "graph" | "mindmap";
  canvasNodes?: CanvasNode[];
  canvasEdges?: CanvasEdge[];
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
  onDraftQuestion?: (draft: string) => void;
  onFindRelated?: (draft: string) => void;
};

function FlowInner({
  onSelect,
  mode,
  canvasNodes = [],
  canvasEdges = [],
  onOpenEvidence,
  onDraftQuestion,
  onFindRelated,
}: GraphCanvasProps) {
  const [collapsedNodes, setCollapsedNodes] = useState<Set<string>>(() => new Set());
  const [prunedNodes, setPrunedNodes] = useState<Set<string>>(() => new Set());
  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const lastMindmapSignature = useRef<string>("");

  const handleToggleNode = useCallback((nodeId: string) => {
    setCollapsedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const childrenMap = useMemo(() => {
    if (mode !== "mindmap") {
      return new Map<string, string[]>();
    }
    const map = new Map<string, string[]>();
    canvasEdges.forEach((edge) => {
      const children = map.get(edge.source) ?? [];
      children.push(edge.target);
      map.set(edge.source, children);
    });
    return map;
  }, [canvasEdges, mode]);

  const collapseDetailNodes = useCallback(() => {
    if (mode !== "mindmap") {
      return;
    }
    setCollapsedNodes(
      new Set(
        canvasNodes
          .filter((node) => (node.depth ?? 0) >= 2 && (childrenMap.get(node.id)?.length ?? 0) > 0)
          .map((node) => node.id)
      )
    );
  }, [canvasNodes, childrenMap, mode]);

  useEffect(() => {
    if (mode !== "mindmap") {
      return;
    }
    const signature = canvasNodes.map((node) => `${node.id}:${node.depth ?? 0}`).join("|");
    if (signature === lastMindmapSignature.current) {
      return;
    }
    lastMindmapSignature.current = signature;
    setPrunedNodes(new Set());
    setSelectedNodeId(null);
    setCollapsedNodes(
      new Set(
        canvasNodes
          .filter((node) => (node.depth ?? 0) >= 2 && (childrenMap.get(node.id)?.length ?? 0) > 0)
          .map((node) => node.id)
      )
    );
  }, [canvasNodes, childrenMap, mode]);

  const hiddenNodes = useMemo(() => {
    if (mode !== "mindmap") {
      return new Set<string>();
    }
    const hidden = new Set<string>();

    const hideDescendants = (nodeId: string) => {
      const stack = [...(childrenMap.get(nodeId) ?? [])];
      while (stack.length) {
        const current = stack.pop();
        if (!current || hidden.has(current)) {
          continue;
        }
        hidden.add(current);
        const children = childrenMap.get(current);
        if (children?.length) {
          stack.push(...children);
        }
      }
    };

    prunedNodes.forEach((nodeId) => {
      hidden.add(nodeId);
      hideDescendants(nodeId);
    });

    collapsedNodes.forEach((nodeId) => {
      hideDescendants(nodeId);
    });

    return hidden;
  }, [childrenMap, collapsedNodes, mode, prunedNodes]);

  const rfNodes = useMemo<Node[]>(
    () =>
      canvasNodes
        .filter((node) => !hiddenNodes.has(node.id))
        .map((node) => {
          const hasChildren = mode === "mindmap" ? (childrenMap.get(node.id)?.length ?? 0) > 0 : false;
          const evidenceRefs = node.evidence_refs ?? [];
          const data =
            mode === "graph"
              ? {
                  label: node.label,
                  entityType: node.type || "concept",
                  confidence: node.confidence ?? null,
                  degree: node.degree ?? 1,
                  evidenceRefs,
                  focused: node.focused,
                }
              : {
                  label: node.label,
                  entityType: node.type || "concept",
                  confidence: node.confidence ?? null,
                  degree: node.degree ?? (hasChildren ? (childrenMap.get(node.id)?.length ?? 0) : 1),
                  evidenceRefs,
                  hasChildren,
                  collapsed: collapsedNodes.has(node.id),
                  onToggle: handleToggleNode,
                  branchColor: node.branchColor,
                  depth: node.depth ?? 0,
                };

          return {
            id: node.id,
            type: mode === "graph" ? "circle" : hasChildren ? "collapsible" : "pill",
            position: node.position ?? { x: 0, y: 0 },
            data,
          };
        }),
    [canvasNodes, childrenMap, collapsedNodes, handleToggleNode, hiddenNodes, mode]
  );

  const rfEdges = useMemo<Edge[]>(
    () =>
      canvasEdges
        .filter((edge) => !hiddenNodes.has(edge.source) && !hiddenNodes.has(edge.target))
        .map((edge, index) => {
          const selectedRelated =
            selectedNodeId !== null && (edge.source === selectedNodeId || edge.target === selectedNodeId);
          const dimmed = mode === "graph" && selectedNodeId !== null && !selectedRelated;
          const semanticLabel =
            mode === "graph" &&
            edge.label &&
            !edge.label.startsWith("co_occurs") &&
            edge.label.length < 24
              ? edge.label.replace(/_/g, " ")
              : undefined;
          const branchColor = edge.branchColor || "#c4cfdd";

          return {
            id: edge.id ?? `${edge.source}__${edge.target}__${index}`,
            source: edge.source,
            target: edge.target,
            type: mode === "mindmap" ? "smoothstep" : "bezier",
            label: selectedRelated ? semanticLabel : undefined,
            animated: false,
            style: {
              opacity: dimmed ? 0.12 : selectedRelated ? 0.95 : mode === "graph" ? 0.38 : 0.75,
              stroke: selectedRelated || edge.focused ? "#0f766e" : mode === "graph" ? "#64748b" : branchColor,
              strokeWidth: selectedRelated ? 2.8 : edge.focused ? 2 : mode === "graph" ? 1.05 : 1.75,
            },
            labelStyle: { fill: "#0f766e", fontSize: 9, fontWeight: 700 },
            labelBgStyle: { fill: "#ffffff", fillOpacity: 0.94 },
            labelBgPadding: [3, 4] as [number, number],
          };
        }),
    [canvasEdges, hiddenNodes, mode, selectedNodeId]
  );

  const layoutedNodes = useMemo(() => {
    if (mode === "mindmap") {
      return applyMindmapTreeLayout(rfNodes, rfEdges);
    }
    const positions = computeForceLayout(rfNodes, rfEdges);
    return rfNodes.map((node) => {
      const position = positions.get(node.id);
      return position ? { ...node, position } : node;
    });
  }, [mode, rfEdges, rfNodes]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rfEdges);

  useEffect(() => {
    setNodes(layoutedNodes);
  }, [layoutedNodes, setNodes]);

  useEffect(() => {
    setEdges(rfEdges);
  }, [rfEdges, setEdges]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => {
      setSelectedNodeId(node.id);
      onSelect(node.id, node.data.label as string);
    },
    [onSelect]
  );

  const openEvidenceFromRefs = useCallback(
    (refs: Array<Record<string, string | number>>) => {
      if (!onOpenEvidence) {
        return false;
      }
      const ref = refs.find((item) => {
        const docId = item.doc_id ?? item.material_id;
        return typeof docId === "string" && docId.length > 0;
      });
      if (!ref) {
        return false;
      }
      const docId = String(ref.doc_id ?? ref.material_id ?? "");
      const pageRaw = ref.page;
      const page = typeof pageRaw === "number" ? pageRaw : Number(pageRaw);
      if (!docId || !Number.isFinite(page) || page <= 0) {
        return false;
      }
      onOpenEvidence({ docId, page, blockId: typeof ref.block_id === "string" ? ref.block_id : null });
      return true;
    },
    [onOpenEvidence]
  );

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      if (mode !== "mindmap") {
        return;
      }
      event.preventDefault();
      setContextMenu({
        nodeId: node.id,
        nodeLabel: node.data.label as string,
        position: { x: event.clientX, y: event.clientY },
        evidenceRefs: ((node.data as MindmapNodeData).evidenceRefs ?? []),
      });
    },
    [mode]
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      if (mode !== "mindmap" || nodeId === "root-topic") {
        return;
      }
      setPrunedNodes((prev) => {
        const next = new Set(prev);
        const stack = [nodeId];
        while (stack.length) {
          const current = stack.pop();
          if (!current || next.has(current)) {
            continue;
          }
          next.add(current);
          const children = childrenMap.get(current);
          if (children?.length) {
            stack.push(...children);
          }
        }
        return next;
      });
      setCollapsedNodes((prev) => {
        const next = new Set(prev);
        next.delete(nodeId);
        return next;
      });
      setContextMenu(null);
    },
    [childrenMap, mode]
  );

  useEffect(() => {
    if (mode !== "mindmap" || contextMenu === null) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setContextMenu(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [contextMenu, mode]);

  return (
    <ReactFlow
      className={mode === "mindmap" ? "mindmap-flow" : "graph-flow"}
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      onNodeContextMenu={handleNodeContextMenu}
      onPaneClick={() => setSelectedNodeId(null)}
      fitView
      fitViewOptions={{ padding: mode === "mindmap" ? 0.24 : 0.14 }}
      minZoom={0.12}
      maxZoom={2.5}
      attributionPosition="bottom-left"
    >
      {mode === "graph" && (
        <MiniMap
          nodeColor={(n) => typeColor((n.data as CircleNodeData)?.entityType ?? "").border}
          nodeStrokeWidth={0}
          pannable
          zoomable
          style={{ background: "#f8fafc" }}
        />
      )}
      <Controls showInteractive={false} />
      <Background gap={mode === "mindmap" ? 32 : 22} color={mode === "mindmap" ? "#f1f5f9" : "#e2e8f0"} />
      {mode === "mindmap" && (
        <Panel position="top-left" className="mindmap-panel">
          <div className="flex items-center gap-2 rounded-lg border border-outline bg-white/95 px-2 py-1.5 shadow-sm backdrop-blur">
            <button
              className="rounded border border-outline px-2.5 py-1 text-[11px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
              onClick={() => setCollapsedNodes(new Set())}
              type="button"
            >
              Mở tất cả
            </button>
            <button
              className="rounded border border-outline px-2.5 py-1 text-[11px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
              onClick={collapseDetailNodes}
              type="button"
            >
              Thu gọn chi tiết
            </button>
          </div>
        </Panel>
      )}
      {contextMenu && (
        <MindmapContextMenu
          nodeId={contextMenu.nodeId}
          nodeLabel={contextMenu.nodeLabel}
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onAskAI={(label) => {
            onDraftQuestion?.(`Hãy giải thích về ${label} dựa trên tài liệu hiện có.`);
            onDraftQuestion?.(`Hãy giải thích về ${label} dựa trên tài liệu hiện có.`);
          }}
          onViewSources={() => {
            if (!openEvidenceFromRefs(contextMenu.evidenceRefs)) {
              onSelect(contextMenu.nodeId, contextMenu.nodeLabel);
            }
          }}
          onFindRelated={(nodeId) => {
            onFindRelated?.(`Các khái niệm liên quan đến ${contextMenu.nodeLabel} là gì?`);
            if (!onFindRelated) {
              onSelect(nodeId, contextMenu.nodeLabel);
            }
            onFindRelated?.(`Các khái niệm liên quan đến ${contextMenu.nodeLabel} là gì?`);
            if (!onFindRelated) {
              onSelect(nodeId, contextMenu.nodeLabel);
            }
          }}
          onDelete={contextMenu.nodeId !== "root-topic" ? handleDeleteNode : undefined}
        />
      )}
    </ReactFlow>
  );
}

export default function GraphCanvas(props: GraphCanvasProps) {
  if (!(props.canvasNodes ?? []).length) {
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
