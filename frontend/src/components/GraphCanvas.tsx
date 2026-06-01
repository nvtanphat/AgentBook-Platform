import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  getRectOfNodes,
  getTransformForBounds,
  Handle,
  MarkerType,
  MiniMap,
  Node,
  NodeProps,
  Panel,
  Position,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";
import { toPng } from "html-to-image";
import { Download, Layers } from "lucide-react";
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
  importance?: number;  // Phase 2 — PageRank score [0, 1]
  community?: number;   // Phase 2 — Louvain community id
  is_hub?: boolean;     // Phase 2 — top 10% by importance
  is_focused?: boolean; // Focus mode — primary entity from citations
  source_docs?: string[];
  evidence_refs?: Array<Record<string, string | number>>;
  focused?: boolean;
  branchColor?: string;
  depth?: number;
  // NotebookLM-style hover preview — short summary + source attribution.
  summary?: string | null;
  source_label?: string | null;
};

export type CanvasEdge = {
  id?: string;
  source: string;
  target: string;
  label: string;
  confidence?: number | null;
  evidence_count?: number;
  evidence_refs?: Array<Record<string, string | number>>;
  evidence_text_chunk?: string | null;
  source_label?: string | null;
  target_label?: string | null;
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

// Solid saturated colors — Tableau / D3-style palette. Each entity type gets a
// distinct hue that pops on both light and dark backgrounds. Bright text on solid
// fill is more readable than dark text on pastel.
const TYPE_COLORS: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
  // Core entity types (from new LLM extractor)
  model:        { bg: "#0284c7", border: "#0369a1", text: "#ffffff", gradient: "linear-gradient(135deg, #38bdf8, #0284c7)" },
  algorithm:    { bg: "#16a34a", border: "#15803d", text: "#ffffff", gradient: "linear-gradient(135deg, #4ade80, #16a34a)" },
  metric:       { bg: "#ca8a04", border: "#a16207", text: "#ffffff", gradient: "linear-gradient(135deg, #fbbf24, #ca8a04)" },
  dataset:      { bg: "#a855f7", border: "#7e22ce", text: "#ffffff", gradient: "linear-gradient(135deg, #c084fc, #a855f7)" },
  framework:    { bg: "#06b6d4", border: "#0891b2", text: "#ffffff", gradient: "linear-gradient(135deg, #22d3ee, #06b6d4)" },
  author:       { bg: "#8b5cf6", border: "#6d28d9", text: "#ffffff", gradient: "linear-gradient(135deg, #a78bfa, #8b5cf6)" },
  field:        { bg: "#f43f5e", border: "#be123c", text: "#ffffff", gradient: "linear-gradient(135deg, #fb7185, #f43f5e)" },
  // Legacy / general types
  concept:      { bg: "#3b82f6", border: "#1d4ed8", text: "#ffffff", gradient: "linear-gradient(135deg, #60a5fa, #3b82f6)" },
  person:       { bg: "#8b5cf6", border: "#6d28d9", text: "#ffffff", gradient: "linear-gradient(135deg, #a78bfa, #8b5cf6)" },
  event:        { bg: "#10b981", border: "#065f46", text: "#ffffff", gradient: "linear-gradient(135deg, #34d399, #10b981)" },
  location:     { bg: "#f59e0b", border: "#b45309", text: "#ffffff", gradient: "linear-gradient(135deg, #fbbf24, #f59e0b)" },
  date:         { bg: "#ec4899", border: "#9d174d", text: "#ffffff", gradient: "linear-gradient(135deg, #f472b6, #ec4899)" },
  technology:   { bg: "#06b6d4", border: "#0891b2", text: "#ffffff", gradient: "linear-gradient(135deg, #22d3ee, #06b6d4)" },
  method:       { bg: "#16a34a", border: "#15803d", text: "#ffffff", gradient: "linear-gradient(135deg, #4ade80, #16a34a)" },
  organization: { bg: "#ef4444", border: "#991b1b", text: "#ffffff", gradient: "linear-gradient(135deg, #f87171, #ef4444)" },
  org:          { bg: "#ef4444", border: "#991b1b", text: "#ffffff", gradient: "linear-gradient(135deg, #f87171, #ef4444)" },
  entity:       { bg: "#64748b", border: "#475569", text: "#ffffff", gradient: "linear-gradient(135deg, #94a3b8, #64748b)" },
  root:         { bg: "#1e3a8a", border: "#1e3a8a", text: "#ffffff", gradient: "linear-gradient(135deg, #3b82f6, #1e3a8a)" },
  cluster:      { bg: "#f1f5f9", border: "#64748b", text: "#1e293b", gradient: "linear-gradient(135deg, #f8fafc, #e2e8f0)" },
};

function typeColor(type: string) {
  const key = type.toLowerCase().split(/[_\s]/)[0];
  return TYPE_COLORS[key] ?? {
    bg: "#64748b",
    border: "#475569",
    text: "#ffffff",
    gradient: "linear-gradient(135deg, #94a3b8, #64748b)",
  };
}

// ── Verification status (verify mode only) ──────────────────────────────────
// Derived from evidence count + confidence + whether the node was cited
// (focused). Thresholds live here so they are easy to tweak.
const VERIFY_STRONG_CONF = 0.7;   // ≥ → verified (green)
const VERIFY_PARTIAL_CONF = 0.4;  // ≥ → partial (amber); below w/ evidence → weak (red)
const VERIFY_ARC_MAX_NODES = 50;  // above this, skip per-node confidence arcs (perf)

type VerifyStatus = "verified" | "partial" | "weak" | "unverified";

const VERIFY_COLORS: Record<VerifyStatus, { ring: string; badge: string; icon: string; label: string }> = {
  verified:   { ring: "#22c55e", badge: "#16a34a", icon: "✓", label: "Đã xác minh" },
  partial:    { ring: "#f59e0b", badge: "#d97706", icon: "≈", label: "Xác minh một phần" },
  weak:       { ring: "#ef4444", badge: "#dc2626", icon: "!", label: "Bằng chứng yếu" },
  unverified: { ring: "#94a3b8", badge: "#64748b", icon: "?", label: "Chưa có bằng chứng" },
};


function confidenceTierColor(confidence: number | null | undefined): string {
  const c = confidence ?? 0;
  if (c >= VERIFY_STRONG_CONF) return "#22c55e";
  if (c >= VERIFY_PARTIAL_CONF) return "#f59e0b";
  return "#ef4444";
}

type CircleNodeData = {
  label: string;
  entityType: string;
  confidence: number | null;
  degree: number;
  importance?: number;
  community?: number;
  isHub?: boolean;
  focused?: boolean;
  dimmed?: boolean;
  searchMatch?: boolean;
  // G4 — GraphRAG answer trace highlight
  answerHighlight?: boolean;
  evidenceRefs?: Array<Record<string, string | number>>;
  // Verify mode — set only when graphFocusOnAnswer is on
  verifyMode?: boolean;
  verifyStatus?: VerifyStatus;
  evidenceCount?: number;
  totalNodes?: number;  // for arc perf fallback
  sourceLabel?: string | null;  // doc · page, for hover tooltip
};

// Distinct community colors (Tableau 20 palette subset)
const COMMUNITY_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#a855f7",
  "#84cc16", "#eab308", "#dc2626", "#0891b2", "#7c3aed",
];

function communityColor(id: number | undefined): string {
  if (id === undefined || id === null) return "#94a3b8";
  return COMMUNITY_COLORS[id % COMMUNITY_COLORS.length];
}

function CircleNode({ data, selected }: NodeProps<CircleNodeData>) {
  const color = typeColor(data.entityType);
  const deg = Math.min(data.degree || 1, 12);
  const isFocused = Boolean(data.focused);
  const isSearchMatch = Boolean(data.searchMatch);
  const isDimmed = Boolean(data.dimmed);
  const isHub = Boolean(data.isHub);
  const isAnswerHighlight = Boolean(data.answerHighlight);
  const isPrimaryFocus = isFocused;  // semantic alias for clarity
  // Hub nodes are 30% larger; primary focused entities (from citations) are 40% larger
  const importance = data.importance ?? 0;
  const sizeBoost = isPrimaryFocus ? 40 : isHub ? 30 : Math.round(importance * 15);
  // Verify mode: focused (cited) nodes get a noticeably bigger boost so they
  // stand apart from search highlight.
  const isVerifyMode = Boolean(data.verifyMode);
  const vStatus = data.verifyStatus;
  const vColor = vStatus ? VERIFY_COLORS[vStatus] : null;
  const evidenceCount = data.evidenceCount ?? 0;
  const verifyBoost = isVerifyMode && isPrimaryFocus ? 18 : 0;
  const size = 72 + Math.min(deg, 10) * 4 + sizeBoost + verifyBoost;
  const commColor = communityColor(data.community);

  // Confidence arc geometry (verify mode, small graphs only). Stroke-dashoffset
  // fills the ring proportional to confidence; colour by tier.
  const showArc = isVerifyMode && (data.totalNodes ?? 0) <= VERIFY_ARC_MAX_NODES && data.confidence != null;
  const arcPct = Math.max(0, Math.min(1, data.confidence ?? 0));
  const arcR = size / 2 + 6;
  const arcC = 2 * Math.PI * arcR;

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

  // Truncate label nicely — try to break at word boundary
  const label = data.label.length > 22 ? `${data.label.slice(0, 20).trimEnd()}…` : data.label;

  // Community color provides a soft outer ring even when not selected
  // Primary focused entity (from citations) gets a distinct gold ring
  // GraphRAG answer-trace nodes get a magenta ring + glow, outranking other highlights.
  // In verify mode the status ring colour takes priority so the user can read
  // verification state at a glance. Focused (cited) nodes keep a thick ring.
  const borderColor = isVerifyMode && vColor
    ? vColor.ring
    : isAnswerHighlight
      ? "#d946ef"
      : isPrimaryFocus
        ? "#fbbf24"
        : isSearchMatch
          ? "#fbbf24"
          : selected
            ? "#ffffff"
            : commColor;
  const borderWidth = isVerifyMode
    ? (isPrimaryFocus ? 5 : 3)
    : isAnswerHighlight ? 5 : isPrimaryFocus ? 4 : selected || isFocused || isSearchMatch ? 3 : isHub ? 3 : 2;
  const verifyGlow = isVerifyMode && vColor
    ? (isPrimaryFocus
        ? `0 0 0 6px ${vColor.ring}33, 0 0 26px ${vColor.ring}88, 0 10px 26px ${vColor.ring}55`
        : `0 0 0 3px ${vColor.ring}22, 0 0 14px ${vColor.ring}55`)
    : null;
  return (
    <div
      className={isSearchMatch ? "node-search-pulse" : undefined}
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background: color.gradient,
        border: `${borderWidth}px solid ${borderColor}`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 3,
        padding: 8,
        opacity: isDimmed ? 0.18 : 1,
        boxShadow: verifyGlow
          ? verifyGlow
          : isAnswerHighlight
          ? `0 0 0 7px rgba(217,70,239,0.30), 0 0 22px rgba(217,70,239,0.55), 0 12px 30px rgba(192,38,211,0.40)`
          : isSearchMatch
            ? `0 0 0 6px rgba(251,191,36,0.25), 0 10px 28px rgba(245,158,11,0.35)`
            : selected
              ? `0 0 0 4px ${color.border}55, 0 12px 28px rgba(15, 23, 42, .35)`
              : isFocused
                ? `0 0 0 5px ${color.border}40, 0 8px 20px rgba(15, 23, 42, .25)`
                : isHub
                  ? `0 0 0 4px ${commColor}30, 0 6px 20px ${commColor}40`
                  : `0 4px 14px ${color.border}40, inset 0 1px 0 rgba(255,255,255,0.2)`,
        cursor: "pointer",
        transition: "all .2s cubic-bezier(0.4, 0, 0.2, 1)",
        transform: selected ? "scale(1.05)" : "scale(1)",
        position: "relative",
      }}
    >
      {/* Confidence arc (verify mode, small graphs) — thin ring just outside node */}
      {showArc && (
        <svg
          width={arcR * 2}
          height={arcR * 2}
          viewBox={`0 0 ${arcR * 2} ${arcR * 2}`}
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%) rotate(-90deg)",
            pointerEvents: "none",
            overflow: "visible",
            zIndex: 0,
          }}
        >
          <circle cx={arcR} cy={arcR} r={arcR} fill="none" stroke="rgba(148,163,184,0.25)" strokeWidth={3} />
          <circle
            cx={arcR}
            cy={arcR}
            r={arcR}
            fill="none"
            stroke={confidenceTierColor(data.confidence)}
            strokeWidth={3}
            strokeLinecap="round"
            strokeDasharray={arcC}
            strokeDashoffset={arcC * (1 - arcPct)}
          />
        </svg>
      )}
      {/* Verify-status badge (top-left) — replaces nothing, glanceable state */}
      {isVerifyMode && vColor && (
        <span
          title={vColor.label}
          style={{
            position: "absolute",
            top: -6,
            left: -4,
            background: vColor.badge,
            color: "#ffffff",
            fontSize: 11,
            fontWeight: 900,
            width: 18,
            height: 18,
            lineHeight: "18px",
            textAlign: "center",
            borderRadius: 999,
            border: "2px solid #ffffff",
            boxShadow: "0 2px 4px rgba(0,0,0,0.3)",
            zIndex: 3,
          }}
        >
          {vColor.icon}
        </span>
      )}
      {/* Evidence count badge (top-right) — in verify mode prefer this over hub star */}
      {isVerifyMode && evidenceCount > 0 ? (
        <span
          title={`${evidenceCount} bằng chứng`}
          style={{
            position: "absolute",
            top: -6,
            right: -4,
            background: vColor ? vColor.badge : "#16a34a",
            color: "#ffffff",
            fontSize: 9,
            fontWeight: 800,
            padding: "1px 5px",
            borderRadius: 999,
            border: "2px solid #ffffff",
            boxShadow: "0 2px 4px rgba(0,0,0,0.25)",
            zIndex: 3,
          }}
        >
          {evidenceCount} 📎
        </span>
      ) : isHub && (
        <span
          title="Hub node — high centrality"
          style={{
            position: "absolute",
            top: -6,
            right: -2,
            background: "#fbbf24",
            color: "#7c2d12",
            fontSize: 9,
            fontWeight: 800,
            padding: "1px 5px",
            borderRadius: 999,
            border: "2px solid #ffffff",
            boxShadow: "0 2px 4px rgba(0,0,0,0.25)",
            zIndex: 2,
          }}
        >
          ★
        </span>
      )}
      <Handle type="target" position={Position.Left} style={centerHandle} />
      <p
        style={{
          fontSize: size > 100 ? 12 : 11,
          fontWeight: 700,
          color: color.text,
          textAlign: "center",
          lineHeight: 1.2,
          wordBreak: "normal",
          overflowWrap: "anywhere",
          hyphens: "auto",
          maxWidth: size - 18,
          userSelect: "none",
          zIndex: 1,
          position: "relative",
          textShadow: "0 1px 2px rgba(0,0,0,0.2)",
          margin: 0,
        }}
      >
        {label}
      </p>
      <span
        style={{
          alignItems: "center",
          background: "rgba(255,255,255,0.25)",
          backdropFilter: "blur(4px)",
          borderRadius: 999,
          color: color.text,
          display: "inline-flex",
          fontSize: 10,
          fontWeight: 800,
          height: 18,
          justifyContent: "center",
          minWidth: 22,
          padding: "0 6px",
          textShadow: "0 1px 1px rgba(0,0,0,0.15)",
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
  width = 1800,
  height = 1200
): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>();
  const n = nodes.length;
  if (n === 0) {
    return pos;
  }

  nodes.forEach((node, i) => {
    const angle = (i / n) * 2 * Math.PI;
    const r = Math.min(width, height) * 0.35;
    const jitter = seededJitter(node.id);
    pos.set(node.id, {
      x: width / 2 + r * Math.cos(angle) + jitter.x,
      y: height / 2 + r * Math.sin(angle) + jitter.y,
    });
  });

  // Tuned for ~30-40 nodes with 90-120px circles. Higher REPEL + IDEAL → cleaner spread.
  const REPEL = 120000;
  const IDEAL = 320;
  const SPRING = 0.04;
  const GRAVITY = 0.008;
  const ITERS = 280;

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
      p.x = Math.max(120, Math.min(width - 200, p.x + f.x * cool));
      p.y = Math.max(100, Math.min(height - 120, p.y + f.y * cool));
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

function applyMindmapTreeLayout(nodes: Node[], edges: Edge[]): Node[] {
  const children = new Map<string, string[]>();
  edges.forEach((edge) => {
    children.set(edge.source, [...(children.get(edge.source) ?? []), edge.target]);
  });

  const root = nodes.find((node) => (node.data as MindmapNodeData).entityType === "root") ?? nodes[0];
  if (!root) return nodes;

  // Generous spacing so pill labels never overlap. Wider level gaps give edges
  // room to curve cleanly; taller leaf rows keep sibling text legible.
  const leafHeight = 58;
  const branchGap = 28;
  const levelX = [40, 380, 700, 1000, 1280];
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
  onEdgeSelect?: (edge: CanvasEdge) => void;
  mode: "graph" | "mindmap";
  canvasNodes?: CanvasNode[];
  canvasEdges?: CanvasEdge[];
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
  onDraftQuestion?: (draft: string) => void;
  onFindRelated?: (draft: string) => void;
  searchQuery?: string;
  // G4 — GraphRAG answer provenance highlight (slug-form ids the LLM actually used)
  answerEntityIds?: string[];
  verifyMode?: boolean;
};

function FlowInner({
  onSelect,
  onEdgeSelect,
  mode,
  canvasNodes = [],
  canvasEdges = [],
  onOpenEvidence,
  onDraftQuestion,
  onFindRelated,
  searchQuery,
  answerEntityIds,
}: GraphCanvasProps) {
  const [collapsedNodes, setCollapsedNodes] = useState<Set<string>>(() => new Set());
  const [prunedNodes, setPrunedNodes] = useState<Set<string>>(() => new Set());
  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const lastMindmapSignature = useRef<string>("");
  const { fitView } = useReactFlow();

  // Adjacency (both directions) for focus-mode dimming — cheap to recompute,
  // and deliberately kept OUT of the layout pipeline so hovering never
  // re-runs the force simulation.
  const adjacency = useMemo<Map<string, Set<string>>>(() => {
    const adj = new Map<string, Set<string>>();
    for (const e of canvasEdges) {
      (adj.get(e.source) ?? adj.set(e.source, new Set()).get(e.source)!).add(e.target);
      (adj.get(e.target) ?? adj.set(e.target, new Set()).get(e.target)!).add(e.source);
    }
    return adj;
  }, [canvasEdges]);

  const matchingNodeIds = useMemo<Set<string> | null>(() => {
    const q = searchQuery?.trim().toLowerCase();
    if (!q || mode !== "graph") return null;
    return new Set(canvasNodes.filter((n) => n.label.toLowerCase().includes(q)).map((n) => n.id));
  }, [searchQuery, canvasNodes, mode]);

  // G4 — slug-form ids the last GraphRAG answer actually used. Highlight these
  // distinctly from search matches: search = blue ring; answer = gold ring.
  const answerHighlightIds = useMemo<Set<string>>(
    () => new Set((answerEntityIds ?? []).map((id) => id)),
    [answerEntityIds],
  );

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
          const isSearchMatch = matchingNodeIds !== null && matchingNodeIds.has(node.id);
          const isDimmedBySearch = matchingNodeIds !== null && !matchingNodeIds.has(node.id);
          const isAnswerHighlight = answerHighlightIds.has(node.id);
          const data =
            mode === "graph"
              ? {
                  label: node.label,
                  entityType: node.type || "concept",
                  confidence: node.confidence ?? null,
                  degree: node.degree ?? 1,
                  importance: node.importance ?? 0,
                  community: node.community ?? 0,
                  isHub: node.is_hub ?? false,
                  evidenceRefs,
                  // Answer-trace highlight outranks citation focus styling.
                  focused: node.focused || isAnswerHighlight,
                  answerHighlight: isAnswerHighlight,
                  dimmed: isDimmedBySearch,
                  searchMatch: isSearchMatch,
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
                  summary: node.summary ?? null,
                  sourceLabel: node.source_label ?? null,
                };

          return {
            id: node.id,
            type: mode === "graph" ? "circle" : hasChildren ? "collapsible" : "pill",
            position: node.position ?? { x: 0, y: 0 },
            data,
          };
        }),
    [canvasNodes, childrenMap, collapsedNodes, handleToggleNode, hiddenNodes, matchingNodeIds, mode, answerHighlightIds]
  );

  const rfEdges = useMemo<Edge[]>(
    () =>
      canvasEdges
        .filter((edge) => !hiddenNodes.has(edge.source) && !hiddenNodes.has(edge.target))
        .map((edge, index) => {
          const selectedRelated =
            selectedNodeId !== null && (edge.source === selectedNodeId || edge.target === selectedNodeId);
          const dimmedBySelection = mode === "graph" && selectedNodeId !== null && !selectedRelated;
          const searchEndpointMatch = matchingNodeIds !== null &&
            (matchingNodeIds.has(edge.source) || matchingNodeIds.has(edge.target));
          const dimmedBySearch = matchingNodeIds !== null && !searchEndpointMatch;
          const dimmed = dimmedBySelection || dimmedBySearch;
          const semanticLabel =
            mode === "graph" &&
            edge.label &&
            !edge.label.startsWith("co_occurs") &&
            edge.label.length < 24
              ? edge.label.replace(/_/g, " ")
              : undefined;
          const branchColor = edge.branchColor || "#c4cfdd";

          const isDark = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
          const defaultStroke = isDark ? "#46557a" : "#94a3b8";
          const highlightStroke = isDark ? "#22d3ee" : "#0891b2";
          const strokeColor = selectedRelated || edge.focused ? highlightStroke : mode === "graph" ? defaultStroke : branchColor;

          return {
            id: edge.id ?? `${edge.source}__${edge.target}__${index}`,
            source: edge.source,
            target: edge.target,
            type: mode === "mindmap" ? "smoothstep" : "bezier",
            data: { canvasEdge: edge },
            label: selectedRelated ? semanticLabel : undefined,
            animated: selectedRelated,
            // Directional arrow on knowledge-graph relations (source → target).
            // Mindmap stays arrow-free: its left→right hierarchy is already clear.
            markerEnd: mode === "graph"
              ? { type: MarkerType.ArrowClosed, width: 16, height: 16, color: strokeColor }
              : undefined,
            style: {
              opacity: dimmed ? 0.08 : selectedRelated ? 1 : mode === "graph" ? 0.4 : 0.7,
              stroke: strokeColor,
              strokeWidth: selectedRelated ? 2.5 : edge.focused ? 2 : mode === "graph" ? 1.4 : 1.75,
            },
            labelStyle: { fill: highlightStroke, fontSize: 10, fontWeight: 700 },
            labelBgStyle: { fill: isDark ? "#131c2e" : "#ffffff", fillOpacity: 0.95 },
            labelBgPadding: [4, 6] as [number, number],
            labelBgBorderRadius: 4,
          };
        }),
    [canvasEdges, hiddenNodes, matchingNodeIds, mode, selectedNodeId]
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

  // Combined sync + focus decoration. Positions always come from `layoutedNodes`
  // (hoveredNodeId is NOT a layout dependency), so hovering only re-paints
  // dim/highlight — it never re-runs the force simulation or shifts nodes.
  useEffect(() => {
    if (mode !== "graph" || !hoveredNodeId) {
      setNodes(layoutedNodes);
      setEdges(rfEdges);
      return;
    }
    const neighbors = adjacency.get(hoveredNodeId) ?? new Set<string>();
    const isDark = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
    const focusStroke = isDark ? "#22d3ee" : "#0891b2";

    setNodes(
      layoutedNodes.map((n) => {
        const active = n.id === hoveredNodeId || neighbors.has(n.id);
        return {
          ...n,
          data: {
            ...n.data,
            dimmed: !active,
            focused: (n.data as { focused?: boolean }).focused || n.id === hoveredNodeId,
          },
        };
      }),
    );
    setEdges(
      rfEdges.map((e) => {
        const related = e.source === hoveredNodeId || e.target === hoveredNodeId;
        return {
          ...e,
          animated: related,
          style: {
            ...e.style,
            opacity: related ? 1 : 0.06,
            stroke: related ? focusStroke : (e.style?.stroke as string),
            strokeWidth: related ? 2.5 : (e.style?.strokeWidth as number),
          },
        };
      }),
    );
  }, [hoveredNodeId, layoutedNodes, rfEdges, adjacency, mode, setNodes, setEdges]);

  // Re-center the canvas whenever the underlying data set changes (new graph,
  // collapse/expand, mode switch) — but not on hover/selection re-paints.
  const dataSignature = useMemo(
    () => `${mode}|${layoutedNodes.map((n) => n.id).join(",")}`,
    [mode, layoutedNodes],
  );
  useEffect(() => {
    const t = setTimeout(
      () => fitView({ padding: mode === "mindmap" ? 0.24 : 0.14, duration: 400 }),
      90,
    );
    return () => clearTimeout(t);
  }, [dataSignature, fitView, mode]);

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => {
      setSelectedNodeId(node.id);
      onSelect(node.id, node.data.label as string);
    },
    [onSelect]
  );

  // Focus mode — hovering a node dims everything except it and its neighbours.
  const handleNodeMouseEnter = useCallback(
    (_: unknown, node: Node) => {
      if (mode === "graph") setHoveredNodeId(node.id);
    },
    [mode]
  );
  const handleNodeMouseLeave = useCallback(() => setHoveredNodeId(null), []);

  const handleEdgeClick = useCallback(
    (_: unknown, edge: Edge) => {
      const canvasEdge = (edge.data as { canvasEdge?: CanvasEdge } | undefined)?.canvasEdge;
      if (!canvasEdge) return;
      setSelectedNodeId(null);
      onEdgeSelect?.(canvasEdge);
    },
    [onEdgeSelect]
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
      event.preventDefault();
      setContextMenu({
        nodeId: node.id,
        nodeLabel: node.data.label as string,
        position: { x: event.clientX, y: event.clientY },
        evidenceRefs: ((node.data as MindmapNodeData).evidenceRefs ?? []),
      });
    },
    []
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

  // ── PNG export ──────────────────────────────────────────────────────────────
  const [exporting, setExporting] = useState(false);
  const exportToPng = useCallback(async () => {
    setExporting(true);
    try {
      const viewport = document.querySelector(".react-flow__viewport") as HTMLElement | null;
      const wrapper = document.querySelector(".react-flow") as HTMLElement | null;
      if (!viewport || !wrapper) return;
      const bounds = getRectOfNodes(nodes);
      const imageWidth = Math.max(1024, bounds.width + 200);
      const imageHeight = Math.max(720, bounds.height + 200);
      const transform = getTransformForBounds(bounds, imageWidth, imageHeight, 0.5, 2, 0.1);
      const isDark = document.documentElement.classList.contains("dark");
      const dataUrl = await toPng(viewport, {
        backgroundColor: isDark ? "#0b1220" : "#ffffff",
        width: imageWidth,
        height: imageHeight,
        style: {
          width: `${imageWidth}px`,
          height: `${imageHeight}px`,
          transform: `translate(${transform[0]}px, ${transform[1]}px) scale(${transform[2]})`,
        },
      });
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `noelys-${mode}-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.png`;
      a.click();
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setExporting(false);
    }
  }, [nodes, mode]);

  // ── Legend: entity types visible in current graph ──────────────────────────
  const [legendOpen, setLegendOpen] = useState(false);
  const visibleTypes = useMemo(() => {
    const seen = new Map<string, { bg: string; border: string; text: string }>();
    for (const n of nodes) {
      const t = ((n.data as { entityType?: string })?.entityType || "").toLowerCase().split(/[_\s]/)[0];
      if (!t || t === "root" || t === "cluster") continue;
      if (!seen.has(t)) seen.set(t, typeColor(t));
    }
    return Array.from(seen.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [nodes]);

  // ── Dark-mode aware background ─────────────────────────────────────────────
  const isDark = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
  const bgColor = mode === "mindmap"
    ? (isDark ? "#1c2842" : "#f1f5f9")
    : (isDark ? "#22304d" : "#e2e8f0");
  const minimapBg = isDark ? "#131c2e" : "#f8fafc";

  return (
    <ReactFlow
      className={mode === "mindmap" ? "mindmap-flow" : "graph-flow"}
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={handleNodeClick}
      onNodeMouseEnter={handleNodeMouseEnter}
      onNodeMouseLeave={handleNodeMouseLeave}
      onEdgeClick={handleEdgeClick}
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
          maskColor={isDark ? "rgba(11,18,32,0.65)" : "rgba(15,23,42,0.06)"}
          style={{ background: minimapBg, border: `1px solid ${isDark ? "#2f3b5c" : "#dbe5f0"}` }}
        />
      )}
      <Controls showInteractive={false} />
      <Background gap={mode === "mindmap" ? 32 : 22} color={bgColor} />

      {/* ── Export PNG button (top-right) ── */}
      <Panel position="top-right" className="!m-2 flex gap-2">
        <button
          type="button"
          onClick={exportToPng}
          disabled={exporting}
          title="Tải hình ảnh (.png)"
          className="flex items-center gap-1.5 rounded-lg border border-outline bg-surface px-3 py-1.5 text-[11px] font-semibold text-text shadow-sm transition hover:border-primary hover:text-primary disabled:opacity-50"
        >
          <Download size={12} />
          {exporting ? "Đang xuất..." : "PNG"}
        </button>
        <button
          type="button"
          onClick={() => setLegendOpen((v) => !v)}
          title="Chú thích loại thực thể"
          className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[11px] font-semibold shadow-sm transition ${
            legendOpen
              ? "border-primary bg-primary/10 text-primary"
              : "border-outline bg-surface text-text hover:border-primary hover:text-primary"
          }`}
        >
          <Layers size={12} />
          Chú thích
        </button>
      </Panel>

      {/* ── Legend panel (bottom-right) ── */}
      {legendOpen && visibleTypes.length > 0 && (
        <Panel position="bottom-right" className="!mr-2 !mb-12">
          <div className="rounded-lg border border-outline bg-surface/95 p-3 shadow-md backdrop-blur max-w-[200px]">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted">Loại thực thể</p>
            <div className="space-y-1.5">
              {visibleTypes.map(([type, color]) => (
                <div key={type} className="flex items-center gap-2">
                  <span
                    className="inline-block h-3 w-3 shrink-0 rounded-full border-2"
                    style={{ background: color.bg, borderColor: color.border }}
                  />
                  <span className="text-[11px] font-medium capitalize text-text">{type}</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      )}

      {mode === "mindmap" && (
        <Panel position="top-left" className="mindmap-panel">
          <div className="flex items-center gap-2 rounded-lg border border-outline bg-surface/95 px-2 py-1.5 shadow-sm backdrop-blur">
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
          subtitle={mode === "graph" ? "Knowledge Graph" : "Mindmap"}
          onClose={() => setContextMenu(null)}
          onAskAI={(label) => {
            onDraftQuestion?.(`Hãy giải thích về ${label} dựa trên tài liệu hiện có.`);
          }}
          onViewSources={() => {
            if (!openEvidenceFromRefs(contextMenu.evidenceRefs)) {
              onSelect(contextMenu.nodeId, contextMenu.nodeLabel);
            }
          }}
          onFindRelated={(nodeId) => {
            if (onFindRelated) {
              onFindRelated(`Các khái niệm liên quan đến ${contextMenu.nodeLabel} là gì?`);
            } else {
              onSelect(nodeId, contextMenu.nodeLabel);
            }
          }}
          onDelete={mode === "mindmap" && contextMenu.nodeId !== "root-topic" ? handleDeleteNode : undefined}
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
