import React, { useCallback, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Circle } from "lucide-react";
import { Handle, NodeProps, Position } from "reactflow";
import { createPortal } from "react-dom";

interface CollapsibleNodeData {
  label: string;
  entityType: string;
  hasChildren: boolean;
  collapsed: boolean;
  onToggle: (nodeId: string) => void;
  degree?: number;
  branchColor?: string;
  depth?: number;
  /** NotebookLM-style preview shown on hover. */
  summary?: string | null;
  /** Short source attribution e.g. "DeAn.docx · p.12". */
  sourceLabel?: string | null;
}

/** Tier-based visual hierarchy:
 *  root  → hero gradient pill, biggest, with halo
 *  topic → solid-color rounded box (high prominence)
 *  branch → outlined box with accent stripe (medium)
 *  leaf  → minimal chip with bullet (low)
 */
export function CollapsibleMindmapNode({ id, data, selected }: NodeProps<CollapsibleNodeData>) {
  const [hovered, setHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const nodeRef = useRef<HTMLDivElement | null>(null);
  const isRoot = data.entityType === "root";
  const isTopic = data.entityType === "topic";
  const isBranch = data.entityType === "branch";
  const isLeaf = !isRoot && !isTopic && !isBranch;
  const childCount = Math.max(0, data.degree ?? 0);
  const accent = data.branchColor || "#3b82f6";
  const hasPreview = Boolean((data.summary && data.summary.trim()) || (data.sourceLabel && data.sourceLabel.trim()));

  const handleToggle = useCallback((event: React.MouseEvent) => {
    event.stopPropagation();
    data.onToggle(id);
  }, [data, id]);

  const handleMouseEnter = useCallback(() => {
    setHovered(true);
    if (!hasPreview || !nodeRef.current) return;
    const rect = nodeRef.current.getBoundingClientRect();
    setTooltipPos({ x: rect.left + rect.width / 2, y: rect.bottom + 8 });
  }, [hasPreview]);

  const handleMouseLeave = useCallback(() => {
    setHovered(false);
    setTooltipPos(null);
  }, []);

  // ── Visual tiers ───────────────────────────────────────────────────────
  let tierStyles: React.CSSProperties;
  let textColor: string;
  let fontWeight: number;
  let fontSize: number;
  let bulletEl: React.ReactNode = null;

  if (isRoot) {
    fontSize = 14;
    fontWeight = 800;
    textColor = "#ffffff";
    tierStyles = {
      background: `linear-gradient(135deg, ${accent} 0%, ${shadeColor(accent, -25)} 100%)`,
      border: `0`,
      boxShadow: hovered || selected
        ? `0 0 0 6px ${accent}25, 0 14px 32px -8px ${accent}80`
        : `0 8px 24px -10px ${accent}66`,
      padding: "12px 18px",
      minWidth: 240,
      maxWidth: 340,
      borderRadius: 999,
    };
  } else if (isTopic) {
    fontSize = 12;
    fontWeight = 700;
    textColor = "#ffffff";
    tierStyles = {
      background: accent,
      border: `0`,
      boxShadow: hovered || selected
        ? `0 0 0 4px ${accent}30, 0 8px 18px -8px ${accent}90`
        : `0 4px 12px -6px ${accent}50`,
      padding: "8px 14px",
      minWidth: 200,
      maxWidth: 290,
      borderRadius: 14,
    };
  } else if (isBranch) {
    fontSize = 11;
    fontWeight = 650;
    textColor = "var(--c-text)";
    tierStyles = {
      background: "var(--c-surface)",
      border: `2px solid ${accent}`,
      boxShadow: hovered || selected
        ? `0 0 0 3px ${accent}20, 0 4px 10px -4px rgba(0,0,0,0.18)`
        : "0 2px 6px -2px rgba(0,0,0,0.08)",
      padding: "7px 12px 7px 8px",
      minWidth: 170,
      maxWidth: 260,
      borderRadius: 10,
      position: "relative",
    };
    // Accent stripe left for branch
    bulletEl = (
      <span
        style={{
          position: "absolute",
          left: -2,
          top: 6,
          bottom: 6,
          width: 4,
          background: accent,
          borderRadius: 2,
        }}
      />
    );
  } else {
    // Leaf
    fontSize = 11;
    fontWeight = 500;
    textColor = "var(--c-text)";
    tierStyles = {
      background: "var(--c-surface-low)",
      border: `1px solid var(--c-outline)`,
      boxShadow: hovered || selected
        ? `0 0 0 2px ${accent}25, 0 2px 6px -2px rgba(0,0,0,0.12)`
        : "0 1px 3px -1px rgba(0,0,0,0.06)",
      padding: "5px 11px 5px 9px",
      minWidth: 130,
      maxWidth: 220,
      borderRadius: 999,
    };
    bulletEl = (
      <span
        style={{
          background: accent,
          borderRadius: 999,
          boxShadow: `0 0 0 3px ${accent}1a`,
          flexShrink: 0,
          height: 7,
          width: 7,
        }}
      />
    );
  }

  const baseStyle: React.CSSProperties = {
    alignItems: "center",
    color: textColor,
    cursor: "pointer",
    display: "flex",
    fontSize,
    fontWeight,
    gap: 8,
    lineHeight: 1.35,
    transition: "all .2s cubic-bezier(0.4, 0, 0.2, 1)",
    transform: hovered || selected ? "translateY(-2px)" : "none",
    ...tierStyles,
  };

  return (
    <div
      ref={nodeRef}
      style={baseStyle}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      {bulletEl}

      {data.hasChildren && (
        <button
          aria-label={data.collapsed ? "Expand node" : "Collapse node"}
          onClick={handleToggle}
          type="button"
          style={{
            alignItems: "center",
            background: isRoot || isTopic ? "rgba(255,255,255,0.22)" : "var(--c-surface-low)",
            border: isRoot || isTopic ? "1px solid rgba(255,255,255,0.35)" : "1px solid var(--c-outline)",
            borderRadius: 999,
            color: isRoot || isTopic ? "#ffffff" : "var(--c-muted)",
            cursor: "pointer",
            display: "flex",
            flexShrink: 0,
            height: 20,
            justifyContent: "center",
            padding: 0,
            transition: "background .15s, transform .15s",
            transform: data.collapsed ? "rotate(0deg)" : "rotate(0deg)",
            width: 20,
          }}
        >
          {data.collapsed
            ? <ChevronRight size={13} strokeWidth={2.4} />
            : <ChevronDown size={13} strokeWidth={2.4} />}
        </button>
      )}

      <span style={{ flex: 1, minWidth: 0, overflowWrap: "anywhere", textAlign: "left", userSelect: "none" }}>
        {data.label}
      </span>

      {data.hasChildren && childCount > 0 && (
        <span
          title={`${childCount} child nodes`}
          style={{
            alignItems: "center",
            background: isRoot || isTopic ? "rgba(255,255,255,0.25)" : `${accent}1a`,
            border: isRoot || isTopic ? "1px solid rgba(255,255,255,0.35)" : `1px solid ${accent}40`,
            borderRadius: 999,
            color: isRoot || isTopic ? "#ffffff" : accent,
            display: "inline-flex",
            flexShrink: 0,
            fontSize: 10,
            fontWeight: 800,
            height: 18,
            justifyContent: "center",
            minWidth: 20,
            padding: "0 6px",
          }}
        >
          {childCount}
        </span>
      )}

      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />

      {hovered && hasPreview && tooltipPos && createPortal(
        <div
          style={{
            position: "fixed",
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: "translateX(-50%)",
            background: "rgba(15, 23, 42, 0.96)",
            border: `1px solid ${accent}`,
            borderRadius: 10,
            color: "#f1f5f9",
            padding: "10px 12px",
            maxWidth: 320,
            zIndex: 9999,
            boxShadow: "0 12px 28px -8px rgba(0,0,0,0.45)",
            fontSize: 11,
            lineHeight: 1.5,
            pointerEvents: "none",
            whiteSpace: "normal",
          }}
        >
          {data.summary && data.summary.trim() ? (
            <div style={{ fontStyle: "italic", color: "#e2e8f0" }}>
              {data.summary.length > 240 ? `${data.summary.slice(0, 240)}…` : data.summary}
            </div>
          ) : null}
          {data.sourceLabel && data.sourceLabel.trim() ? (
            <div
              style={{
                marginTop: data.summary ? 6 : 0,
                color: accent,
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: 0.4,
                textTransform: "uppercase",
              }}
            >
              {data.sourceLabel}
            </div>
          ) : null}
        </div>,
        document.body,
      )}
    </div>
  );
}

// Shade a hex color by percent (-100..+100)
function shadeColor(color: string, percent: number): string {
  let R = parseInt(color.substring(1, 3), 16);
  let G = parseInt(color.substring(3, 5), 16);
  let B = parseInt(color.substring(5, 7), 16);
  R = Math.round(R * (100 + percent) / 100);
  G = Math.round(G * (100 + percent) / 100);
  B = Math.round(B * (100 + percent) / 100);
  R = Math.max(0, Math.min(255, R));
  G = Math.max(0, Math.min(255, G));
  B = Math.max(0, Math.min(255, B));
  return "#" + R.toString(16).padStart(2, "0") + G.toString(16).padStart(2, "0") + B.toString(16).padStart(2, "0");
}

export default CollapsibleMindmapNode;
