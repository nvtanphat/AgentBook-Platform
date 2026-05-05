import React, { useCallback } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Handle, NodeProps, Position } from "reactflow";

interface CollapsibleNodeData {
  label: string;
  entityType: string;
  hasChildren: boolean;
  collapsed: boolean;
  onToggle: (nodeId: string) => void;
  degree?: number;
  branchColor?: string;
}

export function CollapsibleMindmapNode({ id, data, selected }: NodeProps<CollapsibleNodeData>) {
  const isRoot = data.entityType === "root";
  const isTopic = data.entityType === "topic";
  const isBranch = data.entityType === "branch";
  const childCount = Math.max(0, data.degree ?? 0);
  const accent = data.branchColor || "#5f6368";

  const handleToggle = useCallback((event: React.MouseEvent) => {
    event.stopPropagation();
    data.onToggle(id);
  }, [data, id]);

  const nodeStyle: React.CSSProperties = {
    alignItems: "center",
    background: "#ffffff",
    border: `1px solid ${selected ? accent : "#d8dee8"}`,
    borderRadius: 999,
    boxShadow: selected ? `0 0 0 3px ${accent}18, 0 8px 18px -16px rgba(15,23,42,.65)` : "0 6px 16px -18px rgba(15,23,42,.58)",
    color: isRoot ? "#172033" : "#2f3a4a",
    cursor: "pointer",
    display: "flex",
    fontSize: isRoot ? 13 : isTopic ? 12 : 11,
    fontWeight: isRoot ? 750 : isTopic ? 700 : isBranch ? 625 : 550,
    gap: 8,
    lineHeight: 1.35,
    minWidth: isRoot ? 240 : isTopic ? 208 : isBranch ? 170 : 128,
    maxWidth: isRoot ? 330 : isTopic ? 286 : isBranch ? 250 : 220,
    padding: isRoot ? "10px 14px" : isTopic ? "8px 12px" : "6px 10px",
    transition: "border-color .15s ease, box-shadow .15s ease, transform .15s ease",
    transform: selected ? "translateY(-1px)" : "none",
  };

  return (
    <div style={nodeStyle}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />

      {!isRoot && (
        <span
          style={{
            background: accent,
            borderRadius: 999,
            boxShadow: `0 0 0 3px ${accent}14`,
            flexShrink: 0,
            height: isTopic ? 10 : 8,
            width: isTopic ? 10 : 8,
          }}
        />
      )}

      {data.hasChildren && (
        <button
          aria-label={data.collapsed ? "Expand node" : "Collapse node"}
          onClick={handleToggle}
          style={{
            alignItems: "center",
            background: "#f8fafc",
            border: "1px solid #dbe3ef",
            borderRadius: 999,
            cursor: "pointer",
            display: "flex",
            flexShrink: 0,
            height: 18,
            justifyContent: "center",
            padding: 0,
            width: 18,
          }}
          type="button"
        >
          {data.collapsed ? <ChevronRight size={13} color="#5f6b7a" /> : <ChevronDown size={13} color="#5f6b7a" />}
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
            background: "#f8fafc",
            border: "1px solid #e2e8f0",
            borderRadius: 999,
            color: "#64748b",
            display: "inline-flex",
            flexShrink: 0,
            fontSize: 10,
            fontWeight: 750,
            height: 18,
            justifyContent: "center",
            minWidth: 18,
            padding: "0 6px",
          }}
        >
          {childCount}
        </span>
      )}

      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </div>
  );
}

export default CollapsibleMindmapNode;
