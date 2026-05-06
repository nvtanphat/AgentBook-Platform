import React, { useCallback } from 'react';
import { MessageSquare, FileText, Link2, Trash2, Eye } from 'lucide-react';

interface MindmapContextMenuProps {
  nodeId: string;
  nodeLabel: string;
  position: { x: number; y: number };
  onClose: () => void;
  onAskAI: (label: string) => void;
  onViewSources: (nodeId: string) => void;
  onFindRelated: (nodeId: string) => void;
  onDelete?: (nodeId: string) => void;
  subtitle?: string;
}

/**
 * Context menu for mindmap nodes.
 *
 * Features:
 * - Ask AI about this concept
 * - View source documents
 * - Find related concepts
 * - Delete from mindmap
 */
export function MindmapContextMenu({
  nodeId,
  nodeLabel,
  position,
  onClose,
  onAskAI,
  onViewSources,
  onFindRelated,
  onDelete,
  subtitle = "Hành động nhanh",
}: MindmapContextMenuProps) {
  const handleAction = useCallback((action: () => void) => {
    action();
    onClose();
  }, [onClose]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
      />

      {/* Menu */}
      <div
        className="fixed z-50 min-w-[240px] max-w-[320px] rounded-lg border border-outline bg-white py-1 shadow-xl"
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
        }}
      >
        <div className="border-b border-outline px-3 py-2">
          <p className="truncate text-xs font-semibold text-text" title={nodeLabel}>{nodeLabel}</p>
          <p className="mt-0.5 text-[10px] font-medium uppercase text-muted">{subtitle}</p>
        </div>

        {/* Ask AI */}
        <button
          onClick={() => handleAction(() => onAskAI(nodeLabel))}
          className="flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-text transition hover:bg-slate-50"
        >
          <MessageSquare size={16} className="text-primary" />
          <span>Hỏi AI về node này</span>
        </button>

        {/* View Sources */}
        <button
          onClick={() => handleAction(() => onViewSources(nodeId))}
          className="flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-text transition hover:bg-slate-50"
        >
          <FileText size={16} className="text-blue-600" />
          <span>Xem nguồn trích dẫn</span>
        </button>

        {/* Find Related */}
        <button
          onClick={() => handleAction(() => onFindRelated(nodeId))}
          className="flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-text transition hover:bg-slate-50"
        >
          <Link2 size={16} className="text-purple-600" />
          <span>Tìm khái niệm liên quan</span>
        </button>

        {/* Highlight in documents */}
        <button
          onClick={() => handleAction(() => onViewSources(nodeId))}
          className="flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-text transition hover:bg-slate-50"
        >
          <Eye size={16} className="text-amber-600" />
          <span>Mở vị trí trong tài liệu</span>
        </button>

        <div className="h-px bg-outline my-1" />

        {/* Delete */}
        {onDelete && (
          <button
            onClick={() => handleAction(() => onDelete(nodeId))}
            className="flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm text-red-600 transition hover:bg-red-50"
          >
            <Trash2 size={16} />
            <span>Ẩn khỏi mindmap</span>
          </button>
        )}
      </div>
    </>
  );
}

export default MindmapContextMenu;
