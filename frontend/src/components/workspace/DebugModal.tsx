import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, X, Eye, EyeOff, FileText, Layers, Hash } from "lucide-react";
import { DebugBlock, MaterialDebugResponse, getMaterialDebug, getMaterialRawUrl } from "../../api/client";

type Props = {
  materialId: string;
  ownerId: string;
  originalName: string;
  onClose: () => void;
};

type Tab = "blocks" | "chunks" | "info";

function blockColor(type: string): string {
  switch (type) {
    case "heading": return "rgb(239, 68, 68)";
    case "paragraph": return "rgb(59, 130, 246)";
    case "table": return "rgb(168, 85, 247)";
    case "figure": return "rgb(245, 158, 11)";
    case "list": return "rgb(16, 185, 129)";
    case "equation": return "rgb(236, 72, 153)";
    case "handwriting": return "rgb(99, 102, 241)";
    case "ocr_text": return "rgb(20, 184, 166)";
    default: return "rgb(100, 116, 139)";
  }
}

function confidenceColor(conf: number | null | undefined): string {
  if (conf == null) return "text-muted";
  if (conf >= 0.9) return "text-emerald-600";
  if (conf >= 0.7) return "text-amber-600";
  return "text-red-600";
}

function pageUnit(fileType: string, count: number, titleCase = false): string {
  const type = fileType.toLowerCase();
  let label: string;
  if (type === "docx") label = count === 1 ? "logical section" : "logical sections";
  else if (type === "pptx") label = count === 1 ? "slide" : "slides";
  else if (type === "xlsx" || type === "xls" || type === "csv") label = count === 1 ? "sheet" : "sheets";
  else if (type === "png" || type === "jpg" || type === "jpeg") label = count === 1 ? "image" : "images";
  else label = count === 1 ? "page" : "pages";
  return titleCase ? label.charAt(0).toUpperCase() + label.slice(1) : label;
}

function pageRefLabel(fileType: string, pageNumber: number): string {
  const type = fileType.toLowerCase();
  if (type === "docx") return `Section ${pageNumber}`;
  if (type === "pptx") return `Slide ${pageNumber}`;
  if (type === "xlsx" || type === "xls" || type === "csv") return `Sheet ${pageNumber}`;
  if (type === "png" || type === "jpg" || type === "jpeg") return `Image ${pageNumber}`;
  return `P.${pageNumber}`;
}

export default function DebugModal({ materialId, ownerId, originalName, onClose }: Props) {
  const [data, setData] = useState<MaterialDebugResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("blocks");
  const [pageIndex, setPageIndex] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [hoveredBlockId, setHoveredBlockId] = useState<string | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Escape to close + focus modal on mount
  useEffect(() => {
    dialogRef.current?.focus();
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getMaterialDebug(materialId, ownerId)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load debug data");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [materialId, ownerId]);

  const currentPage = data?.pages[pageIndex];
  const rawUrl = useMemo(() => {
    if (!data?.raw_image_url) return null;
    return getMaterialRawUrl(materialId, ownerId);
  }, [data, materialId, ownerId]);

  // Natural image dimensions for SVG viewBox — bbox coords are in this space.
  // SVG is stretched to 100% of the overlay div so the browser handles scaling.
  const naturalSize = useMemo(() => {
    if (!imgSize) return null;
    return { w: currentPage?.width ?? imgSize.w, h: currentPage?.height ?? imgSize.h };
  }, [imgSize, currentPage]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={`Debug — ${originalName}`}
        tabIndex={-1}
        className="bg-white rounded-xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col overflow-hidden outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-outline shrink-0">
          <div className="min-w-0">
            <h2 className="font-heading font-semibold text-sm truncate" title={originalName}>
              Debug — {originalName}
            </h2>
            {data && (
              <div className="flex items-center gap-3 text-[11px] text-muted mt-0.5">
                <span className="flex items-center gap-1"><Layers size={10} /> {data.page_count} {pageUnit(data.file_type, data.page_count)}</span>
                <span className="flex items-center gap-1"><FileText size={10} /> {data.pages.reduce((n, p) => n + p.blocks.length, 0)} blocks</span>
                <span className="flex items-center gap-1"><Hash size={10} /> {data.chunks.length} chunks</span>
                <span className="text-emerald-600">{data.qdrant_vector_count} vectors</span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Đóng debug"
            className="rounded p-1.5 text-muted hover:bg-slate-100 hover:text-text"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-outline bg-slate-50 shrink-0">
          {(["blocks", "chunks", "info"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-xs font-semibold uppercase tracking-wide transition ${
                tab === t ? "text-primary border-b-2 border-primary" : "text-muted hover:text-text"
              }`}
            >
              {t === "blocks" ? "OCR Blocks" : t === "chunks" ? "Chunks" : "Info"}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {loading && (
            <div className="h-full flex items-center justify-center">
              <Loader2 size={20} className="animate-spin text-primary" />
            </div>
          )}
          {error && (
            <div className="h-full flex items-center justify-center text-red-600 text-sm px-6 text-center">
              {error}
            </div>
          )}
          {data && !loading && !error && (
            <>
              {tab === "blocks" && (
                <div className="h-full grid grid-cols-1 lg:grid-cols-[1fr_400px] overflow-hidden">
                  {/* Image + overlay */}
                  <div className="relative bg-slate-100 overflow-auto p-4">
                    {data.pages.length > 1 && (
                      <div className="mb-2 flex items-center gap-1">
                        {data.pages.map((p, i) => (
                          <button
                            key={p.page_number}
                            onClick={() => setPageIndex(i)}
                            className={`px-2 py-1 text-[11px] rounded ${
                              i === pageIndex ? "bg-primary text-white" : "bg-white border border-outline text-muted hover:text-text"
                            }`}
                          >
                            {pageRefLabel(data.file_type, p.page_number)}
                          </button>
                        ))}
                      </div>
                    )}
                    {data.file_type.toLowerCase() === "docx" && (
                      <div className="mb-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-5 text-amber-800">
                        DOCX không có pagination cố định trong file gốc; các mục ở đây là logical sections do parser tạo, không phải số trang in thực tế.
                      </div>
                    )}
                    <div className="mb-2 flex items-center gap-2">
                      <button
                        onClick={() => setShowOverlay((v) => !v)}
                        className="flex items-center gap-1 text-[11px] px-2 py-1 rounded border border-outline bg-white text-muted hover:text-text"
                      >
                        {showOverlay ? <EyeOff size={11} /> : <Eye size={11} />}
                        {showOverlay ? "Hide bboxes" : "Show bboxes"}
                      </button>
                      <span className="text-[11px] text-muted">
                        Hover row → highlight box
                      </span>
                    </div>
                    {rawUrl ? (
                      <div className="relative inline-block">
                        <img
                          ref={imgRef}
                          src={rawUrl}
                          alt={originalName}
                          className="max-w-full h-auto block"
                          onLoad={(e) => {
                            const t = e.currentTarget;
                            // Natural pixel dimensions — bbox coords are in this space
                            setImgSize({ w: t.naturalWidth, h: t.naturalHeight });
                          }}
                        />
                        {showOverlay && currentPage && naturalSize && (
                          <svg
                            className="absolute inset-0 pointer-events-none"
                            width="100%"
                            height="100%"
                            viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
                          >
                            {currentPage.blocks.map((b) => {
                              if (!b.bbox) return null;
                              const isHover = hoveredBlockId === b.block_id;
                              return (
                                <rect
                                  key={b.block_id}
                                  x={b.bbox.x1}
                                  y={b.bbox.y1}
                                  width={b.bbox.x2 - b.bbox.x1}
                                  height={b.bbox.y2 - b.bbox.y1}
                                  fill={isHover ? blockColor(b.block_type) : "transparent"}
                                  fillOpacity={isHover ? 0.25 : 0}
                                  stroke={blockColor(b.block_type)}
                                  strokeWidth={isHover ? 2 : 1}
                                />
                              );
                            })}
                          </svg>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-muted italic">No image preview (file is not an image).</div>
                    )}
                  </div>

                  {/* Block list */}
                  <div className="border-l border-outline overflow-y-auto bg-white">
                    {currentPage?.blocks.length === 0 && (
                      <div className="p-4 text-xs text-muted">No blocks on this page.</div>
                    )}
                    <ul className="divide-y divide-outline">
                      {currentPage?.blocks.map((b: DebugBlock) => (
                        <li
                          key={b.block_id}
                          onMouseEnter={() => setHoveredBlockId(b.block_id)}
                          onMouseLeave={() => setHoveredBlockId(null)}
                          className={`p-3 text-xs cursor-pointer transition ${
                            hoveredBlockId === b.block_id ? "bg-amber-50" : "hover:bg-slate-50"
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span
                              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                              style={{ backgroundColor: blockColor(b.block_type) }}
                            />
                            <span className="font-mono text-[10px] text-muted truncate">{b.block_id}</span>
                            <span className="text-[10px] uppercase font-semibold text-muted">{b.block_type}</span>
                            {b.ocr_confidence != null && (
                              <span className={`text-[10px] ml-auto font-mono ${confidenceColor(b.ocr_confidence)}`}>
                                {(b.ocr_confidence * 100).toFixed(1)}%
                              </span>
                            )}
                          </div>
                          <p className="text-text break-words whitespace-pre-wrap">{b.content}</p>
                          {b.bbox && (
                            <p className="text-[10px] text-muted mt-1 font-mono">
                              [{b.bbox.x1.toFixed(0)}, {b.bbox.y1.toFixed(0)}] → [{b.bbox.x2.toFixed(0)}, {b.bbox.y2.toFixed(0)}]
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {tab === "chunks" && (
                <div className="h-full overflow-y-auto p-4 bg-slate-50">
                  {data.chunks.length === 0 ? (
                    <div className="text-center text-muted text-sm py-8">No chunks indexed.</div>
                  ) : (
                    <div className="space-y-3">
                      {data.chunks.map((c, idx) => (
                        <div key={c.chunk_id} className="bg-white rounded-lg border border-outline p-3">
                          <div className="flex items-center gap-2 text-[11px] text-muted mb-2">
                            <span className="font-bold text-primary">#{idx + 1}</span>
                            <span className="font-mono">{c.chunk_id}</span>
                            <span className="ml-auto">{c.token_count ?? "?"} tokens</span>
                          </div>
                          <p className="text-xs text-text whitespace-pre-wrap break-words leading-relaxed">{c.content}</p>
                          <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
                            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-muted">
                              {pageUnit(data.file_type, 2)}: {c.source_pages.join(", ") || "—"}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-muted">
                              {c.source_block_ids.length} blocks
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-muted">{c.language}</span>
                            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-muted">{c.modality}</span>
                            <span className="px-1.5 py-0.5 rounded bg-slate-100 text-muted">{c.chunk_strategy}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {tab === "info" && (
                <div className="h-full overflow-y-auto p-6 text-xs">
                  <dl className="grid grid-cols-[180px_1fr] gap-y-2 max-w-2xl">
                    <dt className="text-muted">Material ID</dt><dd className="font-mono">{data.material_id}</dd>
                    <dt className="text-muted">Collection ID</dt><dd className="font-mono">{data.collection_id}</dd>
                    <dt className="text-muted">Owner</dt><dd>{data.owner_id}</dd>
                    <dt className="text-muted">File type</dt><dd>{data.file_type}</dd>
                    <dt className="text-muted">Modality</dt><dd>{data.modality}</dd>
                    <dt className="text-muted">Language</dt><dd>{data.language}</dd>
                    <dt className="text-muted">Status</dt><dd>{data.status}</dd>
                    <dt className="text-muted">{pageUnit(data.file_type, data.page_count, true)}</dt><dd>{data.page_count}</dd>
                    <dt className="text-muted">Chunks (Mongo)</dt><dd>{data.chunks.length}</dd>
                    <dt className="text-muted">Vectors (Qdrant)</dt><dd>{data.qdrant_vector_count}</dd>
                  </dl>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
