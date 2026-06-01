/**
 * EvidencePanel — production-grade evidence viewer.
 *
 * Features:
 *  • Keyboard navigation: ←/→ arrows to switch citations, Esc to clear
 *  • Skeleton loading with shimmer animation
 *  • Visited-citation tracking (✓ on chips already opened)
 *  • Position counter [3 / 5] in the strip header
 *  • Search with live match-count ("3 of 13")
 *  • Prev/Next page navigation within the document
 *  • Error state with one-click retry
 *  • Premium Minimalist visual design (Claude Artifacts / Perplexity style)
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check, ChevronDown, ChevronLeft, ChevronRight, ChevronUp,
  Copy, FileAudio, FileSpreadsheet, FileText, Image, Loader2,
  Presentation, RotateCcw, X,
} from "lucide-react";
import { API_BASE_URL, Citation, EvidenceBlock, EvidencePageResponse, loadEvidencePage } from "../api/client";
import SnippetRenderer from "./SnippetRenderer";
import { AudioSegmentList } from "./AudioCitationPlayer";
import { useWorkspace } from "../state/workspace";

// ─── Text utilities ───────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  "và","của","là","trong","có","được","cho","với","các","một",
  "này","đó","từ","theo","để","như","khi","về","bằng","trên",
  "the","a","an","of","in","is","are","to","for","with","by",
  "that","this","on","at","as","be","from","or","and","not",
]);

function extractKeywords(text: string): Set<string> {
  const words = text
    .toLowerCase()
    .replace(/[.,;:!?()\[\]{}"']/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 4 && !STOP_WORDS.has(w));
  return new Set(words.sort((a, b) => b.length - a.length).slice(0, 8));
}

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildExactPattern(text: string, snippet?: string | null): RegExp | null {
  const needle = (snippet ?? "").trim();
  if (!needle) return null;
  const tc = text.replace(/\s+/g, " ").trim();
  const nc = needle.replace(/\s+/g, " ").trim();
  if (!tc || !nc || nc.length < 16) return null;
  if (nc.length / tc.length > 0.9) return null;
  return new RegExp(`(${nc.split(/\s+/).map(escapeRegExp).join("\\s+")})`, "i");
}

function HighlightedText({
  text, exactSnippet, className,
}: {
  text: string; keywords?: Set<string>; exactSnippet?: string | null; className?: string;
}) {
  const pat = buildExactPattern(text, exactSnippet);
  if (pat) {
    const parts = text.split(pat);
    if (parts.length > 1) {
      return (
        <span className={className}>
          {parts.map((p, i) =>
            i % 2 === 1
              ? <mark key={i} className="rounded-sm bg-primary/15 px-0.5 text-primary not-italic">{p}</mark>
              : <span key={i}>{p}</span>
          )}
        </span>
      );
    }
  }
  return <span className={className}>{text}</span>;
}

// ─── Skeleton shimmer ─────────────────────────────────────────────────────────

function Shimmer({ className }: { className?: string }) {
  return <div className={`shimmer-bg rounded ${className ?? ""}`} />;
}

function PageSkeleton() {
  return (
    <div className="p-4">
      <div className="rounded-xl border border-slate-200/80 bg-white p-5 space-y-5">
        <div className="space-y-2">
          <Shimmer className="h-3 w-1/4" />
          <Shimmer className="h-3.5 w-full" />
          <Shimmer className="h-3.5 w-5/6" />
          <Shimmer className="h-3.5 w-4/5" />
        </div>
        <div className="space-y-2 pt-2 border-t border-slate-100">
          <Shimmer className="h-3 w-1/3" />
          <Shimmer className="h-3.5 w-full" />
          <Shimmer className="h-3.5 w-11/12" />
        </div>
        <div className="space-y-2 pt-2 border-t border-slate-100">
          <Shimmer className="h-3.5 w-full" />
          <Shimmer className="h-3.5 w-3/4" />
        </div>
      </div>
    </div>
  );
}

// ─── Copy button ──────────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handle = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [text]);
  return (
    <button
      onClick={handle}
      title="Copy"
      className="flex items-center gap-1 text-[11px] font-medium text-slate-400 transition hover:text-slate-600"
    >
      {copied ? <Check size={11} className="text-emerald-500" /> : <Copy size={11} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

// ─── Matched snippet ──────────────────────────────────────────────────────────

const SNIPPET_COLLAPSE_LINES = 6;

function MatchedSnippet({ citation, ownerId }: { citation: Citation; ownerId: string }) {
  const lines = citation.snippet_original.split("\n");
  const isLong = lines.length > SNIPPET_COLLAPSE_LINES;
  const [expanded, setExpanded] = useState(false);
  const text = isLong && !expanded
    ? lines.slice(0, SNIPPET_COLLAPSE_LINES).join("\n") + "…"
    : citation.snippet_original;

  const audioBlock = citation.evidence_blocks?.find(
    (b) => b.audio_start_seconds != null && b.audio_end_seconds != null,
  );

  return (
    <div className="shrink-0 border-b border-slate-100 px-4 py-3.5">
      {audioBlock && citation.evidence_blocks && (
        <div className="mb-3">
          <AudioSegmentList evidenceBlocks={citation.evidence_blocks} ownerId={ownerId} />
        </div>
      )}

      {/* Premium blockquote */}
      <div className="border-l-[3px] border-primary bg-slate-50/60 px-3.5 py-3 rounded-r-lg">
        <SnippetRenderer
          text={text}
          blockType={citation.block_type ?? undefined}
          maxRows={SNIPPET_COLLAPSE_LINES}
          textClassName="text-[13px] leading-relaxed text-slate-800"
        />
        {isLong && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2 flex items-center gap-1 text-[11px] font-medium text-primary/70 hover:text-primary transition"
          >
            {expanded ? <><ChevronUp size={11} />Thu gọn</> : <><ChevronDown size={11} />Xem thêm</>}
          </button>
        )}
      </div>

      {/* Meta row */}
      <div className="mt-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {citation.role && (
            <span className={`text-[10px] font-semibold uppercase tracking-wide ${
              citation.role === "primary" ? "text-primary" : "text-slate-400"
            }`}>
              {citation.role === "primary" ? "Primary" : "Supporting"}
            </span>
          )}
          {citation.confidence != null && (
            <span className="text-[10px] tabular-nums text-slate-400">
              {Math.round(citation.confidence * 100)}%
            </span>
          )}
        </div>
        <CopyButton text={citation.snippet_original} />
      </div>

      {/* Translation */}
      {citation.snippet_translated && (
        <div className="mt-2.5 border-l-2 border-slate-200 pl-3 text-[12px] italic leading-relaxed text-slate-500">
          <span className="not-italic text-[9px] font-bold uppercase tracking-wider text-slate-400 mr-1.5">VI</span>
          {citation.snippet_translated}
        </div>
      )}
    </div>
  );
}

// ─── Source image viewer ──────────────────────────────────────────────────────

type BBoxOverlay = { x1: number; y1: number; x2: number; y2: number } | null;

function SourceImageViewer({
  url, alt, highlightBbox, citationIndex,
}: {
  url: string; alt: string; highlightBbox: BBoxOverlay; citationIndex?: number | null;
}) {
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgError, setImgError] = useState(false);
  const absoluteUrl = url.startsWith("http") ? url : `${API_BASE_URL}${url}`;

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width));
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (imgError) return null;
  const scale = imgSize && containerWidth ? containerWidth / imgSize.w : 1;

  return (
    <div className="border-b border-slate-100 bg-slate-900 p-3" ref={containerRef}>
      <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-400">Source Image</p>
      <div className="relative overflow-hidden rounded border border-slate-700">
        <img
          src={absoluteUrl} alt={alt} className="w-full object-contain"
          onLoad={(e) => { const img = e.currentTarget; setImgSize({ w: img.naturalWidth, h: img.naturalHeight }); }}
          onError={() => setImgError(true)}
        />
        {highlightBbox && imgSize && (
          <>
            <div
              className="pointer-events-none absolute rounded border-2 border-yellow-300 bg-yellow-300/25 shadow-[0_0_0_9999px_rgba(15,23,42,0.25)]"
              style={{
                left: `${highlightBbox.x1 * scale}px`, top: `${highlightBbox.y1 * scale}px`,
                width: `${(highlightBbox.x2 - highlightBbox.x1) * scale}px`,
                height: `${(highlightBbox.y2 - highlightBbox.y1) * scale}px`,
              }}
            />
            <span
              className="pointer-events-none absolute rounded bg-yellow-300 px-1.5 py-0.5 text-[10px] font-bold text-slate-900 shadow"
              style={{ left: `${highlightBbox.x1 * scale}px`, top: `${Math.max(0, highlightBbox.y1 * scale - 20)}px` }}
            >
              Evidence{citationIndex ? ` [${citationIndex}]` : ""}
            </span>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Block type label ─────────────────────────────────────────────────────────

const BLOCK_TYPE_LABELS: Record<string, string> = {
  heading: "Heading", table: "Table", image: "Figure",
  formula: "Formula", equation: "Formula", math: "Formula",
  code: "Code", caption: "Caption", list: "List",
};

// ─── Document Canvas block ────────────────────────────────────────────────────

function DocumentBlock({
  block, highlighted, keywords, exactSnippet,
}: {
  block: EvidenceBlock; highlighted: boolean;
  keywords: Set<string>; exactSnippet?: string | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (highlighted) {
      setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
    }
  }, [highlighted]);

  const bt = block.block_type.toLowerCase();
  const isTable   = bt === "table" || block.snippet_original.trimStart().startsWith("|");
  const isHeading = bt === "heading";
  const isFormula = bt === "formula" || bt === "equation" || bt === "math";
  const isCode    = bt === "code";
  const isImage   = bt === "image" || bt === "figure";
  const typeLabel = BLOCK_TYPE_LABELS[bt] ?? null;

  const content = (() => {
    if (isTable)   return <SnippetRenderer text={block.snippet_original} blockType={block.block_type} maxRows={6} compact />;
    if (isHeading) {
      const clean = block.snippet_original.replace(/^#{1,4}\s+/, "");
      return (
        <p className={`font-semibold leading-snug ${highlighted ? "text-slate-900" : "text-slate-700"}`}>
          <HighlightedText text={clean} exactSnippet={highlighted ? exactSnippet : null} />
        </p>
      );
    }
    if (isFormula) return (
      <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs italic leading-relaxed text-slate-700">
        {block.snippet_original}
      </pre>
    );
    if (isCode) return (
      <pre className="overflow-x-auto rounded bg-slate-900 px-3 py-2 font-mono text-xs leading-5 text-slate-100">
        {block.snippet_original}
      </pre>
    );
    if (isImage) return (
      <div className="flex items-start gap-2 text-slate-500">
        <Image size={13} className="mt-0.5 shrink-0 text-slate-400" />
        <p className="whitespace-pre-wrap text-xs italic leading-relaxed">{block.snippet_original || "Hình minh họa"}</p>
      </div>
    );
    return (
      <p className={`whitespace-pre-wrap leading-relaxed ${
        highlighted
          ? "text-[13px] text-slate-800"
          : "text-[12.5px] text-slate-600 hover:text-slate-800 transition-colors"
      }`}>
        <HighlightedText
          text={block.snippet_original}
          exactSnippet={highlighted ? exactSnippet : null}
          keywords={highlighted ? keywords : new Set()}
        />
      </p>
    );
  })();

  if (highlighted) {
    return (
      <div
        ref={ref}
        className="bg-primary/5 -mx-3 px-3 py-2.5 rounded-lg border-l-4 border-primary transition-all duration-300 shadow-[0_2px_8px_rgba(0,101,145,0.04)]"
      >
        {typeLabel && (
          <span className="mb-1.5 inline-block text-[9px] font-bold uppercase tracking-widest text-primary/60">
            {typeLabel}
          </span>
        )}
        {content}
        {block.confidence != null && (
          <div className="mt-2 flex items-center gap-1.5">
            <div className="h-1 w-10 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full rounded-full bg-primary/50" style={{ width: `${Math.round(block.confidence * 100)}%` }} />
            </div>
            <span className="text-[10px] tabular-nums text-slate-400">{Math.round(block.confidence * 100)}%</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div ref={ref} className="group">
      {typeLabel && (
        <span className="mb-0.5 inline-block text-[9px] font-semibold uppercase tracking-widest text-slate-300 group-hover:text-slate-400 transition-colors">
          {typeLabel}
        </span>
      )}
      {content}
    </div>
  );
}

// ─── Page view with search + collapsible context ──────────────────────────────

function PageView({
  pageData, highlightBlockId, keywords, exactSnippet,
}: {
  pageData: EvidencePageResponse; highlightBlockId?: string | null;
  keywords: Set<string>; exactSnippet?: string | null;
}) {
  const [searchText, setSearchText] = useState("");
  const [restExpanded, setRestExpanded] = useState(false);
  useEffect(() => { setRestExpanded(false); }, [highlightBlockId]);

  const searchLower = searchText.trim().toLowerCase();
  const isSearching = searchLower.length > 0;

  const citedBlock   = highlightBlockId ? (pageData.blocks.find((b) => b.block_id === highlightBlockId) ?? null) : null;
  const otherBlocks  = pageData.blocks.filter((b) => b.block_id !== highlightBlockId);
  const showSplit    = Boolean(citedBlock) && !isSearching;

  const matchFn = (b: EvidenceBlock) => !isSearching || (b.snippet_original || "").toLowerCase().includes(searchLower);
  const filteredOthers = otherBlocks.filter(matchFn);
  const filteredAll    = pageData.blocks.filter(matchFn);

  const matchCount = isSearching ? filteredAll.length : 0;

  return (
    <div>
      {/* Search bar */}
      {pageData.blocks.length > 3 && (
        <div className="sticky top-0 z-10 border-b border-slate-100 bg-white/95 backdrop-blur px-4 py-2">
          <div className="relative">
            <input
              type="text"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              placeholder="Tìm trong trang này…"
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 pr-16 text-xs text-text placeholder:text-slate-400 focus:border-primary/40 focus:bg-white focus:outline-none transition"
            />
            {isSearching ? (
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <span className="text-[10px] tabular-nums text-slate-400">{matchCount} kết quả</span>
                <button type="button" onClick={() => setSearchText("")} className="text-slate-400 hover:text-slate-600">
                  <X size={11} />
                </button>
              </div>
            ) : (
              <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-slate-300">
                {pageData.blocks.length} blocks
              </span>
            )}
          </div>
        </div>
      )}

      {/* Document Canvas */}
      <div className="p-4">
        <div className="bg-white rounded-xl border border-slate-200/80 shadow-sm p-5 space-y-4">
          {showSplit ? (
            <>
              {citedBlock && (
                <DocumentBlock block={citedBlock} highlighted keywords={keywords} exactSnippet={exactSnippet} />
              )}
              {otherBlocks.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => setRestExpanded((v) => !v)}
                    className="flex w-full items-center justify-between rounded-lg bg-slate-50 px-3.5 py-2.5 text-[11px] font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
                  >
                    <span className="flex items-center gap-1.5">
                      {restExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      {restExpanded ? "Thu gọn trang" : `Xem toàn bộ trang · ${otherBlocks.length} đoạn khác`}
                    </span>
                    <span className="text-[10px] text-slate-400">p.{pageData.page}</span>
                  </button>
                  {restExpanded && (
                    <div className="space-y-4 pt-1 border-t border-slate-100">
                      {filteredOthers.map((block) => (
                        <DocumentBlock key={block.block_id} block={block} highlighted={false} keywords={keywords} exactSnippet={null} />
                      ))}
                    </div>
                  )}
                </>
              )}
            </>
          ) : (
            filteredAll.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400">
                {pageData.blocks.length === 0 ? "Không có nội dung trên trang này." : "Không tìm thấy kết quả."}
              </p>
            ) : (
              filteredAll.map((block) => (
                <DocumentBlock
                  key={block.block_id} block={block}
                  highlighted={block.block_id === highlightBlockId}
                  keywords={keywords} exactSnippet={exactSnippet}
                />
              ))
            )
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Citation pills strip ─────────────────────────────────────────────────────

function ActiveEvidenceStrip({
  citations, currentIndex, onSelect, visitedIndices,
}: {
  citations: Citation[];
  currentIndex: number;
  onSelect: (i: number) => void;
  visitedIndices: Set<number>;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeChipRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!activeChipRef.current || !scrollRef.current) return;
    const container = scrollRef.current;
    const chip = activeChipRef.current;
    const chipLeft = chip.offsetLeft;
    const chipRight = chipLeft + chip.offsetWidth;
    const visLeft = container.scrollLeft;
    const visRight = visLeft + container.clientWidth;
    if (chipLeft < visLeft) {
      container.scrollTo({ left: chipLeft - 8, behavior: "smooth" });
    } else if (chipRight > visRight) {
      container.scrollTo({ left: chipRight - container.clientWidth + 8, behavior: "smooth" });
    }
  }, [currentIndex]);

  if (citations.length === 0) return null;
  const sourceCount  = new Set(citations.map((c) => c.doc_id)).size;
  const primaryCount = citations.filter((c) => c.role === "primary").length;
  const suppCount    = citations.length - primaryCount;
  const summaryParts = [
    primaryCount > 0 ? `${primaryCount} primary` : "",
    suppCount > 0     ? `${suppCount} supporting` : "",
  ].filter(Boolean);

  return (
    <div className="shrink-0 border-b border-slate-100 bg-white px-4 py-2.5">
      {/* Header row */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Sources</p>
          <span className="text-[10px] font-semibold text-slate-500 tabular-nums">
            {currentIndex >= 0 ? currentIndex + 1 : "—"} / {citations.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-400">
            {sourceCount} {sourceCount === 1 ? "doc" : "docs"} · {summaryParts.join(" · ")}
          </span>
          {/* Keyboard hint */}
          <span className="hidden sm:inline-flex items-center gap-0.5 text-[9px] text-slate-300" title="Dùng ← → để chuyển">
            <kbd className="rounded border border-slate-200 bg-slate-50 px-1 py-0.5 font-mono text-[9px] text-slate-400">←</kbd>
            <kbd className="rounded border border-slate-200 bg-slate-50 px-1 py-0.5 font-mono text-[9px] text-slate-400">→</kbd>
          </span>
        </div>
      </div>

      {/* Pills row */}
      <div ref={scrollRef} className="flex gap-1.5 overflow-x-auto pb-0.5 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
        {citations.map((item, index) => {
          const active  = index === currentIndex;
          const visited = visitedIndices.has(index) && !active;
          return (
            <button
              key={`${item.doc_id}-${item.block_id ?? item.page ?? index}-${index}`}
              ref={active ? activeChipRef : null}
              onClick={() => onSelect(index)}
              title={`${item.doc_name}${item.page ? ` · p.${item.page}` : ""}`}
              className={`group relative shrink-0 rounded-full border px-2.5 py-1 text-[11px] font-semibold transition-all whitespace-nowrap ${
                active
                  ? "border-primary/20 bg-primary/10 text-primary shadow-sm"
                  : "border-slate-200/60 bg-slate-50 text-slate-500 hover:bg-slate-100 hover:text-slate-700 hover:border-slate-300"
              }`}
            >
              {/* Visited checkmark */}
              {visited && (
                <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-emerald-500 text-white shadow-sm">
                  <Check size={8} strokeWidth={3} />
                </span>
              )}
              [{index + 1}]{item.page ? ` p.${item.page}` : ""} · {Math.round(item.confidence * 100)}%
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Page navigation footer ───────────────────────────────────────────────────

function PageNavFooter({
  currentPage, onPrev, onNext, canPrev, canNext, loading,
}: {
  currentPage: number; onPrev: () => void; onNext: () => void;
  canPrev: boolean; canNext: boolean; loading: boolean;
}) {
  return (
    <div className="shrink-0 border-t border-slate-100 bg-white px-4 py-2 flex items-center justify-between">
      <button
        type="button"
        onClick={onPrev}
        disabled={!canPrev || loading}
        className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-30"
      >
        <ChevronLeft size={12} />
        Trang trước
      </button>
      <span className="text-[11px] font-semibold text-slate-500 tabular-nums">
        {loading ? <Loader2 size={12} className="animate-spin text-primary/60" /> : `p.${currentPage}`}
      </span>
      <button
        type="button"
        onClick={onNext}
        disabled={!canNext || loading}
        className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-slate-500 transition hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-30"
      >
        Trang sau
        <ChevronRight size={12} />
      </button>
    </div>
  );
}

// ─── File type icon ───────────────────────────────────────────────────────────

function fileIcon(docName: string) {
  const ext = (docName.split(".").pop() || "").toLowerCase();
  if (/^(mp3|wav|m4a|ogg|flac|webm|aac)$/.test(ext))  return <FileAudio size={13} className="shrink-0 text-pink-400" />;
  if (/^(png|jpe?g|gif|webp|bmp)$/.test(ext))          return <Image size={13} className="shrink-0 text-purple-400" />;
  if (/^(xlsx?|csv)$/.test(ext))                        return <FileSpreadsheet size={13} className="shrink-0 text-emerald-500" />;
  if (/^pptx?$/.test(ext))                              return <Presentation size={13} className="shrink-0 text-amber-400" />;
  if (ext === "pdf")                                     return <FileText size={13} className="shrink-0 text-red-400" />;
  if (/^docx?$/.test(ext))                              return <FileText size={13} className="shrink-0 text-blue-400" />;
  return <FileText size={13} className="shrink-0 text-slate-400" />;
}

// ─── Main panel ───────────────────────────────────────────────────────────────

type EvidencePanelProps = {
  citation?: Citation | null;
  docId?: string | null;
  page?: number | null;
};

export default function EvidencePanel({ citation: citationProp, docId: docIdProp, page: pageProp }: EvidencePanelProps) {
  const { selectedCitation, setSelectedCitation, activeCitations, workspace } = useWorkspace();
  const citation = citationProp ?? selectedCitation;

  const currentIndex = activeCitations.findIndex(
    (c) => c.doc_id === citation?.doc_id && c.block_id === citation?.block_id,
  );

  // Visited tracking — remember which chips the user has opened
  const [visitedIndices, setVisitedIndices] = useState<Set<number>>(new Set());

  function navigateTo(i: number) {
    const c = activeCitations[i];
    if (c) {
      setSelectedCitation(c);
      setVisitedIndices((prev) => new Set(prev).add(i));
    }
  }

  // Reset visited when answer changes
  useEffect(() => { setVisitedIndices(new Set()); }, [activeCitations]);

  // Mark initial selection as visited
  useEffect(() => {
    if (currentIndex >= 0) {
      setVisitedIndices((prev) => new Set(prev).add(currentIndex));
    }
  }, [currentIndex]);

  // Resolve doc + page
  const targetDocId = docIdProp || citation?.doc_id || null;
  const basePage    = pageProp || citation?.page || citation?.pages?.[0] || null;
  const [displayPage, setDisplayPage] = useState<number | null>(null);

  // Sync displayed page when citation changes
  useEffect(() => { setDisplayPage(basePage); }, [basePage, targetDocId]);

  const targetPage      = displayPage;
  const highlightBlockId = displayPage === basePage ? (citation?.block_id || null) : null;
  const keywords         = citation ? extractKeywords(citation.snippet_original) : new Set<string>();
  const exactSnippet     = displayPage === basePage ? (citation?.snippet_original ?? null) : null;

  // Page data state
  const [pageData, setPageData] = useState<EvidencePageResponse | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const loadAttempt              = useRef(0);

  const doLoad = useCallback((docId: string, page: number) => {
    const attempt = ++loadAttempt.current;
    setLoading(true); setError(null); setPageData(null);
    loadEvidencePage(docId, page, workspace.ownerId, workspace.collectionId || null)
      .then((data) => { if (loadAttempt.current === attempt) setPageData(data); })
      .catch((err) => { if (loadAttempt.current === attempt) setError(err instanceof Error ? err.message : "Không tải được trang."); })
      .finally(() => { if (loadAttempt.current === attempt) setLoading(false); });
  }, [workspace.ownerId, workspace.collectionId]);

  useEffect(() => {
    if (!targetDocId || !targetPage) { setPageData(null); setError(null); return; }
    doLoad(targetDocId, targetPage);
  }, [targetDocId, targetPage, doLoad]);

  // ── Keyboard navigation ──
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Don't intercept when user is typing in an input / textarea
      const tag = (e.target as HTMLElement).tagName.toLowerCase();
      if (tag === "input" || tag === "textarea") return;

      if (e.key === "ArrowRight" && currentIndex < activeCitations.length - 1) {
        e.preventDefault();
        navigateTo(currentIndex + 1);
      }
      if (e.key === "ArrowLeft" && currentIndex > 0) {
        e.preventDefault();
        navigateTo(currentIndex - 1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, activeCitations.length]);

  // ── Empty state ──
  if (!citation && !targetDocId) {
    return (
      <aside className="panel flex h-full flex-col bg-white">
        <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-4">
          <FileText size={16} className="text-slate-400" />
          <h3 className="text-sm font-semibold text-slate-700">Evidence</h3>
        </div>
        <div className="flex flex-1 flex-col items-center justify-center gap-5 p-8 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-100">
            <FileText size={24} className="text-slate-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-700">Chưa có evidence</p>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-400 max-w-[220px]">
              Click vào số trích dẫn{" "}
              <span className="font-mono font-bold text-primary">[1]</span>{" "}
              trong câu trả lời để mở trang nguồn.
            </p>
          </div>
          <div className="w-full max-w-[240px] space-y-2 text-left">
            {["Hỏi câu hỏi trong chat", "Click [N] để mở evidence", "Dùng ← → để chuyển nguồn"].map((step, i) => (
              <div key={i} className="flex items-center gap-2.5 rounded-lg border border-slate-100 bg-slate-50/60 px-3 py-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[10px] font-bold text-primary">
                  {i + 1}
                </span>
                <span className="text-[11px] text-slate-500">{step}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>
    );
  }

  const docName        = pageData?.doc_name ?? citation?.doc_name ?? "Evidence";
  const citationNumber = currentIndex >= 0 ? currentIndex + 1 : null;
  const canPrevPage    = Boolean(targetPage && targetPage > 1);
  // Allow forward nav up to ~200 pages — backend will 404 if page doesn't exist
  const canNextPage    = Boolean(targetDocId && targetPage);

  return (
    <aside className="panel flex h-full flex-col bg-white">

      {/* ── Header ── */}
      <div className="shrink-0 border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-1.5">
          {fileIcon(docName)}
          <h3 className="truncate text-[13px] font-semibold text-slate-800" title={docName}>
            {docName}
          </h3>
        </div>
        <div className="mt-1 flex items-center gap-1 pl-5 text-[11px] text-slate-400">
          {citation?.source_language && (
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide">
              {citation.source_language}
            </span>
          )}
          {targetPage && (
            <>
              <ChevronRight size={9} className="text-slate-300" />
              <span>p.{targetPage}</span>
            </>
          )}
          {highlightBlockId && (
            <>
              <ChevronRight size={9} className="text-slate-300" />
              <span className="font-mono">#{highlightBlockId.slice(-6)}</span>
            </>
          )}
        </div>
      </div>

      {/* ── Citation pills ── */}
      <ActiveEvidenceStrip
        citations={activeCitations}
        currentIndex={currentIndex}
        onSelect={navigateTo}
        visitedIndices={visitedIndices}
      />

      {/* ── Matched snippet ── */}
      {citation && displayPage === basePage && (
        <MatchedSnippet citation={citation} ownerId={workspace.ownerId} />
      )}

      {/* ── Full page (Document Canvas) ── */}
      <div className="flex-1 overflow-y-auto">
        {loading && <PageSkeleton />}

        {error && (
          <div className="m-4 rounded-xl border border-red-100 bg-red-50/60 p-4">
            <p className="text-xs font-semibold text-red-600 mb-2">Không tải được trang</p>
            <p className="text-[11px] text-red-500 mb-3 leading-relaxed">{error}</p>
            <button
              type="button"
              onClick={() => targetDocId && targetPage && doLoad(targetDocId, targetPage)}
              className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-white px-3 py-1.5 text-[11px] font-semibold text-red-600 transition hover:bg-red-50"
            >
              <RotateCcw size={11} />
              Thử lại
            </button>
          </div>
        )}

        {pageData && !loading && (
          <>
            {pageData.raw_image_url && (
              <SourceImageViewer
                url={pageData.raw_image_url}
                alt={pageData.doc_name}
                highlightBbox={citation?.bbox ?? null}
                citationIndex={citationNumber}
              />
            )}
            <PageView
              pageData={pageData}
              highlightBlockId={highlightBlockId}
              keywords={keywords}
              exactSnippet={exactSnippet}
            />
          </>
        )}
      </div>

      {/* ── Page navigation footer ── */}
      {targetDocId && targetPage && (
        <PageNavFooter
          currentPage={targetPage}
          loading={loading}
          canPrev={canPrevPage}
          canNext={canNextPage}
          onPrev={() => setDisplayPage((p) => (p && p > 1 ? p - 1 : p))}
          onNext={() => setDisplayPage((p) => (p ? p + 1 : p))}
        />
      )}
    </aside>
  );
}
