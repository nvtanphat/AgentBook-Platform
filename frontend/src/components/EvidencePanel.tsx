import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlignLeft, Check, ChevronDown, ChevronLeft, ChevronRight, ChevronUp,
  Copy, FileText, Heading, Image, Loader2, Table2, Sigma,
} from "lucide-react";
import { Citation, EvidenceBlock, EvidencePageResponse, loadEvidencePage } from "../api/client";
import SnippetRenderer from "./SnippetRenderer";
import MarkdownRenderer from "./MarkdownRenderer";
import { useWorkspace } from "../state/workspace";

// ─── Keyword highlight ────────────────────────────────────────────────────────

const STOP_WORDS = new Set([
  "và", "của", "là", "trong", "có", "được", "cho", "với", "các", "một",
  "này", "đó", "từ", "theo", "để", "như", "khi", "về", "bằng", "trên",
  "the", "a", "an", "of", "in", "is", "are", "to", "for", "with", "by",
  "that", "this", "on", "at", "as", "be", "from", "or", "and", "not",
]);

function extractKeywords(text: string): Set<string> {
  const words = text
    .toLowerCase()
    .replace(/[.,;:!?()\[\]{}"']/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 4 && !STOP_WORDS.has(w));
  return new Set(words);
}

function HighlightedText({ text, keywords, className }: { text: string; keywords: Set<string>; className?: string }) {
  if (keywords.size === 0) return <span className={className}>{text}</span>;

  const pattern = new RegExp(
    `(${Array.from(keywords).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
    "gi"
  );
  const parts = text.split(pattern);

  return (
    <span className={className}>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="rounded bg-yellow-200/70 px-0.5 text-yellow-900 not-italic">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </span>
  );
}

// ─── Block type icon ──────────────────────────────────────────────────────────

const BLOCK_TYPE_CONFIG: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  text:     { icon: <AlignLeft size={10} />, color: "bg-slate-100 text-slate-600",   label: "text" },
  heading:  { icon: <Heading size={10} />,   color: "bg-blue-50 text-blue-600",      label: "heading" },
  table:    { icon: <Table2 size={10} />,    color: "bg-purple-50 text-purple-600",  label: "table" },
  image:    { icon: <Image size={10} />,     color: "bg-orange-50 text-orange-600",  label: "image" },
  formula:  { icon: <Sigma size={10} />,     color: "bg-teal-50 text-teal-700",      label: "formula" },
  equation: { icon: <Sigma size={10} />,     color: "bg-teal-50 text-teal-700",      label: "equation" },
  math:     { icon: <Sigma size={10} />,     color: "bg-teal-50 text-teal-700",      label: "math" },
  code:     { icon: <AlignLeft size={10} />, color: "bg-slate-900/10 text-slate-700",label: "code" },
  caption:  { icon: <AlignLeft size={10} />, color: "bg-amber-50 text-amber-700",    label: "caption" },
};

function BlockTypeBadge({ type }: { type: string }) {
  const cfg = BLOCK_TYPE_CONFIG[type.toLowerCase()] ?? { icon: <FileText size={10} />, color: "bg-slate-100 text-slate-500", label: type };
  return (
    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide ${cfg.color}`}>
      {cfg.icon}{cfg.label}
    </span>
  );
}

// ─── Confidence bar ───────────────────────────────────────────────────────────

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const { barColor, textColor, tier } =
    pct >= 70
      ? { barColor: "bg-emerald-400", textColor: "text-emerald-600", tier: "High" }
      : pct >= 40
      ? { barColor: "bg-yellow-400",  textColor: "text-yellow-600",  tier: "Med"  }
      : { barColor: "bg-red-400",     textColor: "text-red-500",     tier: "Low"  };
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-[10px] font-bold tabular-nums ${textColor}`}>{pct}%</span>
      <span className="text-[9px] font-semibold uppercase tracking-wide text-muted/70">{tier}</span>
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
      title="Copy snippet"
      className="flex items-center gap-1 rounded px-1.5 py-1 text-[10px] text-muted hover:bg-slate-100 hover:text-text transition"
    >
      {copied ? <Check size={11} className="text-emerald-500" /> : <Copy size={11} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

// ─── Matched snippet (collapsible) ───────────────────────────────────────────

const SNIPPET_COLLAPSE_LINES = 6;

function MatchedSnippet({ citation }: { citation: Citation }) {
  const lines = citation.snippet_original.split("\n");
  const isLong = lines.length > SNIPPET_COLLAPSE_LINES;
  const [expanded, setExpanded] = useState(false);
  const text = isLong && !expanded
    ? lines.slice(0, SNIPPET_COLLAPSE_LINES).join("\n") + "…"
    : citation.snippet_original;

  const rolePill = citation.role === "primary"
    ? <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-primary">Primary</span>
    : citation.role === "supporting"
    ? <span className="rounded-full bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-muted">Supporting</span>
    : null;

  return (
    <div className="shrink-0 border-b border-outline bg-slate-50 px-4 py-3">
      <div className="mb-2 flex items-center gap-2">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted">Grounded evidence</p>
        {rolePill}
        <div className="ml-auto">
          <CopyButton text={citation.snippet_original} />
        </div>
      </div>
      <div className="rounded-md border border-secondary/30 bg-white px-3 py-2">
        <div className="border-l-2 border-secondary pl-2">
          <SnippetRenderer
            text={text}
            blockType={citation.block_type ?? undefined}
            maxRows={SNIPPET_COLLAPSE_LINES}
            textClassName="text-xs leading-5 text-text"
          />
        </div>
        {isLong && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-1.5 flex items-center gap-1 text-[11px] font-semibold text-primary hover:opacity-70"
          >
            {expanded ? <><ChevronUp size={12} /> Show less</> : <><ChevronDown size={12} /> Show more</>}
          </button>
        )}
      </div>
      {citation.snippet_translated && (
        <div className="mt-2 rounded border border-dashed border-outline bg-white px-3 py-2 text-xs italic leading-5 text-muted">
          <span className="mr-1.5 not-italic text-[9px] font-bold uppercase tracking-wide text-muted/60">VI</span>
          {citation.snippet_translated}
        </div>
      )}
    </div>
  );
}

// ─── Specialised block content renderers ─────────────────────────────────────

function HeadingBlock({ text, keywords }: { text: string; keywords: Set<string> }) {
  const level = text.match(/^(#{1,4})\s/)?.[1]?.length ?? 2;
  const clean = text.replace(/^#{1,4}\s+/, "");
  const cls = level === 1
    ? "text-base font-bold text-text font-heading"
    : level === 2
    ? "text-sm font-bold text-text font-heading"
    : "text-xs font-semibold uppercase tracking-wide text-muted";
  return (
    <div className={`border-l-2 border-primary/40 pl-2 ${cls}`}>
      <HighlightedText text={clean} keywords={keywords} />
    </div>
  );
}

function FormulaBlock({ text }: { text: string }) {
  return (
    <div className="rounded border border-teal-200 bg-teal-50 px-3 py-2">
      <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs italic leading-relaxed text-teal-900">
        {text}
      </pre>
    </div>
  );
}

function CodeBlock({ text }: { text: string }) {
  return (
    <pre className="overflow-x-auto rounded bg-slate-900 px-3 py-2 font-mono text-xs leading-5 text-slate-100">
      {text}
    </pre>
  );
}

function ImageBlock({ text }: { text: string }) {
  return (
    <div className="flex flex-col gap-1.5 rounded border border-orange-200 bg-orange-50 p-2">
      <div className="flex items-center gap-1.5">
        <Image size={12} className="text-orange-500 shrink-0" />
        <span className="text-[10px] font-semibold uppercase tracking-wide text-orange-600">Figure / Image</span>
      </div>
      {text && <p className="whitespace-pre-wrap text-xs italic leading-relaxed text-muted">{text}</p>}
    </div>
  );
}

// ─── Source image viewer ──────────────────────────────────────────────────────

import { API_BASE_URL } from "../api/client";

type BBoxOverlay = { x1: number; y1: number; x2: number; y2: number } | null;

function SourceImageViewer({ url, alt, highlightBbox }: { url: string; alt: string; highlightBbox: BBoxOverlay }) {
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imgError, setImgError] = useState(false);

  // Build absolute URL (backend may return relative path)
  const absoluteUrl = url.startsWith("http") ? url : `${API_BASE_URL}${url}`;

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setContainerWidth(entry.contentRect.width);
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  if (imgError) return null;

  const scale = imgSize && containerWidth ? containerWidth / imgSize.w : 1;

  return (
    <div className="border-b border-outline bg-slate-900 p-3" ref={containerRef}>
      <div className="mb-1.5 flex items-center gap-1.5">
        <Image size={11} className="text-orange-400" />
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Source Image</span>
      </div>
      <div className="relative overflow-hidden rounded border border-slate-700">
        <img
          src={absoluteUrl}
          alt={alt}
          className="w-full object-contain"
          onLoad={(e) => {
            const img = e.currentTarget;
            setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
          onError={() => setImgError(true)}
        />
        {/* Highlight bounding box overlay */}
        {highlightBbox && imgSize && (
          <div
            className="pointer-events-none absolute rounded border-2 border-yellow-400 bg-yellow-300/20"
            style={{
              left: `${highlightBbox.x1 * scale}px`,
              top: `${highlightBbox.y1 * scale}px`,
              width: `${(highlightBbox.x2 - highlightBbox.x1) * scale}px`,
              height: `${(highlightBbox.y2 - highlightBbox.y1) * scale}px`,
            }}
          />
        )}
      </div>
    </div>
  );
}

// ─── Block card ───────────────────────────────────────────────────────────────

function BlockCard({
  block, highlighted, keywords,
}: {
  block: EvidenceBlock;
  highlighted: boolean;
  keywords: Set<string>;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (highlighted) ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [highlighted]);

  const bt = block.block_type.toLowerCase();
  const isTable   = bt === "table" || block.snippet_original.trimStart().startsWith("|");
  const isHeading = bt === "heading";
  const isFormula = bt === "formula" || bt === "equation" || bt === "math";
  const isCode    = bt === "code";
  const isImage   = bt === "image" || bt === "figure";

  return (
    <div
      ref={ref}
      className={`rounded-lg border p-3 transition-all ${
        highlighted
          ? "border-secondary bg-teal-50 shadow-sm ring-1 ring-secondary/30"
          : "border-outline bg-white hover:border-slate-300"
      }`}
    >
      {/* Block meta row */}
      <div className="mb-2 flex items-center gap-2 flex-wrap">
        <BlockTypeBadge type={block.block_type} />
        <span className="text-[10px] text-muted">p.{block.page}</span>
        {block.confidence != null && (
          <div className="ml-auto">
            <ConfidenceBar value={block.confidence} />
          </div>
        )}
      </div>

      {/* Block content — render by block type */}
      {isTable ? (
        <SnippetRenderer text={block.snippet_original} blockType={block.block_type} maxRows={6} compact />
      ) : isHeading ? (
        <HeadingBlock text={block.snippet_original} keywords={highlighted ? keywords : new Set()} />
      ) : isFormula ? (
        <FormulaBlock text={block.snippet_original} />
      ) : isCode ? (
        <CodeBlock text={block.snippet_original} />
      ) : isImage ? (
        <ImageBlock text={block.snippet_original} />
      ) : (
        <p className="whitespace-pre-wrap text-xs leading-relaxed text-text">
          <HighlightedText text={block.snippet_original} keywords={highlighted ? keywords : new Set()} />
        </p>
      )}
    </div>
  );
}

// ─── Page view ────────────────────────────────────────────────────────────────

function PageView({
  pageData, highlightBlockId, keywords,
}: {
  pageData: EvidencePageResponse;
  highlightBlockId?: string | null;
  keywords: Set<string>;
}) {
  return (
    <div className="space-y-2 p-4">
      {pageData.blocks.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted">No blocks found on this page.</p>
      ) : (
        pageData.blocks.map((block) => (
          <BlockCard
            key={block.block_id}
            block={block}
            highlighted={block.block_id === highlightBlockId}
            keywords={keywords}
          />
        ))
      )}
    </div>
  );
}

// ─── Citation nav ─────────────────────────────────────────────────────────────

function CitationNav({
  citations, currentIndex, onSelect,
}: {
  citations: Citation[];
  currentIndex: number;
  onSelect: (i: number) => void;
}) {
  if (citations.length <= 1) return null;
  const cur = citations[currentIndex];
  return (
    <div className="flex shrink-0 items-center gap-2 border-b border-outline bg-white px-4 py-2">
      <button
        disabled={currentIndex === 0}
        onClick={() => onSelect(currentIndex - 1)}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-outline text-muted hover:text-text disabled:opacity-30"
      >
        <ChevronLeft size={13} />
      </button>
      <div className="min-w-0 flex-1 text-center">
        <p className="truncate text-[10px] font-semibold text-text" title={cur?.doc_name}>
          {cur?.doc_name ?? "—"}
        </p>
        <p className="text-[9px] text-muted">
          [{currentIndex + 1}/{citations.length}]
          {cur?.page ? ` · p.${cur.page}` : ""}
          {cur?.role ? ` · ${cur.role}` : ""}
        </p>
      </div>
      <button
        disabled={currentIndex === citations.length - 1}
        onClick={() => onSelect(currentIndex + 1)}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-outline text-muted hover:text-text disabled:opacity-30"
      >
        <ChevronRight size={13} />
      </button>
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────────

type EvidencePanelProps = {
  citation?: Citation | null;
  docId?: string | null;
  page?: number | null;
};

export default function EvidencePanel({ citation: citationProp, docId: docIdProp, page: pageProp }: EvidencePanelProps) {
  const { selectedCitation, setSelectedCitation, activeCitations, workspace } = useWorkspace();

  // Resolve current citation: prop → selected → null
  const citation = citationProp ?? selectedCitation;

  // Citation navigation index
  const currentIndex = activeCitations.findIndex(
    (c) => c.doc_id === citation?.doc_id && c.block_id === citation?.block_id
  );
  const navIndex = currentIndex >= 0 ? currentIndex : 0;

  function navigateTo(i: number) {
    const c = activeCitations[i];
    if (c) setSelectedCitation(c);
  }

  // Resolve doc + page
  const targetDocId = docIdProp || citation?.doc_id || null;
  const targetPage = pageProp || citation?.page || citation?.pages?.[0] || null;
  const highlightBlockId = citation?.block_id || null;

  // Keywords extracted from the citation snippet for highlighting
  const keywords = citation ? extractKeywords(citation.snippet_original) : new Set<string>();

  // Page load state
  const [pageData, setPageData] = useState<EvidencePageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!targetDocId || !targetPage) {
      setPageData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPageData(null);
    loadEvidencePage(targetDocId, targetPage, workspace.ownerId, workspace.collectionId || null)
      .then((data) => { if (!cancelled) setPageData(data); })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load evidence page."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [targetDocId, targetPage, workspace.ownerId, workspace.collectionId]);

  // Empty state
  if (!citation && !targetDocId) {
    return (
      <aside className="panel flex h-full flex-col bg-surface-low">
        <div className="flex items-center gap-2 border-b border-outline bg-white px-5 py-4">
          <FileText size={18} className="text-primary" />
          <h3 className="font-heading text-lg font-semibold text-text">Evidence</h3>
        </div>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
          <FileText size={32} className="text-slate-200" />
          <p className="text-sm font-semibold text-text">No evidence selected</p>
          <p className="text-xs text-muted">Click a citation [N] in the chat to inspect grounded evidence.</p>
        </div>
      </aside>
    );
  }

  const docName = pageData?.doc_name ?? citation?.doc_name ?? "Evidence";

  return (
    <aside className="panel flex h-full flex-col bg-surface-low">
      {/* ── Header ── */}
      <div className="flex shrink-0 items-start justify-between border-b border-outline bg-white px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <FileText size={14} className="shrink-0 text-primary" />
            <h3 className="truncate text-sm font-semibold text-text" title={docName}>{docName}</h3>
          </div>
          {/* Source chain breadcrumb */}
          <div className="mt-1.5 flex items-center gap-1 pl-5">
            {citation?.source_language && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-muted">
                {citation.source_language}
              </span>
            )}
            {targetPage && (
              <>
                <ChevronRight size={9} className="text-muted/40" />
                <span className="text-[10px] font-semibold text-muted">p.{targetPage}</span>
              </>
            )}
            {highlightBlockId && (
              <>
                <ChevronRight size={9} className="text-muted/40" />
                <span className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[9px] text-muted">
                  {highlightBlockId.slice(-8)}
                </span>
              </>
            )}
          </div>
        </div>
        {citation && (
          <div className="ml-3 shrink-0 flex flex-col items-end gap-1">
            <ConfidenceBar value={citation.confidence} />
          </div>
        )}
      </div>

      {/* ── Citation navigation ── */}
      <CitationNav citations={activeCitations} currentIndex={navIndex} onSelect={navigateTo} />

      {/* ── Matched snippet ── */}
      {citation && <MatchedSnippet citation={citation} />}

      {/* ── Full page ── */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted">
            <Loader2 size={16} className="animate-spin text-primary" />
            Loading page blocks…
          </div>
        )}
        {error && (
          <div className="m-4 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            {error}
          </div>
        )}
        {pageData && !loading && (
          <>
            {/* ── Source image viewer (for uploaded image files) ── */}
            {pageData.raw_image_url && (
              <SourceImageViewer
                url={pageData.raw_image_url}
                alt={pageData.doc_name}
                highlightBbox={citation?.bbox ?? null}
              />
            )}

            <div className="flex items-center justify-between border-b border-outline bg-white px-4 py-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">
                {pageData.raw_image_url ? "OCR blocks" : `Page ${pageData.page} · all blocks`}
              </span>
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-muted">
                {pageData.blocks.length}
              </span>
            </div>
            <PageView pageData={pageData} highlightBlockId={highlightBlockId} keywords={keywords} />
          </>
        )}
      </div>
    </aside>
  );
}
