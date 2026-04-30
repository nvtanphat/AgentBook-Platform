import { FormEvent, KeyboardEvent, useState } from "react";
import { AlertCircle, ChevronDown, ChevronUp, Loader2, Plus, X } from "lucide-react";
import { Citation, CompareResponse, compareDocuments } from "../../../api/client";
import { useWorkspace } from "../../../state/workspace";

const DEFAULT_DIMENSIONS = ["definition", "intuition", "example", "limitation"];

// ─── Dimension tag input ──────────────────────────────────────────────────────

function DimensionTags({
  tags, onChange,
}: {
  tags: string[];
  onChange: (tags: string[]) => void;
}) {
  const [input, setInput] = useState("");

  function add() {
    const trimmed = input.trim();
    if (trimmed && !tags.includes(trimmed)) onChange([...tags, trimmed]);
    setInput("");
  }

  function remove(tag: string) {
    onChange(tags.filter((t) => t !== tag));
  }

  function handleKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") { e.preventDefault(); add(); }
    if (e.key === "Backspace" && !input && tags.length) remove(tags[tags.length - 1]);
  }

  return (
    <div className="flex flex-wrap gap-1.5 rounded-md border border-outline bg-white px-2 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all min-h-[38px]">
      {tags.map((tag) => (
        <span key={tag} className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">
          {tag}
          <button type="button" onClick={() => remove(tag)} className="hover:text-red-500 transition">
            <X size={10} />
          </button>
        </span>
      ))}
      <div className="flex items-center gap-1 flex-1 min-w-[100px]">
        <input
          className="flex-1 bg-transparent text-xs outline-none placeholder:text-slate-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={tags.length === 0 ? "Type a dimension, press Enter…" : "Add more…"}
        />
        {input.trim() && (
          <button type="button" onClick={add} className="text-muted hover:text-primary transition">
            <Plus size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Confidence pill ──────────────────────────────────────────────────────────

function ConfidencePill({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 70 ? "bg-emerald-100 text-emerald-700" : pct >= 40 ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700";
  return <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${cls}`}>{pct}%</span>;
}

// ─── Result card ─────────────────────────────────────────────────────────────

const CARD_COLLAPSE_LINES = 5;

function ResultCard({
  dimension, value, source, citation, confidence, onCitationClick,
}: {
  dimension: string;
  value: string;
  source: string;
  citation: Citation | null;
  confidence: number;
  onCitationClick: (c: Citation) => void;
}) {
  const lines = value.split("\n");
  const isLong = lines.length > CARD_COLLAPSE_LINES || value.length > 400;
  const [expanded, setExpanded] = useState(false);
  const displayValue = isLong && !expanded
    ? (value.length > 400 ? value.slice(0, 400) + "…" : lines.slice(0, CARD_COLLAPSE_LINES).join("\n") + "…")
    : value;

  const noEvidence = confidence === 0;

  return (
    <div className={`rounded-xl border bg-white shadow-sm overflow-hidden ${noEvidence ? "opacity-60" : ""}`}>
      {/* Card header */}
      <div className="flex items-center justify-between border-b border-outline bg-slate-50 px-4 py-2.5">
        <span className="text-xs font-bold capitalize text-text">{dimension}</span>
        <ConfidencePill value={confidence} />
      </div>

      {/* Value */}
      <div className="px-4 py-3">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-text">{displayValue}</p>
        {isLong && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-1.5 flex items-center gap-1 text-[11px] font-semibold text-primary hover:opacity-70"
          >
            {expanded ? <><ChevronUp size={11} /> Show less</> : <><ChevronDown size={11} /> Show more</>}
          </button>
        )}
      </div>

      {/* Footer: source + citation */}
      {!noEvidence && (
        <div className="flex items-center justify-between border-t border-outline bg-slate-50/50 px-4 py-2">
          <span className="truncate text-[11px] text-muted max-w-[60%]" title={source}>{source}</span>
          {citation && (
            <button
              onClick={() => onCitationClick(citation)}
              className="flex shrink-0 items-center gap-1 rounded-full border border-outline bg-white px-2 py-0.5 text-[11px] font-semibold text-muted hover:border-primary/40 hover:text-primary transition"
            >
              p.{citation.page ?? "?"}
              <span className="ml-0.5 opacity-50">↗</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main tab ─────────────────────────────────────────────────────────────────

type CompareTabProps = {
  onOpenEvidence?: () => void;
};

export default function CompareTab({ onOpenEvidence }: CompareTabProps) {
  const { workspace, scopedMaterialIds, setSelectedCitation, setActiveCitations } = useWorkspace();
  const [topic, setTopic] = useState("");
  const [dimensions, setDimensions] = useState<string[]>(DEFAULT_DIMENSIONS);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleCitationClick(citation: Citation) {
    setSelectedCitation(citation);
    if (result) setActiveCitations(result.citations);
    onOpenEvidence?.();
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!topic.trim() || dimensions.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await compareDocuments({
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        material_ids: workspace.collectionId ? [] : scopedMaterialIds,
        topic: topic.trim(),
        dimensions,
        top_k: workspace.topK,
      });
      if (response.citations.length > 0) {
        setActiveCitations(response.citations);
        setSelectedCitation(response.citations[0]);
      }
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setLoading(false);
    }
  }

  const hasScope = Boolean(workspace.collectionId || scopedMaterialIds.length);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-slate-50">
      {/* ── Form ── */}
      <form className="shrink-0 space-y-3 border-b border-outline bg-white p-4" onSubmit={handleSubmit}>
        <label className="block">
          <span className="label-caps">Topic to compare</span>
          <input
            className="mt-1 w-full rounded-md border border-outline px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary transition"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. Dropout vs Batch Normalization"
          />
        </label>

        <div>
          <span className="label-caps">Dimensions</span>
          <div className="mt-1">
            <DimensionTags tags={dimensions} onChange={setDimensions} />
          </div>
          <p className="mt-1 text-[10px] text-muted">Press Enter to add · Backspace to remove last</p>
        </div>

        <button
          type="submit"
          className="flex w-full items-center justify-center gap-2 rounded-md bg-primary py-2 text-sm font-semibold text-white transition hover:bg-primary/90 disabled:opacity-50"
          disabled={loading || !topic.trim() || dimensions.length === 0 || !hasScope}
        >
          {loading ? <><Loader2 className="animate-spin" size={14} /> Comparing…</> : "Compare Sources"}
        </button>

        {!hasScope && (
          <p className="text-center text-[11px] text-amber-600">Select a collection first to enable comparison.</p>
        )}
      </form>

      {/* ── Results ── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            <AlertCircle size={14} className="mt-0.5 shrink-0" /> {error}
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-muted">
            <Loader2 size={24} className="animate-spin text-primary" />
            <p className="text-xs">Retrieving and synthesizing {dimensions.length} dimension{dimensions.length !== 1 ? "s" : ""}…</p>
            <p className="text-[10px] text-muted/70">This may take a moment</p>
          </div>
        )}

        {result && !loading && (
          <>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
                {result.topic}
              </h3>
              <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-semibold text-muted">
                {result.comparison_table.length} dims
              </span>
            </div>

            {result.comparison_table.map((row) => (
              <ResultCard
                key={row.dimension}
                dimension={row.dimension}
                value={row.value}
                source={row.source}
                citation={row.citation}
                confidence={row.confidence}
                onCitationClick={handleCitationClick}
              />
            ))}

            {result.conflicts.length > 0 && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                <h4 className="mb-2 text-xs font-bold text-amber-800">Conflicts Detected</h4>
                <ul className="space-y-1">
                  {result.conflicts.map((item) => (
                    <li key={item} className="text-xs text-amber-700">· {item}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
            <p className="text-sm font-semibold text-text">Compare across sources</p>
            <p className="text-xs text-muted max-w-[200px]">
              Enter a topic and dimensions to get a grounded side-by-side analysis.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
