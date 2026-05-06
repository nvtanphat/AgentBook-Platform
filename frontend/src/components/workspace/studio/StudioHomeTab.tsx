import { useState } from "react";
import {
  AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Download, FileText,
  Loader2, Sparkles, Trash2, Wand2,
} from "lucide-react";
import { CoverageReport, summarizeCollection, buildStudyGuide, SummaryResponse, StudyGuideResponse } from "../../../api/client";
import { useWorkspace } from "../../../state/workspace";

// ─── Download helper ──────────────────────────────────────────────────────────

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function summaryToText(data: SummaryResponse): string {
  return `SUMMARY\n${"=".repeat(60)}\n\n${data.summary}\n`;
}

function guideToText(data: StudyGuideResponse): string {
  const lines: string[] = ["STUDY GUIDE", "=".repeat(60), ""];
  if (data.overview) lines.push(data.overview, "");
  if (data.key_concepts?.length) {
    lines.push("KEY CONCEPTS", "-".repeat(40));
    data.key_concepts.forEach((c) => lines.push(`• ${c}`));
    lines.push("");
  }
  if (data.outline?.length) {
    lines.push("OUTLINE", "-".repeat(40));
    data.outline.forEach((item, i) => lines.push(`${i + 1}. ${item}`));
  }
  return lines.join("\n");
}

// ─── Artifact card ────────────────────────────────────────────────────────────

function ArtifactCard({
  icon, label, accentClass, onDelete, onDownload, children,
}: {
  icon: React.ReactNode;
  label: string;
  accentClass: string;
  onDelete: () => void;
  onDownload: () => void;
  children: React.ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="rounded-xl border border-outline/30 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-outline/30 bg-slate-50/80 px-3 py-2">
        <span className={accentClass}>{icon}</span>
        <span className="text-[10px] font-bold uppercase tracking-wider text-muted/70 flex-1">{label}</span>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand" : "Collapse"}
          className="flex items-center justify-center rounded-md p-1 text-muted/50 hover:bg-slate-200 hover:text-text transition"
        >
          {collapsed ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </button>

        {/* Download */}
        <button
          onClick={onDownload}
          title="Download as .txt"
          className="flex items-center justify-center rounded-md p-1 text-muted/50 hover:bg-slate-200 hover:text-text transition"
        >
          <Download size={13} />
        </button>

        {/* Delete */}
        <button
          onClick={onDelete}
          title="Remove artifact"
          className="flex items-center justify-center rounded-md p-1 text-muted/50 hover:bg-red-50 hover:text-red-500 transition"
        >
          <Trash2 size={13} />
        </button>
      </div>

      {/* Body */}
      {!collapsed && <div className="p-4">{children}</div>}
    </div>
  );
}

function CoveragePanel({ coverage }: { coverage?: CoverageReport | null }) {
  if (!coverage || coverage.requested_count === 0) return null;
  const complete = coverage.covered_count >= coverage.requested_count;
  const missing = coverage.sources.filter((source) => !source.covered);
  return (
    <div className={`mb-3 rounded-lg border px-3 py-2 ${complete ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {complete ? <CheckCircle2 size={14} className="text-emerald-600" /> : <AlertTriangle size={14} className="text-amber-600" />}
          <span className={`text-[11px] font-bold ${complete ? "text-emerald-700" : "text-amber-700"}`}>
            Covered sources: {coverage.covered_count}/{coverage.requested_count}
          </span>
        </div>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${complete ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
          {complete ? "Complete" : "Missing evidence"}
        </span>
      </div>
      {missing.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {missing.slice(0, 6).map((source) => (
            <span key={source.material_id} className="max-w-full truncate rounded border border-amber-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-amber-700" title={source.name}>
              {source.name}
            </span>
          ))}
          {missing.length > 6 && (
            <span className="rounded border border-amber-200 bg-white px-2 py-0.5 text-[10px] font-semibold text-amber-700">
              +{missing.length - 6}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function SummaryCitationStrip({ data }: { data: SummaryResponse }) {
  if (!data.citations.length) return null;
  const grouped = data.citations.reduce<Record<string, number>>((acc, citation) => {
    acc[citation.doc_name] = (acc[citation.doc_name] ?? 0) + 1;
    return acc;
  }, {});
  return (
    <div className="mt-3 border-t border-outline pt-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h4 className="text-[10px] font-bold uppercase tracking-wider text-muted">Evidence used</h4>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-muted">{data.citations.length} refs</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(grouped).map(([name, count]) => (
          <span key={name} className="max-w-full truncate rounded border border-outline bg-slate-50 px-2 py-0.5 text-[10px] font-semibold text-muted" title={name}>
            {name} · {count}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Main tab ─────────────────────────────────────────────────────────────────

export default function StudioHomeTab() {
  const { workspace, scopedMaterialIds, readySources, setActiveCitations, setSelectedCitation } = useWorkspace();
  const [loading, setLoading] = useState<"summary" | "guide" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [summaryData, setSummaryData] = useState<SummaryResponse | null>(null);
  const [guideData, setGuideData] = useState<StudyGuideResponse | null>(null);

  const hasScope = Boolean(workspace.collectionId || scopedMaterialIds.length);

  async function handleSummarize() {
    if (!hasScope) return;
    setLoading("summary");
    setError(null);
    try {
      const response = await summarizeCollection({
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        material_ids: scopedMaterialIds,
        scope: workspace.collectionId ? "collection" : "document",
        top_k: Math.max(workspace.topK, scopedMaterialIds.length || readySources.length || workspace.topK),
      });
      if (response.citations.length > 0) {
        setActiveCitations(response.citations);
        setSelectedCitation(response.citations[0]);
      }
      setSummaryData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate summary.");
    } finally {
      setLoading(null);
    }
  }

  async function handleStudyGuide() {
    if (!hasScope) return;
    setLoading("guide");
    setError(null);
    try {
      const response = await buildStudyGuide({
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        scope: "collection",
        format: "outline",
      });
      setGuideData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate study guide.");
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-slate-50/50">
      {/* ── Toolbar (fixed) ── */}
      <div className="shrink-0 grid grid-cols-2 gap-3 bg-white/80 p-4 section-divider" style={{ backdropFilter: 'blur(8px)' }}>
        <button
          onClick={handleSummarize}
          disabled={loading !== null || !hasScope}
          className="group flex flex-col items-center justify-center gap-2 rounded-xl border border-outline/30 bg-white p-4 hover:border-primary/40 hover:shadow-md transition-all disabled:opacity-50 disabled:pointer-events-none"
        >
          {loading === "summary"
            ? <Loader2 size={22} className="animate-spin text-primary" />
            : <Sparkles size={22} className="text-primary group-hover:scale-110 transition-transform" />}
          <span className="text-[11px] font-semibold text-text">Summarize</span>
        </button>

        <button
          onClick={handleStudyGuide}
          disabled={loading !== null || !hasScope}
          className="group flex flex-col items-center justify-center gap-2 rounded-xl border border-outline/30 bg-white p-4 hover:border-secondary/40 hover:shadow-md transition-all disabled:opacity-50 disabled:pointer-events-none"
        >
          {loading === "guide"
            ? <Loader2 size={22} className="animate-spin text-secondary" />
            : <Wand2 size={22} className="text-secondary group-hover:scale-110 transition-transform" />}
          <span className="text-[11px] font-semibold text-text">Study Guide</span>
        </button>
      </div>

      {/* ── Scrollable content ── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            <AlertCircle size={14} className="shrink-0 mt-0.5" /> {error}
          </div>
        )}

        {summaryData && (
          <ArtifactCard
            icon={<Sparkles size={13} />}
            label="Summary"
            accentClass="text-primary"
            onDelete={() => setSummaryData(null)}
            onDownload={() => downloadText("summary.txt", summaryToText(summaryData))}
          >
            <CoveragePanel coverage={summaryData.coverage} />
            <p className="text-sm leading-relaxed whitespace-pre-wrap text-text">{summaryData.summary}</p>
            <SummaryCitationStrip data={summaryData} />
          </ArtifactCard>
        )}

        {guideData && (
          <ArtifactCard
            icon={<Wand2 size={13} />}
            label="Study Guide"
            accentClass="text-secondary"
            onDelete={() => setGuideData(null)}
            onDownload={() => downloadText("study-guide.txt", guideToText(guideData))}
          >
            <div className="space-y-4">
              {guideData.overview && (
                <p className="text-sm leading-relaxed whitespace-pre-wrap text-text">{guideData.overview}</p>
              )}
              {guideData.key_concepts && guideData.key_concepts.length > 0 && (
                <div className="border-t border-outline pt-3">
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Key Concepts</h4>
                  <ul className="space-y-1">
                    {guideData.key_concepts.map((concept, idx) => (
                      <li key={idx} className="text-xs leading-relaxed text-text">• {concept}</li>
                    ))}
                  </ul>
                </div>
              )}
              {guideData.outline && guideData.outline.length > 0 && (
                <div className="border-t border-outline pt-3">
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Outline</h4>
                  <ol className="space-y-1">
                    {guideData.outline.map((item, idx) => (
                      <li key={idx} className="text-xs leading-relaxed text-text">{idx + 1}. {item}</li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          </ArtifactCard>
        )}

        {!summaryData && !guideData && !loading && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100">
              <FileText size={24} className="empty-state-icon text-slate-400" />
            </div>
            <p className="text-sm font-semibold text-text">Studio Artifacts</p>
            <p className="text-xs text-muted/60 mt-1.5 max-w-[220px] leading-relaxed">
              Generate summaries, study guides, and comparisons from your sources.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
