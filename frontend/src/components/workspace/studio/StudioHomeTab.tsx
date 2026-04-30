import { useState } from "react";
import {
  AlertCircle, ChevronDown, ChevronUp, Download, FileText,
  Loader2, Sparkles, Trash2, Wand2,
} from "lucide-react";
import { summarizeCollection, buildStudyGuide, SummaryResponse, StudyGuideResponse } from "../../../api/client";
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
    <div className="rounded-xl border border-outline bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-outline bg-slate-50 px-3 py-2">
        <span className={accentClass}>{icon}</span>
        <span className="text-xs font-semibold uppercase tracking-wider text-muted flex-1">{label}</span>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? "Expand" : "Collapse"}
          className="flex items-center justify-center rounded p-1 text-muted hover:bg-slate-200 hover:text-text transition"
        >
          {collapsed ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </button>

        {/* Download */}
        <button
          onClick={onDownload}
          title="Download as .txt"
          className="flex items-center justify-center rounded p-1 text-muted hover:bg-slate-200 hover:text-text transition"
        >
          <Download size={13} />
        </button>

        {/* Delete */}
        <button
          onClick={onDelete}
          title="Remove artifact"
          className="flex items-center justify-center rounded p-1 text-muted hover:bg-red-50 hover:text-red-500 transition"
        >
          <Trash2 size={13} />
        </button>
      </div>

      {/* Body */}
      {!collapsed && <div className="p-4">{children}</div>}
    </div>
  );
}

// ─── Main tab ─────────────────────────────────────────────────────────────────

export default function StudioHomeTab() {
  const { workspace, scopedMaterialIds } = useWorkspace();
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
        scope: workspace.collectionId ? "collection" : "document",
      });
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
    <div className="flex h-full flex-col overflow-hidden bg-slate-50">
      {/* ── Toolbar (fixed) ── */}
      <div className="shrink-0 grid grid-cols-2 gap-3 border-b border-outline bg-white p-4">
        <button
          onClick={handleSummarize}
          disabled={loading !== null || !hasScope}
          className="flex flex-col items-center justify-center gap-2 rounded-xl border border-outline bg-white p-4 hover:border-primary/50 hover:bg-primary/5 transition disabled:opacity-50 disabled:pointer-events-none"
        >
          {loading === "summary"
            ? <Loader2 size={24} className="animate-spin text-primary" />
            : <Sparkles size={24} className="text-primary" />}
          <span className="text-xs font-semibold text-text">Summarize</span>
        </button>

        <button
          onClick={handleStudyGuide}
          disabled={loading !== null || !hasScope}
          className="flex flex-col items-center justify-center gap-2 rounded-xl border border-outline bg-white p-4 hover:border-primary/50 hover:bg-primary/5 transition disabled:opacity-50 disabled:pointer-events-none"
        >
          {loading === "guide"
            ? <Loader2 size={24} className="animate-spin text-secondary" />
            : <Wand2 size={24} className="text-secondary" />}
          <span className="text-xs font-semibold text-text">Study Guide</span>
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
            <p className="text-sm leading-relaxed whitespace-pre-wrap text-text">{summaryData.summary}</p>
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
            <FileText size={32} className="text-slate-200 mb-3" />
            <p className="text-sm font-semibold text-text">Studio Artifacts</p>
            <p className="text-xs text-muted mt-1 max-w-[200px]">
              Generate summaries, study guides, and comparisons from your sources.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
