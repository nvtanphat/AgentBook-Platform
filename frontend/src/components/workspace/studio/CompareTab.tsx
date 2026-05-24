import { FormEvent, KeyboardEvent, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Copy, Download, GitBranch, Loader2, Plus, X } from "lucide-react";
import { Citation, CompareMatrixCell, CompareResponse, CoverageReport, compareDocuments } from "../../../api/client";
import { useWorkspace } from "../../../state/workspace";

const DEFAULT_DIMENSIONS = ["ý chính", "điểm giống", "điểm khác", "bằng chứng"];
const DIMENSION_PRESETS = [
  { label: "Tổng quan", values: ["ý chính", "định nghĩa", "ví dụ", "bằng chứng"] },
  { label: "So sánh", values: ["điểm giống", "điểm khác", "ưu điểm", "hạn chế"] },
  { label: "Kiểm chứng", values: ["luận điểm", "bằng chứng", "nguồn trích dẫn", "độ tin cậy"] },
];

function citationKey(citation: Citation): string {
  return `${citation.doc_id}:${citation.page ?? "None"}:${citation.block_id ?? "None"}`;
}

function DimensionTags({ tags, onChange }: { tags: string[]; onChange: (tags: string[]) => void }) {
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
    if (e.key === "Enter") {
      e.preventDefault();
      add();
    }
    if (e.key === "Backspace" && !input && tags.length) remove(tags[tags.length - 1]);
  }

  return (
    <div className="flex min-h-[38px] flex-wrap gap-1.5 rounded-md border border-outline bg-white px-2 py-1.5 transition-all focus-within:border-primary focus-within:ring-1 focus-within:ring-primary">
      {tags.map((tag) => (
        <span key={tag} className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">
          {tag}
          <button type="button" onClick={() => remove(tag)} className="transition hover:text-red-500" aria-label={`Xóa ${tag}`}>
            <X size={10} />
          </button>
        </span>
      ))}
      <div className="flex min-w-[100px] flex-1 items-center gap-1">
        <input
          className="flex-1 bg-transparent text-xs outline-none placeholder:text-slate-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={tags.length === 0 ? "Nhập khía cạnh, Enter để thêm" : "Thêm khía cạnh"}
        />
        {input.trim() && (
          <button type="button" onClick={add} className="text-muted transition hover:text-primary" aria-label="Thêm khía cạnh">
            <Plus size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

function ConfidencePill({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls = pct >= 70 ? "bg-emerald-100 text-emerald-700" : pct >= 40 ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700";
  return <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${cls}`}>{pct}%</span>;
}

function CoveragePanel({ coverage }: { coverage?: CoverageReport | null }) {
  if (!coverage || coverage.requested_count === 0) return null;
  const complete = coverage.covered_count >= coverage.requested_count;
  const missing = coverage.sources.filter((source) => !source.covered);
  return (
    <div className={`rounded-lg border px-3 py-2 ${complete ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {complete ? <CheckCircle2 size={14} className="text-emerald-600" /> : <AlertTriangle size={14} className="text-amber-600" />}
          <span className={`text-[11px] font-bold ${complete ? "text-emerald-700" : "text-amber-700"}`}>
            Độ phủ bằng chứng: {coverage.covered_count}/{coverage.requested_count}
          </span>
        </div>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${complete ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
          {complete ? "Đủ tất cả nguồn" : "Thiếu một số nguồn"}
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

function DimensionCoveragePanel({ result }: { result: CompareResponse }) {
  const rows = result.dimension_coverage ?? [];
  if (!rows.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {rows.map((row) => {
        const complete = row.covered_count >= row.requested_count;
        return (
          <span
            key={row.dimension}
            className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${
              complete ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"
            }`}
          >
            {row.dimension}: {row.covered_count}/{row.requested_count}
          </span>
        );
      })}
    </div>
  );
}

function CompareCellView({
  cell,
  citations,
  onCitationClick,
}: {
  cell?: CompareMatrixCell;
  citations: Citation[];
  onCitationClick: (c: Citation) => void;
}) {
  if (!cell) return <span className="text-muted">-</span>;
  const citationById = new Map(citations.map((citation) => [citationKey(citation), citation]));
  const cellCitations = cell.citation_ids.map((id) => citationById.get(id)).filter((citation): citation is Citation => Boolean(citation));

  return (
    <div className="space-y-2">
      <div className={cell.missing_evidence ? "text-amber-700" : "text-text"}>
        {cell.value}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <ConfidencePill value={cell.confidence} />
        {cell.missing_evidence && (
          <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold text-amber-700">
            thiếu bằng chứng
          </span>
        )}
        {cellCitations.slice(0, 2).map((citation, index) => (
          <button
            key={`${citationKey(citation)}-${index}`}
            type="button"
            onClick={() => onCitationClick(citation)}
            className="rounded-full border border-outline bg-white px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
          >
            p.{citation.page ?? "?"}
          </button>
        ))}
      </div>
    </div>
  );
}

function CompareMatrixTable({
  result,
  onCitationClick,
}: {
  result: CompareResponse;
  onCitationClick: (c: Citation) => void;
}) {
  const sources = result.sources ?? [];
  const matrix = result.matrix ?? {};
  const dimensions = result.dimension_coverage?.map((row) => row.dimension) ?? [];
  if (!sources.length || !dimensions.length) return null;

  return (
    <div className="rounded-lg border border-outline bg-surface shadow-sm overflow-hidden">
      <div className="border-b border-outline bg-surface-low px-3 py-2 flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted flex-1">
          Bảng so sánh theo nguồn · mỗi ô có citation riêng
        </span>
        <span className="text-[10px] text-muted">{sources.length} × {dimensions.length}</span>
      </div>
      <div className="w-full overflow-x-auto">
        <table className="w-max min-w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-outline bg-surface-low">
              <th className="sticky left-0 z-10 w-[190px] min-w-[190px] border-r border-outline bg-surface-low px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted">
                Nguồn
              </th>
              {dimensions.map((dimension) => (
                <th key={dimension} className="w-[300px] min-w-[300px] border-r border-outline px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted last:border-r-0">
                  {dimension}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-outline">
            {sources.map((source, rowIdx) => (
              <tr key={source.source_id} className={`align-top transition-colors hover:bg-primary/5 ${rowIdx % 2 === 1 ? "bg-surface-low/50" : ""}`}>
                <th className="sticky left-0 z-10 w-[190px] min-w-[190px] border-r border-outline bg-inherit px-3 py-3 text-xs font-semibold leading-snug text-text">
                  <p className="break-words">{source.name}</p>
                </th>
                {dimensions.map((dimension) => (
                  <td key={`${source.source_id}-${dimension}`} className="w-[300px] min-w-[300px] border-r border-outline px-3 py-3 text-xs leading-relaxed last:border-r-0">
                    <div className="max-h-40 overflow-y-auto pr-1">
                      <CompareCellView cell={matrix[source.source_id]?.[dimension]} citations={result.citations} onCitationClick={onCitationClick} />
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


// ─── Export helpers ────────────────────────────────────────────────────────

function exportToMarkdown(result: CompareResponse): string {
  const lines: string[] = [`# So sánh: ${result.topic}`, ""];
  const sources = result.sources ?? [];
  const dimensions = result.dimension_coverage?.map((d) => d.dimension) ?? [];
  const matrix = result.matrix ?? {};
  if (sources.length && dimensions.length) {
    lines.push("| Nguồn | " + dimensions.join(" | ") + " |");
    lines.push("|" + "---|".repeat(dimensions.length + 1));
    for (const src of sources) {
      const row = [src.name];
      for (const dim of dimensions) {
        const cell = matrix[src.source_id]?.[dim];
        const val = (cell?.value || "—").replace(/\n+/g, " ").replace(/\|/g, "\\|");
        row.push(val);
      }
      lines.push("| " + row.join(" | ") + " |");
    }
  } else {
    for (const row of result.comparison_table) {
      lines.push(`**${row.dimension}** · *${row.source}*: ${row.value}`);
    }
  }
  if (result.conflicts?.length) {
    lines.push("", "## Điểm có thể mâu thuẫn");
    for (const c of result.conflicts) lines.push(`- ${c}`);
  }
  return lines.join("\n");
}

function exportToCsv(result: CompareResponse): string {
  const sources = result.sources ?? [];
  const dimensions = result.dimension_coverage?.map((d) => d.dimension) ?? [];
  const matrix = result.matrix ?? {};
  const esc = (s: string) => `"${(s || "").replace(/"/g, '""').replace(/\n/g, " ")}"`;
  const lines: string[] = ["Nguồn," + dimensions.map(esc).join(",")];
  for (const src of sources) {
    const row = [esc(src.name)];
    for (const dim of dimensions) {
      const cell = matrix[src.source_id]?.[dim];
      row.push(esc(cell?.value || ""));
    }
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

function downloadFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime + ";charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const CARD_COLLAPSE_LINES = 5;

function ResultCard({
  dimension,
  value,
  source,
  citation,
  confidence,
  missingEvidence,
  onCitationClick,
}: {
  dimension: string;
  value: string;
  source: string;
  citation: Citation | null;
  confidence: number;
  missingEvidence?: boolean;
  onCitationClick: (c: Citation) => void;
}) {
  const lines = value.split("\n");
  const isLong = lines.length > CARD_COLLAPSE_LINES || value.length > 400;
  const [expanded, setExpanded] = useState(false);
  const displayValue = isLong && !expanded
    ? (value.length > 400 ? `${value.slice(0, 400)}...` : `${lines.slice(0, CARD_COLLAPSE_LINES).join("\n")}...`)
    : value;
  const noEvidence = confidence === 0 || missingEvidence;

  return (
    <div className={`overflow-hidden rounded-lg border bg-white shadow-sm ${noEvidence ? "opacity-70" : ""}`}>
      <div className="flex items-center justify-between gap-3 border-b border-outline bg-slate-50 px-4 py-2.5">
        <div className="min-w-0">
          <span className="block truncate text-xs font-bold capitalize text-text">{dimension}</span>
          <span className="text-[10px] text-muted">{source}</span>
        </div>
        <ConfidencePill value={confidence} />
      </div>

      <div className="px-4 py-3">
        <p className={`whitespace-pre-wrap text-sm leading-relaxed ${noEvidence ? "text-amber-700" : "text-text"}`}>{displayValue}</p>
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="mt-1.5 flex items-center gap-1 text-[11px] font-semibold text-primary hover:opacity-70"
          >
            {expanded ? <><ChevronUp size={11} /> Thu gọn</> : <><ChevronDown size={11} /> Xem thêm</>}
          </button>
        )}
      </div>

      {!noEvidence && citation && (
        <div className="flex items-center justify-between border-t border-outline bg-slate-50/50 px-4 py-2">
          <span className="max-w-[60%] truncate text-[11px] text-muted" title={source}>{source}</span>
          <button
            type="button"
            onClick={() => onCitationClick(citation)}
            className="flex shrink-0 items-center gap-1 rounded-full border border-outline bg-white px-2 py-0.5 text-[11px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
          >
            p.{citation.page ?? "?"}
            <span className="ml-0.5 opacity-50">mở</span>
          </button>
        </div>
      )}
    </div>
  );
}

type CompareTabProps = {
  onOpenEvidence?: () => void;
};

function buildDefaultTopic(collectionName: string, readySources: Array<{ materialId: string; name: string }>, selectedIds: string[], selectedOnly: boolean) {
  const sourcePool = selectedOnly ? readySources.filter((source) => selectedIds.includes(source.materialId)) : [];
  const names = sourcePool.map((source) => source.name).filter(Boolean).slice(0, 3);
  if (names.length >= 2) return `So sánh ${names.join(" và ")}`;
  if (names.length === 1) return `Phân tích và đối chiếu ${names[0]}`;
  return collectionName ? `So sánh các tài liệu trong ${collectionName}` : "So sánh các tài liệu đang chọn";
}

export default function CompareTab({ onOpenEvidence }: CompareTabProps) {
  const { workspace, scopedMaterialIds, sourceScopeMode, selectedSourceIds, readySources, setSelectedCitation, setActiveCitations } = useWorkspace();
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
    const compareTopic = topic.trim() || buildDefaultTopic(workspace.collectionName, readySources, selectedSourceIds, sourceScopeMode === "selected");
    if (!compareTopic || dimensions.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await compareDocuments({
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        material_ids: scopedMaterialIds,
        topic: compareTopic,
        dimensions,
        top_k: workspace.topK,
      });
      if (response.citations.length > 0) {
        setActiveCitations(response.citations);
        setSelectedCitation(response.citations[0]);
      }
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không thể tạo bảng so sánh.");
    } finally {
      setLoading(false);
    }
  }

  const hasScope = Boolean(workspace.collectionId ? sourceScopeMode === "all" || scopedMaterialIds.length : scopedMaterialIds.length);
  const defaultTopic = buildDefaultTopic(workspace.collectionName, readySources, selectedSourceIds, sourceScopeMode === "selected");
  const sourceHint = sourceScopeMode === "selected"
    ? `${scopedMaterialIds.length} nguồn đang chọn`
    : readySources.length
      ? `${readySources.length} nguồn sẵn sàng`
      : workspace.collectionName || "collection hiện tại";
  const hasMatrix = Boolean(result?.sources?.length && result?.matrix && Object.keys(result.matrix).length > 0);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-surface-low">
      <form className="shrink-0 space-y-3 border-b border-outline bg-surface p-4" onSubmit={handleSubmit}>
        <label className="block">
          <span className="label-caps">Chủ đề</span>
          <input
            className="mt-1 w-full rounded-md border border-outline px-3 py-2 text-sm transition focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder={defaultTopic}
          />
          <p className="mt-1 text-[10px] text-muted">Để trống thì Noelys tự so sánh theo tài liệu đang chọn: {sourceHint}.</p>
        </label>

        <div>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="label-caps">Khía cạnh</span>
            <div className="flex flex-wrap justify-end gap-1">
              {DIMENSION_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => setDimensions(preset.values)}
                  className="rounded border border-outline px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary/50 hover:text-primary"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <DimensionTags tags={dimensions} onChange={setDimensions} />
          <p className="mt-1 text-[10px] text-muted">Enter để thêm, Backspace để xóa nhanh.</p>
        </div>

        <button
          type="submit"
          className="flex w-full items-center justify-center gap-2 rounded-md bg-primary py-2 text-sm font-semibold text-white transition hover:bg-primary/90 disabled:opacity-50"
          disabled={loading || dimensions.length === 0 || !hasScope}
        >
          {loading ? <><Loader2 className="animate-spin" size={14} /> Đang so sánh...</> : "Tạo bảng so sánh"}
        </button>

        {!hasScope && (
          <p className="text-center text-[11px] text-amber-600">Chọn collection hoặc nguồn đã sẵn sàng trước khi so sánh.</p>
        )}
      </form>

      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            <AlertCircle size={14} className="mt-0.5 shrink-0" /> {error}
          </div>
        )}

        {loading && (
          <div className="rounded-lg border border-outline bg-surface p-6 shadow-sm">
            <div className="flex flex-col items-center gap-3 text-muted">
              <div className="relative">
                <Loader2 size={28} className="animate-spin text-primary" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-[10px] font-bold text-primary">{dimensions.length}</span>
                </div>
              </div>
              <p className="text-sm font-semibold text-text">Đang so sánh tài liệu...</p>
              <p className="text-xs text-center max-w-[300px]">
                Truy xuất bằng chứng cho <span className="font-bold text-primary">{dimensions.length}</span> khía cạnh
                trên <span className="font-bold text-primary">{scopedMaterialIds.length || readySources.length}</span> nguồn
              </p>
              <div className="mt-2 w-full max-w-[280px] space-y-1.5">
                {dimensions.slice(0, 4).map((d, i) => (
                  <div key={d} className="flex items-center gap-2 text-[10px]">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" style={{ animationDelay: `${i * 200}ms` }} />
                    <span className="text-muted truncate">{d}</span>
                  </div>
                ))}
                {dimensions.length > 4 && (
                  <span className="text-[9px] text-muted/60 italic pl-3.5">+{dimensions.length - 4} khía cạnh khác</span>
                )}
              </div>
            </div>
          </div>
        )}

        {result && !loading && (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-muted flex-1 min-w-0 truncate" title={result.topic}>
                {result.topic}
              </h3>
              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
                {result.sources?.length || result.coverage?.requested_count || 0} nguồn
              </span>
              <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
                {result.dimension_coverage?.length || dimensions.length} khía cạnh
              </span>
              {result.citations.length > 0 && (
                <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
                  {result.citations.length} trích dẫn
                </span>
              )}
              {/* Export buttons */}
              <div className="ml-auto flex gap-1">
                <button
                  type="button"
                  onClick={() => {
                    try {
                      navigator.clipboard.writeText(exportToMarkdown(result));
                    } catch {}
                  }}
                  title="Copy Markdown bảng so sánh"
                  className="flex items-center gap-1 rounded border border-outline bg-surface px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary hover:text-primary"
                >
                  <Copy size={10} /> MD
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const stamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
                    downloadFile(`compare-${stamp}.csv`, exportToCsv(result), "text/csv");
                  }}
                  title="Tải bảng dạng CSV"
                  className="flex items-center gap-1 rounded border border-outline bg-surface px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary hover:text-primary"
                >
                  <Download size={10} /> CSV
                </button>
              </div>
            </div>

            <CoveragePanel coverage={result.coverage} />
            <DimensionCoveragePanel result={result} />

            {hasMatrix ? (
              <CompareMatrixTable result={result} onCitationClick={handleCitationClick} />
            ) : (
              result.comparison_table.map((row, index) => (
                <ResultCard
                  key={`${row.source}-${row.dimension}-${index}`}
                  dimension={row.dimension}
                  value={row.value}
                  source={row.source}
                  citation={row.citation}
                  confidence={row.confidence}
                  missingEvidence={row.missing_evidence}
                  onCitationClick={handleCitationClick}
                />
              ))
            )}

            {result.conflicts.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                <div className="mb-2 flex items-center gap-1.5">
                  <AlertTriangle size={13} className="text-amber-700" />
                  <h4 className="text-xs font-bold text-amber-800">Điểm có thể mâu thuẫn ({result.conflicts.length})</h4>
                </div>
                <ul className="space-y-1.5">
                  {result.conflicts.map((item) => (
                    <li key={item} className="flex items-start gap-1.5 text-xs text-amber-700">
                      <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-amber-500" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center gap-4 py-10 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/8">
              <GitBranch size={26} className="text-primary/60" />
            </div>
            <div>
              <p className="text-sm font-semibold text-text">So sánh đa tài liệu</p>
              <p className="mt-1 max-w-[280px] text-xs text-muted">
                Noelys đối chiếu các nguồn đang chọn theo từng khía cạnh, mỗi ô có citation riêng để verify.
              </p>
            </div>
            <div className="grid gap-2 max-w-[300px] w-full text-left">
              <div className="flex items-start gap-2 rounded-lg border border-outline/50 bg-surface px-3 py-2">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold">1</span>
                <span className="text-[11px] text-muted leading-relaxed">Chọn nguồn ở panel Sources (mặc định: tất cả)</span>
              </div>
              <div className="flex items-start gap-2 rounded-lg border border-outline/50 bg-surface px-3 py-2">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold">2</span>
                <span className="text-[11px] text-muted leading-relaxed">Đặt khía cạnh muốn so (preset hoặc tự gõ)</span>
              </div>
              <div className="flex items-start gap-2 rounded-lg border border-outline/50 bg-surface px-3 py-2">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold">3</span>
                <span className="text-[11px] text-muted leading-relaxed">Tạo bảng → click citation để mở evidence panel</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
