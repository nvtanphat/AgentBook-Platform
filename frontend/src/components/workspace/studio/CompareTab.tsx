import { FormEvent, KeyboardEvent, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Loader2, Plus, X } from "lucide-react";
import { Citation, CompareResponse, CoverageReport, compareDocuments } from "../../../api/client";
import { useWorkspace } from "../../../state/workspace";

const DEFAULT_DIMENSIONS = ["ý chính", "điểm giống", "điểm khác", "bằng chứng"];
const DIMENSION_PRESETS = [
  { label: "Tổng quan", values: ["ý chính", "định nghĩa", "ví dụ", "bằng chứng"] },
  { label: "So sánh", values: ["điểm giống", "điểm khác", "ưu điểm", "hạn chế"] },
  { label: "Kiểm chứng", values: ["luận điểm", "bằng chứng", "nguồn trích dẫn", "độ tin cậy"] },
];

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
    <div className="flex min-h-[38px] flex-wrap gap-1.5 rounded-md border border-outline bg-white px-2 py-1.5 transition-all focus-within:border-primary focus-within:ring-1 focus-within:ring-primary">
      {tags.map((tag) => (
        <span key={tag} className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-semibold text-primary">
          {tag}
          <button type="button" onClick={() => remove(tag)} className="transition hover:text-red-500" aria-label={`Remove ${tag}`}>
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
          <button type="button" onClick={add} className="text-muted transition hover:text-primary" aria-label="Add dimension">
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

const CARD_COLLAPSE_LINES = 5;

type SourceEvidenceRow = {
  source: string;
  snippet: string;
};

type CompareMatrix = {
  sources: string[];
  dimensions: string[];
  cells: Record<string, Record<string, string>>;
};

function parseSourceEvidence(value: string): { heading: string; rows: SourceEvidenceRow[] } {
  const [firstLine, ...rest] = value.split("\n");
  const heading = firstLine?.replace(/:$/, "").trim() || "Bằng chứng";
  const rows = rest
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => {
      const body = line.slice(2);
      const separator = body.indexOf(": ");
      if (separator <= 0) return { source: "Nguồn", snippet: body };
      return {
        source: body.slice(0, separator).trim(),
        snippet: body.slice(separator + 2).trim(),
      };
    });
  return { heading, rows };
}

function buildCompareMatrix(rows: CompareResponse["comparison_table"]): CompareMatrix | null {
  const parsed = rows.map((row) => ({
    dimension: row.dimension,
    rows: parseSourceEvidence(row.value).rows,
  }));
  if (!parsed.length || parsed.some((item) => item.rows.length === 0)) return null;

  const sources = Array.from(new Set(parsed.flatMap((item) => item.rows.map((row) => row.source))));
  const dimensions = parsed.map((item) => item.dimension);
  const cells: CompareMatrix["cells"] = {};
  for (const source of sources) cells[source] = {};
  for (const item of parsed) {
    for (const row of item.rows) {
      cells[row.source][item.dimension] = row.snippet;
    }
  }
  return { sources, dimensions, cells };
}

function CompareMatrixTable({
  matrix, citations, onCitationClick,
}: {
  matrix: CompareMatrix;
  citations: Citation[];
  onCitationClick: (c: Citation) => void;
}) {
  const citationByDoc = new Map(citations.map((citation) => [citation.doc_name, citation]));
  return (
    <div className="rounded-lg border border-outline bg-white shadow-sm">
      <div className="border-b border-outline bg-slate-50 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
        Bảng so sánh theo nguồn · cuộn ngang để xem các khía cạnh
      </div>
      <div className="w-full overflow-x-auto">
        <table className="w-max min-w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-outline bg-slate-50">
              <th className="w-[180px] min-w-[180px] border-r border-outline bg-slate-50 px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted">
                Nguồn
              </th>
              {matrix.dimensions.map((dimension) => (
                <th key={dimension} className="w-[260px] min-w-[260px] border-r border-outline px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted last:border-r-0">
                  {dimension}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-outline">
            {matrix.sources.map((source) => {
              const citation = citationByDoc.get(source);
              return (
                <tr key={source} className="align-top">
                  <th className="w-[180px] min-w-[180px] border-r border-outline bg-white px-3 py-3 text-xs font-semibold leading-snug text-text">
                    <div className="space-y-2">
                      <p className="break-words">{source}</p>
                      {citation && (
                        <button
                          type="button"
                          onClick={() => onCitationClick(citation)}
                          className="rounded-full border border-outline px-2 py-0.5 text-[10px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
                        >
                          Bằng chứng p.{citation.page ?? "?"}
                        </button>
                      )}
                    </div>
                  </th>
                  {matrix.dimensions.map((dimension) => (
                    <td key={`${source}-${dimension}`} className="w-[260px] min-w-[260px] border-r border-outline px-3 py-3 text-xs leading-relaxed text-text last:border-r-0">
                      <div className="max-h-32 overflow-y-auto pr-1">
                        {matrix.cells[source]?.[dimension] || <span className="text-muted">-</span>}
                      </div>
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SourceEvidenceList({ rows, expanded }: { rows: SourceEvidenceRow[]; expanded: boolean }) {
  const visibleRows = expanded ? rows : rows.slice(0, 4);
  return (
    <div className="overflow-hidden rounded-md border border-outline">
      <div className="grid grid-cols-[minmax(120px,0.85fr)_minmax(0,2fr)] border-b border-outline bg-slate-50 text-[10px] font-bold uppercase tracking-wider text-muted">
        <div className="border-r border-outline px-3 py-2">Nguồn</div>
        <div className="px-3 py-2">Bằng chứng</div>
      </div>
      <div className="divide-y divide-outline bg-white">
        {visibleRows.map((row, index) => (
          <div key={`${row.source}-${index}`} className="grid grid-cols-[minmax(120px,0.85fr)_minmax(0,2fr)]">
            <div className="border-r border-outline bg-slate-50/50 px-3 py-2">
              <p className="break-words text-xs font-semibold leading-snug text-text">{row.source}</p>
            </div>
            <div className="px-3 py-2">
              <p className="text-xs leading-relaxed text-text">{row.snippet}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

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
  const parsed = parseSourceEvidence(value);
  const hasSourceRows = parsed.rows.length > 0;
  const lines = value.split("\n");
  const isLong = hasSourceRows ? parsed.rows.length > 4 : lines.length > CARD_COLLAPSE_LINES || value.length > 400;
  const [expanded, setExpanded] = useState(false);
  const visibleRowCount = Math.min(4, parsed.rows.length);
  const displayValue = isLong && !expanded
    ? (value.length > 400 ? `${value.slice(0, 400)}...` : `${lines.slice(0, CARD_COLLAPSE_LINES).join("\n")}...`)
    : value;

  const noEvidence = confidence === 0;

  return (
    <div className={`overflow-hidden rounded-lg border bg-white shadow-sm ${noEvidence ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between gap-3 border-b border-outline bg-slate-50 px-4 py-2.5">
        <div className="min-w-0">
          <span className="block truncate text-xs font-bold capitalize text-text">{dimension}</span>
          {hasSourceRows && <span className="text-[10px] text-muted">{parsed.rows.length} bằng chứng theo nguồn</span>}
        </div>
        <ConfidencePill value={confidence} />
      </div>

      <div className="px-4 py-3">
        {hasSourceRows ? (
          <SourceEvidenceList rows={parsed.rows} expanded={expanded} />
        ) : (
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-text">{displayValue}</p>
        )}
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="mt-1.5 flex items-center gap-1 text-[11px] font-semibold text-primary hover:opacity-70"
          >
            {expanded ? <><ChevronUp size={11} /> Thu gọn</> : <><ChevronDown size={11} /> Xem thêm{hasSourceRows ? ` (${parsed.rows.length - visibleRowCount})` : ""}</>}
          </button>
        )}
      </div>

      {!noEvidence && (
        <div className="flex items-center justify-between border-t border-outline bg-slate-50/50 px-4 py-2">
          <span className="max-w-[60%] truncate text-[11px] text-muted" title={source}>{source}</span>
          {citation && (
            <button
              type="button"
              onClick={() => onCitationClick(citation)}
              className="flex shrink-0 items-center gap-1 rounded-full border border-outline bg-white px-2 py-0.5 text-[11px] font-semibold text-muted transition hover:border-primary/40 hover:text-primary"
            >
              p.{citation.page ?? "?"}
              <span className="ml-0.5 opacity-50">mở</span>
            </button>
          )}
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
      setError(err instanceof Error ? err.message : "Comparison failed.");
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
  const matrix = result ? buildCompareMatrix(result.comparison_table) : null;

  return (
    <div className="flex h-full flex-col overflow-hidden bg-slate-50">
      <form className="shrink-0 space-y-3 border-b border-outline bg-white p-4" onSubmit={handleSubmit}>
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
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-muted">
            <Loader2 size={24} className="animate-spin text-primary" />
            <p className="text-xs">Đang truy xuất bằng chứng cho {dimensions.length} khía cạnh...</p>
            <p className="text-[10px] text-muted/70">Lần đầu có thể chậm hơn vì phải warm model.</p>
          </div>
        )}

        {result && !loading && (
          <>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-muted">
                {result.topic}
              </h3>
              <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-semibold text-muted">
                {result.comparison_table.length} khía cạnh
              </span>
              {result.citations.length > 0 && (
                <span className="rounded bg-slate-200 px-1.5 py-0.5 text-[10px] font-semibold text-muted">
                  {result.citations.length} trích dẫn
                </span>
              )}
            </div>

            <CoveragePanel coverage={result.coverage} />

            {matrix ? (
              <CompareMatrixTable matrix={matrix} citations={result.citations} onCitationClick={handleCitationClick} />
            ) : (
              result.comparison_table.map((row) => (
                <ResultCard
                  key={row.dimension}
                  dimension={row.dimension}
                  value={row.value}
                  source={row.source}
                  citation={row.citation}
                  confidence={row.confidence}
                  onCitationClick={handleCitationClick}
                />
              ))
            )}

            {result.conflicts.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                <h4 className="mb-2 text-xs font-bold text-amber-800">Điểm có thể mâu thuẫn</h4>
                <ul className="space-y-1">
                  {result.conflicts.map((item) => (
                    <li key={item} className="text-xs text-amber-700">- {item}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
            <p className="text-sm font-semibold text-text">So sánh trên bằng chứng</p>
            <p className="max-w-[220px] text-xs text-muted">
              Bấm tạo bảng để Noelys đối chiếu các tài liệu đang chọn theo từng khía cạnh.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
