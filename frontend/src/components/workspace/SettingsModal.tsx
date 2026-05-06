import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, X, Zap } from "lucide-react";
import { API_BASE_URL, CollectionSummary, checkHealth, getAdminMetrics, listCollections } from "../../api/client";
import { useWorkspace } from "../../state/workspace";
import { useSearchParams, useNavigate } from "react-router-dom";

// ─── Pipeline settings API ────────────────────────────────────────────────────

async function fetchPipelineSettings(): Promise<{ contextual_retrieval_enabled: boolean }> {
  const res = await fetch(`${API_BASE_URL}/admin/settings`);
  if (!res.ok) throw new Error("Failed to load pipeline settings");
  return res.json();
}

async function patchPipelineSettings(patch: { contextual_retrieval_enabled?: boolean }): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/admin/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error("Failed to update pipeline settings");
}

// ─── Toggle switch ────────────────────────────────────────────────────────────

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none disabled:opacity-40 ${
        checked ? "bg-primary" : "bg-slate-300"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
          checked ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  );
}

// ─── Main modal ───────────────────────────────────────────────────────────────

export default function SettingsModal() {
  const { workspace, updateWorkspace, materials } = useWorkspace();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isOpen = searchParams.get("settings") === "open";
  const dialogRef = useRef<HTMLDivElement>(null);

  const [draft, setDraft] = useState(workspace);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [collections, setCollections] = useState<CollectionSummary[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);

  // Pipeline settings
  const [contextualRetrieval, setContextualRetrieval] = useState(true);
  const [loadingPipeline, setLoadingPipeline] = useState(false);
  const [savingPipeline, setSavingPipeline] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setDraft(workspace);
    setLoadingCollections(true);
    listCollections(workspace.ownerId)
      .then(setCollections)
      .catch(() => {})
      .finally(() => setLoadingCollections(false));

    setLoadingPipeline(true);
    fetchPipelineSettings()
      .then((s) => setContextualRetrieval(s.contextual_retrieval_enabled))
      .catch(() => {})
      .finally(() => setLoadingPipeline(false));
  }, [isOpen, workspace.ownerId]);

  function close() {
    const params = new URLSearchParams(searchParams);
    params.delete("settings");
    navigate({ search: params.toString() });
  }

  // Focus modal + Escape to close
  useEffect(() => {
    if (!isOpen) return;
    dialogRef.current?.focus();
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") close();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  function handleCollectionSelect(collectionId: string) {
    if (!collectionId) {
      setDraft((d) => ({ ...d, collectionId: "", collectionName: "" }));
      return;
    }
    const col = collections.find((c) => c.collection_id === collectionId);
    setDraft((d) => ({
      ...d,
      collectionId,
      collectionName: col?.name ?? "",
      subject: col?.subject ?? d.subject,
    }));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    updateWorkspace(draft);

    // Save pipeline settings
    setSavingPipeline(true);
    try {
      await patchPipelineSettings({ contextual_retrieval_enabled: contextualRetrieval });
    } catch {
      // non-fatal
    } finally {
      setSavingPipeline(false);
    }

    setStatus("Settings saved.");
    setError(null);
    setTimeout(() => { setStatus(null); close(); }, 1500);
  }

  async function testConnection() {
    setChecking(true);
    setError(null);
    setStatus(null);
    try {
      const health = await checkHealth();
      const metrics = await getAdminMetrics().catch(() => null);
      setStatus(`${health.service} is ${health.status}. ${metrics ? `${metrics.indexed_docs} docs indexed.` : ""}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed.");
    } finally {
      setChecking(false);
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4" onClick={close}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Workspace Settings"
        tabIndex={-1}
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl bg-white shadow-2xl outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-outline px-6 py-4 sticky top-0 bg-white z-10">
          <h2 className="font-heading text-lg font-bold">Workspace Settings</h2>
          <button onClick={close} className="text-muted hover:text-text p-1"><X size={20} /></button>
        </div>

        <div className="p-6">
          <form className="space-y-5" onSubmit={save}>
            {/* API URL */}
            <div>
              <p className="label-caps">API base URL</p>
              <p className="mt-1 rounded-md bg-slate-50 px-3 py-2 text-sm font-mono text-muted border border-outline">{API_BASE_URL}</p>
            </div>

            {/* Workspace fields */}
            <div className="grid gap-4 md:grid-cols-2">
              <label>
                <span className="label-caps">Owner ID</span>
                <input className="mt-1 w-full rounded-md border border-outline px-3 py-2 text-sm" value={draft.ownerId} onChange={(e) => setDraft({ ...draft, ownerId: e.target.value })} />
              </label>

              <label>
                <span className="label-caps">Collection</span>
                <select className="mt-1 w-full rounded-md border border-outline bg-white px-3 py-2 text-sm" value={draft.collectionId} onChange={(e) => handleCollectionSelect(e.target.value)} disabled={loadingCollections}>
                  <option value="">{loadingCollections ? "Loading..." : "— No collection —"}</option>
                  {collections.map((col) => (
                    <option key={col.collection_id} value={col.collection_id}>{col.name} ({col.retrievable_chunk_count} chunks)</option>
                  ))}
                </select>
              </label>

              <label>
                <span className="label-caps">Answer Language</span>
                <select className="mt-1 w-full rounded-md border border-outline px-3 py-2 text-sm" value={draft.language} onChange={(e) => setDraft({ ...draft, language: e.target.value })}>
                  <option value="vi">Tiếng Việt</option>
                  <option value="en">English</option>
                  <option value="zh">中文 (Chinese)</option>
                  <option value="ja">日本語 (Japanese)</option>
                  <option value="ko">한국어 (Korean)</option>
                  <option value="fr">Français (French)</option>
                  <option value="de">Deutsch (German)</option>
                </select>
              </label>

              <label>
                <span className="label-caps">Top K (Retrieval)</span>
                <input className="mt-1 w-full rounded-md border border-outline px-3 py-2 text-sm" type="number" min="1" max="20" value={draft.topK} onChange={(e) => setDraft({ ...draft, topK: Number(e.target.value) || 5 })} />
              </label>
            </div>

            {/* ── Pipeline Settings ── */}
            <div className="rounded-lg border border-outline bg-slate-50 p-4 space-y-3">
              <div className="flex items-center gap-2 mb-1">
                <Zap size={13} className="text-primary" />
                <p className="text-xs font-bold uppercase tracking-wider text-text">Pipeline Settings</p>
              </div>

              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-text">Contextual Retrieval</p>
                  <p className="text-xs text-muted mt-0.5">
                    LLM tự động thêm context cho từng chunk trước khi index.
                    Tăng độ chính xác tìm kiếm nhưng <span className="font-semibold text-yellow-600">làm chậm pipeline ~10× </span>
                    (mỗi chunk gọi Ollama 1 lần).
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1 pt-0.5">
                  {loadingPipeline ? (
                    <Loader2 size={16} className="animate-spin text-muted" />
                  ) : (
                    <Toggle checked={contextualRetrieval} onChange={setContextualRetrieval} />
                  )}
                  <span className={`text-[10px] font-semibold ${contextualRetrieval ? "text-primary" : "text-muted"}`}>
                    {contextualRetrieval ? "Bật" : "Tắt"}
                  </span>
                </div>
              </div>

              {contextualRetrieval && (
                <div className="rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 text-xs text-yellow-700">
                  ⚠️ Bật sẽ làm pipeline chậm đáng kể. Chỉ nên dùng khi demo hoặc production.
                </div>
              )}
            </div>

            {/* Status / Error */}
            {status && (
              <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                <CheckCircle2 size={16} /> {status}
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <AlertCircle size={16} /> {error}
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-between items-center pt-4 border-t border-outline">
              <button type="button" className="flex items-center gap-2 rounded-md border border-outline bg-white px-4 py-2 text-sm font-semibold text-muted hover:bg-slate-50" onClick={testConnection} disabled={checking}>
                {checking ? <Loader2 className="animate-spin" size={16} /> : null} Test API
              </button>
              <div className="flex gap-2">
                <button type="button" onClick={close} className="rounded-md px-4 py-2 text-sm font-semibold text-muted hover:bg-slate-50">Cancel</button>
                <button type="submit" disabled={savingPipeline} className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-bright disabled:opacity-60">
                  {savingPipeline && <Loader2 size={14} className="animate-spin" />}
                  Save
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
