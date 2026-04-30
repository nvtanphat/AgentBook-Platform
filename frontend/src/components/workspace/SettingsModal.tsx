import { FormEvent, useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2, X } from "lucide-react";
import { API_BASE_URL, CollectionSummary, checkHealth, getAdminMetrics, listCollections } from "../../api/client";
import { useWorkspace } from "../../state/workspace";
import { useSearchParams, useNavigate } from "react-router-dom";

export default function SettingsModal() {
  const { workspace, updateWorkspace, materials } = useWorkspace();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isOpen = searchParams.get("settings") === "open";

  const [draft, setDraft] = useState(workspace);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [collections, setCollections] = useState<CollectionSummary[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setDraft(workspace);
      setLoadingCollections(true);
      listCollections(workspace.ownerId)
        .then(setCollections)
        .catch(() => {})
        .finally(() => setLoadingCollections(false));
    }
  }, [isOpen, workspace.ownerId]);

  function close() {
    const params = new URLSearchParams(searchParams);
    params.delete("settings");
    navigate({ search: params.toString() });
  }

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
      subject: col?.subject ?? d.subject
    }));
  }

  function save(event: FormEvent) {
    event.preventDefault();
    updateWorkspace(draft);
    setStatus("Settings saved locally.");
    setError(null);
    setTimeout(() => {
      setStatus(null);
      close();
    }, 1500);
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
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-xl bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-outline px-6 py-4 sticky top-0 bg-white z-10">
          <h2 className="font-heading text-lg font-bold">Workspace Settings</h2>
          <button onClick={close} className="text-muted hover:text-text p-1"><X size={20} /></button>
        </div>

        <div className="p-6">
          <form className="space-y-5" onSubmit={save}>
            <div>
              <p className="label-caps">API base URL</p>
              <p className="mt-1 rounded-md bg-slate-50 px-3 py-2 text-sm font-mono text-muted border border-outline">{API_BASE_URL}</p>
            </div>

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

            <div className="flex justify-between items-center pt-4 border-t border-outline">
              <button type="button" className="flex items-center gap-2 rounded-md border border-outline bg-white px-4 py-2 text-sm font-semibold text-muted hover:bg-slate-50" onClick={testConnection} disabled={checking}>
                {checking ? <Loader2 className="animate-spin" size={16} /> : null} Test API
              </button>
              <div className="flex gap-2">
                <button type="button" onClick={close} className="rounded-md px-4 py-2 text-sm font-semibold text-muted hover:bg-slate-50">Cancel</button>
                <button type="submit" className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-bright">Save</button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
