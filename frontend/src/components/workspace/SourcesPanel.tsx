import { DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, BookOpen, CheckCircle2, FileText, Image, Loader2, Plus, RefreshCw, Search, Table2, Trash2, UploadCloud, X } from "lucide-react";
import { CollectionSummary, MaterialInfo, MaterialUploadMetadata, createCollection, deleteCollection, deleteMaterial, getMaterialStatus, listCollections, listMaterials, uploadMaterialsBatchWithProgress } from "../../api/client";
import StatusBadge from "../StatusBadge";
import { useWorkspace } from "../../state/workspace";
import DebugModal from "./DebugModal";

const ACCEPTED_TYPES = ".pdf,.docx,.pptx,.png,.jpg,.jpeg,.csv,.xlsx";
const ACCEPTED_MIME = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "image/png",
  "image/jpeg",
  "text/csv",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
]);

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

type FileIconInfo = { icon: React.ReactNode; colorClass: string };

function getFileIconInfo(filename: string, fileType = ""): FileIconInfo {
  const name = filename.toLowerCase();
  const mime = fileType.toLowerCase();
  if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || mime.includes("image"))
    return { icon: <Image size={14} />, colorClass: "text-purple-500" };
  if (name.endsWith(".csv") || name.endsWith(".xlsx") || mime.includes("spreadsheet") || mime.includes("csv"))
    return { icon: <Table2 size={14} />, colorClass: "text-emerald-600" };
  if (name.endsWith(".pdf") || mime.includes("pdf"))
    return { icon: <FileText size={14} />, colorClass: "text-red-500" };
  if (name.endsWith(".docx") || name.endsWith(".doc") || mime.includes("word"))
    return { icon: <FileText size={14} />, colorClass: "text-blue-500" };
  if (name.endsWith(".pptx") || name.endsWith(".ppt") || mime.includes("presentation"))
    return { icon: <FileText size={14} />, colorClass: "text-amber-500" };
  return { icon: <FileText size={14} />, colorClass: "text-primary" };
}

function detectLanguage(filename: string): string {
  if (/[àáảãạăắặẳẵặâấầẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/i.test(filename)) return "vi";
  return "unknown";
}

function isNotFoundError(err: unknown) {
  const msg = err instanceof Error ? err.message.toLowerCase() : "";
  return msg.includes("not found") || msg.includes("404");
}

export default function SourcesPanel({ onCloseMobile }: { onCloseMobile?: () => void }) {
  const {
    workspace,
    updateWorkspace,
    materials,
    addUploadedMaterial,
    updateMaterialStatus,
    removeUploadedMaterial,
    clearUploadedMaterialsForCollection,
  } = useWorkspace();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const didAutoSelectCollectionRef = useRef(false);

  const [collections, setCollections] = useState<CollectionSummary[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);
  const [serverMaterials, setServerMaterials] = useState<MaterialInfo[]>([]);
  const [loadingMaterials, setLoadingMaterials] = useState(false);

  const [files, setFiles] = useState<File[]>([]);
  const [handwritingFiles, setHandwritingFiles] = useState<Set<string>>(new Set());
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deletingCollection, setDeletingCollection] = useState(false);
  const [creatingCollection, setCreatingCollection] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deletingMaterial, setDeletingMaterial] = useState<string | null>(null);
  const [confirmDeleteMaterial, setConfirmDeleteMaterial] = useState<string | null>(null);
  const [debugMaterial, setDebugMaterial] = useState<MaterialInfo | null>(null);

  const loadCollections = useCallback(async () => {
    setLoadingCollections(true);
    try {
      const items = await listCollections(workspace.ownerId);
      setCollections(items);
      const currentStillExists = items.some((c) => c.collection_id === workspace.collectionId);
      if (workspace.collectionId && !currentStillExists) {
        // Stale collectionId (e.g. deleted) — clear it so uploads create a new collection
        updateWorkspace({ collectionId: "", collectionName: "" });
      } else if (!workspace.collectionId && !didAutoSelectCollectionRef.current) {
        const best = [...items]
          .filter((c) => c.retrievable_chunk_count > 0)
          .sort((a, b) => b.retrievable_chunk_count - a.retrievable_chunk_count)[0];
        if (best) {
          didAutoSelectCollectionRef.current = true;
          updateWorkspace({ collectionId: best.collection_id, collectionName: best.name, subject: best.subject ?? workspace.subject });
        }
      }
    } catch {}
    finally { setLoadingCollections(false); }
  }, [workspace.ownerId, workspace.collectionId, workspace.subject, updateWorkspace]);

  const loadMaterialsFromServer = useCallback(async (collectionId: string | null) => {
    if (!collectionId) { setServerMaterials([]); return; }
    setLoadingMaterials(true);
    try {
      const items = await listMaterials(workspace.ownerId, collectionId);
      setServerMaterials(items);
    } catch {
      setServerMaterials([]);
    } finally {
      setLoadingMaterials(false);
    }
  }, [workspace.ownerId]);

  const TERMINAL_STATUSES = new Set(["indexed", "failed"]);
  const isTerminalStatus = (status: string) => TERMINAL_STATUSES.has(status.toLowerCase());
  const serverIds = new Set(serverMaterials.map((m) => m.material_id));
  const sessionOnlyMaterials = materials
    .filter((m) => m.collectionId === workspace.collectionId && !serverIds.has(m.materialId));
  const pendingSessionMaterials = sessionOnlyMaterials.filter((m) => !isTerminalStatus(m.status));
  const pendingSessionKey = useMemo(
    () => pendingSessionMaterials.map((m) => `${m.materialId}:${m.status}:${m.stage}`).join("|"),
    [pendingSessionMaterials]
  );
  const pendingCount = serverMaterials.filter((m) => !isTerminalStatus(m.status)).length + pendingSessionMaterials.length;

  useEffect(() => { loadCollections(); }, [loadCollections]);

  useEffect(() => {
    loadMaterialsFromServer(workspace.collectionId || null);
  }, [loadMaterialsFromServer, workspace.collectionId]);

  // Auto-poll while any material is still processing
  useEffect(() => {
    if (!workspace.collectionId || pendingCount === 0) return;
    const interval = setInterval(() => {
      loadMaterialsFromServer(workspace.collectionId || null);
    }, 4000);
    return () => clearInterval(interval);
  }, [workspace.collectionId, pendingCount, loadMaterialsFromServer]);

  useEffect(() => {
    if (!pendingSessionMaterials.length) return;
    let cancelled = false;
    async function refreshSessionStatuses() {
      await Promise.all(
        pendingSessionMaterials.map(async (material) => {
          try {
            const status = await getMaterialStatus(material.materialId, workspace.ownerId);
            if (!cancelled) updateMaterialStatus(material.materialId, status.status, status.stage);
          } catch (err) {
            // If the server says "not found", the material was deleted — remove the stale
            // session entry so the spinner clears. Ignore transient network errors.
            if (!cancelled && isNotFoundError(err)) {
              removeUploadedMaterial(material.materialId);
            }
          }
        })
      );
    }
    refreshSessionStatuses();
    const interval = setInterval(refreshSessionStatuses, 4000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingSessionKey, updateMaterialStatus, workspace.ownerId]);

  function addFiles(incoming: FileList | null) {
    if (!incoming) return;
    const valid = Array.from(incoming).filter(
      (f) => ACCEPTED_MIME.has(f.type) || ACCEPTED_TYPES.split(",").some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...valid.filter((f) => !names.has(f.name))];
    });
    setError(null);
    setSuccess(null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragOver(false);
    addFiles(event.dataTransfer.files);
  }

  async function uploadFiles(e?: { preventDefault(): void }) {
    if (e) e.preventDefault();
    if (!files.length || uploading) return;
    
    setUploading(true);
    setError(null);
    setSuccess(null);

    let latestCollectionId: string | null = workspace.collectionId || null;
    const uploadErrors: string[] = [];
    setUploadingName(files.length === 1 ? files[0].name : `${files.length} files`);
    const metadata = files.map((file): MaterialUploadMetadata => {
      const lang = detectLanguage(file.name);
      const isImage = /\.(png|jpe?g)$/i.test(file.name);
      const isHandwriting = isImage && handwritingFiles.has(file.name);
      return {
        owner_id: workspace.ownerId,
        collection_id: workspace.collectionId || null,
        collection_name: workspace.collectionName || null,
        subject: workspace.subject || null,
        language: lang,
        modality: isHandwriting ? "handwriting" : "mixed",
        source_type: isHandwriting ? "handwriting" : isImage ? "printed_scan" : null,
        version: "v1.0",
        extra_metadata: { uploaded_from: "workspace_sources" }
      };
    });
    let count = 0;
    try {
      setUploadProgress(0);
      const response = await uploadMaterialsBatchWithProgress(files, metadata, setUploadProgress);
      response.results.forEach((item, index) => {
        if (item.success && item.data) {
          const lang = detectLanguage(files[index]?.name ?? item.filename);
          addUploadedMaterial(item.data, { topic: "", language: lang });
          latestCollectionId = item.data.collection_id;
          count++;
        } else {
          uploadErrors.push(`${item.filename}: ${item.error ?? "Upload failed."}`);
        }
      });
    } catch (err) {
      uploadErrors.push(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      if (uploadProgress < 100) {
        setUploadProgress(100);
      }
    }
    setUploadProgress(0);
    if (uploadErrors.length) setError(uploadErrors.join("\n"));

    if (count > 0) {
      setSuccess(`${count} file(s) uploaded successfully!`);
      setFiles([]);
      setHandwritingFiles(new Set());
      await Promise.all([
        loadCollections(),
        loadMaterialsFromServer(latestCollectionId),
      ]);
    }
    setUploadingName(null);
    setUploading(false);
  }

  async function handleDeleteCollection() {
    if (!workspace.collectionId) return;
    if (!confirmDelete) { setConfirmDelete(true); return; }
    setDeletingCollection(true);
    setConfirmDelete(false);
    setError(null);
    try {
      await deleteCollection(workspace.collectionId, workspace.ownerId);
      clearUploadedMaterialsForCollection(workspace.collectionId);
      updateWorkspace({ collectionId: "", collectionName: "" });
      setServerMaterials([]);
      setSuccess("Collection deleted.");
      await loadCollections();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setDeletingCollection(false);
    }
  }

  async function handleCreateCollection() {
    const name = workspace.collectionName.trim();
    if (!name || creatingCollection) return;
    setCreatingCollection(true);
    setError(null);
    setSuccess(null);
    try {
      const collection = await createCollection({
        owner_id: workspace.ownerId,
        name,
        subject: workspace.subject || null,
      });
      didAutoSelectCollectionRef.current = true;
      updateWorkspace({
        collectionId: collection.collection_id,
        collectionName: collection.name,
        subject: collection.subject ?? workspace.subject,
      });
      setCollections((current) => [collection, ...current.filter((item) => item.collection_id !== collection.collection_id)]);
      setServerMaterials([]);
      setSuccess("Collection created.");
      await loadCollections();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Create collection failed.");
    } finally {
      setCreatingCollection(false);
    }
  }

  async function handleDeleteMaterial(materialId: string) {
    if (confirmDeleteMaterial !== materialId) { setConfirmDeleteMaterial(materialId); return; }
    setDeletingMaterial(materialId);
    setConfirmDeleteMaterial(null);
    setError(null);
    try {
      await deleteMaterial(materialId, workspace.ownerId);
      removeUploadedMaterial(materialId);
      setSuccess("Source deleted.");
      await Promise.all([
        loadCollections(),
        loadMaterialsFromServer(workspace.collectionId || null),
      ]);
    } catch (err) {
      if (isNotFoundError(err)) {
        removeUploadedMaterial(materialId);
        setSuccess("Source was already removed.");
        await Promise.all([
          loadCollections(),
          loadMaterialsFromServer(workspace.collectionId || null),
        ]);
        return;
      }
      setError(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setDeletingMaterial(null);
    }
  }

  const selectedCollection = collections.find((c) => c.collection_id === workspace.collectionId);

  const displayMaterials = [...serverMaterials, ...sessionOnlyMaterials.map((m) => ({
    material_id: m.materialId,
    collection_id: m.collectionId,
    owner_id: workspace.ownerId,
    filename: m.filename,
    original_name: m.originalName,
    file_type: "",
    status: m.status,
    subject: null,
    topic: m.topic || null,
    page_count: null,
    version: "v1.0",
  } satisfies MaterialInfo))];

  return (
    <div className="flex flex-col h-full bg-slate-50 relative">
      {/* Mobile header */}
      <div className="lg:hidden flex items-center justify-between p-4 border-b border-outline bg-white">
        <h2 className="font-heading font-semibold flex items-center gap-2">
          <BookOpen size={16} className="text-primary" /> Sources
        </h2>
        <button
          type="button"
          aria-label="Đóng Sources"
          onClick={onCloseMobile}
          className="p-2 -mr-2 text-muted hover:text-text"
        >
          <X size={20} />
        </button>
      </div>

      {/* Collection Selector */}
      <div className="p-4 border-b border-outline bg-white shrink-0">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="label-caps">Active Collection</span>
            <div className="flex items-center gap-2">
              {workspace.collectionId && (
                confirmDelete ? (
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] text-red-600 font-semibold">Delete?</span>
                    <button type="button" onClick={handleDeleteCollection} disabled={deletingCollection} className="text-[10px] font-bold text-red-600 hover:text-red-700 disabled:opacity-50">
                      {deletingCollection ? <Loader2 size={11} className="animate-spin" /> : "Yes"}
                    </button>
                    <button type="button" onClick={() => setConfirmDelete(false)} className="text-[10px] text-muted hover:text-text">No</button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={handleDeleteCollection}
                    title="Delete collection"
                    aria-label="Xóa collection hiện tại"
                    className="rounded p-1 text-muted hover:bg-red-50 hover:text-red-500 disabled:opacity-50"
                    disabled={deletingCollection}
                  >
                    <Trash2 size={12} />
                  </button>
                )
              )}
              <button
                type="button"
                onClick={loadCollections}
                disabled={loadingCollections}
                aria-label="Tải lại danh sách collection"
                className="rounded p-1 text-muted hover:bg-slate-100 hover:text-primary disabled:opacity-50"
              >
                <RefreshCw size={12} className={loadingCollections ? "animate-spin" : ""} />
              </button>
            </div>
          </div>
          <select
            className="w-full rounded-md border border-outline bg-slate-50 px-2 py-1.5 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-primary"
            value={workspace.collectionId}
            onChange={(e) => {
              const id = e.target.value;
              setConfirmDelete(false);
              if (!id) {
                didAutoSelectCollectionRef.current = true;
                updateWorkspace({ collectionId: "", collectionName: "" });
                return;
              }
              const col = collections.find((c) => c.collection_id === id);
              updateWorkspace({ collectionId: id, collectionName: col?.name ?? "" });
            }}
          >
            <option value="">+ New Collection</option>
            {collections.map((c) => (
              <option key={c.collection_id} value={c.collection_id}>{c.name}</option>
            ))}
          </select>
          
          {!workspace.collectionId && (
            <div className="flex gap-2">
             <input
               className="min-w-0 flex-1 rounded-md border border-outline px-2 py-1.5 text-xs"
               placeholder="Name for new collection..."
               value={workspace.collectionName}
               onChange={(e) => updateWorkspace({ collectionName: e.target.value })}
             />
              <button
                type="button"
                onClick={handleCreateCollection}
                disabled={creatingCollection || !workspace.collectionName.trim()}
                className="flex items-center justify-center gap-1 rounded-md bg-primary px-2 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
              >
                {creatingCollection ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
                Create
              </button>
            </div>
          )}

          {selectedCollection && (
            <div className="flex gap-3 text-[10px] text-muted font-medium mt-1">
              <span>{selectedCollection.material_count} docs</span>
              <span>{selectedCollection.retrievable_chunk_count} chunks</span>
            </div>
          )}
        </div>
      </div>

      {/* Dropzone */}
      <div className="p-4 shrink-0">
        <div
          className={`cursor-pointer rounded-lg border-2 border-dashed p-4 text-center transition ${
            dragOver ? "border-primary bg-primary/5" : "border-outline bg-white hover:border-primary/50"
          } ${uploading ? "opacity-50 pointer-events-none" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !uploading && fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" className="hidden" multiple accept={ACCEPTED_TYPES} onChange={(e) => { addFiles(e.target.files); if(e.target) e.target.value = ""; }} />
          <UploadCloud size={20} className="mx-auto mb-2 text-primary" />
          <p className="text-xs font-semibold text-text">Click to upload or drag files</p>
          <p className="text-[10px] text-muted mt-1">PDF, DOCX, PPTX, Images</p>
        </div>

        {/* Queue */}
        {files.length > 0 && (
          <div className="mt-3 bg-white border border-outline rounded-lg p-2 space-y-1.5">
            {files.map(f => {
              const isImage = /\.(png|jpe?g)$/i.test(f.name);
              const isHW = handwritingFiles.has(f.name);
              return (
              <div key={f.name} className="flex items-center gap-2 text-xs">
                <FileText size={12} className="text-primary shrink-0" />
                <span className="truncate flex-1" title={f.name}>{f.name}</span>
                {isImage && (
                  <button
                    type="button"
                    title={isHW ? "Đang chọn: viết tay. Nhấn để chuyển sang in ấn" : "Đang chọn: in ấn. Nhấn để chuyển sang viết tay"}
                    onClick={() => setHandwritingFiles(prev => {
                      const next = new Set(prev);
                      if (next.has(f.name)) next.delete(f.name); else next.add(f.name);
                      return next;
                    })}
                    className={`shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold border transition ${isHW ? "bg-purple-50 text-purple-700 border-purple-300" : "bg-slate-50 text-slate-500 border-slate-200 hover:border-primary/40"}`}
                  >
                    {isHW ? "✍ Viết tay" : "🖨 In ấn"}
                  </button>
                )}
                <span className="text-muted shrink-0 text-[10px]">{formatBytes(f.size)}</span>
                <button
                  type="button"
                  onClick={() => { setFiles(prev => prev.filter(x => x.name !== f.name)); setHandwritingFiles(prev => { const n = new Set(prev); n.delete(f.name); return n; }); }}
                  aria-label={`Xóa ${f.name} khỏi hàng đợi upload`}
                  className="rounded p-1 text-muted hover:bg-red-50 hover:text-red-500"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            );
            })}
            <button onClick={() => uploadFiles()} className="w-full flex items-center justify-center gap-2 mt-2 bg-primary text-white py-1.5 rounded text-xs font-semibold disabled:opacity-50" disabled={uploading}>
              {uploading ? <Loader2 size={12} className="animate-spin" /> : <UploadCloud size={12} />}
              Upload {files.length} file{files.length > 1 ? "s" : ""}
            </button>
            {uploading && uploadingName && (
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-between text-[10px] text-muted">
                  <span className="truncate max-w-[70%]">{uploadingName}</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div className="h-full rounded-full bg-primary transition-all duration-200" style={{ width: `${uploadProgress}%` }} />
                </div>
              </div>
            )}
          </div>
        )}

        {error && <div className="mt-2 text-[10px] text-red-600 bg-red-50 p-2 rounded flex items-start gap-1 whitespace-pre-wrap"><AlertCircle size={12} className="shrink-0 mt-0.5" /> {error}</div>}
        {success && !uploading && <div className="mt-2 text-[10px] text-emerald-600 bg-emerald-50 p-2 rounded flex items-center gap-1"><CheckCircle2 size={12} /> {success}</div>}
      </div>

      {/* Materials List */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <div className="px-2 mb-2 flex items-center justify-between">
          <h3 className="label-caps">
            Sources ({loadingMaterials ? "…" : displayMaterials.length})
          </h3>
          {workspace.collectionId && (
            <button
              type="button"
              onClick={() => loadMaterialsFromServer(workspace.collectionId || null)}
              disabled={loadingMaterials}
              className="text-muted hover:text-primary disabled:opacity-50"
              title="Refresh sources"
              aria-label="Tải lại danh sách nguồn"
            >
              <RefreshCw size={11} className={loadingMaterials ? "animate-spin" : ""} />
            </button>
          )}
        </div>

        {/* Indexing warning banner */}
        {pendingCount > 0 && (
          <div className="mx-2 mb-3 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
            <Loader2 size={12} className="mt-0.5 shrink-0 animate-spin text-amber-500" />
            <p className="text-[10px] leading-snug text-amber-700">
              <span className="font-bold">{pendingCount} tài liệu</span> đang xử lý — chưa tìm kiếm được. Chat sẽ chỉ dùng các nguồn đã index xong.
            </p>
          </div>
        )}
        {displayMaterials.length === 0 ? (
           <div className="text-center p-6 text-muted">
             <FileText size={24} className="mx-auto mb-2 opacity-50" />
             <p className="text-xs">{workspace.collectionId ? "No sources in this collection" : "Select a collection"}</p>
           </div>
        ) : (
          <div className="space-y-1">
            {displayMaterials.map(item => {
              const isPending = !isTerminalStatus(item.status);
              return (
              <div key={item.material_id} className={`flex items-start gap-2 p-2 rounded-lg hover:bg-black/5 bg-white border transition group ${isPending ? "border-amber-200 bg-amber-50/50" : "border-transparent hover:border-outline"}`}>
                {isPending
                  ? <Loader2 size={14} className="mt-0.5 shrink-0 animate-spin text-amber-500" />
                  : (() => { const fi = getFileIconInfo(item.original_name, item.file_type); return <span className={`mt-0.5 shrink-0 ${fi.colorClass}`}>{fi.icon}</span>; })()}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-semibold text-text" title={item.original_name}>{item.original_name}</p>
                  <div className="flex items-center gap-1 mt-1">
                    <StatusBadge status={item.status} />
                    {item.topic && <span className="text-[10px] text-muted uppercase ml-1">{item.topic}</span>}
                  </div>
                </div>
                <div className="shrink-0 flex items-center gap-1 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition">
                  {!isPending && (
                    <button
                      type="button"
                      onClick={() => setDebugMaterial(item)}
                      title="Inspect OCR / chunks"
                      aria-label={`Xem chi tiết OCR và chunks của ${item.original_name}`}
                      className="rounded p-1 text-muted hover:bg-primary/10 hover:text-primary"
                    >
                      <Search size={11} />
                    </button>
                  )}
                  {confirmDeleteMaterial === item.material_id ? (
                    <>
                      <button
                        type="button"
                        onClick={() => handleDeleteMaterial(item.material_id)}
                        disabled={deletingMaterial === item.material_id}
                        aria-label={`Xác nhận xóa nguồn ${item.original_name}`}
                        className="text-[10px] font-bold text-red-600 hover:text-red-700 disabled:opacity-50"
                      >
                        {deletingMaterial === item.material_id ? <Loader2 size={11} className="animate-spin" /> : "Yes"}
                      </button>
                      <button type="button" onClick={() => setConfirmDeleteMaterial(null)} className="text-[10px] text-muted hover:text-text">No</button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleDeleteMaterial(item.material_id)}
                      disabled={deletingMaterial === item.material_id}
                      title="Delete source"
                      aria-label={`Xóa nguồn ${item.original_name}`}
                      className="rounded p-1 text-muted hover:bg-red-50 hover:text-red-500 disabled:opacity-50"
                    >
                      <Trash2 size={11} />
                    </button>
                  )}
                </div>
              </div>
            );
            })}
          </div>
        )}
      </div>
      {debugMaterial && (
        <DebugModal
          materialId={debugMaterial.material_id}
          ownerId={workspace.ownerId}
          originalName={debugMaterial.original_name}
          onClose={() => setDebugMaterial(null)}
        />
      )}
    </div>
  );
}
