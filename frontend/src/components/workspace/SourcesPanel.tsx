import { DragEvent, useCallback, useEffect, useMemo, useRef, useState, useId } from "react";
import { AlertCircle, BookOpen, Check, FileAudio, FileText, Image, Loader2, Pencil, Plus, RefreshCw, Search, Table2, Trash2, UploadCloud, X } from "lucide-react";
import { CollectionSummary, MaterialInfo, MaterialUploadMetadata, createCollection, deleteCollection, deleteMaterial, getMaterialStatus, listCollections, listMaterials, retryMaterial, updateCollection, uploadMaterialsBatchWithProgress } from "../../api/client";
import StatusBadge from "../StatusBadge";
import { useWorkspace } from "../../state/workspace";
import DebugModal from "./DebugModal";
import { useToast } from "../Toast";

const ACCEPTED_TYPES = ".pdf,.docx,.pptx,.png,.jpg,.jpeg,.csv,.xlsx,.mp3,.wav,.m4a,.ogg,.flac,.webm,.aac";
const ACCEPTED_MIME = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "image/png",
  "image/jpeg",
  "text/csv",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  // Audio MIME types
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/mp4",
  "audio/x-m4a",
  "audio/m4a",
  "audio/ogg",
  "application/ogg",
  "audio/flac",
  "audio/x-flac",
  "audio/webm",
  "video/webm",
  "audio/aac",
  "audio/x-aac",
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
  if (/\.(mp3|wav|m4a|ogg|flac|webm|aac)$/.test(name) || mime.includes("audio"))
    return { icon: <FileAudio size={14} />, colorClass: "text-pink-500" };
  return { icon: <FileText size={14} />, colorClass: "text-primary" };
}

// ─── Pipeline stage → progress % ─────────────────────────────────────────────

const STAGE_PCT: Record<string, number> = {
  uploaded:  8,
  parsing:   20,
  parsed:    38,
  chunking:  52,
  embedding: 66,
  indexing:  82,
  indexed:   100,
  failed:    0,
};

function stagePct(status: string, stage?: string | null): number {
  const key = (stage || status || "").toLowerCase();
  return STAGE_PCT[key] ?? STAGE_PCT[status?.toLowerCase()] ?? 10;
}

// ─── Smooth real-time progress hook ──────────────────────────────────────────

function useSmoothedProgress(status: string, stage: string | null): number {
  const target = stagePct(status, stage);

  // Find the next stage's % so we know where to stop creeping
  const sortedStages = Object.entries(STAGE_PCT).sort((a, b) => a[1] - b[1]);
  const ceiling = (() => {
    const above = sortedStages.filter(([, v]) => v > target);
    return above.length > 0 ? above[0][1] - 2 : target;
  })();

  const [displayed, setDisplayed] = useState<number>(target);

  // Jump forward immediately when a real stage change arrives
  useEffect(() => {
    setDisplayed((prev) => Math.max(prev, target));
  }, [target]);

  // Creep slowly within stage (~0.2% per 600ms ≈ fills gap in ~1 min)
  useEffect(() => {
    const terminal = status === "indexed" || status === "failed";
    if (terminal) return;
    const id = setInterval(() => {
      setDisplayed((prev) => (prev < ceiling ? +(prev + 0.2).toFixed(1) : prev));
    }, 600);
    return () => clearInterval(id);
  }, [ceiling, status]);

  return Math.min(Math.round(displayed), 99); // never show 100 until truly indexed
}

// ─── Circular progress ring ───────────────────────────────────────────────────

function CircularProgress({ status, stage }: { status: string; stage?: string | null }) {
  const SIZE   = 32;
  const STROKE = 3.2;
  const r      = (SIZE - STROKE) / 2;
  const circ   = 2 * Math.PI * r;

  const pct  = useSmoothedProgress(status, stage ?? null);
  const dash = (pct / 100) * circ;

  const isTerminal = status === "indexed" || status === "failed";
  const realPct    = isTerminal ? stagePct(status, stage ?? null) : pct;
  const color      = realPct === 0 ? "#ef4444" : realPct >= 100 ? "#10b981" : "#f59e0b";
  const label      = stage || status || "";

  return (
    <div
      className="relative shrink-0 mt-0.5"
      style={{ width: SIZE, height: SIZE }}
      title={`${label} — ${realPct}%`}
    >
      <svg width={SIZE} height={SIZE} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={SIZE / 2} cy={SIZE / 2} r={r} fill="none" stroke="#e2e8f0" strokeWidth={STROKE} />
        <circle
          cx={SIZE / 2} cy={SIZE / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeDasharray={`${(realPct / 100) * circ} ${circ}`}
          strokeLinecap="round"
        />
      </svg>
      <span
        className="absolute inset-0 flex items-center justify-center font-bold tabular-nums"
        style={{ fontSize: 7, color, letterSpacing: "-0.02em" }}
      >
        {realPct}%
      </span>
    </div>
  );
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
    sourceScopeMode,
    selectedSourceIds,
    setSourceScopeMode,
    setSelectedSourceIds,
    setReadySources,
  } = useWorkspace();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const didAutoSelectCollectionRef = useRef(false);
  const checkboxIdPrefix = useId();

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
  const [deletingCollection, setDeletingCollection] = useState(false);
  const [creatingCollection, setCreatingCollection] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [renamingCollection, setRenamingCollection] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [savingRename, setSavingRename] = useState(false);
  const [deletingMaterial, setDeletingMaterial] = useState<string | null>(null);
  const [confirmDeleteMaterial, setConfirmDeleteMaterial] = useState<string | null>(null);
  const [debugMaterial, setDebugMaterial] = useState<MaterialInfo | null>(null);
  const [retryingMaterial, setRetryingMaterial] = useState<string | null>(null);
  const prevPendingCountRef = useRef<number>(0);

  const loadCollections = useCallback(async () => {
    setLoadingCollections(true);
    try {
      const items = await listCollections(workspace.ownerId);
      setCollections(items);
      const currentStillExists = items.some((c) => c.collection_id === workspace.collectionId);
      if (workspace.collectionId && !currentStillExists) {
        updateWorkspace({ collectionId: "", collectionName: "" });
      } else if (workspace.collectionId && !workspace.collectionName) {
        const current = items.find((c) => c.collection_id === workspace.collectionId);
        if (current) {
          updateWorkspace({ collectionName: current.name, subject: current.subject ?? workspace.subject });
        }
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

  useEffect(() => {
    setRenamingCollection(false);
    setRenameValue(workspace.collectionName);
  }, [workspace.collectionId, workspace.collectionName]);

  // Notify when all pending materials finish processing
  useEffect(() => {
    if (prevPendingCountRef.current > 0 && pendingCount === 0 && workspace.collectionId) {
      toast(`Tài liệu trong "${workspace.collectionName || "collection"}" đã sẵn sàng để hỏi đáp!`, "success");
    }
    prevPendingCountRef.current = pendingCount;
  }, [pendingCount, workspace.collectionId, workspace.collectionName, toast]);

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
      toast(`${count} file${count > 1 ? "s" : ""} đã upload thành công!`, "success");
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
      toast("Collection đã xóa.", "success");
      await loadCollections();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Xóa collection thất bại.", "error");
    } finally {
      setDeletingCollection(false);
    }
  }

  async function handleCreateCollection() {
    const name = workspace.collectionName.trim();
    if (!name || creatingCollection) return;
    setCreatingCollection(true);
    setError(null);
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
      toast(`Collection "${collection.name}" đã tạo.`, "success");
      await loadCollections();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Tạo collection thất bại.", "error");
    } finally {
      setCreatingCollection(false);
    }
  }

  async function handleRenameCollection() {
    const name = renameValue.trim();
    if (!workspace.collectionId || !name || savingRename) return;
    setSavingRename(true);
    setError(null);
    try {
      const updated = await updateCollection(workspace.collectionId, {
        owner_id: workspace.ownerId,
        name,
        subject: selectedCollection?.subject ?? (workspace.subject || null),
        description: selectedCollection?.description ?? null,
      });
      updateWorkspace({ collectionName: updated.name, subject: updated.subject ?? workspace.subject });
      setCollections((current) => current.map((item) => item.collection_id === updated.collection_id ? { ...item, ...updated } : item));
      setRenamingCollection(false);
      toast(`Đổi tên thành "${updated.name}".`, "success");
      await loadCollections();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Đổi tên thất bại.", "error");
    } finally {
      setSavingRename(false);
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
      toast("Đã xóa tài liệu.", "success");
      await Promise.all([
        loadCollections(),
        loadMaterialsFromServer(workspace.collectionId || null),
      ]);
    } catch (err) {
      if (isNotFoundError(err)) {
        removeUploadedMaterial(materialId);
        toast("Tài liệu đã bị xóa trước đó.", "info");
        await Promise.all([
          loadCollections(),
          loadMaterialsFromServer(workspace.collectionId || null),
        ]);
        return;
      }
      toast(err instanceof Error ? err.message : "Xóa tài liệu thất bại.", "error");
    } finally {
      setDeletingMaterial(null);
    }
  }

  async function handleRetryMaterial(materialId: string) {
    setRetryingMaterial(materialId);
    setError(null);
    try {
      await retryMaterial(materialId, workspace.ownerId);
      toast("Pipeline retry đã được xếp hàng.", "info");
      await loadMaterialsFromServer(workspace.collectionId || null);
    } catch (err) {
      toast(err instanceof Error ? err.message : "Retry thất bại.", "error");
    } finally {
      setRetryingMaterial(null);
    }
  }

  const selectedCollection = collections.find((c) => c.collection_id === workspace.collectionId);

  function selectCollection(collectionId: string) {
    setConfirmDelete(false);
    if (!collectionId) {
      didAutoSelectCollectionRef.current = true;
      updateWorkspace({ collectionId: "", collectionName: "" });
      setSourceScopeMode("all");
      return;
    }
    const col = collections.find((c) => c.collection_id === collectionId);
    updateWorkspace({ collectionId: collectionId, collectionName: col?.name ?? "", subject: col?.subject ?? workspace.subject });
    setSourceScopeMode("all");
  }

  const displayMaterials = useMemo(() => [
    ...serverMaterials,
    ...sessionOnlyMaterials.map((m) => ({
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
    } satisfies MaterialInfo)),
  // sessionOnlyMaterials already derived from materials + serverIds
  // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [serverMaterials, sessionOnlyMaterials]);
  const indexedMaterialIds = displayMaterials
    .filter((item) => item.status.toLowerCase() === "indexed")
    .map((item) => item.material_id);
  const activeSourceIds = sourceScopeMode === "selected"
    ? new Set(selectedSourceIds)
    : new Set(indexedMaterialIds);
  const activeCount = indexedMaterialIds.filter((id) => activeSourceIds.has(id)).length;

  useEffect(() => {
    setReadySources(
      displayMaterials
        .filter((item) => item.status.toLowerCase() === "indexed")
        .map((item) => ({
          materialId: item.material_id,
          name: item.original_name,
          topic: item.topic,
        }))
    );
  }, [displayMaterials, setReadySources]);

  function toggleSource(materialId: string, enabled: boolean) {
    const base = sourceScopeMode === "selected" ? selectedSourceIds : indexedMaterialIds;
    const next = enabled
      ? Array.from(new Set([...base, materialId]))
      : base.filter((id) => id !== materialId);
    if (next.length === indexedMaterialIds.length && indexedMaterialIds.every((id) => next.includes(id))) {
      setSourceScopeMode("all");
    } else {
      setSelectedSourceIds(next);
    }
  }

  return (
    <div className="flex flex-col h-full bg-slate-50/80 relative">
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

      {/* Collection Manager */}
      <div className="p-4 bg-white/90 shrink-0 section-divider" style={{ backdropFilter: 'blur(8px)' }}>
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="label-caps">Bộ tài liệu</span>
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
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => { setRenameValue(workspace.collectionName); setRenamingCollection(true); }}
                      title="Rename collection"
                      aria-label="Rename collection"
                      className="rounded p-1 text-muted hover:bg-slate-100 hover:text-primary disabled:opacity-50"
                      disabled={deletingCollection}
                    >
                      <Pencil size={12} />
                    </button>
                    <button
                      type="button"
                      onClick={handleDeleteCollection}
                      title="Delete collection"
                      aria-label="Delete collection"
                      className="rounded p-1 text-muted hover:bg-red-50 hover:text-red-500 disabled:opacity-50"
                      disabled={deletingCollection}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                )
              )}
              <button
                type="button"
                onClick={loadCollections}
                disabled={loadingCollections}
                aria-label="Tải lại danh sách bộ tài liệu"
                className="rounded p-1 text-muted hover:bg-slate-100 hover:text-primary disabled:opacity-50"
              >
                <RefreshCw size={12} className={loadingCollections ? "animate-spin" : ""} />
              </button>
            </div>
          </div>

          <div className="flex gap-2">
            <select
              className="min-w-0 flex-1 rounded-md border border-outline bg-slate-50 px-2 py-1.5 text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-primary"
              value={workspace.collectionId}
              onChange={(e) => selectCollection(e.target.value)}
              disabled={loadingCollections}
            >
              <option value="">{loadingCollections ? "Loading..." : "+ Tạo bộ tài liệu"}</option>
              {collections.map((collection) => (
                <option key={collection.collection_id} value={collection.collection_id}>
                  {collection.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => selectCollection("")}
              title="Tạo bộ tài liệu"
              aria-label="Tạo bộ tài liệu"
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary text-white hover:bg-primary-bright"
            >
              <Plus size={14} />
            </button>
          </div>

          {!workspace.collectionId && (
            <div className="flex gap-2">
              <input
                className="min-w-0 flex-1 rounded-md border border-outline px-2 py-1.5 text-xs"
                placeholder="Tên bộ tài liệu..."
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
                Tạo
              </button>
            </div>
          )}

          {workspace.collectionId && renamingCollection && (
            <div className="flex gap-2">
              <input
                className="min-w-0 flex-1 rounded-md border border-outline px-2 py-1.5 text-xs"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleRenameCollection();
                  if (e.key === "Escape") setRenamingCollection(false);
                }}
                autoFocus
              />
              <button
                type="button"
                onClick={handleRenameCollection}
                disabled={savingRename || !renameValue.trim()}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary text-white disabled:opacity-50"
                title="Save name"
                aria-label="Save collection name"
              >
                {savingRename ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
              </button>
              <button
                type="button"
                onClick={() => setRenamingCollection(false)}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-outline text-muted hover:bg-slate-50"
                title="Cancel"
                aria-label="Cancel rename"
              >
                <X size={13} />
              </button>
            </div>
          )}

          {selectedCollection && (
            <div className="flex items-center gap-3 text-[10px] font-medium text-muted">
              <span>{selectedCollection.material_count} tài liệu</span>
              <span>{selectedCollection.retrievable_chunk_count} chunks</span>
              {selectedCollection.latest_material_name && (
                <span className="min-w-0 flex-1 truncate text-right" title={selectedCollection.latest_material_name}>
                  {selectedCollection.latest_material_name}
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Dropzone */}
      <div className="p-4 shrink-0">
        <div
          className={`cursor-pointer rounded-xl border-2 border-dashed p-4 text-center transition-all ${
            dragOver ? "border-primary bg-primary/5 shadow-sm" : "border-outline/50 bg-white/80 hover:border-primary/40 hover:shadow-sm"
          } ${uploading ? "opacity-50 pointer-events-none" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !uploading && fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" className="hidden" multiple accept={ACCEPTED_TYPES} onChange={(e) => { addFiles(e.target.files); if(e.target) e.target.value = ""; }} />
          <UploadCloud size={18} className="mx-auto mb-2 text-primary/60" />
          <p className="text-xs font-semibold text-text/80">Click to upload or drag files</p>
          <p className="text-[10px] text-muted/50 mt-1">PDF, DOCX, PPTX, XLSX, Images, Audio (MP3/WAV/M4A)</p>
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

        {displayMaterials.length > 0 && (
          <div className="mx-2 mb-3 rounded-lg border border-outline bg-white p-2">
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <p className="text-[10px] font-bold uppercase tracking-wide text-muted">Query scope</p>
                <p className="truncate text-xs font-semibold text-text">
                  {sourceScopeMode === "selected"
                    ? `${activeCount} selected source${activeCount === 1 ? "" : "s"}`
                    : `All indexed sources (${indexedMaterialIds.length})`}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSourceScopeMode("all")}
                className="shrink-0 rounded-md border border-outline px-2 py-1 text-[10px] font-semibold text-muted hover:border-primary/40 hover:text-primary"
              >
                Select all
              </button>
            </div>
            {sourceScopeMode === "selected" && activeCount === 0 && (
              <p className="mt-2 rounded bg-amber-50 px-2 py-1 text-[10px] text-amber-700">
                No active source selected. Select all or choose a file before asking.
              </p>
            )}
          </div>
        )}

        {/* Indexing warning banner */}
        {pendingCount > 0 && (
          <div className="mx-2 mb-3 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
            <Loader2 size={12} className="mt-0.5 shrink-0 animate-spin text-amber-500" />
            <p className="text-[10px] leading-snug text-amber-700">
              <span className="font-bold">{pendingCount} tài liệu</span> đang xử lý — chưa tìm kiếm được. Chat sẽ chỉ dùng các nguồn đã index xong.
            </p>
          </div>
        )}
        {loadingMaterials && displayMaterials.length === 0 ? (
          <div className="space-y-1 px-0" aria-label="Đang tải danh sách tài liệu">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="flex items-start gap-2 p-2 rounded-lg bg-white border border-transparent animate-pulse">
                <div className="mt-0.5 h-5 w-5 shrink-0 rounded bg-slate-200" />
                <div className="h-4 w-4 shrink-0 rounded bg-slate-200" />
                <div className="flex-1 space-y-1.5 pt-0.5">
                  <div className="h-3 rounded bg-slate-200" style={{ width: `${60 + (i * 13) % 30}%` }} />
                  <div className="h-2.5 w-16 rounded bg-slate-100" />
                </div>
              </div>
            ))}
          </div>
        ) : displayMaterials.length === 0 ? (
           <div className="text-center p-6 text-muted">
             <FileText size={24} className="mx-auto mb-2 opacity-50" />
             <p className="text-xs">{workspace.collectionId ? "Chưa có tài liệu trong bộ này" : "Chọn một bộ tài liệu"}</p>
           </div>
        ) : (
          <div className="space-y-1">
            {displayMaterials.map(item => {
              const isPending = !isTerminalStatus(item.status);
              const isActiveSource = activeSourceIds.has(item.material_id) && !isPending;
              return (
              <div key={item.material_id} className={`source-item flex items-start gap-2 p-2.5 rounded-xl bg-white border transition group ${isPending ? "border-amber-200/60 bg-amber-50/30" : isActiveSource ? "border-primary/20 bg-primary/[0.03]" : "border-transparent hover:border-outline/40 opacity-70"}`}>
                {isPending ? (
                  <CircularProgress status={item.status} stage={(item as any).stage ?? null} />
                ) : (
                  <label
                    htmlFor={`${checkboxIdPrefix}-${item.material_id}`}
                    className="mt-0.5 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center"
                    title={isActiveSource ? "Đang dùng — nhấn để bỏ chọn" : "Nhấn để chọn nguồn này"}
                  >
                    <input
                      id={`${checkboxIdPrefix}-${item.material_id}`}
                      type="checkbox"
                      checked={isActiveSource}
                      onChange={(e) => toggleSource(item.material_id, e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-outline text-primary focus:ring-primary"
                      aria-label={`Use ${item.original_name} when asking`}
                    />
                  </label>
                )}
                {!isPending && (() => { const fi = getFileIconInfo(item.original_name, item.file_type); return <span className={`mt-0.5 shrink-0 ${fi.colorClass}`}>{fi.icon}</span>; })()}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-semibold text-text" title={item.original_name}>{item.original_name}</p>
                  <div className="flex items-center gap-1 mt-1">
                    <StatusBadge status={item.status} />
                    {!isPending && !isActiveSource && <span className="text-[10px] text-muted">not used</span>}
                    {item.topic && <span className="text-[10px] text-muted uppercase ml-1">{item.topic}</span>}
                  </div>
                </div>
                <div className="shrink-0 flex items-center gap-1">
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
                  {(item.status === "failed" || item.status === "parsed" || item.status === "parsing" || item.status === "chunking" || item.status === "embedding") && (
                    <button
                      type="button"
                      onClick={() => handleRetryMaterial(item.material_id)}
                      disabled={retryingMaterial === item.material_id}
                      title="Retry pipeline"
                      aria-label={`Thử lại pipeline cho ${item.original_name}`}
                      className="rounded p-1 text-muted hover:bg-amber-50 hover:text-amber-600 disabled:opacity-50"
                    >
                      {retryingMaterial === item.material_id ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
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
