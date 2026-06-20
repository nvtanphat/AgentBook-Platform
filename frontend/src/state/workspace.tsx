import { ReactNode, createContext, useContext, useEffect, useMemo, useState } from "react";
import { Citation, MaterialUploadResponse, QueryResponse } from "../api/client";
import { useAuth } from "./auth";

const WORKSPACE_STORAGE_KEY = "prism.workspace.v1";
const LEGACY_WORKSPACE_STORAGE_KEY = "agentbook.workspace.v2";
const MATERIALS_STORAGE_KEY = "prism.materials.v1";
const SOURCE_SCOPE_STORAGE_KEY = "prism.sourceScope.v1";

export type WorkspaceSettings = {
  ownerId: string;
  collectionId: string;
  collectionName: string;
  subject: string;
  language: string;
  topK: number;
};

export type UploadedMaterial = {
  materialId: string;
  docId: string;
  collectionId: string;
  jobId: string;
  status: string;
  stage: string;
  originalName: string;
  filename: string;
  fileSizeBytes: number;
  topic: string;
  language: string;
  uploadedAt: string;
};

export type ActiveQueryContext = {
  question: string;
  response: QueryResponse;
  createdAt: string;
};

export type ReadySourceSummary = {
  materialId: string;
  name: string;
  topic?: string | null;
};

type WorkspaceContextValue = {
  workspace: WorkspaceSettings;
  updateWorkspace: (settings: Partial<WorkspaceSettings>) => void;
  materials: UploadedMaterial[];
  addUploadedMaterial: (response: MaterialUploadResponse, metadata: { topic: string; language: string }) => void;
  updateMaterialStatus: (materialId: string, status: string, stage: string) => void;
  removeUploadedMaterial: (materialId: string) => void;
  clearUploadedMaterialsForCollection: (collectionId: string) => void;
  selectedCitation: Citation | null;
  setSelectedCitation: (citation: Citation) => void;
  activeCitations: Citation[];
  setActiveCitations: (citations: Citation[]) => void;
  activeQueryContext: ActiveQueryContext | null;
  setActiveQueryContext: (context: ActiveQueryContext | null) => void;
  chatDraft: string | null;
  setChatDraft: (draft: string | null) => void;
  // When true, GraphTab will fetch a focused subgraph filtered to entities
  // that back the last answer (using citations as evidence anchors).
  graphFocusOnAnswer: boolean;
  setGraphFocusOnAnswer: (value: boolean) => void;
  sourceScopeMode: "all" | "selected";
  selectedSourceIds: string[];
  setSourceScopeMode: (mode: "all" | "selected") => void;
  setSelectedSourceIds: (ids: string[]) => void;
  scopedMaterialIds: string[];
  readySourceCount: number;
  readySources: ReadySourceSummary[];
  setReadySources: (sources: ReadySourceSummary[]) => void;
};

const defaultWorkspace: WorkspaceSettings = {
  ownerId: "user_demo",
  collectionId: "",
  collectionName: "",
  subject: "",
  language: "vi",
  topK: 5
};

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function readWorkspaceStorage(): WorkspaceSettings {
  const stored = readStorage<WorkspaceSettings | null>(WORKSPACE_STORAGE_KEY, null);
  if (stored) return { ...defaultWorkspace, ...stored };

  const legacy = readStorage<WorkspaceSettings | null>(LEGACY_WORKSPACE_STORAGE_KEY, null);
  if (!legacy) return defaultWorkspace;

  const migrated = { ...defaultWorkspace, ...legacy };
  if (migrated.collectionName === "Machine Learning") {
    migrated.collectionName = migrated.collectionId ? "Active collection" : "";
  }
  if (migrated.subject === "Machine Learning") {
    migrated.subject = "";
  }
  return migrated;
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [workspace, setWorkspace] = useState<WorkspaceSettings>(() => readWorkspaceStorage());

  // When logged-in user changes, sync owner_id so scoped queries use the auth user
  useEffect(() => {
    if (user && user.user_id !== workspace.ownerId) {
      setWorkspace((prev) => ({ ...prev, ownerId: user.user_id, collectionId: "", collectionName: "" }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.user_id]);
  const [materials, setMaterials] = useState<UploadedMaterial[]>(() => readStorage(MATERIALS_STORAGE_KEY, []));
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const [activeQueryContext, setActiveQueryContext] = useState<ActiveQueryContext | null>(null);
  const [chatDraft, setChatDraft] = useState<string | null>(null);
  const [graphFocusOnAnswer, setGraphFocusOnAnswer] = useState<boolean>(false);
  const [readySources, setReadySourcesState] = useState<ReadySourceSummary[]>([]);
  const [sourceScopeMode, setSourceScopeModeState] = useState<"all" | "selected">(() => {
    const stored = readStorage<{ mode: string; ids: string[] }>(SOURCE_SCOPE_STORAGE_KEY, { mode: "all", ids: [] });
    return stored.mode === "selected" ? "selected" : "all";
  });
  const [selectedSourceIds, setSelectedSourceIdsState] = useState<string[]>(() => {
    const stored = readStorage<{ mode: string; ids: string[] }>(SOURCE_SCOPE_STORAGE_KEY, { mode: "all", ids: [] });
    return Array.isArray(stored.ids) ? stored.ids : [];
  });

  useEffect(() => {
    localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(workspace));
  }, [workspace]);

  useEffect(() => {
    localStorage.setItem(MATERIALS_STORAGE_KEY, JSON.stringify(materials));
  }, [materials]);

  useEffect(() => {
    localStorage.setItem(SOURCE_SCOPE_STORAGE_KEY, JSON.stringify({ mode: sourceScopeMode, ids: selectedSourceIds }));
  }, [selectedSourceIds, sourceScopeMode]);

  const value = useMemo<WorkspaceContextValue>(() => {
    const updateWorkspace = (settings: Partial<WorkspaceSettings>) => {
      setWorkspace((current) => ({ ...current, ...settings }));
    };

    const addUploadedMaterial = (response: MaterialUploadResponse, metadata: { topic: string; language: string }) => {
      const item: UploadedMaterial = {
        materialId: response.material_id,
        docId: response.doc_id,
        collectionId: response.collection_id,
        jobId: response.job_id,
        status: response.status,
        stage: response.stage,
        originalName: response.original_name,
        filename: response.filename,
        fileSizeBytes: response.file_size_bytes,
        topic: metadata.topic,
        language: metadata.language,
        uploadedAt: new Date().toISOString()
      };
      setMaterials((current) => [item, ...current.filter((material) => material.materialId !== item.materialId)]);
      setWorkspace((current) => ({ ...current, collectionId: response.collection_id }));
    };

    const updateMaterialStatus = (materialId: string, status: string, stage: string) => {
      setMaterials((current) =>
        current.map((material) =>
          material.materialId === materialId ? { ...material, status, stage } : material
        )
      );
    };

    const removeUploadedMaterial = (materialId: string) => {
      setMaterials((current) => current.filter((material) => material.materialId !== materialId));
      setSelectedSourceIdsState((current) => current.filter((id) => id !== materialId));
    };

    const clearUploadedMaterialsForCollection = (collectionId: string) => {
      setMaterials((current) => current.filter((material) => material.collectionId !== collectionId));
    };

    const setSourceScopeMode = (mode: "all" | "selected") => {
      setSourceScopeModeState(mode);
      if (mode === "all") setSelectedSourceIdsState([]);
    };

    const setSelectedSourceIds = (ids: string[]) => {
      const unique = Array.from(new Set(ids.filter(Boolean)));
      setSelectedSourceIdsState(unique);
      setSourceScopeModeState("selected");
    };

    const setReadySources = (sources: ReadySourceSummary[]) => {
      setReadySourcesState((current) => {
        const next = sources.map((source) => ({
          materialId: source.materialId,
          name: source.name,
          topic: source.topic ?? null,
        }));
        return JSON.stringify(current) === JSON.stringify(next) ? current : next;
      });
    };

    const localIndexedIds = materials
      .filter((item) => item.status.toLowerCase() === "indexed")
      .filter((item) => !workspace.collectionId || item.collectionId === workspace.collectionId)
      .map((item) => item.materialId);
    const readySourceCount = sourceScopeMode === "selected" ? selectedSourceIds.length : localIndexedIds.length;

    return {
      workspace,
      updateWorkspace,
      materials,
      addUploadedMaterial,
      updateMaterialStatus,
      removeUploadedMaterial,
      clearUploadedMaterialsForCollection,
      selectedCitation,
      setSelectedCitation,
      activeCitations,
      setActiveCitations,
      activeQueryContext,
      setActiveQueryContext,
      chatDraft,
      setChatDraft,
      graphFocusOnAnswer,
      setGraphFocusOnAnswer,
      sourceScopeMode,
      selectedSourceIds,
      setSourceScopeMode,
      setSelectedSourceIds,
      scopedMaterialIds: sourceScopeMode === "selected" ? selectedSourceIds : (workspace.collectionId ? [] : localIndexedIds),
      readySourceCount,
      readySources,
      setReadySources,
    };
  }, [activeCitations, activeQueryContext, chatDraft, graphFocusOnAnswer, materials, readySources, selectedCitation, selectedSourceIds, sourceScopeMode, workspace]);

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace() {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used inside WorkspaceProvider");
  }
  return context;
}
