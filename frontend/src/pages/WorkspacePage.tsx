import { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { BookOpen, GitBranch, Network, Sparkles, X } from "lucide-react";
import { Group, Panel, Separator } from "react-resizable-panels";
import SourcesPanel from "../components/workspace/SourcesPanel";
import ChatPanel from "../components/workspace/ChatPanel";
import StudioPanel from "../components/workspace/StudioPanel";

export type StudioTab = "evidence" | "studio" | "visualize" | "compare";

const MOBILE_TABS: { id: StudioTab; label: string; icon: React.ReactNode }[] = [
  { id: "studio",    label: "Studio",    icon: <Sparkles size={16} /> },
  { id: "evidence",  label: "Evidence",  icon: <BookOpen size={16} /> },
  { id: "compare",   label: "Compare",   icon: <GitBranch size={16} /> },
  { id: "visualize", label: "Visualize", icon: <Network size={16} /> },
];

// Panel layout persistence
const STORAGE_KEY = "prism.panel-sizes.v4";
const PANEL_IDS = { left: "sources", center: "chat", right: "studio" } as const;
const DEFAULT_LAYOUT = { [PANEL_IDS.left]: 20, [PANEL_IDS.center]: 48, [PANEL_IDS.right]: 32 };

type LayoutMap = typeof DEFAULT_LAYOUT;

function loadLayout(): LayoutMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as LayoutMap;
      const total =
        parsed[PANEL_IDS.left] +
        parsed[PANEL_IDS.center] +
        parsed[PANEL_IDS.right];
      if (
        parsed &&
        typeof parsed === "object" &&
        typeof parsed[PANEL_IDS.left] === "number" &&
        typeof parsed[PANEL_IDS.center] === "number" &&
        typeof parsed[PANEL_IDS.right] === "number" &&
        parsed[PANEL_IDS.left] >= 16 &&
        parsed[PANEL_IDS.left] <= 36 &&
        parsed[PANEL_IDS.center] >= 30 &&
        parsed[PANEL_IDS.right] >= 20 &&
        parsed[PANEL_IDS.right] <= 52 &&
        Math.abs(total - 100) <= 2
      ) {
        return parsed;
      }
    }
  } catch {}
  return { ...DEFAULT_LAYOUT };
}

// True when viewport ≥ lg (1024px). Updates on resize.
function useIsDesktop() {
  const [isDesktop, setIsDesktop] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 1024 : true
  );
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    setIsDesktop(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return isDesktop;
}

function ResizeHandle({ onDoubleClick }: { onDoubleClick?: () => void }) {
  return (
    <Separator
      onDoubleClick={onDoubleClick}
      title="Kéo để thay đổi kích thước · Double-click để reset"
      className="resize-handle"
    >
      <div className="resize-handle-bar" />
    </Separator>
  );
}

export default function WorkspacePage() {
  const [searchParams] = useSearchParams();
  const [rightTab, setRightTab] = useState<StudioTab>("studio");
  const [visualizeMode, setVisualizeMode] = useState<"graph" | "mindmap">("graph");
  const [showSourcesMobile, setShowSourcesMobile] = useState(false);
  const [showStudioMobile, setShowStudioMobile] = useState(false);
  const [evidenceDocId, setEvidenceDocId] = useState<string | null>(null);
  const [evidencePage, setEvidencePage] = useState<number | null>(null);
  // Key trick: increment to force PanelGroup remount when resetting layout
  const [layoutKey, setLayoutKey] = useState(0);

  const isDesktop = useIsDesktop();

  useEffect(() => {
    const panel = searchParams.get("panel");
    const mapped = panel === "graph" || panel === "mindmap" ? "visualize" : panel;
    if (mapped === "evidence" || mapped === "studio" || mapped === "visualize" || mapped === "compare") {
      setRightTab(mapped);
    }
    if (panel && !isDesktop) setShowStudioMobile(true);
    if (panel === "evidence") {
      setEvidenceDocId(searchParams.get("doc") || null);
      const page = searchParams.get("page");
      setEvidencePage(page ? parseInt(page, 10) : null);
    }
  }, [searchParams, isDesktop]);

  function handleTabChange(tab: StudioTab) {
    setRightTab(tab);
    if (!isDesktop) setShowStudioMobile(true);
  }

  function handleTraceAnswerGraph() {
    setVisualizeMode("graph");
    handleTabChange("visualize");
  }

  function handleOpenEvidence(target: { docId: string; page: number; blockId?: string | null }) {
    setEvidenceDocId(target.docId);
    setEvidencePage(target.page);
    handleTabChange("evidence");
  }

  const resetLayout = useCallback(() => {
    try { localStorage.removeItem(STORAGE_KEY); } catch {}
    setLayoutKey((k) => k + 1);
  }, []);

  const savedLayout = loadLayout();

  // ── Desktop layout ──────────────────────────────────────────────────────────
  if (isDesktop) {
    return (
      <div
        className="flex h-full w-full overflow-hidden p-2 gap-0"
        title="Double-click vung trong de reset layout"
        onDoubleClick={(event) => {
          if (event.target === event.currentTarget) resetLayout();
        }}
      >
        <Group
          key={layoutKey}
          orientation="horizontal"
          defaultLayout={savedLayout}
          onLayoutChanged={(layout) => {
            try { localStorage.setItem(STORAGE_KEY, JSON.stringify(layout)); } catch {}
          }}
          className="flex h-full w-full"
        >
          <Panel
            id={PANEL_IDS.left}
            defaultSize={`${savedLayout[PANEL_IDS.left]}%`}
            minSize="16%"
            maxSize="36%"
            className="flex flex-col rounded-xl overflow-hidden bg-white/90 shadow-sm border border-white/60"
            style={{ backdropFilter: 'blur(8px)' }}
          >
            <SourcesPanel />
          </Panel>

          <ResizeHandle onDoubleClick={resetLayout} />

          <Panel
            id={PANEL_IDS.center}
            defaultSize={`${savedLayout[PANEL_IDS.center]}%`}
            minSize="30%"
            className="flex flex-col rounded-xl overflow-hidden bg-white shadow-sm border border-outline/30"
          >
            <ChatPanel
              onOpenSources={() => {}}
              onOpenEvidence={() => handleTabChange("evidence")}
              onTraceGraph={handleTraceAnswerGraph}
              onTabChange={handleTabChange}
            />
          </Panel>

          <ResizeHandle onDoubleClick={resetLayout} />

          <Panel
            id={PANEL_IDS.right}
            defaultSize={`${savedLayout[PANEL_IDS.right]}%`}
            minSize="20%"
            maxSize="52%"
            className="flex flex-col rounded-xl overflow-hidden bg-white shadow-sm border border-outline/30"
          >
            <StudioPanel
              activeTab={rightTab}
              onTabChange={setRightTab}
              visualizeMode={visualizeMode}
              onVisualizeModeChange={setVisualizeMode}
              evidenceDocId={evidenceDocId}
              evidencePage={evidencePage}
              onOpenEvidence={handleOpenEvidence}
            />
          </Panel>
        </Group>
      </div>
    );
  }

  // ── Mobile layout ───────────────────────────────────────────────────────────
  return (
    <div className="flex h-full overflow-hidden">
      {/* Backdrop */}
      {showSourcesMobile && (
        <button
          type="button"
          aria-label="Đóng Sources"
          className="fixed inset-0 z-10 bg-black/30"
          onClick={() => setShowSourcesMobile(false)}
        />
      )}

      {/* Sources slide-over — always mounted so workspace state is preserved */}
      {showSourcesMobile && (
        <div className="fixed inset-y-0 left-0 z-20 flex w-full flex-col border-r border-outline bg-slate-50">
          <SourcesPanel onCloseMobile={() => setShowSourcesMobile(false)} />
        </div>
      )}

      {/* Chat */}
      <div className="flex min-w-0 flex-1 flex-col bg-background relative z-10 pb-[calc(3.5rem+env(safe-area-inset-bottom))]">
        <ChatPanel
          onOpenSources={() => setShowSourcesMobile(true)}
          onOpenEvidence={() => handleTabChange("evidence")}
          onTraceGraph={handleTraceAnswerGraph}
          onTabChange={handleTabChange}
        />
      </div>

      {/* Studio slide-over */}
      {showStudioMobile && (
        <div className="fixed inset-0 z-40 flex flex-col bg-white">
          <div className="flex shrink-0 items-center justify-between border-b border-outline px-4 py-3 bg-white">
            <span className="text-sm font-semibold text-text capitalize">{rightTab}</span>
            <button
              type="button"
              aria-label="Đóng panel Studio"
              onClick={() => setShowStudioMobile(false)}
              className="rounded-md p-1.5 text-muted hover:bg-slate-100 transition"
            >
              <X size={18} />
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <StudioPanel
              activeTab={rightTab}
              onTabChange={(tab) => setRightTab(tab)}
              visualizeMode={visualizeMode}
              onVisualizeModeChange={setVisualizeMode}
              evidenceDocId={evidenceDocId}
              evidencePage={evidencePage}
              onOpenEvidence={handleOpenEvidence}
            />
          </div>
        </div>
      )}

      {/* Bottom Tab Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-30 flex border-t border-outline bg-white shadow-lg pb-[env(safe-area-inset-bottom)]">
        {MOBILE_TABS.map((tab) => (
          <button
            type="button"
            key={tab.id}
            onClick={() => { setRightTab(tab.id); setShowStudioMobile(true); }}
            aria-pressed={showStudioMobile && rightTab === tab.id}
            className={`flex flex-1 flex-col items-center justify-center gap-0.5 py-2 text-[10px] font-semibold transition ${
              showStudioMobile && rightTab === tab.id
                ? "text-primary border-t-2 border-primary -mt-px"
                : "text-muted"
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
