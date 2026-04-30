import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { BookOpen, GitBranch, Network, Share2, Sparkles, X } from "lucide-react";
import SourcesPanel from "../components/workspace/SourcesPanel";
import ChatPanel from "../components/workspace/ChatPanel";
import StudioPanel from "../components/workspace/StudioPanel";
import { useWorkspace } from "../state/workspace";

export type StudioTab = "evidence" | "studio" | "graph" | "mindmap" | "compare";

const MOBILE_TABS: { id: StudioTab; label: string; icon: React.ReactNode }[] = [
  { id: "studio",   label: "Studio",   icon: <Sparkles size={16} /> },
  { id: "evidence", label: "Evidence", icon: <BookOpen size={16} /> },
  { id: "compare",  label: "Compare",  icon: <GitBranch size={16} /> },
  { id: "graph",    label: "Graph",    icon: <Network size={16} /> },
  { id: "mindmap",  label: "Mindmap",  icon: <Share2 size={16} /> },
];

export default function WorkspacePage() {
  const [searchParams] = useSearchParams();
  const [rightTab, setRightTab] = useState<StudioTab>("studio");
  const [showSourcesMobile, setShowSourcesMobile] = useState(false);
  const [showStudioMobile, setShowStudioMobile] = useState(false);
  const [evidenceDocId, setEvidenceDocId] = useState<string | null>(null);
  const [evidencePage, setEvidencePage] = useState<number | null>(null);

  useEffect(() => {
    const panel = searchParams.get("panel");
    if (panel === "evidence" || panel === "studio" || panel === "graph" || panel === "mindmap" || panel === "compare") {
      setRightTab(panel);
    }
    if (panel === "evidence") {
      const doc = searchParams.get("doc");
      const page = searchParams.get("page");
      setEvidenceDocId(doc || null);
      setEvidencePage(page ? parseInt(page, 10) : null);
    }
  }, [searchParams]);

  function handleMobileTabChange(tab: StudioTab) {
    setRightTab(tab);
    setShowStudioMobile(true);
  }

  function handleTabChange(tab: StudioTab) {
    setRightTab(tab);
    // On mobile, opening a tab via suggestion chip or citation should also open the drawer
    if (window.innerWidth < 1024) setShowStudioMobile(true);
  }

  return (
    <div className="flex h-full overflow-hidden">
      {showSourcesMobile && (
        <button
          type="button"
          aria-label="Đóng Sources"
          className="fixed inset-0 z-10 bg-black/30 lg:hidden"
          onClick={() => setShowSourcesMobile(false)}
        />
      )}

      {/* ── Left Panel: Sources ── */}
      <div
        className={`shrink-0 w-full lg:w-[340px] border-r border-outline bg-slate-50 flex flex-col absolute lg:relative z-20 h-full transition-transform duration-200 ${showSourcesMobile ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
      >
        <SourcesPanel onCloseMobile={() => setShowSourcesMobile(false)} />
      </div>

      {/* ── Center Panel: Chat ── */}
      {/* pb-14 on mobile for bottom tab bar */}
      <div className="flex min-w-0 flex-1 flex-col bg-background relative z-10 pb-14 lg:pb-0">
        <ChatPanel
          onOpenSources={() => setShowSourcesMobile(true)}
          onOpenEvidence={() => handleTabChange("evidence")}
          onTabChange={handleTabChange}
        />
      </div>

      {/* ── Right Panel: Studio/Evidence (desktop) ── */}
      <div className="hidden w-[420px] shrink-0 border-l border-outline bg-white lg:flex lg:flex-col">
        <StudioPanel
          activeTab={rightTab}
          onTabChange={setRightTab}
          evidenceDocId={evidenceDocId}
          evidencePage={evidencePage}
        />
      </div>

      {/* ── Mobile Studio Slide-Over ── */}
      {showStudioMobile && (
        <div className="lg:hidden fixed inset-0 z-40 flex flex-col bg-white">
          {/* Close bar */}
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
              onTabChange={(tab) => { setRightTab(tab); }}
              evidenceDocId={evidenceDocId}
              evidencePage={evidencePage}
            />
          </div>
        </div>
      )}

      {/* ── Mobile Bottom Tab Bar ── */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-30 flex border-t border-outline bg-white shadow-lg">
        {MOBILE_TABS.map((tab) => (
          <button
            type="button"
            key={tab.id}
            onClick={() => handleMobileTabChange(tab.id)}
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
