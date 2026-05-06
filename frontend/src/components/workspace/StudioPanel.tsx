import { BookOpen, GitBranch, Network, Sparkles } from "lucide-react";
import EvidencePanel from "../EvidencePanel";
import StudioHomeTab from "./studio/StudioHomeTab";
import CompareTab from "./studio/CompareTab";
import GraphTab from "./studio/GraphTab";
import { StudioTab } from "../../pages/WorkspacePage";

type StudioPanelProps = {
  activeTab: StudioTab;
  onTabChange: (tab: StudioTab) => void;
  evidenceDocId?: string | null;
  evidencePage?: number | null;
  onOpenEvidence?: (target: { docId: string; page: number; blockId?: string | null }) => void;
  visualizeMode: "graph" | "mindmap";
  onVisualizeModeChange: (mode: "graph" | "mindmap") => void;
};

const TABS: { id: StudioTab; label: string; shortLabel: string; icon: React.ReactNode }[] = [
  { id: "studio",    label: "Studio",    shortLabel: "Studio",   icon: <Sparkles size={13} /> },
  { id: "evidence",  label: "Evidence",  shortLabel: "Evidence", icon: <BookOpen size={13} /> },
  { id: "compare",   label: "Compare",   shortLabel: "Compare",  icon: <GitBranch size={13} /> },
  { id: "visualize", label: "Visualize", shortLabel: "Visual",   icon: <Network size={13} /> },
];

export default function StudioPanel({ activeTab, onTabChange, evidenceDocId, evidencePage, onOpenEvidence, visualizeMode, onVisualizeModeChange }: StudioPanelProps) {
  return (
    <div className="flex h-full flex-col bg-white">
      {/* Tabs Header */}
      <div className="flex h-[48px] shrink-0 items-stretch overflow-x-auto no-scrollbar section-divider">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`studio-tab flex flex-1 items-center justify-center gap-1.5 px-1.5 text-[10px] font-semibold uppercase tracking-wider transition whitespace-nowrap ${
              activeTab === tab.id ? "active text-primary" : "text-muted/60 hover:text-text"
            }`}
            onClick={() => onTabChange(tab.id)}
            title={
              tab.id === "studio" ? "Tóm tắt, study guide, hành động AI" :
              tab.id === "evidence" ? "Xem trang tài liệu nguồn trích dẫn" :
              tab.id === "compare" ? "So sánh nội dung giữa các tài liệu" :
              "Knowledge Graph & Mindmap trực quan"
            }
          >
            <span className={`transition ${activeTab === tab.id ? "text-primary" : "text-muted/40"}`}>
              {tab.icon}
            </span>
            <span className="hidden xl:inline">{tab.label}</span>
            <span className="xl:hidden">{tab.shortLabel}</span>
          </button>
        ))}
      </div>

      {/* Visualize sub-toggle */}
      {activeTab === "visualize" && (
        <div className="shrink-0 flex items-center gap-1 border-b border-outline/30 bg-slate-50/80 px-3 py-1.5">
          <div className="inline-flex overflow-hidden rounded-lg border border-outline/50 bg-white p-0.5">
            <button
              type="button"
              onClick={() => onVisualizeModeChange("graph")}
              className={`px-3 py-1 text-[10px] font-semibold transition rounded-md ${
                visualizeMode === "graph" ? "text-white shadow-sm" : "text-muted hover:text-text"
              }`}
              style={visualizeMode === "graph" ? { background: 'linear-gradient(135deg, #006591 0%, #0ea5e9 100%)' } : undefined}
            >
              Knowledge Graph
            </button>
            <button
              type="button"
              onClick={() => onVisualizeModeChange("mindmap")}
              className={`px-3 py-1 text-[10px] font-semibold transition rounded-md ${
                visualizeMode === "mindmap" ? "text-white shadow-sm" : "text-muted hover:text-text"
              }`}
              style={visualizeMode === "mindmap" ? { background: 'linear-gradient(135deg, #006591 0%, #0ea5e9 100%)' } : undefined}
            >
              Mindmap
            </button>
          </div>
        </div>
      )}

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "studio" && <StudioHomeTab />}
        {activeTab === "evidence" && <EvidencePanel docId={evidenceDocId} page={evidencePage} />}
        {activeTab === "compare" && <CompareTab onOpenEvidence={() => onTabChange("evidence")} />}
        {activeTab === "visualize" && <GraphTab mode={visualizeMode} onOpenEvidence={onOpenEvidence} />}
      </div>
    </div>
  );
}
