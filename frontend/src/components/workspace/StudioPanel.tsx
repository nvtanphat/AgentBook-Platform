import { BookOpen, GitBranch, Network, Share2, Sparkles } from "lucide-react";
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
};

const TABS: { id: StudioTab; label: string; icon: React.ReactNode }[] = [
  { id: "studio", label: "Studio", icon: <Sparkles size={13} /> },
  { id: "evidence", label: "Evidence", icon: <BookOpen size={13} /> },
  { id: "compare", label: "Compare", icon: <GitBranch size={13} /> },
  { id: "graph", label: "Graph", icon: <Network size={13} /> },
  { id: "mindmap", label: "Mindmap", icon: <Share2 size={13} /> },
];

export default function StudioPanel({ activeTab, onTabChange, evidenceDocId, evidencePage, onOpenEvidence }: StudioPanelProps) {
  return (
    <div className="flex h-full flex-col bg-white">
      {/* Tabs Header */}
      <div className="flex shrink-0 border-b border-outline overflow-x-auto no-scrollbar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`flex flex-1 items-center justify-center gap-1 py-3 px-2 text-[10px] font-semibold uppercase tracking-wider transition whitespace-nowrap ${
              activeTab === tab.id ? "border-b-2 border-primary text-primary" : "text-muted hover:text-text"
            }`}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "studio" && <StudioHomeTab />}
        {activeTab === "evidence" && <EvidencePanel docId={evidenceDocId} page={evidencePage} />}
        {activeTab === "compare" && <CompareTab onOpenEvidence={() => onTabChange("evidence")} />}
        {activeTab === "graph" && <GraphTab mode="graph" onOpenEvidence={onOpenEvidence} />}
        {activeTab === "mindmap" && <GraphTab mode="mindmap" onOpenEvidence={onOpenEvidence} />}
      </div>
    </div>
  );
}
