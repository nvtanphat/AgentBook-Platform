import { Outlet, NavLink } from "react-router-dom";
import { useEffect, useState } from "react";
import { Activity, Database, Settings, Triangle } from "lucide-react";
import { API_BASE_URL, checkHealth, getAdminMetrics } from "../api/client";
import { useWorkspace } from "../state/workspace";
import SettingsModal from "./workspace/SettingsModal";

export default function AppShell() {
  const { workspace } = useWorkspace();
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [indexedDocs, setIndexedDocs] = useState<number | null>(null);

  useEffect(() => {
    let ignore = false;
    async function loadStatus() {
      try {
        await checkHealth();
        const metrics = await getAdminMetrics().catch(() => null);
        if (!ignore) {
          setHealth("online");
          setIndexedDocs(metrics?.indexed_docs ?? null);
        }
      } catch {
        if (!ignore) setHealth("offline");
      }
    }
    loadStatus();
    const timer = window.setInterval(loadStatus, 30000);
    return () => {
      ignore = true;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="flex h-screen flex-col bg-background overflow-hidden">
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-outline bg-white px-5 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
            <svg 
              className="w-6 h-6 text-primary" 
              viewBox="0 0 24 24" 
              fill="none" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M12 2L2 19H22L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 2V19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M2 19L12 12L22 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <h1 className="font-heading text-lg font-bold leading-tight text-text">Prism</h1>
            <p className="text-[10px] font-medium uppercase tracking-wide text-muted">Workspace</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Status */}
          <div className="hidden items-center gap-3 md:flex">
            <div className="flex items-center gap-1.5 text-xs font-medium text-muted" title={`API: ${API_BASE_URL}`}>
              <Activity size={14} className={health === "online" ? "text-emerald-500" : health === "checking" ? "text-amber-500" : "text-red-500"} />
              {health === "online" ? "Online" : health === "checking" ? "Checking" : "Offline"}
            </div>
            {indexedDocs !== null && (
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted">
                <Database size={14} />
                {indexedDocs} indexed
              </div>
            )}
          </div>

          <div className="h-5 w-px bg-outline" />

          {/* User / Settings */}
          <NavLink
            to="/workspace?settings=open"
            className="flex items-center gap-2 rounded-full border border-outline bg-surface-high pl-3 pr-1 py-1 text-xs font-semibold text-text transition hover:border-primary"
          >
            {workspace.ownerId}
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Settings size={14} />
            </div>
          </NavLink>
        </div>
      </header>
      
      <main className="flex-1 min-h-0">
        <Outlet />
      </main>
      
      <SettingsModal />
    </div>
  );
}
