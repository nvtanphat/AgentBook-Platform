import { Outlet, NavLink } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { Activity, Database, Settings } from "lucide-react";
import { API_BASE_URL, checkHealth, getAdminMetrics } from "../api/client";
import { useWorkspace } from "../state/workspace";
import SettingsModal from "./workspace/SettingsModal";

export default function AppShell() {
  const { workspace } = useWorkspace();
  const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
  const [indexedDocs, setIndexedDocs] = useState<number | null>(null);

  const ownerInitials = useMemo(() => {
    const id = workspace.ownerId || "?";
    return id.slice(0, 2).toUpperCase();
  }, [workspace.ownerId]);

  const ownerLabel = useMemo(() => {
    const id = workspace.ownerId || "";
    return id.length > 12 ? id.slice(0, 10) + "…" : id;
  }, [workspace.ownerId]);

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
    <div className="flex h-screen flex-col overflow-hidden" style={{ background: 'linear-gradient(135deg, #f0f2f8 0%, #e8ecf4 100%)' }}>
      {/* ── Premium Header ── */}
      <header className="app-header flex h-14 shrink-0 items-center justify-between px-5">
        <div className="flex items-center gap-3">
          {/* Logo */}
          <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: 'linear-gradient(135deg, #006591 0%, #0ea5e9 100%)' }}>
            <svg
              className="w-5 h-5 text-white"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M5 18V6L19 18V6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M7 20H17" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <h1 className="font-heading text-[15px] font-bold leading-tight text-text tracking-tight">Noelys</h1>
            <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-muted/70">Evidence workspace</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Status indicators */}
          <div className="hidden items-center gap-3 md:flex">
            <div
              className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium"
              style={{
                background: health === "online" ? 'rgba(16, 185, 129, 0.08)' : health === "checking" ? 'rgba(245, 158, 11, 0.08)' : 'rgba(239, 68, 68, 0.08)',
              }}
              title={`API: ${API_BASE_URL}`}
            >
              <div className="relative flex h-2 w-2">
                {health === "online" && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />}
                <span className={`relative inline-flex h-2 w-2 rounded-full ${health === "online" ? "bg-emerald-500" : health === "checking" ? "bg-amber-500" : "bg-red-500"}`} />
              </div>
              <span className={health === "online" ? "text-emerald-700" : health === "checking" ? "text-amber-700" : "text-red-600"}>
                {health === "online" ? "Online" : health === "checking" ? "Checking" : "Offline"}
              </span>
            </div>
            {indexedDocs !== null && (
              <div className="flex items-center gap-1.5 rounded-full bg-primary/5 px-2.5 py-1 text-[11px] font-medium text-primary">
                <Database size={12} />
                <span>{indexedDocs} indexed</span>
              </div>
            )}
          </div>

          <div className="h-5 w-px bg-outline/50" />

          {/* User / Settings */}
          <NavLink
            to="/workspace?settings=open"
            title={`Owner: ${workspace.ownerId} — Click để mở Settings`}
            className="user-avatar-ring"
          >
            <div className="flex items-center gap-2 bg-white pl-3 pr-1 py-1 text-xs font-semibold text-text">
              <span className="hidden sm:inline">{ownerLabel}</span>
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold sm:hidden">
                {ownerInitials}
              </div>
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary transition-colors hover:bg-primary/20">
                <Settings size={13} />
              </div>
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
