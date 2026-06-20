import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { Database, LogOut, Settings } from "lucide-react";
import { API_BASE_URL, checkHealth, getAdminMetrics } from "../api/client";
import { useWorkspace } from "../state/workspace";
import { useAuth } from "../state/auth";
import SettingsModal from "./workspace/SettingsModal";

export default function AppShell() {
  const { workspace } = useWorkspace();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
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
    <div className="flex h-screen flex-col overflow-hidden" style={{ background: 'var(--c-app-grad-from)' }}>
      {/* ── Clean & Academic Header ── */}
      <header className="app-header flex h-14 shrink-0 items-center justify-between px-5 z-50">
        <div className="flex items-center gap-3">
          {/* Stunning Ink Slate Logo Box */}
          <div 
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#0f172a] dark:bg-slate-800 border border-slate-700/50 shadow-sm transition-all duration-200 hover:scale-105 cursor-pointer" 
            onClick={() => navigate("/workspace")}
          >
            <svg className="w-[18px] h-[18px] text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="font-heading text-[15px] font-extrabold tracking-[-0.01em] text-text leading-none">Noelys</span>
              <span className="inline-flex items-center rounded-full bg-primary/10 px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider text-primary">RAG</span>
            </div>
            <span className="text-[9px] font-medium tracking-[0.06em] text-muted/70 mt-0.5">Evidence Workspace</span>
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
              <span className="text-muted">
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
            <div className="flex items-center gap-2 bg-surface pl-3 pr-1 py-1 text-xs font-semibold text-text">
              <span className="hidden sm:inline">{user?.display_name || ownerLabel}</span>
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold sm:hidden">
                {ownerInitials}
              </div>
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary transition-colors hover:bg-primary/20">
                <Settings size={13} />
              </div>
            </div>
          </NavLink>

          {/* Logout */}
          <button
            type="button"
            title="Đăng xuất"
            onClick={() => { logout(); navigate("/login", { replace: true }); }}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-outline text-muted transition hover:border-red-300 hover:bg-red-50 hover:text-red-600"
          >
            <LogOut size={13} />
          </button>
        </div>
      </header>

      <main className="flex-1 min-h-0">
        <Outlet />
      </main>

      <SettingsModal />
    </div>
  );
}
