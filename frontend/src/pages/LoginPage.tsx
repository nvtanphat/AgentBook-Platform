import { FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { AlertCircle, Eye, EyeOff, Loader2, LogIn, UserPlus } from "lucide-react";
import { useAuth } from "../state/auth";

type Mode = "login" | "register";

export default function LoginPage() {
  const { user, login, register, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (user) {
    const target = (location.state as { from?: string })?.from ?? "/workspace";
    return <Navigate to={target} replace />;
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      if (mode === "login") {
        await login(email.trim(), password);
      } else {
        await register(email.trim(), password, displayName.trim());
      }
      navigate("/workspace", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đã xảy ra lỗi");
    }
  }

  return (
    <div
      className="flex min-h-screen items-center justify-center p-4"
      style={{ background: "linear-gradient(135deg, var(--c-app-grad-from) 0%, var(--c-app-grad-to) 100%)" }}
    >
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="mb-6 flex items-center justify-center gap-3">
          <div className="relative flex h-12 w-12 items-center justify-center rounded-[14px] bg-gradient-to-br from-[#006591] via-[#0ea5e9] to-[#14b8a6] shadow-[0_8px_24px_rgba(14,165,233,0.35)] border border-white/20">
            <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 21V5a2 2 0 0 1 2-2h1l8 14h1a2 2 0 0 0 2-2V3" />
            </svg>
          </div>
          <div>
            <h1 className="font-heading text-2xl font-bold text-text">Noelys</h1>
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">Evidence workspace</p>
          </div>
        </div>

        <div className="rounded-2xl border border-outline bg-surface p-7 shadow-soft">
          <div className="mb-5">
            <h2 className="font-heading text-xl font-bold text-text">
              {mode === "login" ? "Đăng nhập" : "Tạo tài khoản"}
            </h2>
            <p className="mt-1 text-xs text-muted">
              {mode === "login"
                ? "Đăng nhập để truy cập tài liệu của bạn"
                : "Tạo tài khoản mới để bắt đầu sử dụng Noelys"}
            </p>
          </div>

          <form className="space-y-3.5" onSubmit={onSubmit}>
            {mode === "register" && (
              <label className="block">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">Tên hiển thị</span>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Nguyễn Văn A"
                  maxLength={128}
                  className="mt-1 w-full rounded-md border border-outline bg-surface px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </label>
            )}

            <label className="block">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">Email</span>
              <input
                type="email"
                required
                autoComplete={mode === "login" ? "username" : "email"}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="mt-1 w-full rounded-md border border-outline bg-surface px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </label>

            <label className="block">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">Mật khẩu</span>
              <div className="relative mt-1">
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={mode === "register" ? "Tối thiểu 6 ký tự" : ""}
                  minLength={mode === "register" ? 6 : 1}
                  className="w-full rounded-md border border-outline bg-surface px-3 py-2 pr-9 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-text"
                  tabIndex={-1}
                  aria-label={showPassword ? "Ẩn mật khẩu" : "Hiện mật khẩu"}
                >
                  {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </label>

            {error && (
              <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                <AlertCircle size={13} className="mt-0.5 shrink-0" /> {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-primary py-2.5 text-sm font-semibold text-white transition hover:bg-primary-bright disabled:opacity-60"
            >
              {loading ? (
                <Loader2 size={15} className="animate-spin" />
              ) : mode === "login" ? (
                <><LogIn size={14} /> Đăng nhập</>
              ) : (
                <><UserPlus size={14} /> Đăng ký</>
              )}
            </button>
          </form>

          <div className="mt-4 flex items-center justify-center gap-1 text-xs text-muted">
            {mode === "login" ? (
              <>
                <span>Chưa có tài khoản?</span>
                <button
                  type="button"
                  onClick={() => { setMode("register"); setError(null); }}
                  className="font-semibold text-primary hover:underline"
                >
                  Đăng ký ngay
                </button>
              </>
            ) : (
              <>
                <span>Đã có tài khoản?</span>
                <button
                  type="button"
                  onClick={() => { setMode("login"); setError(null); }}
                  className="font-semibold text-primary hover:underline"
                >
                  Đăng nhập
                </button>
              </>
            )}
          </div>
        </div>

        <p className="mt-4 text-center text-[10px] text-muted/70">
          Noelys · Evidence-grounded learning assistant
        </p>
      </div>
    </div>
  );
}
