import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

type ToastVariant = "success" | "error" | "info";

interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
}

interface ToastContextValue {
  toast: (message: string, variant?: ToastVariant) => void;
}

// ─── Context ──────────────────────────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

// ─── Single toast item ────────────────────────────────────────────────────────

const ICONS: Record<ToastVariant, React.ReactNode> = {
  success: <CheckCircle2 size={15} className="shrink-0 text-emerald-500" />,
  error:   <AlertCircle  size={15} className="shrink-0 text-red-500" />,
  info:    <Info         size={15} className="shrink-0 text-primary" />,
};

const DURATION = 4000;

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    // Mount → fade in
    requestAnimationFrame(() => setVisible(true));

    timerRef.current = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onDismiss(toast.id), 300);
    }, DURATION);

    return () => clearTimeout(timerRef.current);
  }, [toast.id, onDismiss]);

  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex items-start gap-2.5 rounded-lg border border-outline bg-white px-4 py-3 shadow-lg transition-all duration-300 ${
        visible ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"
      }`}
    >
      {ICONS[toast.variant]}
      <p className="flex-1 text-sm font-medium text-text">{toast.message}</p>
      <button
        type="button"
        aria-label="Dismiss"
        onClick={() => {
          clearTimeout(timerRef.current);
          setVisible(false);
          setTimeout(() => onDismiss(toast.id), 300);
        }}
        className="ml-1 rounded p-0.5 text-muted transition hover:text-text"
      >
        <X size={13} />
      </button>
    </div>
  );
}

// ─── Provider + Container ─────────────────────────────────────────────────────

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback((message: string, variant: ToastVariant = "info") => {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev.slice(-4), { id, message, variant }]);
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {/* Toast container */}
      <div
        aria-label="Notifications"
        className="fixed bottom-20 right-4 z-[200] flex w-[340px] max-w-[calc(100vw-2rem)] flex-col gap-2 lg:bottom-6"
      >
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
