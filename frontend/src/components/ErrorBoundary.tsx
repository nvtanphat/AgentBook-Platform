import { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
          <div className="rounded-full bg-red-100 p-4">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-text">Đã xảy ra lỗi</h2>
            <p className="mt-1 text-sm text-muted">{this.state.error?.message}</p>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary/90 transition"
          >
            Tải lại trang
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
