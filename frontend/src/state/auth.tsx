import { ReactNode, createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { API_V1_BASE_URL } from "../api/client";

const ACCESS_KEY = "prism.auth.access";
const REFRESH_KEY = "prism.auth.refresh";
const USER_KEY = "prism.auth.user";

export type AuthUser = {
  user_id: string;
  email: string;
  display_name: string;
  role: string;
  created_at?: string;
};

type AuthState = {
  user: AuthUser | null;
  accessToken: string | null;
  loading: boolean;
};

type AuthContextValue = AuthState & {
  login: (emailOrId: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName?: string) => Promise<void>;
  logout: () => void;
  setSession: (user: AuthUser, access: string, refresh?: string | null) => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function readStored<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function storeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

function storeString(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {}
}

async function apiAuthCall<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_V1_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data?.success === false) {
    const detail = data?.detail || data?.message || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return (data?.data ?? data) as T;
}

type TokenResponse = {
  access_token: string;
  refresh_token: string | null;
  user: AuthUser;
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => readStored<AuthUser>(USER_KEY));
  const [accessToken, setAccessToken] = useState<string | null>(() => {
    try {
      return localStorage.getItem(ACCESS_KEY);
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(false);

  const setSession = useCallback((u: AuthUser, access: string, refresh?: string | null) => {
    setUser(u);
    setAccessToken(access);
    storeString(ACCESS_KEY, access);
    storeJson(USER_KEY, u);
    if (refresh !== undefined && refresh !== null) {
      storeString(REFRESH_KEY, refresh);
    }
  }, []);

  const login = useCallback(async (emailOrId: string, password: string) => {
    setLoading(true);
    try {
      const data = await apiAuthCall<TokenResponse>("/auth/login", {
        email_or_id: emailOrId,
        password,
      });
      setSession(data.user, data.access_token, data.refresh_token);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const register = useCallback(async (email: string, password: string, displayName?: string) => {
    setLoading(true);
    try {
      const data = await apiAuthCall<TokenResponse>("/auth/register", {
        email,
        password,
        display_name: displayName || "",
      });
      setSession(data.user, data.access_token, data.refresh_token);
    } finally {
      setLoading(false);
    }
  }, [setSession]);

  const logout = useCallback(() => {
    setUser(null);
    setAccessToken(null);
    storeString(ACCESS_KEY, null);
    storeString(REFRESH_KEY, null);
    try {
      localStorage.removeItem(USER_KEY);
    } catch {}
  }, []);

  // Verify token on load; clear session if expired
  useEffect(() => {
    if (!accessToken) return;
    const ac = new AbortController();
    fetch(`${API_V1_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${accessToken}` },
      signal: ac.signal,
    })
      .then((res) => {
        if (res.status === 401) {
          logout();
        }
      })
      .catch(() => {});
    return () => ac.abort();
  }, [accessToken, logout]);

  const value = useMemo<AuthContextValue>(
    () => ({ user, accessToken, loading, login, register, logout, setSession }),
    [user, accessToken, loading, login, register, logout, setSession],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

export function getStoredAccessToken(): string | null {
  try {
    return localStorage.getItem(ACCESS_KEY);
  } catch {
    return null;
  }
}
