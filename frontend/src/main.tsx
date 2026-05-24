import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { Navigate, Outlet, RouterProvider, createBrowserRouter, useLocation, useNavigate } from "react-router-dom";
import AppShell from "./components/AppShell";
import WorkspacePage from "./pages/WorkspacePage";
import LoginPage from "./pages/LoginPage";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { WorkspaceProvider } from "./state/workspace";
import { ThemeProvider } from "./state/theme";
import { AuthProvider, useAuth } from "./state/auth";
import { ToastProvider } from "./components/Toast";
import "./styles.css";

function EvidenceRedirect() {
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  params.set("panel", "evidence");
  return <Navigate to={`/workspace?${params.toString()}`} replace />;
}

/** Gate that redirects unauthenticated users to /login. */
function RequireAuth() {
  const { user } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  // Listen for 401s from the API client and bounce to login
  useEffect(() => {
    function onUnauthorized() {
      navigate("/login", { state: { from: location.pathname + location.search }, replace: true });
    }
    window.addEventListener("prism:auth:unauthorized", onUnauthorized);
    return () => window.removeEventListener("prism:auth:unauthorized", onUnauthorized);
  }, [navigate, location.pathname, location.search]);

  if (!user) {
    return <Navigate to="/login" state={{ from: location.pathname + location.search }} replace />;
  }
  return <Outlet />;
}

const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  {
    path: "/",
    element: <RequireAuth />,
    children: [
      {
        path: "",
        element: <AppShell />,
        children: [
          { index: true, element: <Navigate to="/workspace" replace /> },
          { path: "workspace", element: <WorkspacePage /> },
          { path: "upload", element: <Navigate to="/workspace" replace /> },
          { path: "chat", element: <Navigate to="/workspace" replace /> },
          { path: "evidence", element: <EvidenceRedirect /> },
          { path: "compare", element: <Navigate to="/workspace?panel=compare" replace /> },
          { path: "graph", element: <Navigate to="/workspace?panel=graph" replace /> },
          { path: "mindmap", element: <Navigate to="/workspace?panel=mindmap" replace /> },
          { path: "settings", element: <Navigate to="/workspace?settings=open" replace /> },
        ],
      },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <WorkspaceProvider>
            <ToastProvider>
              <RouterProvider router={router} />
            </ToastProvider>
          </WorkspaceProvider>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  </React.StrictMode>
);
