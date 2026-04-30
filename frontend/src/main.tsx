import React from "react";
import ReactDOM from "react-dom/client";
import { Navigate, RouterProvider, createBrowserRouter, useLocation } from "react-router-dom";
import AppShell from "./components/AppShell";
import WorkspacePage from "./pages/WorkspacePage";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { WorkspaceProvider } from "./state/workspace";
import "./styles.css";

function EvidenceRedirect() {
  const { search } = useLocation();
  const params = new URLSearchParams(search);
  params.set("panel", "evidence");
  return <Navigate to={`/workspace?${params.toString()}`} replace />;
}

const router = createBrowserRouter([
  {
    path: "/",
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
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <WorkspaceProvider>
        <RouterProvider router={router} />
      </WorkspaceProvider>
    </ErrorBoundary>
  </React.StrictMode>
);
