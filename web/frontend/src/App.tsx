import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NavBar } from "./components/NavBar";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { NewRunPage } from "./pages/NewRunPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsListPage } from "./pages/RunsListPage";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1 } },
});

function Layout() {
  return (
    <>
      <NavBar />
      <main className="main-content">
        <Routes>
          <Route index element={<Navigate to="/runs" replace />} />
          <Route path="runs" element={<RunsListPage />} />
          <Route path="runs/new" element={<NewRunPage />} />
          <Route path="runs/:id" element={<RunDetailPage />} />
        </Routes>
      </main>
    </>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/*" element={<Layout />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
