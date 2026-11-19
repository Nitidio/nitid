import { lazy, Suspense, useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router";
import { detectBackend, type BackendInfo } from "./api/backend";
import { BackendContext, useBackend } from "./backend-context";
import { About } from "./pages/About";
import { Home } from "./pages/Home";
import { NotFound } from "./pages/NotFound";

// The workbenches are only needed on their routes: keep them out of the first load.
const RunLocal = lazy(() => import("./pages/RunLocal").then((m) => ({ default: m.RunLocal })));
const RunBrowser = lazy(() => import("./pages/RunBrowser").then((m) => ({ default: m.RunBrowser })));
const RunCompare = lazy(() => import("./pages/RunCompare").then((m) => ({ default: m.RunCompare })));

/** Navigation shared by both deployments; entries that need a local server only appear when one exists. */
export function MainNav() {
  const backend = useBackend();
  return (
    <nav aria-label="Main">
      <NavLink to="/" end>
        Home
      </NavLink>
      {backend?.mode === "local" && <NavLink to="/run/local">Local</NavLink>}
      {backend !== null && <NavLink to="/run/browser">Browser</NavLink>}
      {backend?.mode === "local" && <NavLink to="/compare">Compare</NavLink>}
      <NavLink to="/about">About</NavLink>
    </nav>
  );
}

const loading = <p role="status">Loading...</p>;

export function AppRoutes() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <span className="brand">Nitid app</span>
        <MainNav />
      </header>
      <main id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/run/local" element={<Suspense fallback={loading}><RunLocal /></Suspense>} />
          <Route path="/run/browser" element={<Suspense fallback={loading}><RunBrowser /></Suspense>} />
          <Route path="/compare" element={<Suspense fallback={loading}><RunCompare /></Suspense>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </>
  );
}

export function App() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    void detectBackend().then((info) => {
      if (!cancelled) setBackend(info);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <BackendContext.Provider value={backend}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </BackendContext.Provider>
  );
}
