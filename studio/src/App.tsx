import { lazy, Suspense, useEffect, useState } from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router";
import { detectBackend, type BackendInfo } from "./api/backend";
import { BackendContext } from "./backend-context";
import { About } from "./pages/About";
import { Home } from "./pages/Home";
import { RunPlaceholder } from "./pages/RunPlaceholder";

// The workbench is only needed on /run/local: keep it out of the first load.
const RunLocal = lazy(() => import("./pages/RunLocal").then((m) => ({ default: m.RunLocal })));

export function AppRoutes() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <span className="brand">Nitid Studio</span>
        <nav aria-label="Main">
          <NavLink to="/" end>
            Home
          </NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>
      <main id="main" tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route
            path="/run/local"
            element={
              <Suspense fallback={<p role="status">Loading...</p>}>
                <RunLocal />
              </Suspense>
            }
          />
          <Route path="/run/:mode" element={<RunPlaceholder />} />
          <Route path="*" element={<RunPlaceholder />} />
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
