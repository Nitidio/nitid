import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initToken } from "./token";
import { applyTheme, readStoredTheme } from "./theme";
import "./styles.css";

// Must run before the router mounts so the `#t=...` fragment is consumed and removed.
initToken();
// The visitor's saved light / dark choice (else the system preference, via CSS), applied before the first paint of the app.
applyTheme(readStoredTheme());

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
