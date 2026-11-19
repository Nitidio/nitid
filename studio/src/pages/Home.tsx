import { useNavigate } from "react-router";
import { useBackend } from "../backend-context";
import { usePageTitle } from "../use-page-title";

export function Home() {
  usePageTitle();
  const backend = useBackend();
  const navigate = useNavigate();

  const detecting = backend === null;
  const localAvailable = backend?.mode === "local";
  const browserAvailable = !detecting;

  return (
    <>
      <h1>Nitid Studio</h1>
      <p className="lead">Choose where the model runs.</p>

      <p role="status" className="status">
        {detecting && "Checking for a local server..."}
        {backend?.mode === "local" && "Connected to a local server."}
        {backend?.mode === "hosted" &&
          "No local server found. Only in-browser mode is available."}
      </p>

      {backend?.versionMismatch && (
        <p role="alert" className="notice">
          This page expects a different API version than the server reports
          {backend.health ? ` (server: ${backend.health.api_version})` : ""}. Some features may not
          work; update Nitid or reload this page.
        </p>
      )}

      <div className="cards">
        <section className="card" aria-labelledby="card-local">
          <h2 id="card-local">Local</h2>
          <p>
            Runs on this computer through the Nitid Python server (PyTorch or OpenVINO). Images stay
            on this machine.
          </p>
          <button
            type="button"
            disabled={!localAvailable}
            aria-describedby="card-local-hint"
            onClick={() => navigate("/run/local")}
          >
            Use local mode
          </button>
          <p id="card-local-hint" className="hint">
            {localAvailable
              ? "Available."
              : "Requires a running local server (nitid studio)."}
          </p>
        </section>

        <section className="card" aria-labelledby="card-browser">
          <h2 id="card-browser">Browser</h2>
          <p>
            Runs inside this page with ONNX Runtime Web. The image and the computation stay in your
            browser.
          </p>
          <button
            type="button"
            disabled={!browserAvailable}
            aria-describedby="card-browser-hint"
            onClick={() => navigate("/run/browser")}
          >
            Use browser mode
          </button>
          <p id="card-browser-hint" className="hint">
            {browserAvailable ? "Available." : "Waiting for the start-up check."}
          </p>
        </section>
        {localAvailable && (
          <section className="card" aria-labelledby="card-compare">
            <h2 id="card-compare">Compare</h2>
            <p>Run one image on the server and in the browser (or with two models) and see where they agree.</p>
            <button type="button" onClick={() => navigate("/compare")}>
              Open comparison
            </button>
          </section>
        )}
      </div>
    </>
  );
}
