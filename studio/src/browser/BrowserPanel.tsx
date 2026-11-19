import { useCallback, useEffect, useId, useRef, useState, type ComponentType } from "react";
import { Link } from "react-router";
import { useBackend } from "../backend-context";
import { createApiClient } from "../api/client";
import { fetchOnnxMeta } from "../api/local";
import { createInferenceClient } from "../engine/client";
import { getToken } from "../token";
import { ErrorNotice } from "../local/ErrorNotice";
import { ImageDropzone } from "../local/ImageDropzone";
import { ResultsView } from "../local/ResultsView";
import type { DescribedError } from "../local/errors";
import type { ExportMenuProps, ResultItem, RunResults } from "../local/results-model";
import { ResultsExportMenu } from "../exports/ResultsExportMenu";
import type { BrowserModel } from "./catalog";
import { describeBrowserError } from "./errors";
import { BROWSER_CONFIDENCE_FLOOR, BrowserRunner, type EngineLike, type LoadedModelInfo } from "./runner";
import { useBrowserModels, type BrowserModelsDeps } from "./use-browser-models";

export interface BrowserPanelProps extends BrowserModelsDeps {
  /** Replaces the Web Worker (tests). */
  createEngine?: () => EngineLike;
  /** Defaults to `ResultsExportMenu`. */
  ExportMenu?: ComponentType<ExportMenuProps>;
}

type Phase =
  | { name: "idle" }
  | { name: "loading"; loaded: number; total: number | undefined }
  | { name: "running" }
  | { name: "failed"; error: DescribedError };

/** Runner wired to the real worker, the bearer token and `GET /onnx/{id}/meta`. */
function useRunner(createEngine: () => EngineLike) {
  const ref = useRef<BrowserRunner | null>(null);
  const factory = useRef(createEngine);
  useEffect(() => {
    factory.current = createEngine;
  }, [createEngine]);
  const get = useCallback(() => {
    ref.current ??= new BrowserRunner({
      createEngine: () => factory.current(),
      getToken,
      fetchServerMeta: (m: BrowserModel) => fetchOnnxMeta(createApiClient(), m.id, m.task),
    });
    return ref.current;
  }, []);
  const close = useCallback(() => {
    ref.current?.close();
    ref.current = null;
  }, []);
  useEffect(() => close, [close]);
  return { get, close };
}

/** In-browser mode: ONNX Runtime Web in a Web Worker. Available in both deployments. */
export function BrowserPanel({ createEngine = createInferenceClient, ExportMenu = ResultsExportMenu, ...modelDeps }: BrowserPanelProps) {
  const id = useId();
  const backend = useBackend();
  const models = useBrowserModels(backend, modelDeps);
  const runner = useRunner(createEngine);

  const [pickedKey, setPickedKey] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  /** The results together with the files they were computed from (item ids index into `files`). */
  const [done, setDone] = useState<{ results: RunResults; files: File[] } | null>(null);
  const token = useRef(0);
  const [loadedInfo, setLoadedInfo] = useState<LoadedModelInfo | null>(null);

  const list = models.status === "ready" ? models.models : [];
  const model = list.find((m) => m.key === pickedKey) ?? list.find((m) => m.task === "detect") ?? list[0];
  const busy = phase.name === "loading" || phase.name === "running";
  const canRun = !!model && files.length > 0 && !busy;

  const doneFiles = done?.files;
  const loadImage = useCallback(
    async (item: ResultItem): Promise<ImageBitmap | null> => {
      const file = doneFiles?.[item.id];
      return file && typeof createImageBitmap === "function" ? createImageBitmap(file) : null;
    },
    [doneFiles],
  );

  async function onRun() {
    if (!model) return;
    const r = runner.get();
    const runFiles = files;
    const mine = ++token.current;
    setDone(null);
    try {
      if (r.current?.model.key !== model.key) {
        setLoadedInfo(null);
        setPhase({ name: "loading", loaded: 0, total: undefined });
        const info = await r.load(model, (loaded, total) => setPhase({ name: "loading", loaded, total }));
        setLoadedInfo(info);
      }
      if (mine !== token.current) return;
      setPhase({ name: "running" });
      const results = await r.runFiles(runFiles, { conf: BROWSER_CONFIDENCE_FLOOR });
      if (mine !== token.current) return;
      setDone({ results, files: runFiles });
      setPhase({ name: "idle" });
    } catch (e) {
      // After "Stop" the worker is gone and its pending requests reject: that is not an error to show.
      if (mine === token.current) setPhase({ name: "failed", error: describeBrowserError(e) });
    }
  }

  function onStop() {
    token.current += 1;
    runner.close();
    setLoadedInfo(null);
    setPhase({ name: "idle" });
  }

  const dlPct = phase.name === "loading" && phase.total ? Math.round((phase.loaded / phase.total) * 100) : null;
  const results = done?.results ?? null;
  const hasResults = results !== null;

  return (
    <>
      <h1>Browser mode</h1>
      <p className="lead">Run a model inside this page. The image and the computation never leave your browser.</p>

      {models.status === "loading" && <p role="status">Loading the model list...</p>}

      {models.status === "unconfigured" && (
        <div role="status" className="notice" data-testid="no-models">
          <strong>No models are configured for this site.</strong>
          <p>
            The operator of this deployment has not set a model catalogue (<code>STUDIO_MODELS_BASE_URL</code> at
            build time), so there is nothing to load. To run models on your own computer, install Nitid and start{" "}
            <code>nitid studio</code>.
          </p>
        </div>
      )}

      {models.status === "error" && <ErrorNotice error={models.error} title="Could not get the model list." />}

      {models.status === "ready" && list.length === 0 && (
        <p role="status">
          No models are available yet. Download one in <Link to="/run/local">server mode</Link>, then come back.
        </p>
      )}

      {models.status === "ready" && list.length > 0 && (
        <form
          className="local-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (canRun) void onRun();
          }}
        >
          <div className="field">
            <label htmlFor={`${id}-model`}>Model and task</label>
            <select
              id={`${id}-model`}
              value={model?.key ?? ""}
              disabled={busy}
              onChange={(e) => setPickedKey(e.target.value)}
            >
              {list.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
            <div className="field-state">
              {models.origin === "server"
                ? "Loaded from the local server and cached in this browser."
                : "Loaded from the model catalogue of this site and cached in this browser."}
            </div>
          </div>

          <ImageDropzone
            files={files}
            onChange={setFiles}
            disabled={busy}
            hint="JPEG, PNG or WebP. Up to 25 MB and 20 megapixels each. Images never leave this page."
          />

          <div className="actions">
            <button type="submit" disabled={!canRun}>
              {busy ? "Running..." : "Run"}
            </button>
            {busy && (
              <button type="button" className="secondary" onClick={onStop}>
                Stop
              </button>
            )}
            {hasResults && !busy && (
              <button type="button" className="secondary" onClick={() => setDone(null)}>
                Clear results
              </button>
            )}
          </div>
          {!canRun && !busy && files.length === 0 && <p className="hint">Add at least one image.</p>}
        </form>
      )}

      <div role="status" aria-live="polite" className="run-status">
        {phase.name === "loading" && (
          <>
            Loading the model{dlPct !== null ? ` (${dlPct}%)` : "..."}
            {phase.total ? <progress max={phase.total} value={phase.loaded} aria-label="Model download" /> : null}
          </>
        )}
        {phase.name === "running" && "Running the model in the browser..."}
        {phase.name === "idle" && loadedInfo && !hasResults && `Model ready (${loadedInfo.backend}, from ${loadedInfo.source}).`}
      </div>

      {phase.name === "failed" && <ErrorNotice error={phase.error} title="The run failed." />}

      {results && (
        <ResultsView
          key={results.runId}
          results={results}
          floorConfidence={BROWSER_CONFIDENCE_FLOOR}
          loadImage={loadImage}
          ExportMenu={ExportMenu}
        />
      )}
    </>
  );
}
