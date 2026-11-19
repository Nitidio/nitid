import { useCallback, useMemo, useState, type ComponentType } from "react";
import { Link } from "react-router";
import { createLocalApi, type LocalApi } from "../api/local";
import { DevicePicker } from "./DevicePicker";
import { ErrorNotice } from "./ErrorNotice";
import { ImageDropzone } from "./ImageDropzone";
import { ModelPicker } from "./ModelPicker";
import { ResultsView } from "./ResultsView";
import { buildBackendOptions, defaultDeviceChoice, isChoiceAvailable, type DeviceChoice } from "./devices";

import { defaultModelKey, findModel } from "./models";
import type { ExportMenuProps, ResultItem } from "./results-model";
import { useDownloads } from "./use-downloads";
import { useLocalData } from "./use-local-data";
import { useRun } from "./use-run";

/** Confidence sent to the server. Low on purpose: the slider then filters client-side, instantly. */
export const RUN_CONFIDENCE_FLOOR = 0.1;

export interface LocalPanelProps {
  /** Injected in tests; defaults to the real client. */
  api?: LocalApi;
  /** Slot for exports; wired by the unified UI. */
  ExportMenu?: ComponentType<ExportMenuProps>;
}

/** The Local deployment workbench: pick model and device, upload images, run, inspect. */
export function LocalPanel({ api: apiProp, ExportMenu }: LocalPanelProps) {
  const api = useMemo(() => apiProp ?? createLocalApi(), [apiProp]);
  const { devices, models, reloadModels, reloadAll } = useLocalData(api);
  const { downloads, start: startDownload } = useDownloads(api, reloadModels);
  const run = useRun(api);

  const [pickedKey, setPickedKey] = useState<string | null>(null);
  const [pickedChoice, setPickedChoice] = useState<DeviceChoice | null>(null);
  const [files, setFiles] = useState<File[]>([]);

  const modelList = models.status === "ready" ? models.data : null;
  const backendOptions = useMemo(
    () => (devices.status === "ready" ? buildBackendOptions(devices.data) : null),
    [devices],
  );

  // Derived, not synced: the user's pick wins while it is still valid, else a sensible default.
  const modelKey =
    modelList && pickedKey && findModel(modelList, pickedKey) ? pickedKey : modelList ? defaultModelKey(modelList) : null;
  const choice =
    backendOptions && pickedChoice && isChoiceAvailable(backendOptions, pickedChoice)
      ? pickedChoice
      : backendOptions
        ? defaultDeviceChoice(backendOptions)
        : null;
  const model = modelList && modelKey ? findModel(modelList, modelKey) : undefined;
  const busy = run.state.phase === "uploading" || run.state.phase === "processing";
  const canRun = !!model?.downloaded && !!choice && files.length > 0 && !busy;

  const loadImage = useCallback(
    async (item: ResultItem): Promise<ImageBitmap | null> => {
      if (run.state.phase !== "done" || typeof createImageBitmap !== "function") return null;
      const blob = await api.getItemImage(run.state.runId, item.id);
      return createImageBitmap(blob);
    },
    [api, run.state],
  );

  function onRun() {
    if (!model || !choice) return;
    void run.start(files, {
      model: model.id,
      task: model.task,
      backend: choice.backend,
      device: choice.device,
      conf: RUN_CONFIDENCE_FLOOR,
    });
  }

  const fatal = devices.status === "error" ? devices.error : models.status === "error" ? models.error : null;
  const loading = devices.status === "loading" || models.status === "loading";

  return (
    <>
      <h1>Local mode</h1>
      <p className="lead">Run a model on this computer. Images stay on this machine.</p>

      {fatal && (
        <>
          <ErrorNotice error={fatal} title="Could not talk to the local server." />
          <p>
            <button type="button" onClick={reloadAll}>
              Try again
            </button>{" "}
            <Link to="/">Back to start</Link>
          </p>
        </>
      )}

      {loading && !fatal && <p role="status">Loading models and devices...</p>}

      {!fatal && modelList && backendOptions && (
        <form
          className="local-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (canRun) onRun();
          }}
        >
          <ModelPicker
            models={modelList}
            selectedKey={modelKey}
            onSelect={setPickedKey}
            downloads={downloads}
            onDownload={(m) => void startDownload(m)}
            disabled={busy}
          />
          {choice ? (
            <DevicePicker options={backendOptions} choice={choice} onChange={setPickedChoice} disabled={busy} />
          ) : (
            <ErrorNotice error={{ message: "The server reports no usable inference device." }} />
          )}
          <ImageDropzone files={files} onChange={setFiles} disabled={busy} />

          <div className="actions">
            <button type="submit" disabled={!canRun}>
              {busy ? "Running..." : "Run"}
            </button>
            {run.state.phase !== "idle" && !busy && (
              <button type="button" className="secondary" onClick={run.reset}>
                Clear results
              </button>
            )}
            {busy && (
              <button type="button" className="secondary" onClick={run.reset}>
                Stop waiting
              </button>
            )}
          </div>
          {!canRun && !busy && (
            <p className="hint" id="run-hint">
              {!model?.downloaded
                ? "Download the selected model to run it."
                : files.length === 0
                  ? "Add at least one image."
                  : ""}
            </p>
          )}
        </form>
      )}

      <div role="status" aria-live="polite" className="run-status">
        {run.state.phase === "uploading" && "Uploading images..."}
        {run.state.phase === "processing" &&
          (run.state.status === "pending" ? "Queued on the server..." : "Running the model...")}
      </div>

      {run.state.phase === "failed" && (
        <ErrorNotice
          error={{
            message: run.state.message,
            ...(run.state.code ? { code: run.state.code } : {}),
            ...(run.state.hint ? { hint: run.state.hint } : {}),
          }}
          title="The run failed."
        />
      )}

      {run.results && (
        <ResultsView
          key={run.results.runId}
          results={run.results}
          floorConfidence={RUN_CONFIDENCE_FLOOR}
          loadImage={loadImage}
          {...(ExportMenu ? { ExportMenu } : {})}
        />
      )}
    </>
  );
}

