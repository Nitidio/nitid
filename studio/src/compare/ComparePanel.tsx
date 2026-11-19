import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { createApiClient } from "../api/client";
import { createLocalApi, fetchOnnxMeta, type LocalApi } from "../api/local";
import type { Model } from "../api/types";
import type { BrowserModel } from "../browser/catalog";
import { fromServerModels } from "../browser/catalog";
import { BrowserRunner, type EngineLike } from "../browser/runner";
import { createInferenceClient } from "../engine/client";
import { ResultsExportMenu } from "../exports/ResultsExportMenu";
import { DetectionCanvas } from "../local/DetectionCanvas";
import { DevicePicker } from "../local/DevicePicker";
import { ErrorNotice } from "../local/ErrorNotice";
import { ImageDropzone } from "../local/ImageDropzone";
import { buildBackendOptions, defaultDeviceChoice, isChoiceAvailable, type BackendOption, type DeviceChoice } from "../local/devices";
import { clampConfidence, defaultFilter, filterDetections } from "../local/filter";
import { findModel, modelKey, modelLabel } from "../local/models";
import { itemTotalMs, type ResultDetection, type RunResults } from "../local/results-model";
import { useLocalData } from "../local/use-local-data";
import { getToken } from "../token";
import { summarizeAgreement } from "./agreement";
import { COMPARE_CONFIDENCE_FLOOR, laneTitle, runLane, type LaneOutcome, type LaneSpec } from "./lanes";

export interface ComparePanelProps {
  api?: LocalApi;
  /** Replaces the Web Worker (tests). */
  createEngine?: () => EngineLike;
}

type LaneKind = "server" | "browser";
interface LaneChoice {
  kind: LaneKind;
  modelKey: string | null;
  device: DeviceChoice | null;
}

function fmtMs(n: number): string {
  return n >= 100 ? `${Math.round(n)} ms` : `${n.toFixed(1)} ms`;
}

interface LaneConfigProps {
  name: string;
  choice: LaneChoice;
  models: readonly Model[];
  options: readonly BackendOption[];
  disabled: boolean;
  onChange: (next: LaneChoice) => void;
}

function LaneConfig({ name, choice, models, options, disabled, onChange }: LaneConfigProps) {
  const id = useId();
  const usable = models.filter((m) => m.downloaded);
  const device = choice.device ?? defaultDeviceChoice(options);
  return (
    <fieldset className="field lane-config">
      <legend>{name}</legend>
      <div className="field">
        <label htmlFor={`${id}-kind`}>Runs</label>
        <select
          id={`${id}-kind`}
          value={choice.kind}
          disabled={disabled}
          onChange={(e) => onChange({ ...choice, kind: e.target.value as LaneKind })}
        >
          <option value="server">On the server (PyTorch or OpenVINO)</option>
          <option value="browser">In the browser (ONNX Runtime Web)</option>
        </select>
      </div>
      <div className="field">
        <label htmlFor={`${id}-model`}>Model and task</label>
        <select
          id={`${id}-model`}
          value={choice.modelKey ?? ""}
          disabled={disabled}
          onChange={(e) => onChange({ ...choice, modelKey: e.target.value })}
        >
          {usable.map((m) => (
            <option key={modelKey(m)} value={modelKey(m)}>
              {modelLabel(m)} - {m.task}
            </option>
          ))}
        </select>
      </div>
      {choice.kind === "server" && device && (
        <DevicePicker options={options} choice={device} disabled={disabled} onChange={(d) => onChange({ ...choice, device: d })} />
      )}
    </fieldset>
  );
}

function LaneView({
  label,
  outcome,
  image,
  minConfidence,
  visible,
}: {
  label: string;
  outcome: LaneOutcome;
  image: ImageBitmap | null;
  minConfidence: number;
  visible: ResultDetection[];
}) {
  const id = useId();
  if (outcome.status === "error") {
    return (
      <section className="lane" aria-labelledby={`${id}-h`}>
        <h3 id={`${id}-h`}>{label}</h3>
        <ErrorNotice error={outcome.error} title="This side failed." />
      </section>
    );
  }
  const { results, loadMs } = outcome;
  const item = results.items[0];
  if (!item) return null;
  return (
    <section className="lane" aria-labelledby={`${id}-h`}>
      <h3 id={`${id}-h`}>
        {label}: {laneTitle(outcome.spec, results)}
      </h3>
      <DetectionCanvas
        image={image}
        width={item.width}
        height={item.height}
        detections={visible}
        showLabels
        showMasks
        label={`${label}: ${visible.length} detection${visible.length === 1 ? "" : "s"} shown`}
      />
      <p className="hint" data-testid={`${label}-stats`}>
        {visible.length} detection{visible.length === 1 ? "" : "s"} at {Math.round(minConfidence * 100)}% or more. Inference{" "}
        {fmtMs(item.speed.inference)}, pipeline {fmtMs(itemTotalMs(item))}
        {results.wallClockMs !== undefined && <>, wall clock {fmtMs(results.wallClockMs)}</>}
        {loadMs !== null && <>, model load {fmtMs(loadMs)}</>}.
      </p>
      <ResultsExportMenu
        results={results}
        filter={{ ...defaultFilter(minConfidence) }}
        selectedItemId={item.id}
        image={image}
      />
    </section>
  );
}

/** Same image through two lanes (server and/or browser, optionally two models) side by side. */
export function ComparePanel({ api: apiProp, createEngine = createInferenceClient }: ComparePanelProps) {
  const id = useId();
  const api = useMemo(() => apiProp ?? createLocalApi(), [apiProp]);
  const { devices, models } = useLocalData(api);

  const [lanes, setLanes] = useState<[LaneChoice, LaneChoice]>([
    { kind: "server", modelKey: null, device: null },
    { kind: "browser", modelKey: null, device: null },
  ]);
  const [files, setFiles] = useState<File[]>([]);
  const [minConfidence, setMinConfidence] = useState(0.25);
  const [phase, setPhase] = useState<"idle" | "running">("idle");
  const [outcomes, setOutcomes] = useState<[LaneOutcome, LaneOutcome] | null>(null);
  const [shownFile, setShownFile] = useState<File | null>(null);
  const [image, setImage] = useState<ImageBitmap | null>(null);
  const ctl = useRef<AbortController | null>(null);

  const modelList = models.status === "ready" ? models.data : null;
  const options = useMemo(() => (devices.status === "ready" ? buildBackendOptions(devices.data) : null), [devices]);
  const usable = useMemo(() => (modelList ?? []).filter((m) => m.downloaded), [modelList]);

  // Effective choices: the user's pick while valid, else a default (derived, not synced).
  const effective = useMemo(() => {
    if (!modelList || !options) return null;
    const def = usable.find((m) => m.task === "detect") ?? usable[0];
    return lanes.map((l) => {
      const key = l.modelKey && usable.some((m) => modelKey(m) === l.modelKey) ? l.modelKey : def ? modelKey(def) : null;
      const dev = l.device && isChoiceAvailable(options, l.device) ? l.device : defaultDeviceChoice(options);
      return { ...l, modelKey: key, device: dev };
    }) as [LaneChoice, LaneChoice];
  }, [lanes, modelList, options, usable]);

  const specs = useMemo(() => {
    if (!effective || !modelList) return null;
    const out: LaneSpec[] = [];
    for (const l of effective) {
      const model = l.modelKey ? findModel(modelList, l.modelKey) : undefined;
      if (!model) return null;
      if (l.kind === "server") {
        if (!l.device) return null;
        out.push({ kind: "server", model, choice: l.device });
      } else {
        const bm = fromServerModels([model])[0] as BrowserModel | undefined;
        if (!bm) return null;
        out.push({ kind: "browser", model: bm });
      }
    }
    return out as [LaneSpec, LaneSpec];
  }, [effective, modelList]);

  useEffect(() => () => ctl.current?.abort(), []);

  // Decode the image once for both canvases; release it when it changes.
  useEffect(() => {
    if (!shownFile || typeof createImageBitmap !== "function") return;
    let cancelled = false;
    let bmp: ImageBitmap | null = null;
    createImageBitmap(shownFile)
      .then((b) => {
        if (cancelled) b.close?.();
        else {
          bmp = b;
          setImage(b);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      bmp?.close?.();
      setImage(null);
    };
  }, [shownFile]);

  const file = files[0];
  const canRun = !!specs && !!file && phase === "idle";

  async function onRun() {
    if (!specs || !file) return;
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setPhase("running");
    setOutcomes(null);
    setShownFile(file);
    const deps = {
      api,
      createRunner: () =>
        new BrowserRunner({
          createEngine,
          getToken,
          fetchServerMeta: (m) => fetchOnnxMeta(createApiClient(), m.id, m.task),
        }),
    };
    try {
      // One after the other: running both at once would make them compete for the same CPU and
      // distort the timings this view is about.
      const a = await runLane(specs[0], file, deps, c.signal);
      const b = await runLane(specs[1], file, deps, c.signal);
      if (!c.signal.aborted) setOutcomes([a, b]);
    } catch {
      /* aborted */
    } finally {
      if (ctl.current === c) setPhase("idle");
    }
  }

  const fatal = devices.status === "error" ? devices.error : models.status === "error" ? models.error : null;
  const loading = devices.status === "loading" || models.status === "loading";

  const okResults = outcomes?.map((o) => (o.status === "ok" ? o.results : null)) ?? [];
  const shown = (r: RunResults | null): ResultDetection[] =>
    r?.items[0] ? filterDetections(r.items[0].detections, defaultFilter(minConfidence)) : [];
  const visA = shown(okResults[0] ?? null);
  const visB = shown(okResults[1] ?? null);
  const agreement = okResults[0] && okResults[1] ? summarizeAgreement(visA, visB) : null;
  const pct = Math.round(minConfidence * 100);

  return (
    <>
      <h1>Compare</h1>
      <p className="lead">
        Run one image through two setups and compare boxes, counts and timings. Typical use: the same model on the
        server and in the browser, or two models.
      </p>

      {fatal && <ErrorNotice error={fatal} title="Could not talk to the local server." />}
      {loading && !fatal && <p role="status">Loading models and devices...</p>}
      {!fatal && modelList && usable.length === 0 && (
        <p role="status">
          No downloaded models. Download one in <Link to="/run/local">server mode</Link> first.
        </p>
      )}

      {!fatal && modelList && options && effective && usable.length > 0 && (
        <form
          className="local-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (canRun) void onRun();
          }}
        >
          <div className="lanes-config">
            <LaneConfig
              name="Side A"
              choice={effective[0]}
              models={modelList}
              options={options}
              disabled={phase === "running"}
              onChange={(n) => setLanes([n, lanes[1]])}
            />
            <LaneConfig
              name="Side B"
              choice={effective[1]}
              models={modelList}
              options={options}
              disabled={phase === "running"}
              onChange={(n) => setLanes([lanes[0], n])}
            />
          </div>
          <ImageDropzone
            files={files}
            onChange={setFiles}
            disabled={phase === "running"}
            multiple={false}
            hint="One image, JPEG, PNG or WebP, up to 25 MB. It goes to the local server only for sides that run on the server."
          />
          <div className="actions">
            <button type="submit" disabled={!canRun}>
              {phase === "running" ? "Comparing..." : "Compare"}
            </button>
            {phase === "running" && (
              <button type="button" className="secondary" onClick={() => ctl.current?.abort()}>
                Stop waiting
              </button>
            )}
          </div>
          {!file && phase === "idle" && <p className="hint">Add an image.</p>}
        </form>
      )}

      <div role="status" aria-live="polite" className="run-status">
        {phase === "running" && "Running side A, then side B..."}
      </div>

      {outcomes && (
        <section aria-labelledby={`${id}-res`} className="results">
          <h2 id={`${id}-res`}>Comparison</h2>
          <div className="field">
            <label htmlFor={`${id}-conf`}>Minimum confidence for both sides: {pct}%</label>
            <input
              id={`${id}-conf`}
              type="range"
              min={Math.round(COMPARE_CONFIDENCE_FLOOR * 100)}
              max={100}
              step={1}
              value={pct}
              aria-valuetext={`${pct} percent`}
              onChange={(e) => setMinConfidence(clampConfidence(Number(e.target.value) / 100, COMPARE_CONFIDENCE_FLOOR))}
            />
          </div>

          {agreement && (
            <div role="status" className="agreement" data-testid="agreement">
              <p>
                <strong>Agreement {Math.round(agreement.agreement * 100)}%</strong>: {agreement.matched} matched,{" "}
                {agreement.onlyA.length} only in A, {agreement.onlyB.length} only in B
                {agreement.meanIou !== null && <>, mean IoU {agreement.meanIou.toFixed(2)}</>}.
              </p>
              <p className="hint">
                A box matches when it overlaps one from the other side with IoU 0.5 or more and has the same class; each
                box is used once. Agreement is matched / (A + B - matched). Counts include only detections shown at the
                threshold above.
              </p>
              {(agreement.onlyA.length > 0 || agreement.onlyB.length > 0) && (
                <ul className="class-list">
                  {agreement.onlyA.map((i) => (
                    <li key={`a${i}`}>
                      Only in A: {visA[i]?.name} {Math.round((visA[i]?.confidence ?? 0) * 100)}%
                    </li>
                  ))}
                  {agreement.onlyB.map((i) => (
                    <li key={`b${i}`}>
                      Only in B: {visB[i]?.name} {Math.round((visB[i]?.confidence ?? 0) * 100)}%
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="lanes">
            <LaneView label="A" outcome={outcomes[0]} image={image} minConfidence={minConfidence} visible={visA} />
            <LaneView label="B" outcome={outcomes[1]} image={image} minConfidence={minConfidence} visible={visB} />
          </div>
        </section>
      )}
    </>
  );
}
