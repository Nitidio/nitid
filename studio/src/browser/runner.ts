/**
 * Drives the inference worker for the UI: load a model, run images, build `RunResults`.
 *
 * The engine itself is injected (`EngineLike`), so tests use a fake instead of a Web Worker; the
 * real one is `createInferenceClient()` from `src/engine`.
 */
import type { InitParams } from "../engine/client";
import type { InitResult, RunImage } from "../workers/protocol";
import type { RunOptions, RunResult } from "../engine/session";
import { EngineError } from "../engine/errors";
import type { RunResults } from "../local/results-model";
import type { BrowserModel } from "./catalog";
import { buildBrowserResults } from "./results";

/** What the runner uses of `InferenceClient`. */
export interface EngineLike {
  init(params: InitParams, onProgress?: (loaded: number, total: number | undefined) => void): Promise<InitResult>;
  run(image: RunImage, options?: RunOptions): Promise<RunResult>;
  dispose(): Promise<void>;
  terminate(): void;
}

export interface RunnerDeps {
  createEngine: () => EngineLike;
  /** Bearer token for the local server's ONNX endpoints (null on the hosted site). */
  getToken: () => string | null;
  /** `GET /onnx/{id}/meta` as JSON (local deployment only). */
  fetchServerMeta: (model: BrowserModel) => Promise<unknown>;
  decode?: (file: File) => Promise<ImageBitmap>;
  now?: () => number;
}

/** Confidence floor of browser runs, the same as the server panel: the slider then filters instantly. */
export const BROWSER_CONFIDENCE_FLOOR = 0.1;

export interface LoadedModelInfo {
  model: BrowserModel;
  backend: "wasm" | "webgpu";
  source: InitResult["modelSource"];
  cached: boolean;
  imgsz: number;
}

export class BrowserRunner {
  private engine: EngineLike | null = null;
  private loaded: LoadedModelInfo | null = null;
  private runCounter = 0;

  constructor(private readonly deps: RunnerDeps) {}

  get current(): LoadedModelInfo | null {
    return this.loaded;
  }

  /** Download (or read from the Cache API) and start a model. Replaces any previous one. */
  async load(
    model: BrowserModel,
    onProgress?: (loaded: number, total: number | undefined) => void,
  ): Promise<LoadedModelInfo> {
    this.loaded = null;
    this.engine ??= this.deps.createEngine();
    let params: InitParams;
    if (model.origin === "server") {
      const meta = await this.deps.fetchServerMeta(model);
      const sha = (meta as { sha256?: unknown } | null)?.sha256;
      const token = this.deps.getToken();
      params = {
        modelUrl: model.url,
        meta,
        ...(typeof sha === "string" ? { sha256: sha } : {}),
        ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
      };
    } else {
      params = { modelUrl: model.url, ...(model.sha256 ? { sha256: model.sha256 } : {}) };
    }
    const info = await this.engine.init(params, onProgress);
    if (info.meta.task !== model.task) {
      throw new EngineError("invalid_meta", `The file is a ${info.meta.task} model but ${model.task} was expected.`);
    }
    this.loaded = {
      model,
      backend: info.backend,
      source: info.modelSource,
      cached: info.modelCached,
      imgsz: info.meta.imgsz,
    };
    return this.loaded;
  }

  /** Run the loaded model on each file, one after the other. */
  async runFiles(files: readonly File[], options: RunOptions = { conf: BROWSER_CONFIDENCE_FLOOR }): Promise<RunResults> {
    const engine = this.engine;
    const loaded = this.loaded;
    if (!engine || !loaded) throw new EngineError("session_not_ready", "Load a model first.");
    const now = this.deps.now ?? (() => performance.now());
    const decode = this.deps.decode ?? ((f: File) => createImageBitmap(f));
    const started = now();
    const items = [];
    for (const file of files) {
      const bitmap = await decode(file);
      // The worker takes ownership of (and closes) the bitmap it receives.
      const result = await engine.run({ kind: "bitmap", bitmap }, options);
      items.push({ name: file.name, result });
    }
    return buildBrowserResults({
      runId: `browser-${++this.runCounter}`,
      model: loaded.model.id,
      task: loaded.model.task,
      backend: loaded.backend,
      items,
      wallClockMs: now() - started,
    });
  }

  /** Stop the worker and free the model. */
  close(): void {
    this.loaded = null;
    this.engine?.terminate();
    this.engine = null;
  }
}
