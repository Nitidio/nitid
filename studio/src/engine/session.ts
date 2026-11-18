/**
 * Inference session: ONNX Runtime Web + preprocessing + decoding.
 *
 * `onnxruntime-web` is reached only through the small {@link OrtModule}
 * interface, so the session logic (provider choice, thread count, input/output
 * wiring, serialisation of runs, limits) is unit-tested without WebAssembly.
 * The real module is loaded by `src/workers/ort-loader.ts`.
 *
 * Decisions (all same-origin, no CDN; CSP `script-src 'self' 'wasm-unsafe-eval';
 * worker-src 'self' blob:`):
 *
 * - `wasm` EP by default. `webgpu` only on request (`backend: "webgpu" | "auto"`)
 *   and only if `navigator.gpu` yields an adapter; any failure falls back to
 *   `wasm` and reports why (`fallbackReason`). `[NV]` WebGPU was not verified in
 *   a real browser here (headless Chrome without a GPU).
 * - `env.wasm.wasmPaths` must be ABSOLUTE (`/ort/`): a relative prefix is
 *   resolved against the worker/script URL and ends up duplicated.
 * - `numThreads`: multi-threading needs `SharedArrayBuffer`, i.e. a
 *   cross-origin isolated page (COOP/COEP). Without isolation we force 1 thread
 *   instead of letting ORT guess; with it, up to 4 (or fewer cores).
 */
import { decodeOutputs } from "./decode";
import { EngineError } from "./errors";
import { outputNames, parseMeta } from "./meta";
import { readEmbeddedMeta, stripValueInfo } from "./onnx-metadata";
import { preprocessCanvas, preprocessRgba, type CanvasFactory, bitmapToRgba, type ResizeMode, validateDimensions } from "./preprocess";
import type { DecodeOptions, Detection, NamedTensors, OnnxMeta, RgbaImage } from "./types";

export type Backend = "wasm" | "webgpu" | "auto";

/** What the engine uses of `onnxruntime-web` (a structural subset of its API). */
export interface OrtTensorOutput {
  data: ArrayLike<number> | ArrayLike<bigint>;
  dims: readonly number[];
  dispose?: () => void;
}
export interface OrtSessionLike {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorOutput>>;
  release(): Promise<void>;
}
export interface OrtModule {
  env: { wasm: { wasmPaths?: string | Record<string, string>; numThreads?: number; proxy?: boolean } };
  InferenceSession: {
    create(
      model: Uint8Array,
      options: { executionProviders: string[]; graphOptimizationLevel?: string },
    ): Promise<OrtSessionLike>;
  };
  Tensor: new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;
}

export interface EnvironmentInfo {
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
  /** Resolves true when a WebGPU adapter is available. Must never reject. */
  hasWebGpu(): Promise<boolean>;
}

export interface SessionDeps {
  /** Loads the ORT build for the requested provider (`wasm` or `webgpu` bundle). */
  loadOrt(provider: "wasm" | "webgpu"): Promise<OrtModule>;
  env: EnvironmentInfo;
  makeCanvas?: CanvasFactory;
}

export interface SessionInit {
  modelBytes: ArrayBuffer;
  /** Meta supplied by the caller (REST `/onnx/{model}/meta`); else read from the ONNX file. */
  meta?: unknown;
  backend?: Backend;
  /** Absolute URL prefix of the ORT `.wasm`/`.mjs` files. Default `/ort/`. */
  wasmBase?: string;
}

export interface InitInfo {
  backend: "wasm" | "webgpu";
  fallbackReason?: string;
  /** Load-time workarounds that were needed (e.g. `"stripped-value-info"`). */
  workarounds?: string[];
  numThreads: number;
  meta: OnnxMeta;
}

export interface RunOptions extends DecodeOptions {
  /** `"pil"` (default, bit-exact with Python) or `"canvas"` (faster, approximate; ImageBitmap only). */
  resize?: ResizeMode;
}

export type ImageInput =
  | ({ kind: "rgba" } & RgbaImage)
  | { kind: "bitmap"; bitmap: { width: number; height: number } };

export interface RunTimings {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
}

export interface RunResult {
  detections: Detection[];
  imageSize: { width: number; height: number };
  timings: RunTimings;
}

export function assertAbsoluteWasmBase(base: string): string {
  // Root-relative ("/ort/") or with a scheme ("scheme://host/ort/"); "//host/" is protocol-relative: refuse.
  const rooted = base.startsWith("/") && !base.startsWith("//");
  if (!(rooted || /^[a-z][a-z0-9+.-]*:\/\//i.test(base))) {
    throw new EngineError(
      "invalid_options",
      `wasmBase must be absolute (root-relative like "/ort/", or a full URL), got "${base}". A relative path is resolved twice by onnxruntime-web.`,
    );
  }
  return base.endsWith("/") ? base : `${base}/`;
}

/** 1 thread without cross-origin isolation (no SharedArrayBuffer); else min(4, cores). */
export function chooseNumThreads(env: Pick<EnvironmentInfo, "crossOriginIsolated" | "hardwareConcurrency">): number {
  if (!env.crossOriginIsolated) return 1;
  return Math.max(1, Math.min(4, Math.floor(env.hardwareConcurrency || 1)));
}

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class EngineSession {
  private chain: Promise<unknown> = Promise.resolve();
  private disposed = false;

  private constructor(
    private readonly ort: OrtModule,
    private readonly session: OrtSessionLike,
    readonly info: InitInfo,
    private readonly makeCanvas: CanvasFactory | undefined,
  ) {}

  static async create(deps: SessionDeps, init: SessionInit): Promise<EngineSession> {
    const bytes = new Uint8Array(init.modelBytes);
    const meta = init.meta !== undefined ? parseMeta(init.meta) : readEmbeddedMeta(bytes);
    if (!meta) {
      throw new EngineError(
        "invalid_meta",
        'This ONNX file has no "nitid_meta" entry. Export it with nitid >= the version that embeds it, or pass the meta explicitly.',
      );
    }
    const wasmBase = assertAbsoluteWasmBase(init.wasmBase ?? "/ort/");
    const requested = init.backend ?? "wasm";
    const numThreads = chooseNumThreads(deps.env);

    let fallbackReason: string | undefined;
    const workarounds: string[] = [];
    let wantWebGpu = requested === "webgpu" || requested === "auto";
    if (wantWebGpu && !(await safe(deps.env.hasWebGpu()))) {
      wantWebGpu = false;
      if (requested === "webgpu") fallbackReason = "WebGPU is not available in this browser";
    }

    let provider: "wasm" | "webgpu" = wantWebGpu ? "webgpu" : "wasm";
    let ort: OrtModule | undefined;
    let session: OrtSessionLike | undefined;
    if (provider === "webgpu") {
      try {
        ort = await deps.loadOrt("webgpu");
        configureEnv(ort, wasmBase, numThreads);
        session = await createWithWorkarounds(ort, bytes, ["webgpu", "wasm"], workarounds);
      } catch (error) {
        fallbackReason = `WebGPU session could not be created: ${String(error)}`;
        provider = "wasm";
        ort = undefined;
        session = undefined;
      }
    }
    if (!ort || !session) {
      try {
        ort = await deps.loadOrt("wasm");
        configureEnv(ort, wasmBase, numThreads);
        session = await createWithWorkarounds(ort, bytes, ["wasm"], workarounds);
      } catch (error) {
        throw new EngineError("backend_unavailable", `Could not start ONNX Runtime Web: ${String(error)}`, {
          cause: error,
        });
      }
    }

    // The meta must describe what the file really exposes.
    const missing = outputNames(meta).filter((name) => !session!.outputNames.includes(name));
    if (!session.inputNames.includes(meta.input.name) || missing.length > 0) {
      await session.release().catch(() => undefined);
      throw new EngineError(
        "unsupported_output",
        `The ONNX file does not match its meta: inputs [${session.inputNames.join(", ")}], outputs [${session.outputNames.join(", ")}], meta expects input "${meta.input.name}" and outputs [${outputNames(meta).join(", ")}]`,
      );
    }

    const info: InitInfo = {
      backend: provider,
      numThreads,
      meta,
      ...(fallbackReason ? { fallbackReason } : {}),
      ...(workarounds.length > 0 ? { workarounds } : {}),
    };
    return new EngineSession(ort, session, info, deps.makeCanvas);
  }

  get meta(): OnnxMeta {
    return this.info.meta;
  }

  /** Runs one image. Calls are serialised: an ORT session must not run concurrently. */
  run(image: ImageInput, options: RunOptions = {}): Promise<RunResult> {
    if (this.disposed) {
      return Promise.reject(new EngineError("session_not_ready", "The session was disposed"));
    }
    const next = this.chain.then(() => this.runNow(image, options));
    this.chain = next.catch(() => undefined);
    return next;
  }

  private async runNow(image: ImageInput, options: RunOptions): Promise<RunResult> {
    const meta = this.info.meta;
    const size = meta.imgsz;

    const t0 = now();
    let original: { width: number; height: number };
    let chw: Float32Array;
    if (image.kind === "rgba") {
      original = { width: image.width, height: image.height };
      chw = preprocessRgba(image, size);
    } else {
      validateDimensions(image.bitmap.width, image.bitmap.height);
      original = { width: image.bitmap.width, height: image.bitmap.height };
      chw =
        options.resize === "canvas"
          ? preprocessCanvas(image.bitmap, size, this.makeCanvas)
          : preprocessRgba(bitmapToRgba(image.bitmap, this.makeCanvas), size);
    }
    const t1 = now();

    let outputs: Record<string, OrtTensorOutput>;
    const input = new this.ort.Tensor("float32", chw, meta.input.shape);
    try {
      outputs = await this.session.run({ [meta.input.name]: input });
    } catch (error) {
      throw new EngineError("inference_failed", `Inference failed: ${String(error)}`, { cause: error });
    }
    const t2 = now();

    try {
      const named: NamedTensors = {};
      for (const name of outputNames(meta)) {
        const tensor = outputs[name];
        if (tensor) named[name] = { data: tensor.data, dims: tensor.dims };
      }
      const decodeOptions: DecodeOptions = {};
      if (options.conf !== undefined) decodeOptions.conf = options.conf;
      if (options.classes !== undefined) decodeOptions.classes = options.classes;
      if (options.maskThreshold !== undefined) decodeOptions.maskThreshold = options.maskThreshold;
      if (options.maxMaskBytes !== undefined) decodeOptions.maxMaskBytes = options.maxMaskBytes;
      const detections = decodeOutputs(meta, named, original, decodeOptions);
      const t3 = now();
      return {
        detections,
        imageSize: original,
        timings: { preprocessMs: t1 - t0, inferenceMs: t2 - t1, postprocessMs: t3 - t2 },
      };
    } finally {
      for (const tensor of Object.values(outputs)) tensor.dispose?.();
    }
  }

  /** Stops accepting runs, lets the already queued ones finish, then releases the ORT session. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.chain;
    await this.session.release().catch(() => undefined);
  }
}

/**
 * Creates the ORT session; if ORT rejects the graph with a shape-inference
 * error caused by a wrong `value_info` entry, retries once without `value_info`
 * (see `stripValueInfo`). Any other error is rethrown untouched.
 */
async function createWithWorkarounds(
  ort: OrtModule,
  bytes: Uint8Array,
  executionProviders: string[],
  workarounds: string[],
): Promise<OrtSessionLike> {
  const options = { executionProviders, graphOptimizationLevel: "all" };
  try {
    return await ort.InferenceSession.create(bytes, options);
  } catch (error) {
    if (!/ShapeInferenceError|inferred=\d+ declared=\d+/.test(String(error))) throw error;
    const stripped = stripValueInfo(bytes);
    if (!stripped) throw error;
    const session = await ort.InferenceSession.create(stripped, options);
    workarounds.push("stripped-value-info");
    return session;
  }
}

function configureEnv(ort: OrtModule, wasmBase: string, numThreads: number): void {
  ort.env.wasm.wasmPaths = wasmBase;
  ort.env.wasm.numThreads = numThreads;
  // The engine already runs inside our own Web Worker; do not let ORT spawn another proxy worker.
  ort.env.wasm.proxy = false;
}

async function safe(promise: Promise<boolean>): Promise<boolean> {
  try {
    return await promise;
  } catch {
    return false;
  }
}
