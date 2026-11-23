/**
 * Message protocol between the page and `inference.worker.ts`.
 *
 * Every request carries a numeric `id`; the worker answers with exactly one
 * `ok` or `error` message for that id (plus any number of `progress` messages
 * while a model downloads). Messages are plain structured-cloneable objects;
 * large buffers are transferred, not copied.
 */
import type { SerializedEngineError } from "../engine/errors";
import type { Backend, InitInfo, RunOptions, RunResult } from "../engine/session";

export interface InitRequest {
  id: number;
  type: "init";
  /** Fetch the model from this URL (same origin, or the hosted models origin over CORS; cached by sha256/ETag)... */
  modelUrl?: string;
  /** ...or use these bytes (a file the user picked). Transferred. */
  modelBytes?: ArrayBuffer;
  /** Request headers for `modelUrl` (the bearer token of the local server). Not used with `modelBytes`. */
  headers?: Record<string, string>;
  /** Known sha256 of the file (from the meta); cache key and integrity check. */
  sha256?: string;
  /** Meta from the REST API; otherwise read from the ONNX `metadata_props`. */
  meta?: unknown;
  backend?: Backend;
  /** Absolute prefix of the ORT wasm files (default `/ort/`). */
  wasmBase?: string;
}

export type RunImage =
  | { kind: "rgba"; data: ArrayBuffer; width: number; height: number }
  | { kind: "bitmap"; bitmap: ImageBitmap };

export interface RunRequest {
  id: number;
  type: "run";
  image: RunImage;
  options?: RunOptions;
}

export interface DisposeRequest {
  id: number;
  type: "dispose";
}

export type WorkerRequest = InitRequest | RunRequest | DisposeRequest;

export interface InitResult extends InitInfo {
  modelSource: "cache" | "network" | "bytes";
  modelCached: boolean;
}

export type WorkerResponse =
  | { id: number; type: "ok"; result: InitResult | RunResult | null }
  | { id: number; type: "error"; error: SerializedEngineError }
  | { id: number; type: "progress"; loaded: number; total: number | undefined };
