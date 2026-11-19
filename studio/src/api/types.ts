/**
 * Minimal hand-written API types.
 *
 * TODO: these will be generated from `studio/api-contract/openapi.json` once
 * that contract exists (separate work package). Until then this file is the
 * single place that describes what the UI expects from the backend. Anything
 * marked [NV] is an assumption that has not been verified against the server.
 */

/** Major API version this UI build was written against (contract `Health.api_version`, an integer). */
export const EXPECTED_API_VERSION = 1;

/** Body of `GET /api/v1/health` (validated at runtime by `parseHealth`). */
export interface HealthResponse {
  status: "ok";
  nitid_version: string;
  /** Integer per the contract. `parseHealth` also accepts legacy servers that sent `"1"`. */
  api_version: number;
}

/**
 * Error body. The contract envelope is `{"error":{"code","message"}}`; `detail` and
 * `message` at the top level are tolerated for servers that predate it.
 */
export interface ApiErrorBody {
  error?: { code?: string; message?: string };
  detail?: string;
  message?: string;
}

/** Where the model runs. */
export type RunMode = "local" | "browser";

/** Which deployment of the SPA this is. */
export type BackendMode = "local" | "hosted";

/* ------------------------------------------------------------------ */
/* Contract types for the local endpoints (studio/api-contract/openapi.json, v1). */
/* Hand-written mirror of the schemas; keep names identical to the contract.      */
/* ------------------------------------------------------------------ */

export type Task = "detect" | "segment";
export type Backend = "torch" | "openvino";
export type JobStatus = "pending" | "running" | "done" | "failed";
export type RunStatus = JobStatus;

export interface CudaDevice {
  index: number;
  name: string;
}

/** `GET /devices`. */
export interface Devices {
  torch: { cpu: boolean; cuda: CudaDevice[] };
  openvino: { available: boolean; devices: string[] };
}

/** One `(id, task)` pair of `GET /models`. */
export interface Model {
  id: string;
  task: Task;
  size: "n" | "s" | "m" | "l" | "x" | null;
  downloaded: boolean;
  source: "registry" | "local";
  /** Only for `source: "local"`. */
  path?: string;
}

/** `GET /jobs/{id}`. */
export interface Job {
  status: JobStatus;
  /** 0..1 */
  progress: number;
  error?: string;
}

/** JSON in the `params` multipart field of `POST /runs`. */
export interface RunParams {
  model: string;
  task: Task;
  conf?: number;
  iou?: number;
  imgsz?: number;
  classes?: number[] | null;
  mask_threshold?: number;
  backend: Backend;
  device?: string | null;
  frame_step?: number;
}

/** Pixels of the ORIGINAL image. */
export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * Outline of an instance mask (`segments` of the contract): parallel arrays of vertex
 * coordinates in pixels of the ORIGINAL image. One outline per detection (the server sends
 * the largest contour). Not closed explicitly: the last vertex joins the first.
 */
export interface Segments {
  x: number[];
  y: number[];
}

export interface Detection {
  box: Box;
  confidence: number;
  class_id: number;
  name: string;
  has_mask: boolean;
  /** Present only for `task=segment` (`has_mask` true); absent for older servers. */
  segments?: Segments;
}

/** Milliseconds. */
export interface Speed {
  preprocess: number;
  inference: number;
  postprocess: number;
}

export interface RunItem {
  id: number;
  name: string;
  width: number;
  height: number;
  detections: Detection[];
  speed: Speed;
}

export interface Run {
  id: string;
  model: string;
  task: Task;
  backend: Backend;
  device: string | null;
  status: RunStatus;
  error_msg: string | null;
  item_count: number;
  items: RunItem[];
}
