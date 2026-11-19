/**
 * Normalised results model shared by every exporter.
 *
 * Two producers map into it (see `mappers.ts`): the in-browser engine
 * (`RunResult` of `src/engine`) and the REST server (`Run` / `RunItem` of
 * `api-contract/openapi.json`). Every exporter reads only this type, so a new
 * format never needs to know where the numbers came from.
 *
 * Coordinates: `box` is xyxy in pixels of the ORIGINAL image (same convention
 * as the engine and the REST API). A mask is a binary raster cropped to its
 * box: `data` is `width * height` bytes of 0/1, top-left pixel at `(x, y)`.
 */

export type ExportTask = "detect" | "segment";

/** Where the numbers came from. */
export type ExportSource = "browser" | "server";

export interface ExportBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ExportMask {
  x: number;
  y: number;
  width: number;
  height: number;
  data: Uint8Array;
}

export interface ExportDetection {
  box: ExportBox;
  /** 0..1 */
  confidence: number;
  classId: number;
  /** Class name; may be any Unicode text (user models). */
  name: string;
  /**
   * The model produced a mask for this detection. `mask` holds its pixels when
   * they are available; the REST API only reports the flag, so for server
   * results `hasMask` can be true while `mask` is undefined.
   */
  hasMask: boolean;
  mask?: ExportMask;
}

export interface ExportTimings {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
}

export interface ExportResults {
  source: ExportSource;
  image: { name: string; width: number; height: number };
  model: string;
  task: ExportTask;
  /** Browser: "wasm" | "webgpu". Server: "torch" | "openvino". */
  backend: string;
  /** Server device name ("cpu", "cuda:0", "NPU", ...); null when unknown/default. */
  device: string | null;
  timings: ExportTimings | null;
  /** ISO 8601 timestamp of the export (injected, so outputs are reproducible). */
  createdAt: string;
  detections: ExportDetection[];
}

/** The formats `ExportMenu` and `buildExport` know about. */
export type ExportFormat = "png" | "json" | "csv" | "yolo" | "crops" | "pdf" | "zip";

export const ALL_FORMATS: readonly ExportFormat[] = ["png", "json", "csv", "yolo", "crops", "pdf", "zip"];

/** Formats that are the members of a ZIP bundle (everything but "zip"). */
export type BundleFormat = Exclude<ExportFormat, "zip">;
export const BUNDLE_FORMATS: readonly BundleFormat[] = ["png", "json", "csv", "yolo", "crops", "pdf"];

/** Formats that need pixels of the source image. */
export const IMAGE_FORMATS: readonly ExportFormat[] = ["png", "crops", "pdf", "zip"];

/** `street view.JPG` -> `street_view`. Keeps Unicode letters and digits. */
export function baseName(imageName: string): string {
  const stem = imageName.replace(/^.*[\\/]/, "").replace(/\.[A-Za-z0-9]{1,5}$/, "");
  const clean = stem.replace(/[^\p{L}\p{N}_-]+/gu, "_").replace(/^_+|_+$/g, "");
  return clean.length > 0 ? clean : "image";
}

/** Class-name list indexed by class id; ids that never occur get `class_<id>`. */
export function classNameList(results: ExportResults): string[] {
  const names = new Map<number, string>();
  let max = -1;
  for (const d of results.detections) {
    if (!names.has(d.classId)) names.set(d.classId, d.name);
    if (d.classId > max) max = d.classId;
  }
  const out: string[] = [];
  for (let i = 0; i <= max; i++) out.push(names.get(i) ?? `class_${i}`);
  return out;
}
