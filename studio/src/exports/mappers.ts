/**
 * The two producers of `ExportResults`.
 *
 * - `fromEngineResult`: `RunResult` of `src/engine` (browser inference).
 * - `fromServerRun`: one item of `GET /runs/{id}` (`Run` / `RunItem` of
 *   `api-contract/openapi.json`).
 *
 * The server shapes are declared structurally here (the hand-written
 * `src/api/types.ts` has no run types yet) and contain exactly the fields the
 * contract marks as required.
 */
import type { Detection, DetectionMask } from "../engine/types";
import type { RunResult } from "../engine/session";
import type { ExportDetection, ExportMask, ExportResults, ExportTask } from "./model";

export interface EngineContext {
  imageName: string;
  /** Model id (e.g. `nitid1s`). */
  model: string;
  task: ExportTask;
  /** `InitInfo.backend` of the session. */
  backend: "wasm" | "webgpu";
  /** ISO timestamp; defaults to now. */
  createdAt?: string;
}

function mapMask(mask: DetectionMask | undefined): ExportMask | undefined {
  if (!mask) return undefined;
  return { x: mask.x, y: mask.y, width: mask.width, height: mask.height, data: mask.data };
}

function mapEngineDetection(d: Detection): ExportDetection {
  const mask = mapMask(d.mask);
  return {
    box: { x1: d.box.x1, y1: d.box.y1, x2: d.box.x2, y2: d.box.y2 },
    confidence: d.confidence,
    classId: d.class_id,
    name: d.name,
    hasMask: d.has_mask || mask !== undefined,
    ...(mask ? { mask } : {}),
  };
}

export function fromEngineResult(result: RunResult, ctx: EngineContext): ExportResults {
  return {
    source: "browser",
    image: { name: ctx.imageName, width: result.imageSize.width, height: result.imageSize.height },
    model: ctx.model,
    task: ctx.task,
    backend: ctx.backend,
    device: null,
    timings: {
      preprocessMs: result.timings.preprocessMs,
      inferenceMs: result.timings.inferenceMs,
      postprocessMs: result.timings.postprocessMs,
    },
    createdAt: ctx.createdAt ?? new Date().toISOString(),
    detections: result.detections.map(mapEngineDetection),
  };
}

/** `Detection` of the REST contract. */
export interface ServerDetection {
  box: { x1: number; y1: number; x2: number; y2: number };
  confidence: number;
  class_id: number;
  name: string;
  has_mask: boolean;
}

/** `RunItem` of the REST contract. */
export interface ServerRunItem {
  id: number;
  name: string;
  width: number;
  height: number;
  detections: ServerDetection[];
  speed: { preprocess: number; inference: number; postprocess: number };
}

/** The fields of `Run` the exporters need. */
export interface ServerRun {
  id: string;
  model: string;
  task: ExportTask;
  backend: "torch" | "openvino";
  device: string | null;
  items: ServerRunItem[];
}

/**
 * Maps item `itemIndex` (the `id` of the item, zero-based) of a finished run.
 * The REST API reports `has_mask` but not the mask pixels, so server results
 * never carry `mask`: PNG shows boxes only and YOLO TXT falls back to boxes.
 */
export function fromServerRun(run: ServerRun, itemIndex: number, createdAt?: string): ExportResults {
  const item = run.items.find((i) => i.id === itemIndex);
  if (!item) throw new RangeError(`Run ${run.id} has no item ${itemIndex}`);
  return {
    source: "server",
    image: { name: item.name, width: item.width, height: item.height },
    model: run.model,
    task: run.task,
    backend: run.backend,
    device: run.device,
    timings: {
      preprocessMs: item.speed.preprocess,
      inferenceMs: item.speed.inference,
      postprocessMs: item.speed.postprocess,
    },
    createdAt: createdAt ?? new Date().toISOString(),
    detections: item.detections.map((d) => ({
      box: { x1: d.box.x1, y1: d.box.y1, x2: d.box.x2, y2: d.box.y2 },
      confidence: d.confidence,
      classId: d.class_id,
      name: d.name,
      hasMask: d.has_mask,
    })),
  };
}
