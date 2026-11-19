import type { Task } from "../api/types";
import type { RunResult } from "../engine/session";
import { maskToPolygon } from "../exports/contour";
import { fromEngineResult } from "../exports/mappers";
import type { ResultDetection, ResultItem, RunResults } from "../local/results-model";

export interface BrowserRunItem {
  name: string;
  result: RunResult;
}

export interface BuildBrowserResultsArgs {
  runId: string;
  model: string;
  task: Task;
  backend: "wasm" | "webgpu";
  items: readonly BrowserRunItem[];
  wallClockMs?: number;
}

function toDetection(d: RunResult["detections"][number]): ResultDetection {
  const out: ResultDetection = {
    box: { ...d.box },
    confidence: d.confidence,
    classId: d.class_id,
    name: d.name,
    hasMask: d.has_mask || d.mask !== undefined,
  };
  if (d.mask) {
    const poly = maskToPolygon({ ...d.mask }, 1);
    if (poly) out.mask = { polygons: [poly.flatMap(([x, y]) => [x, y])] };
  }
  return out;
}

/**
 * Engine output -> the UI results model (same shape the server panel produces, so the same
 * `ResultsView` and canvas draw it). Masks become outlines through the export module's contour code.
 */
export function buildBrowserResults(args: BuildBrowserResultsArgs): RunResults {
  const items: ResultItem[] = args.items.map((it, id) => ({
    id,
    name: it.name,
    width: it.result.imageSize.width,
    height: it.result.imageSize.height,
    speed: {
      preprocess: it.result.timings.preprocessMs,
      inference: it.result.timings.inferenceMs,
      postprocess: it.result.timings.postprocessMs,
    },
    detections: it.result.detections.map(toDetection),
  }));
  return {
    runId: args.runId,
    source: "browser",
    model: args.model,
    task: args.task,
    backend: args.backend,
    device: null,
    items,
    ...(args.wallClockMs !== undefined ? { wallClockMs: args.wallClockMs } : {}),
    toExport(itemId) {
      const it = args.items[itemId];
      if (!it) throw new RangeError(`No item ${itemId}`);
      return fromEngineResult(it.result, {
        imageName: it.name,
        model: args.model,
        task: args.task,
        backend: args.backend,
      });
    },
  };
}
