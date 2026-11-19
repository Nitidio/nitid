import { ringToFlat, segmentsToRing } from "../api/segments";
import type { Box, Run, RunItem, Task } from "../api/types";
import { fromServerRun } from "../exports/mappers";
import type { ExportResults } from "../exports/model";

/**
 * UI-facing results model. It is deliberately independent of where the numbers come
 * from: the local panel builds it from `GET /runs/{id}` (`fromRun`) and the browser
 * engine (src/engine) can build the same shape, so the unified UI (issue #46) can render
 * either with the same `ResultsView` and hand it to the same export code.
 */

/** Pixel mask of one detection, in coordinates of the ORIGINAL image. */
export interface DetectionMask {
  /** Closed polygons, each a flat `[x0, y0, x1, y1, ...]` array. */
  polygons: number[][];
}

export interface ResultDetection {
  box: Box;
  confidence: number;
  classId: number;
  name: string;
  hasMask: boolean;
  /**
   * Mask geometry when the source provides it. Server runs carry the `segments` outline
   * (one polygon, no pixels); browser runs carry the traced contour of the pixel mask.
   * Undefined when only the `has_mask` flag is known (servers that predate `segments`).
   */
  mask?: DetectionMask;
}

export interface ResultItem {
  id: number;
  name: string;
  width: number;
  height: number;
  detections: ResultDetection[];
  /** Milliseconds per stage, as reported by the engine that ran the model. */
  speed: { preprocess: number; inference: number; postprocess: number };
}

export interface RunResults {
  /** Opaque id of the run in its source (a server run id, or a local id for browser runs). */
  runId: string;
  source: "local" | "browser";
  model: string;
  task: Task;
  backend: string;
  device: string | null;
  items: ResultItem[];
  /** Wall-clock milliseconds from "Run" to results (upload + queue + inference), if measured. */
  wallClockMs?: number;
  /**
   * Builds the `ExportResults` of one item through the shared mappers (`fromServerRun` /
   * `fromEngineResult`), so the export code never needs to know where the numbers came from.
   * Undefined when a producer does not support exports.
   */
  toExport?: (itemId: number) => ExportResults;
}

/** What the user has chosen to see; export code should honour it ("export what I see"). */
export interface ViewFilter {
  /** Detections below this confidence are hidden. */
  minConfidence: number;
  /** Class ids the user switched off. */
  hiddenClasses: ReadonlySet<number>;
}

/** Props of the slot component the unified UI plugs in to offer exports (CSV, JSON, PDF...). */
export interface ExportMenuProps {
  results: RunResults;
  filter: ViewFilter;
  /** The item currently on screen. */
  selectedItemId: number;
  /** Decoded original image of that item, when available (PNG, crops and PDF need it). */
  image?: ImageBitmap | null;
}

function fromItem(item: RunItem): ResultItem {
  return {
    id: item.id,
    name: item.name,
    width: item.width,
    height: item.height,
    speed: { ...item.speed },
    detections: item.detections.map((d) => {
      const ring = segmentsToRing(d.segments);
      return {
        box: { ...d.box },
        confidence: d.confidence,
        classId: d.class_id,
        name: d.name,
        hasMask: d.has_mask,
        ...(ring ? { mask: { polygons: [ringToFlat(ring)] } } : {}),
      };
    }),
  };
}

/** Convert a finished `Run` from the API into the UI model. */
export function fromRun(run: Run, wallClockMs?: number): RunResults {
  return {
    runId: run.id,
    source: "local",
    model: run.model,
    task: run.task,
    backend: run.backend,
    device: run.device,
    items: run.items.map(fromItem),
    toExport: (itemId) => fromServerRun(run, itemId),
    ...(wallClockMs !== undefined ? { wallClockMs } : {}),
  };
}

/** Total engine time of an item in ms (sum of the three stages). */
export function itemTotalMs(item: ResultItem): number {
  return item.speed.preprocess + item.speed.inference + item.speed.postprocess;
}
