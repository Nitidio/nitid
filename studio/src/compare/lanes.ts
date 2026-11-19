/** One side of the comparison: run the same image through the server or through the browser. */
import type { LocalApi } from "../api/local";
import type { Model } from "../api/types";
import type { BrowserModel } from "../browser/catalog";
import { BROWSER_CONFIDENCE_FLOOR, type BrowserRunner } from "../browser/runner";
import { describeBrowserError } from "../browser/errors";
import type { DeviceChoice } from "../local/devices";
import type { DescribedError } from "../local/errors";
import { pollRun } from "../local/polling";
import { fromRun, type RunResults } from "../local/results-model";

/** Same floor as the server panel and the browser panel, so one threshold applies to every lane. */
export const COMPARE_CONFIDENCE_FLOOR = BROWSER_CONFIDENCE_FLOOR;

export type LaneSpec =
  | { kind: "server"; model: Model; choice: DeviceChoice }
  | { kind: "browser"; model: BrowserModel };

export type LaneOutcome =
  | { status: "ok"; spec: LaneSpec; results: RunResults; loadMs: number | null }
  | { status: "error"; spec: LaneSpec; error: DescribedError };

export function laneTitle(spec: LaneSpec, results?: RunResults): string {
  if (spec.kind === "server") {
    const backend = spec.choice.backend === "openvino" ? "OpenVINO" : "PyTorch";
    return `${spec.model.id} (${spec.model.task}), server: ${backend} / ${spec.choice.device}`;
  }
  return `${spec.model.id} (${spec.model.task}), in the browser${results ? ` (${results.backend})` : ""}`;
}

export interface LaneDeps {
  api: LocalApi;
  /** A fresh runner per lane run; the caller closes it afterwards through the returned `close`. */
  createRunner: () => BrowserRunner;
  now?: () => number;
}

/** Runs one lane. Never throws (other than abort): failures become an `error` outcome. */
export async function runLane(spec: LaneSpec, file: File, deps: LaneDeps, signal?: AbortSignal): Promise<LaneOutcome> {
  const now = deps.now ?? (() => performance.now());
  try {
    if (spec.kind === "server") {
      const started = now();
      const runId = await deps.api.createRun(
        [file],
        {
          model: spec.model.id,
          task: spec.model.task,
          backend: spec.choice.backend,
          device: spec.choice.device,
          conf: COMPARE_CONFIDENCE_FLOOR,
        },
        signal,
      );
      const run = await pollRun({ getRun: () => deps.api.getRun(runId, signal), ...(signal ? { signal } : {}) });
      return { status: "ok", spec, results: fromRun(run, now() - started), loadMs: null };
    }
    const runner = deps.createRunner();
    try {
      const t0 = now();
      await runner.load(spec.model);
      const loadMs = now() - t0;
      const results = await runner.runFiles([file], { conf: COMPARE_CONFIDENCE_FLOOR });
      return { status: "ok", spec, results, loadMs };
    } finally {
      runner.close();
    }
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    return { status: "error", spec, error: describeBrowserError(e) };
  }
}
