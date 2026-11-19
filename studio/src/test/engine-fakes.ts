/** Shared fakes for component tests of the browser engine path (no Web Worker, no WebAssembly). */
import { vi } from "vitest";
import type { RunResult } from "../engine/session";
import type { Detection } from "../engine/types";
import type { EngineLike } from "../browser/runner";

export function engineDetection(over: Partial<Detection> = {}): Detection {
  return { box: { x1: 10, y1: 10, x2: 60, y2: 70 }, confidence: 0.9, class_id: 0, name: "person", has_mask: false, ...over };
}

export function engineResult(detections: Detection[] = [engineDetection()]): RunResult {
  return {
    detections,
    imageSize: { width: 200, height: 100 },
    timings: { preprocessMs: 1, inferenceMs: 5, postprocessMs: 1 },
  };
}

export function fakeEngine(task: "detect" | "segment" = "detect", detections?: Detection[]) {
  return {
    init: vi.fn(async (_p: unknown, onProgress?: (l: number, t: number | undefined) => void) => {
      onProgress?.(50, 100);
      return { backend: "wasm", numThreads: 1, modelSource: "network", modelCached: true, meta: { task, imgsz: 64 } } as never;
    }),
    run: vi.fn(async () => engineResult(detections)),
    dispose: vi.fn(async () => undefined),
    terminate: vi.fn(),
  } satisfies EngineLike;
}

/** jsdom has no createImageBitmap. */
export function stubImageBitmap(): void {
  vi.stubGlobal(
    "createImageBitmap",
    vi.fn(async () => ({ width: 200, height: 100, close: vi.fn() }) as unknown as ImageBitmap),
  );
}
