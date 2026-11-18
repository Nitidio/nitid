/**
 * Helpers shared by the engine tests (not part of the runtime bundle).
 *
 * `synthImage` re-implements `synth_image` of `studio/tools/make_parity_fixtures.py`
 * (generator "gradient-v1"); keep both in sync. The fixtures hold what Python
 * produced for that image, so no pixel data or binary is stored in the repo.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Detection, NamedTensors, OnnxMeta, RgbaImage } from "./types";
import { maskToFull } from "./decode";

export function synthImage(width: number, height: number): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  const x0a = Math.floor(width / 5);
  const x1a = Math.floor(width / 2);
  const y0a = Math.floor(height / 4);
  const y1a = Math.floor(height / 2);
  const x0b = Math.floor(width / 2);
  const x1b = Math.floor((width * 4) / 5);
  const y0b = Math.floor(height / 2);
  const y1b = Math.floor((height * 4) / 5);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = Math.floor((x * 255) / Math.max(width - 1, 1));
      let g = Math.floor((y * 255) / Math.max(height - 1, 1));
      let b = (x * 7 + y * 13) % 256;
      if (x >= x0a && x < x1a && y >= y0a && y < y1a) [r, g, b] = [250, 40, 40];
      if (x >= x0b && x < x1b && y >= y0b && y < y1b) [r, g, b] = [20, 30, 220];
      const i = (y * width + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width, height };
}

export interface FixtureDetection {
  box: [number, number, number, number];
  score: number;
  class_id: number;
  name: string;
  mask_rle?: number[];
  mask_area?: number;
}
export interface FixtureCase {
  name: string;
  conf: number;
  classes: number[] | null;
  mask_threshold: number;
  detections: FixtureDetection[];
}
interface FixtureTensor {
  dims: number[];
  data: number[];
}
export interface ParityFixture {
  task: "detect" | "segment";
  imgsz: number;
  image: { generator: string; width: number; height: number };
  meta: OnnxMeta;
  raw_meta: OnnxMeta;
  preprocess: {
    image: { generator: string; width: number; height: number };
    channel_level_sums: number[];
    samples: { c: number; y: number; x: number; level: number }[];
  }[];
  postprocessed_outputs: {
    labels: number[];
    boxes: number[];
    scores: number[];
    masks?: FixtureTensor;
    dims: Record<string, number[]>;
  };
  raw_outputs: Record<string, FixtureTensor>;
  cases: FixtureCase[];
  all_scores: number[];
}

export function loadFixture(task: "detect" | "segment"): ParityFixture {
  const path = resolve(import.meta.dirname, "__fixtures__", `parity.${task}.json`);
  return JSON.parse(readFileSync(path, "utf8")) as ParityFixture;
}

/** The ONNX outputs of the postprocessed layout, as ORT would return them. */
export function postprocessedTensors(fixture: ParityFixture): NamedTensors {
  const out = fixture.postprocessed_outputs;
  const tensors: NamedTensors = {
    labels: { data: BigInt64Array.from(out.labels.map(BigInt)), dims: out.dims.labels! },
    boxes: { data: Float32Array.from(out.boxes), dims: out.dims.boxes! },
    scores: { data: Float32Array.from(out.scores), dims: out.dims.scores! },
  };
  if (out.masks) tensors.masks = { data: Float32Array.from(out.masks.data), dims: out.masks.dims };
  return tensors;
}

export function rawTensors(fixture: ParityFixture): NamedTensors {
  return Object.fromEntries(
    Object.entries(fixture.raw_outputs).map(([name, t]) => [name, { data: Float32Array.from(t.data), dims: t.dims }]),
  );
}

export function boxIou(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union <= 0 ? 1 : inter / union;
}

/** Decodes a run-length mask (alternating 0/1 runs, starting with zeros). */
export function rleToMask(runs: number[], length: number): Uint8Array {
  const out = new Uint8Array(length);
  let pos = 0;
  let value = 0;
  for (const run of runs) {
    if (value === 1) out.fill(1, pos, pos + run);
    pos += run;
    value ^= 1;
  }
  return out;
}

export function maskIou(a: Uint8Array, b: Uint8Array): number {
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++;
    if (a[i] || b[i]) union++;
  }
  return union === 0 ? 1 : inter / union;
}

export interface ParityReport {
  matched: number;
  minBoxIou: number;
  maxScoreDelta: number;
  minMaskIou: number;
}

/**
 * Greedy one-to-one matching of TS detections to Python detections by class and
 * box IoU (scores tie inside a class in the tiny fixture model, so the order of
 * rows is not a reliable key). Throws if any Python detection has no partner.
 */
export function compareWithPython(
  detections: Detection[],
  expected: FixtureDetection[],
  image: { width: number; height: number },
): ParityReport {
  if (detections.length !== expected.length) {
    throw new Error(`expected ${expected.length} detections, got ${detections.length}`);
  }
  const used = new Set<number>();
  const report: ParityReport = { matched: 0, minBoxIou: 1, maxScoreDelta: 0, minMaskIou: 1 };
  for (const want of expected) {
    let best = -1;
    let bestIou = -1;
    detections.forEach((got, i) => {
      if (used.has(i) || got.class_id !== want.class_id) return;
      const iou = boxIou([got.box.x1, got.box.y1, got.box.x2, got.box.y2], want.box);
      if (iou > bestIou) {
        bestIou = iou;
        best = i;
      }
    });
    if (best < 0) throw new Error(`no detection of class ${want.class_id} left to match`);
    used.add(best);
    const got = detections[best]!;
    report.matched++;
    report.minBoxIou = Math.min(report.minBoxIou, bestIou);
    report.maxScoreDelta = Math.max(report.maxScoreDelta, Math.abs(got.confidence - want.score));
    if (want.mask_rle) {
      const total = image.width * image.height;
      const expectedMask = rleToMask(want.mask_rle, total);
      const gotMask = got.mask ? maskToFull(got.mask, image.width, image.height) : new Uint8Array(total);
      report.minMaskIou = Math.min(report.minMaskIou, maskIou(gotMask, expectedMask));
    }
  }
  return report;
}
