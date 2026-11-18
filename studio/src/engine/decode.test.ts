import { describe, expect, it } from "vitest";
import { decodeOutputs, maskToFull } from "./decode";
import { parseMeta } from "./meta";
import { compareWithPython, loadFixture, postprocessedTensors, rawTensors } from "./test-utils";
import type { NamedTensors, OnnxMeta } from "./types";

// Tolerances required by WP-2.3 (the measured values are far tighter, see docs/studio-engine.md).
const BOX_IOU_MIN = 0.98;
const SCORE_DELTA_MAX = 0.01;
const MASK_IOU_MIN = 0.95;

describe("decode: parity with the Python predictor (fixtures made by studio/tools/make_parity_fixtures.py)", () => {
  for (const task of ["detect", "segment"] as const) {
    const fixture = loadFixture(task);
    for (const layout of ["postprocessed", "raw"] as const) {
      const meta = layout === "postprocessed" ? fixture.meta : fixture.raw_meta;
      const tensors = layout === "postprocessed" ? postprocessedTensors(fixture) : rawTensors(fixture);
      for (const testCase of fixture.cases) {
        it(`${task}/${layout}: ${testCase.name}`, () => {
          const detections = decodeOutputs(meta, tensors, fixture.image, {
            conf: testCase.conf,
            classes: testCase.classes,
            maskThreshold: testCase.mask_threshold,
          });
          const report = compareWithPython(detections, testCase.detections, fixture.image);
          expect(report.matched).toBe(testCase.detections.length);
          expect(report.minBoxIou).toBeGreaterThanOrEqual(BOX_IOU_MIN);
          expect(report.maxScoreDelta).toBeLessThanOrEqual(SCORE_DELTA_MAX);
          expect(report.minMaskIou).toBeGreaterThanOrEqual(MASK_IOU_MIN);
        });
      }
    }
  }

  it("the segment fixture really exercises masks (non-empty, threshold-sensitive)", () => {
    const fixture = loadFixture("segment");
    const byName = Object.fromEntries(fixture.cases.map((c) => [c.name, c]));
    const area = (name: string) => byName[name]!.detections.reduce((sum, d) => sum + (d.mask_area ?? 0), 0);
    expect(area("default")).toBeGreaterThan(1000);
    expect(area("low_mask_threshold")).toBeGreaterThan(area("default"));
  });

  it("returns the contract shape (openapi Detection) and zero-based class names", () => {
    const fixture = loadFixture("segment");
    const [first] = decodeOutputs(fixture.meta, postprocessedTensors(fixture), fixture.image, { conf: 0.1 });
    expect(Object.keys(first!).sort()).toEqual(["box", "class_id", "confidence", "has_mask", "mask", "name"]);
    expect(Object.keys(first!.box).sort()).toEqual(["x1", "x2", "y1", "y2"]);
    expect(first!.has_mask).toBe(true);
    expect(["cat", "dog"]).toContain(first!.name);
  });

  it("detect results have no mask and has_mask=false", () => {
    const fixture = loadFixture("detect");
    const detections = decodeOutputs(fixture.meta, postprocessedTensors(fixture), fixture.image, { conf: 0.1 });
    expect(detections.length).toBeGreaterThan(0);
    for (const d of detections) {
      expect(d.has_mask).toBe(false);
      expect("mask" in d).toBe(false);
    }
  });

  it("default conf is 0.5 (the Python default), not 0.25", () => {
    const fixture = loadFixture("detect");
    const tensors = postprocessedTensors(fixture);
    const withDefault = decodeOutputs(fixture.meta, tensors, fixture.image);
    const explicit = decodeOutputs(fixture.meta, tensors, fixture.image, { conf: 0.5 });
    expect(withDefault).toEqual(explicit);
    // The fixture model has class scores ~0.82 and ~0.41: 0.5 keeps one class, 0.25 would keep both.
    expect(decodeOutputs(fixture.meta, tensors, fixture.image, { conf: 0.25 }).length).toBe(2 * withDefault.length);
  });
});

/** A hand-made detect meta and outputs for the unit tests below. */
function detectMeta(overrides: Partial<OnnxMeta> = {}): OnnxMeta {
  return parseMeta({
    schema_version: 1,
    task: "detect",
    model: "unit",
    imgsz: 64,
    input: { name: "images", shape: [1, 3, 64, 64], dtype: "float32", range: "0-1", color: "RGB", resize: "squash" },
    outputs: [
      { name: "labels", dtype: "int64" },
      { name: "boxes", dtype: "float32", format: "xyxy_input_px" },
      { name: "scores", dtype: "float32" },
    ],
    names: { "0": "a", "1": "b", "2": "c" },
    topk: 4,
    default_iou: 0.85,
    nitid_version: "0.0.0",
    ...overrides,
  });
}

function tensors(rows: { label: number; score: number; box: [number, number, number, number] }[]): NamedTensors {
  return {
    labels: { data: BigInt64Array.from(rows.map((r) => BigInt(r.label))), dims: [1, rows.length] },
    boxes: { data: Float32Array.from(rows.flatMap((r) => r.box)), dims: [1, rows.length, 4] },
    scores: { data: Float32Array.from(rows.map((r) => r.score)), dims: [1, rows.length] },
  };
}

describe("decode: filters (predictor.py:419-431)", () => {
  const rows = [
    { label: 0, score: 0.9, box: [0, 0, 32, 32] as [number, number, number, number] },
    { label: 1, score: 0.5, box: [8, 8, 40, 40] as [number, number, number, number] },
    { label: 2, score: 0.3, box: [16, 16, 48, 48] as [number, number, number, number] },
    { label: 0, score: 0.7, box: [4, 4, 36, 36] as [number, number, number, number] },
  ];

  it("keeps score > conf strictly: a score equal to conf is dropped", () => {
    const out = decodeOutputs(detectMeta(), tensors(rows), { width: 64, height: 64 }, { conf: 0.5 });
    expect(out.map((d) => d.confidence)).toEqual([Math.fround(0.9), Math.fround(0.7)]);
  });

  it("keeps everything with conf=0 except non-positive scores", () => {
    const out = decodeOutputs(detectMeta(), tensors([...rows, { label: 1, score: 0, box: [0, 0, 1, 1] }]), { width: 64, height: 64 }, { conf: 0 });
    expect(out.length).toBe(4);
  });

  it("filters by class ids and preserves the original (top-K) order", () => {
    const out = decodeOutputs(detectMeta(), tensors(rows), { width: 64, height: 64 }, { conf: 0, classes: [0, 2] });
    expect(out.map((d) => d.class_id)).toEqual([0, 2, 0]);
    expect(out.map((d) => d.name)).toEqual(["a", "c", "a"]);
  });

  it("an empty classes list keeps nothing; null/undefined keeps all", () => {
    const base = { width: 64, height: 64 };
    expect(decodeOutputs(detectMeta(), tensors(rows), base, { conf: 0, classes: [] })).toEqual([]);
    expect(decodeOutputs(detectMeta(), tensors(rows), base, { conf: 0, classes: null }).length).toBe(4);
  });

  it("drops NaN scores like Python's `scores > conf`", () => {
    const out = decodeOutputs(detectMeta(), tensors([{ label: 0, score: NaN, box: [0, 0, 4, 4] }]), { width: 64, height: 64 }, { conf: 0 });
    expect(out).toEqual([]);
  });

  it("falls back to the class id when names has no entry", () => {
    const out = decodeOutputs(detectMeta(), tensors([{ label: 9, score: 0.9, box: [0, 0, 4, 4] }]), { width: 64, height: 64 });
    expect(out[0]!.name).toBe("9");
  });

  it("validates conf and maskThreshold", () => {
    for (const conf of [-0.1, 1.5, NaN]) {
      expect(() => decodeOutputs(detectMeta(), tensors(rows), { width: 64, height: 64 }, { conf })).toThrowError(/conf must be/);
    }
    expect(() => decodeOutputs(detectMeta(), tensors(rows), { width: 64, height: 64 }, { maskThreshold: 2 })).toThrowError(/maskThreshold/);
  });
});

describe("decode: no NMS in the default path (predictor.py: suppression exists only for augment=True)", () => {
  it("keeps overlapping same-class boxes, even identical ones", () => {
    const same: [number, number, number, number] = [10, 10, 50, 50];
    const out = decodeOutputs(
      detectMeta(),
      tensors([
        { label: 0, score: 0.9, box: same },
        { label: 0, score: 0.8, box: same },
        { label: 0, score: 0.7, box: [11, 11, 50, 50] },
      ]),
      { width: 64, height: 64 },
      { conf: 0.1 },
    );
    expect(out.length).toBe(3);
  });
});

describe("decode: box scaling and clipping (postprocessor.py:47-48, ops.py:59-64)", () => {
  it("rescales from the S x S input to the original size, independently per axis (squash)", () => {
    const out = decodeOutputs(detectMeta(), tensors([{ label: 0, score: 0.9, box: [16, 8, 48, 32] }]), { width: 128, height: 32 });
    expect(out[0]!.box).toEqual({ x1: 32, y1: 4, x2: 96, y2: 16 });
  });

  it("clips boxes to the image after filtering", () => {
    const out = decodeOutputs(detectMeta(), tensors([{ label: 0, score: 0.9, box: [-5, -3, 70, 80] }]), { width: 64, height: 64 });
    expect(out[0]!.box).toEqual({ x1: 0, y1: 0, x2: 64, y2: 64 });
  });

  it("a box entirely outside is clipped to a degenerate box, not dropped (Python keeps it)", () => {
    const out = decodeOutputs(detectMeta(), tensors([{ label: 0, score: 0.9, box: [100, 100, 120, 120] }]), { width: 64, height: 64 });
    expect(out.length).toBe(1);
    expect(out[0]!.box).toEqual({ x1: 64, y1: 64, x2: 64, y2: 64 });
  });

  it("accepts int64 labels as BigInt64Array, Int32Array or plain arrays", () => {
    const base = tensors([{ label: 1, score: 0.9, box: [0, 0, 8, 8] }]);
    for (const data of [BigInt64Array.from([1n]), Int32Array.from([1]), [1]]) {
      const out = decodeOutputs(detectMeta(), { ...base, labels: { data, dims: [1, 1] } }, { width: 64, height: 64 });
      expect(out[0]!.class_id).toBe(1);
    }
  });
});

describe("decode: output validation", () => {
  const original = { width: 64, height: 64 };
  it("reports a missing output as unsupported_output", () => {
    const { labels, boxes } = tensors([{ label: 0, score: 0.9, box: [0, 0, 8, 8] }]);
    expect(() => decodeOutputs(detectMeta(), { labels: labels!, boxes: boxes! }, original)).toThrowError(/"scores" is missing/);
  });

  it("rejects a batch other than 1 and inconsistent K", () => {
    const base = tensors([{ label: 0, score: 0.9, box: [0, 0, 8, 8] }]);
    expect(() => decodeOutputs(detectMeta(), { ...base, scores: { data: new Float32Array(2), dims: [2, 1] } }, original)).toThrowError(/batch 1/);
    expect(() => decodeOutputs(detectMeta(), { ...base, boxes: { data: new Float32Array(8), dims: [1, 2, 4] } }, original)).toThrowError(/dims/);
  });
});

/** A 2x2 mask tensor for one detection of a 4x4 "model input" (mask stride 4 -> 1x1 would be degenerate). */
function segmentMeta(): OnnxMeta {
  return detectMeta({
    task: "segment",
    imgsz: 32,
    input: { name: "images", shape: [1, 3, 32, 32], dtype: "float32", range: "0-1", color: "RGB", resize: "squash" },
    outputs: [
      { name: "labels", dtype: "int64" },
      { name: "boxes", dtype: "float32", format: "xyxy_input_px" },
      { name: "scores", dtype: "float32" },
      { name: "masks", dtype: "float32", stride: 4 },
    ],
  });
}

function segTensors(box: [number, number, number, number], maskValues: number[], mh = 2, mw = 2): NamedTensors {
  return {
    ...tensors([{ label: 0, score: 0.9, box }]),
    masks: { data: Float32Array.from(maskValues), dims: [1, 1, mh, mw] },
  };
}

describe("decode: masks (predictor.py:462-470, ops.py:11-36)", () => {
  it("thresholds with >= (a probability equal to the threshold is foreground)", () => {
    // 1x1 source mask resized to 4x4: every pixel interpolates to the same value.
    const at = (value: number, threshold: number) =>
      decodeOutputs(segmentMeta(), segTensors([0, 0, 32, 32], [value], 1, 1), { width: 4, height: 4 }, { maskThreshold: threshold })[0]!.mask!.data;
    expect(Array.from(at(0.5, 0.5))).toEqual(new Array(16).fill(1));
    expect(Array.from(at(0.49, 0.5))).toEqual(new Array(16).fill(0));
    expect(Array.from(at(0.49, 0.4))).toEqual(new Array(16).fill(1));
  });

  it("interpolates bilinearly with align_corners=False (torch semantics)", () => {
    // 1x2 source [0, 1] -> 1x4 output. Source coords: (i + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, 1.25
    // clamped to [0, 1): values 0, 0.25, 0.75, 1 -> threshold 0.5 gives 0 0 1 1.
    const out = decodeOutputs(segmentMeta(), segTensors([0, 0, 32, 32], [0, 1], 1, 2), { width: 4, height: 1 })[0]!;
    expect(Array.from(out.mask!.data)).toEqual([0, 0, 1, 1]);
  });

  it("zeroes everything outside the box and stores only the box region", () => {
    // Mask is all ones. Box covers x in [1.2, 3.1] -> pixels 2..3; y in [0.5, 2.5] -> pixels 1..2 (ceil rule).
    const original = { width: 4, height: 4 };
    const box: [number, number, number, number] = [(1.2 * 32) / 4, (0.5 * 32) / 4, (3.1 * 32) / 4, (2.5 * 32) / 4];
    const out = decodeOutputs(segmentMeta(), segTensors(box, [1, 1, 1, 1]), original)[0]!;
    expect(out.mask).toMatchObject({ x: 2, y: 1, width: 2, height: 2 });
    const full = maskToFull(out.mask!, 4, 4);
    expect(Array.from(full)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]);
  });

  it("an empty box gives an empty mask", () => {
    const out = decodeOutputs(segmentMeta(), segTensors([10, 10, 10, 10], [1, 1, 1, 1]), { width: 32, height: 32 })[0]!;
    expect(out.mask).toMatchObject({ width: 0, height: 0 });
    expect(out.mask!.data.length).toBe(0);
  });

  it("works when the original image is smaller than the mask (downscale, no antialias)", () => {
    const out = decodeOutputs(segmentMeta(), segTensors([0, 0, 32, 32], [1, 1, 1, 1]), { width: 1, height: 1 })[0]!;
    expect(Array.from(out.mask!.data)).toEqual([1]);
  });

  it("enforces the mask memory budget with a clear error", () => {
    const run = () =>
      decodeOutputs(segmentMeta(), segTensors([0, 0, 32, 32], [1, 1, 1, 1]), { width: 1000, height: 1000 }, { maxMaskBytes: 1000 });
    expect(run).toThrowError(/Raise the confidence threshold/);
    try {
      run();
    } catch (error) {
      expect((error as { code: string }).code).toBe("mask_budget_exceeded");
    }
  });

  it("requires the masks output for segment", () => {
    const { masks: _unused, ...withoutMasks } = segTensors([0, 0, 8, 8], [1, 1, 1, 1]);
    void _unused;
    expect(() => decodeOutputs(segmentMeta(), withoutMasks, { width: 8, height: 8 })).toThrowError(/"masks" is missing/);
  });
});

describe("decode: raw layout replays the post-processor (postprocessor.py:55-62)", () => {
  const rawMeta = (): OnnxMeta =>
    detectMeta({
      outputs: [
        { name: "pred_logits", dtype: "float32" },
        { name: "pred_boxes", dtype: "float32", format: "cxcywh_norm" },
      ],
      topk: 3,
    });

  it("takes the top-K of the flattened [queries x classes] sigmoid scores, label = idx % C", () => {
    // 2 queries, 3 classes. logits chosen so the scores order is q1c2 > q0c1 > q1c0.
    const logits = [-5, 2, -5, 1, -5, 3];
    const out = decodeOutputs(
      rawMeta(),
      {
        pred_logits: { data: Float32Array.from(logits), dims: [1, 2, 3] },
        pred_boxes: { data: Float32Array.from([0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2]), dims: [1, 2, 4] },
      },
      { width: 100, height: 200 },
      { conf: 0.5 },
    );
    expect(out.map((d) => d.class_id)).toEqual([2, 1, 0]);
    expect(out[0]!.confidence).toBeCloseTo(1 / (1 + Math.exp(-3)), 12);
    // query 1 box: cxcywh (0.5, 0.5, 0.2, 0.2) -> xyxy (0.4, 0.4, 0.6, 0.6) * (W, H)
    expect(out[0]!.box.x1).toBeCloseTo(40, 4);
    expect(out[0]!.box.y2).toBeCloseTo(120, 4);
    // query 0 box: (0.25, 0.25, 0.5, 0.5) -> (0, 0, 0.5, 0.5)
    expect(out[1]!.box).toEqual({ x1: 0, y1: 0, x2: 50, y2: 100 });
  });

  it("topk is capped by queries * classes", () => {
    const out = decodeOutputs(
      detectMeta({ outputs: rawMeta().outputs, topk: 300 }),
      {
        pred_logits: { data: Float32Array.from([5, 5]), dims: [1, 1, 2] },
        pred_boxes: { data: Float32Array.from([0.5, 0.5, 0.2, 0.2]), dims: [1, 1, 4] },
      },
      { width: 10, height: 10 },
    );
    expect(out.length).toBe(2);
    // Equal scores: stable order by flat index (class 0 first).
    expect(out.map((d) => d.class_id)).toEqual([0, 1]);
  });
});
