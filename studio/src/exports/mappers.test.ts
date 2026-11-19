import { describe, expect, it } from "vitest";
import { fromEngineResult, fromServerRun, type ServerRun } from "./mappers";
import type { RunResult } from "../engine/session";

describe("mappers", () => {
  it("maps an engine RunResult (with mask)", () => {
    const run: RunResult = {
      detections: [
        {
          box: { x1: 1, y1: 2, x2: 3, y2: 4 },
          confidence: 0.8,
          class_id: 5,
          name: "bus",
          has_mask: true,
          mask: { x: 1, y: 2, width: 2, height: 2, data: new Uint8Array([1, 1, 1, 1]) },
        },
      ],
      imageSize: { width: 640, height: 480 },
      timings: { preprocessMs: 1, inferenceMs: 2, postprocessMs: 3 },
    };
    const r = fromEngineResult(run, { imageName: "x.png", model: "nitid1s", task: "segment", backend: "webgpu", createdAt: "2026-01-01T00:00:00Z" });
    expect(r).toMatchObject({
      source: "browser",
      image: { name: "x.png", width: 640, height: 480 },
      backend: "webgpu",
      device: null,
      timings: { preprocessMs: 1, inferenceMs: 2, postprocessMs: 3 },
    });
    expect(r.detections[0]).toMatchObject({ classId: 5, name: "bus", hasMask: true });
    expect(r.detections[0]?.mask?.data).toHaveLength(4);
  });

  const serverRun: ServerRun = {
    id: "r_01",
    model: "nitid1s",
    task: "detect",
    backend: "openvino",
    device: "NPU",
    items: [
      {
        id: 0,
        name: "street.jpg",
        width: 1280,
        height: 720,
        detections: [{ box: { x1: 12.5, y1: 40, x2: 210, y2: 388.25 }, confidence: 0.91, class_id: 0, name: "person", has_mask: true }],
        speed: { preprocess: 3.1, inference: 28.4, postprocess: 1.2 },
      },
    ],
  };

  it("maps a server run item (flag only, no mask pixels)", () => {
    const r = fromServerRun(serverRun, 0, "2026-01-01T00:00:00Z");
    expect(r).toMatchObject({ source: "server", backend: "openvino", device: "NPU", model: "nitid1s" });
    expect(r.timings).toEqual({ preprocessMs: 3.1, inferenceMs: 28.4, postprocessMs: 1.2 });
    expect(r.detections[0]).toEqual({
      box: { x1: 12.5, y1: 40, x2: 210, y2: 388.25 },
      confidence: 0.91,
      classId: 0,
      name: "person",
      hasMask: true,
    });
  });

  it("maps segments to an export polygon (no pixel mask)", () => {
    const run = {
      ...serverRun,
      task: "segment" as const,
      items: [
        {
          ...serverRun.items[0]!,
          detections: [
            { box: { x1: 0, y1: 0, x2: 10, y2: 10 }, confidence: 0.9, class_id: 1, name: "car", has_mask: true, segments: { x: [0, 10, 10, 0], y: [0, 0, 10, 0] } },
            { box: { x1: 0, y1: 0, x2: 10, y2: 10 }, confidence: 0.8, class_id: 1, name: "car", has_mask: true },
          ],
        },
      ],
    };
    const [a, b] = fromServerRun(run, 0).detections;
    expect(a?.polygon).toEqual([[0, 0], [10, 0], [10, 10]]);
    expect(a?.mask).toBeUndefined();
    expect(a?.hasMask).toBe(true);
    expect("polygon" in (b ?? {})).toBe(false);
  });

  it("throws for a missing item", () => {
    expect(() => fromServerRun(serverRun, 3)).toThrow(RangeError);
  });
});
