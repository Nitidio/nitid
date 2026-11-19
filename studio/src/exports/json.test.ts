import { describe, expect, it } from "vitest";
import { formatJson, toJsonDocument } from "./json";
import { det, rectMask, results } from "./test-fixtures";

describe("toJsonDocument", () => {
  it("matches the golden document", () => {
    expect(toJsonDocument(results())).toEqual({
      schema: "nitid-studio-results",
      version: 1,
      created_at: "2026-11-19T10:00:00.000Z",
      source: "browser",
      image: { name: "street view.JPG", width: 200, height: 100 },
      model: { id: "nitid1s", task: "detect" },
      runtime: { backend: "wasm", device: null },
      timings_ms: { preprocess: 3.1, inference: 28.4, postprocess: 1.2 },
      detection_count: 2,
      detections: [
        { index: 0, class_id: 0, name: "person", confidence: 0.9, box: { x1: 10, y1: 20, x2: 110, y2: 70 }, has_mask: false },
        { index: 1, class_id: 2, name: "car", confidence: 0.5, box: { x1: 100, y1: 0, x2: 200, y2: 50 }, has_mask: false },
      ],
    });
  });

  it("writes the server outline as mask.polygon without a pixel count", () => {
    const doc = toJsonDocument(
      results({
        source: "server",
        task: "segment",
        detections: [det({ hasMask: true, polygon: [[10.123, 20], [110, 20.456], [60, 70]] })],
      }),
    );
    expect(doc.detections[0]?.mask).toEqual({ polygon: [[10.12, 20], [110, 20.46], [60, 70]] });
    expect(doc.detections[0]?.mask && "pixels" in doc.detections[0].mask).toBe(false);
  });

  it("handles empty results and missing timings", () => {
    const doc = toJsonDocument(results({ detections: [], timings: null }));
    expect(doc.detections).toEqual([]);
    expect(doc.detection_count).toBe(0);
    expect(doc.timings_ms).toBeNull();
  });

  it("adds the mask polygon in image pixels and the pixel count", () => {
    const r = results({ detections: [det({ hasMask: true, mask: rectMask(5, 6, 2, 2) })] });
    const m = toJsonDocument(r).detections[0]?.mask;
    expect(m?.pixels).toBe(4);
    expect(new Set(m?.polygon.map((p) => p.join(",")))).toEqual(new Set(["5,6", "7,6", "7,8", "5,8"]));
  });

  it("is valid JSON with a trailing newline and keeps Unicode", () => {
    const text = formatJson(results({ detections: [det({ name: "停止 señal" })] }));
    expect(text.endsWith("\n")).toBe(true);
    expect(JSON.parse(text).detections[0].name).toBe("停止 señal");
  });
});
