import { describe, expect, it } from "vitest";
import { classColor, cropFileName, cropRect, labelText, maskOverlay, renderAnnotated, renderCrops } from "./raster";
import { det, fakeRaster, rectMask, results, type FakeCanvas } from "./test-fixtures";

describe("raster helpers", () => {
  it("gives stable, distinct colours per class", () => {
    expect(classColor(0).css).toBe(classColor(0).css);
    expect(classColor(0).css).not.toBe(classColor(1).css);
    for (const c of classColor(7).rgb) expect(c).toBeGreaterThanOrEqual(0);
  });

  it("builds a mask overlay only where the mask is set", () => {
    const o = maskOverlay({ x: 0, y: 0, width: 2, height: 1, data: new Uint8Array([1, 0]) }, [10, 20, 30], 100);
    expect(Array.from(o)).toEqual([10, 20, 30, 100, 0, 0, 0, 0]);
  });

  it("formats labels and crop rectangles", () => {
    expect(labelText(det({ name: "perro", confidence: 0.876 }))).toBe("perro 88%");
    expect(cropRect(det({ box: { x1: -5, y1: 10.4, x2: 50.2, y2: 10.6 } }), 100, 100)).toEqual({ sx: 0, sy: 10, sw: 51, sh: 1 });
    expect(cropRect(det({ box: { x1: 99.9, y1: 99.9, x2: 120, y2: 120 } }), 100, 100)).toEqual({ sx: 99, sy: 99, sw: 1, sh: 1 });
  });

  it("names crops safely and with Unicode kept", () => {
    expect(cropFileName(results(), det({ name: "señal/tráfico 停" }), 7)).toBe("street_view_0007_señal_tráfico_停.png");
    expect(cropFileName(results(), det({ name: "///", classId: 4 }), 0)).toBe("street_view_0000_class_4.png");
  });
});

describe("renderAnnotated", () => {
  it("draws the image, masks, boxes and labels at original size", () => {
    const { backend, canvases } = fakeRaster();
    const r = results({ detections: [det({ hasMask: true, mask: rectMask(10, 20, 5, 5) }), det({ classId: 2, name: "car" })] });
    const canvas = renderAnnotated(r, "IMG", backend) as unknown as FakeCanvas;
    expect(canvas.width).toBe(200);
    expect(canvas.height).toBe(100);
    const ops = canvas.calls.map((c) => c[0]);
    expect(ops[0]).toBe("drawImage");
    expect(canvas.calls[0]).toEqual(["drawImage", "IMG", 0, 0, 200, 100]);
    expect(ops.filter((o) => o === "strokeRect")).toHaveLength(2);
    expect(ops.filter((o) => o === "fillText").length).toBe(2);
    // The mask went through its own small canvas, drawn at its origin before any box.
    expect(canvases).toHaveLength(2);
    const maskDraw = canvas.calls.findIndex((c) => c[0] === "drawImage" && c[1] !== "IMG");
    expect(canvas.calls[maskDraw]?.slice(2)).toEqual([10, 20]);
    expect(maskDraw).toBeLessThan(ops.indexOf("strokeRect"));
    expect(canvas.calls.find((c) => c[0] === "fillText")?.[1]).toBe("person 90%");
  });

  it("draws a server outline as a translucent fill plus a stroke, before the boxes", () => {
    const { backend, canvases } = fakeRaster();
    const polygon: Array<[number, number]> = [[10, 20], [110, 20], [60, 70]];
    const r = results({ source: "server", detections: [det({ hasMask: true, polygon })] });
    const canvas = renderAnnotated(r, "IMG", backend) as unknown as FakeCanvas;
    const ops = canvas.calls.map((c) => c[0]);
    expect(canvases).toHaveLength(1); // no pixel-mask canvas needed
    expect(ops.slice(1, 7)).toEqual(["beginPath", "moveTo", "lineTo", "lineTo", "closePath", "fill"]);
    expect(canvas.calls.find((c) => c[0] === "moveTo")).toEqual(["moveTo", 10, 20]);
    expect(String(canvas.calls.find((c) => c[0] === "fill")?.[1])).toMatch(/^rgba\(\d+,\d+,\d+,0\.4\)$/);
    expect(ops.indexOf("stroke")).toBeLessThan(ops.indexOf("strokeRect"));
  });

  it("ignores outlines with fewer than 3 vertices", () => {
    const { backend } = fakeRaster();
    const r = results({ detections: [det({ hasMask: true, polygon: [[0, 0], [5, 5]] })] });
    const canvas = renderAnnotated(r, "IMG", backend) as unknown as FakeCanvas;
    expect(canvas.calls.some((c) => c[0] === "fill")).toBe(false);
  });

  it("works with zero detections and can hide labels", () => {
    const { backend } = fakeRaster();
    const empty = renderAnnotated(results({ detections: [] }), "IMG", backend) as unknown as FakeCanvas;
    expect(empty.calls).toHaveLength(1);
    const noLabels = renderAnnotated(results(), "IMG", backend, { showLabels: false }) as unknown as FakeCanvas;
    expect(noLabels.calls.some((c) => c[0] === "fillText")).toBe(false);
  });
});

describe("renderCrops", () => {
  it("makes one PNG per detection from the right source rectangle", async () => {
    const { backend, canvases } = fakeRaster();
    const crops = await renderCrops(results(), "IMG", backend);
    expect(crops.map((c) => c.name)).toEqual(["street_view_0000_person.png", "street_view_0001_car.png"]);
    expect(canvases[0]?.calls[0]).toEqual(["drawImage", "IMG", 10, 20, 100, 50, 0, 0, 100, 50]);
    expect(crops[0]?.blob.type).toBe("image/png");
    expect(await renderCrops(results({ detections: [] }), "IMG", backend)).toEqual([]);
  });
});
