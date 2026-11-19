import { describe, expect, it } from "vitest";
import { maskToPolygon, maskToPolygonLocal, polygonArea } from "./contour";
import { maskFromRows, rectMask } from "./test-fixtures";

describe("maskToPolygon", () => {
  it("traces a rectangle as 4 vertices at pixel corners", () => {
    const poly = maskToPolygonLocal(rectMask(0, 0, 4, 3));
    expect(poly).toHaveLength(4);
    expect(polygonArea(poly ?? [])).toBe(12);
    expect(new Set((poly ?? []).map((p) => p.join(",")))).toEqual(new Set(["0,0", "4,0", "4,3", "0,3"]));
  });

  it("adds the mask origin", () => {
    const poly = maskToPolygon(rectMask(10, 20, 2, 2)) ?? [];
    expect(new Set(poly.map((p) => p.join(",")))).toEqual(new Set(["10,20", "12,20", "12,22", "10,22"]));
  });

  it("preserves the area of an L shape exactly (tolerance 0)", () => {
    const m = maskFromRows(0, 0, ["##..", "##..", "####", "####"]);
    const poly = maskToPolygonLocal(m, 0) ?? [];
    expect(poly).toHaveLength(6);
    expect(polygonArea(poly)).toBe(12);
  });

  it("ignores holes (outer contour only)", () => {
    const m = maskFromRows(0, 0, ["####", "#..#", "#..#", "####"]);
    expect(polygonArea(maskToPolygonLocal(m) ?? [])).toBe(16);
  });

  it("keeps only the largest component", () => {
    const m = maskFromRows(0, 0, ["###..#", "###..#", "###..."]);
    const poly = maskToPolygonLocal(m) ?? [];
    expect(polygonArea(poly)).toBe(9);
    expect(Math.max(...poly.map((p) => p[0]))).toBe(3);
  });

  it("joins diagonal neighbours (8-connectivity) into one outline", () => {
    const m = maskFromRows(0, 0, ["#.", ".#"]);
    const poly = maskToPolygonLocal(m, 0) ?? [];
    expect(polygonArea(poly)).toBe(2);
  });

  it("returns null for empty, 1-pixel-degenerate-free input", () => {
    expect(maskToPolygon(maskFromRows(0, 0, ["...", "..."]))).toBeNull();
    expect(maskToPolygon({ x: 0, y: 0, width: 0, height: 0, data: new Uint8Array(0) })).toBeNull();
    expect(maskToPolygon(rectMask(0, 0, 1, 1))).toHaveLength(4);
  });

  it("approximates a disc within the tolerance and with few vertices", () => {
    const r = 40;
    const size = 2 * r + 1;
    const data = new Uint8Array(size * size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if ((x - r) ** 2 + (y - r) ** 2 <= r * r) data[y * size + x] = 1;
    const pixels = data.reduce((a, b) => a + b, 0);
    const exact = maskToPolygonLocal({ x: 0, y: 0, width: size, height: size, data }, 0) ?? [];
    expect(polygonArea(exact)).toBe(pixels);
    const poly = maskToPolygonLocal({ x: 0, y: 0, width: size, height: size, data }, 1) ?? [];
    expect(poly.length).toBeLessThan(exact.length / 2);
    expect(Math.abs(polygonArea(poly) - pixels) / pixels).toBeLessThan(0.03);
  });

  it("handles a mask filling a whole thin strip and touching all borders", () => {
    expect(polygonArea(maskToPolygonLocal(rectMask(0, 0, 50, 1)) ?? [])).toBe(50);
    expect(polygonArea(maskToPolygonLocal(rectMask(0, 0, 1, 50)) ?? [])).toBe(50);
  });
});
