import { describe, expect, it, vi } from "vitest";
import { drawDetections } from "./draw";
import type { ResultDetection } from "./results-model";

function fakeCtx() {
  const calls: string[] = [];
  const rec = (name: string) =>
    vi.fn((...args: unknown[]) => {
      calls.push(`${name}(${args.join(",")})`);
    });
  const ctx = {
    save: rec("save"),
    restore: rec("restore"),
    beginPath: rec("beginPath"),
    moveTo: rec("moveTo"),
    lineTo: rec("lineTo"),
    closePath: rec("closePath"),
    fill: rec("fill"),
    stroke: rec("stroke"),
    fillRect: rec("fillRect"),
    strokeRect: rec("strokeRect"),
    fillText: rec("fillText"),
    measureText: (t: string) => ({ width: t.length * 6 }),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

const det = (over: Partial<ResultDetection> = {}): ResultDetection => ({
  box: { x1: 20, y1: 40, x2: 120, y2: 90 },
  confidence: 0.9,
  classId: 0,
  name: "person",
  hasMask: false,
  ...over,
});
const base = { scale: 0.5, imageWidth: 200, imageHeight: 100, showLabels: true, showMasks: true };

describe("drawDetections", () => {
  it("strokes the scaled box and draws a label", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det()], base);
    expect(calls).toContain("strokeRect(10,20,50,25)");
    expect(calls.some((c) => c.startsWith("fillText(person 90%"))).toBe(true);
  });

  it("omits labels when switched off", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det()], { ...base, showLabels: false });
    expect(calls.some((c) => c.startsWith("fillText"))).toBe(false);
  });

  it("skips degenerate boxes", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ box: { x1: 5, y1: 5, x2: 5, y2: 5 } })], base);
    expect(calls).toEqual([]);
  });

  it("fills mask polygons when geometry is provided", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ hasMask: true, mask: { polygons: [[0, 0, 100, 0, 100, 100]] } })], base);
    expect(calls).toContain("moveTo(0,0)");
    expect(calls).toContain("lineTo(50,0)");
    expect(calls).toContain("fill()");
  });

  it("strokes the outline after filling it, once per polygon", () => {
    const { ctx, calls } = fakeCtx();
    const mask = { polygons: [[0, 0, 100, 0, 100, 100], [120, 20, 180, 20, 180, 60]] };
    drawDetections(ctx, [det({ hasMask: true, mask })], { ...base, showLabels: false });
    expect(calls.filter((c) => c === "fill()")).toHaveLength(2);
    expect(calls.filter((c) => c === "stroke()")).toHaveLength(2);
    expect(calls.indexOf("fill()")).toBeLessThan(calls.indexOf("stroke()"));
    expect(calls).toContain("lineTo(90,10)"); // second polygon, scaled by 0.5
    // The outline sits under the box stroke.
    expect(calls.indexOf("strokeRect(10,20,50,25)")).toBeGreaterThan(calls.lastIndexOf("stroke()"));
  });

  it("skips outlines with fewer than 3 vertices and falls back to the tinted box", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ hasMask: true, mask: { polygons: [[0, 0, 10, 10]] } })], base);
    expect(calls).not.toContain("fill()");
  });

  it("draws nothing for outlines when masks are hidden", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ hasMask: true, mask: { polygons: [[0, 0, 100, 0, 100, 100]] } })], { ...base, showMasks: false });
    expect(calls).not.toContain("fill()");
    expect(calls).not.toContain("stroke()");
  });

  it("falls back to a tinted box when has_mask is true but no geometry exists", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ hasMask: true })], base);
    expect(calls.filter((c) => c === "fill()")).toHaveLength(0);
    expect(calls.filter((c) => c.startsWith("fillRect(10,20,50,25)"))).toHaveLength(1);
  });

  it("does not tint masks when masks are hidden", () => {
    const { ctx, calls } = fakeCtx();
    drawDetections(ctx, [det({ hasMask: true })], { ...base, showMasks: false, showLabels: false });
    expect(calls.some((c) => c.startsWith("fillRect"))).toBe(false);
  });
});
