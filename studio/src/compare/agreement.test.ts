import { describe, expect, it } from "vitest";
import { iou, matchDetections, summarizeAgreement } from "./agreement";

const det = (x1: number, y1: number, x2: number, y2: number, classId = 0) => ({ box: { x1, y1, x2, y2 }, classId });

describe("iou", () => {
  it("is 1 for identical boxes and 0 for disjoint or touching ones", () => {
    expect(iou(det(0, 0, 10, 10).box, det(0, 0, 10, 10).box)).toBe(1);
    expect(iou(det(0, 0, 10, 10).box, det(20, 20, 30, 30).box)).toBe(0);
    expect(iou(det(0, 0, 10, 10).box, det(10, 0, 20, 10).box)).toBe(0);
  });
  it("computes partial overlap (50 of 150 = 1/3)", () => {
    expect(iou(det(0, 0, 10, 10).box, det(5, 0, 15, 10).box)).toBeCloseTo(1 / 3, 10);
  });
  it("is 0 for degenerate boxes (no division by zero)", () => {
    expect(iou(det(5, 5, 5, 5).box, det(5, 5, 5, 5).box)).toBe(0);
    expect(iou(det(10, 10, 0, 0).box, det(0, 0, 10, 10).box)).toBe(0);
  });
  it("is symmetric", () => {
    const p = det(0, 0, 10, 8).box;
    const q = det(3, 2, 14, 9).box;
    expect(iou(p, q)).toBeCloseTo(iou(q, p), 12);
  });
});

describe("matchDetections", () => {
  it("matches identical lists one-to-one", () => {
    const a = [det(0, 0, 10, 10), det(50, 50, 80, 90, 2)];
    const r = matchDetections(a, a);
    expect(r.matches.map((m) => [m.a, m.b])).toEqual([[0, 0], [1, 1]]);
    expect(r.onlyA).toEqual([]);
    expect(r.onlyB).toEqual([]);
  });

  it("uses the threshold inclusively", () => {
    // 10x10 against 10x5 inside it: intersection 50, union 100, IoU exactly 0.5.
    const r = matchDetections([det(0, 0, 10, 10)], [det(0, 0, 10, 5)]);
    expect(r.matches).toHaveLength(1);
    expect(matchDetections([det(0, 0, 10, 10)], [det(0, 0, 10, 4)]).matches).toHaveLength(0);
  });

  it("does not match different classes unless sameClass is false", () => {
    const a = [det(0, 0, 10, 10, 1)];
    const b = [det(0, 0, 10, 10, 2)];
    expect(matchDetections(a, b).matches).toHaveLength(0);
    expect(matchDetections(a, b, { sameClass: false }).matches).toHaveLength(1);
  });

  it("is one-to-one: two boxes in A competing for one in B", () => {
    const a = [det(0, 0, 10, 10), det(1, 0, 11, 10)];
    const b = [det(0, 0, 10, 10)];
    const r = matchDetections(a, b);
    expect(r.matches).toEqual([{ a: 0, b: 0, iou: 1 }]);
    expect(r.onlyA).toEqual([1]);
    expect(r.onlyB).toEqual([]);
  });

  it("prefers the highest IoU, not the first index", () => {
    const a = [det(0, 0, 10, 10)];
    const b = [det(2, 0, 12, 10), det(0, 0, 10, 10)]; // b[1] is perfect
    const r = matchDetections(a, b);
    expect(r.matches[0]).toMatchObject({ a: 0, b: 1 });
    expect(r.onlyB).toEqual([0]);
  });

  it("breaks IoU ties by lower index", () => {
    const a = [det(0, 0, 10, 10)];
    const b = [det(0, 0, 10, 10), det(0, 0, 10, 10)];
    expect(matchDetections(a, b).matches[0]).toMatchObject({ a: 0, b: 0 });
  });

  it("handles empty lists", () => {
    expect(matchDetections([], [])).toEqual({ matches: [], onlyA: [], onlyB: [] });
    expect(matchDetections([det(0, 0, 1, 1)], [])).toMatchObject({ onlyA: [0], onlyB: [] });
  });

  it("honours a custom threshold", () => {
    const a = [det(0, 0, 10, 10)];
    const b = [det(5, 0, 15, 10)]; // IoU 1/3
    expect(matchDetections(a, b).matches).toHaveLength(0);
    expect(matchDetections(a, b, { iouThreshold: 0.3 }).matches).toHaveLength(1);
  });
});

describe("summarizeAgreement", () => {
  it("reports counts, agreement and mean IoU", () => {
    const a = [det(0, 0, 10, 10), det(100, 100, 120, 120), det(200, 200, 210, 210)];
    const b = [det(0, 0, 10, 10), det(300, 300, 310, 310)];
    const s = summarizeAgreement(a, b);
    expect(s).toMatchObject({ countA: 3, countB: 2, matched: 1, onlyA: [1, 2], onlyB: [1] });
    expect(s.agreement).toBeCloseTo(1 / 4, 10); // 1 / (3 + 2 - 1)
    expect(s.meanIou).toBe(1);
  });
  it("is 1 when both are empty and 0 with no overlap", () => {
    expect(summarizeAgreement([], []).agreement).toBe(1);
    expect(summarizeAgreement([], []).meanIou).toBeNull();
    expect(summarizeAgreement([det(0, 0, 1, 1)], [det(5, 5, 6, 6)]).agreement).toBe(0);
  });
});
