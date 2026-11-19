import { describe, expect, it } from "vitest";
import { ringToFlat, segmentsToRing } from "./segments";

describe("segmentsToRing", () => {
  it("pairs x and y into vertices", () => {
    expect(segmentsToRing({ x: [0, 10, 10], y: [0, 0, 5] })).toEqual([[0, 0], [10, 0], [10, 5]]);
  });

  it("removes a repeated closing vertex", () => {
    expect(segmentsToRing({ x: [0, 10, 10, 0], y: [0, 0, 5, 0] })).toEqual([[0, 0], [10, 0], [10, 5]]);
  });

  it("ignores the surplus of the longer array", () => {
    expect(segmentsToRing({ x: [0, 10, 10, 99], y: [0, 0, 5] })).toEqual([[0, 0], [10, 0], [10, 5]]);
  });

  it("drops non-finite vertices", () => {
    const bad = { x: [0, Number.NaN, 10, 10], y: [0, 3, 0, 5] };
    expect(segmentsToRing(bad)).toEqual([[0, 0], [10, 0], [10, 5]]);
  });

  it("returns null when it is not an area", () => {
    expect(segmentsToRing(undefined)).toBeNull();
    expect(segmentsToRing(null)).toBeNull();
    expect(segmentsToRing({ x: [], y: [] })).toBeNull();
    expect(segmentsToRing({ x: [1, 2], y: [1, 2] })).toBeNull();
    expect(segmentsToRing({ x: [1, 2, 1], y: [1, 2, 1] })).toBeNull(); // closes to 2 distinct vertices
    expect(segmentsToRing({ x: "no", y: [1] } as never)).toBeNull();
  });

  it("flattens to the DetectionMask shape", () => {
    expect(ringToFlat([[1, 2], [3, 4], [5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
