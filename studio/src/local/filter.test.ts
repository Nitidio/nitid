import { describe, expect, it } from "vitest";
import { clampConfidence, defaultFilter, filterDetections, setAllClasses, summarizeClasses, toggleClass } from "./filter";
import type { ResultDetection, ResultItem } from "./results-model";

const det = (classId: number, name: string, confidence: number): ResultDetection => ({
  box: { x1: 0, y1: 0, x2: 10, y2: 10 },
  confidence,
  classId,
  name,
  hasMask: false,
});
const item = (detections: ResultDetection[]): ResultItem => ({
  id: 0,
  name: "a.jpg",
  width: 100,
  height: 100,
  detections,
  speed: { preprocess: 1, inference: 2, postprocess: 3 },
});

describe("filterDetections", () => {
  const ds = [det(0, "person", 0.9), det(2, "car", 0.5), det(0, "person", 0.2)];

  it("keeps detections at or above the threshold (inclusive)", () => {
    expect(filterDetections(ds, defaultFilter(0.5)).map((d) => d.confidence)).toEqual([0.9, 0.5]);
  });

  it("drops hidden classes", () => {
    const f = toggleClass(defaultFilter(0), 0);
    expect(filterDetections(ds, f).map((d) => d.name)).toEqual(["car"]);
  });

  it("returns everything with threshold 0 and no hidden classes", () => {
    expect(filterDetections(ds, defaultFilter(0))).toHaveLength(3);
  });
});

describe("class helpers", () => {
  it("toggleClass is immutable and reversible", () => {
    const f0 = defaultFilter();
    const f1 = toggleClass(f0, 3);
    expect(f0.hiddenClasses.size).toBe(0);
    expect(f1.hiddenClasses.has(3)).toBe(true);
    expect(toggleClass(f1, 3).hiddenClasses.has(3)).toBe(false);
  });

  it("setAllClasses hides or shows every class", () => {
    expect([...setAllClasses(defaultFilter(), [1, 2], false).hiddenClasses]).toEqual([1, 2]);
    expect(setAllClasses(toggleClass(defaultFilter(), 1), [1, 2], true).hiddenClasses.size).toBe(0);
  });

  it("summarizeClasses counts above threshold, sorts, and flags hidden", () => {
    const items = [item([det(0, "person", 0.9), det(2, "car", 0.5), det(0, "person", 0.2)]), item([det(2, "car", 0.8)])];
    const rows = summarizeClasses(items, toggleClass(defaultFilter(0.4), 2));
    expect(rows).toEqual([
      { classId: 2, name: "car", count: 2, hidden: true },
      { classId: 0, name: "person", count: 1, hidden: false },
    ]);
  });

  it("clampConfidence clamps and survives NaN", () => {
    expect(clampConfidence(1.5)).toBe(1);
    expect(clampConfidence(0.05, 0.1)).toBe(0.1);
    expect(clampConfidence(Number.NaN, 0.1)).toBe(0.1);
  });
});
