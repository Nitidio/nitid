import { describe, expect, it } from "vitest";
import { boxToRect, classColor, fitScale, labelPosition, labelText, scalePolygon } from "./geometry";

describe("geometry", () => {
  it("fitScale never upscales unless asked", () => {
    expect(fitScale(3200, 1600)).toBe(0.5);
    expect(fitScale(800, 1600)).toBe(1);
    expect(fitScale(800, 1600, true)).toBe(2);
    expect(fitScale(0, 100)).toBe(1);
  });

  it("boxToRect scales original-pixel boxes", () => {
    expect(boxToRect({ x1: 10, y1: 20, x2: 110, y2: 70 }, 200, 100, 0.5)).toEqual({ x: 5, y: 10, w: 50, h: 25 });
  });

  it("boxToRect clamps to the image and normalises flipped corners", () => {
    expect(boxToRect({ x1: 250, y1: -5, x2: -10, y2: 40 }, 200, 100, 1)).toEqual({ x: 0, y: 0, w: 200, h: 40 });
  });

  it("boxToRect rejects empty, outside and non-finite boxes", () => {
    expect(boxToRect({ x1: 5, y1: 5, x2: 5, y2: 9 }, 100, 100, 1)).toBeNull();
    expect(boxToRect({ x1: 150, y1: 0, x2: 190, y2: 10 }, 100, 100, 1)).toBeNull();
    expect(boxToRect({ x1: Number.NaN, y1: 0, x2: 10, y2: 10 }, 100, 100, 1)).toBeNull();
  });

  it("labelText rounds the score to a percentage", () => {
    expect(labelText({ name: "car", confidence: 0.876 })).toBe("car 88%");
  });

  it("labelPosition goes above the box, or inside when there is no room, and stays inside the canvas", () => {
    expect(labelPosition({ x: 10, y: 50, w: 20, h: 20 }, 40, 20, 300)).toEqual({ x: 10, y: 30 });
    expect(labelPosition({ x: 10, y: 5, w: 20, h: 20 }, 40, 20, 300)).toEqual({ x: 10, y: 5 });
    expect(labelPosition({ x: 290, y: 50, w: 10, h: 10 }, 40, 20, 300)).toEqual({ x: 260, y: 30 });
  });

  it("classColor is stable and differs between neighbouring ids", () => {
    expect(classColor(3)).toBe(classColor(3));
    expect(classColor(0)).not.toBe(classColor(1));
    expect(classColor(5)).toMatch(/^hsl\(\d+ 70% 38%\)$/);
  });

  it("scalePolygon scales every coordinate", () => {
    expect(scalePolygon([10, 20, 30, 40], 0.5)).toEqual([5, 10, 15, 20]);
  });
});
