/**
 * YOLO TXT labels.
 *
 * One line per detection, class id first, all coordinates normalised to the
 * ORIGINAL image size (0..1, 6 decimals, clamped):
 *   box:     `<class> <cx> <cy> <w> <h>`
 *   polygon: `<class> <x1> <y1> <x2> <y2> ... <xn> <yn>`   (n >= 3; Ultralytics segmentation format)
 * Polygon lines are written for detections that carry a mask (see
 * `contour.ts` for the extraction); everything else falls back to a box line.
 * An image without detections gives an empty file (""), the YOLO convention for a negative sample.
 */
import { maskToPolygon } from "./contour";
import type { ExportResults } from "./model";

export interface YoloOptions {
  /** "polygon" (default): masks become polygon lines; "box": always box lines. */
  segmentation?: "polygon" | "box";
  /** Polygon simplification tolerance in image pixels (default 1). */
  tolerance?: number;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const fmt = (v: number): string => clamp01(v).toFixed(6);

export function formatYoloTxt(results: ExportResults, options: YoloOptions = {}): string {
  const { width, height } = results.image;
  const usePolygons = (options.segmentation ?? "polygon") === "polygon";
  const lines: string[] = [];
  for (const d of results.detections) {
    const poly = usePolygons && d.mask ? maskToPolygon(d.mask, options.tolerance ?? 1) : null;
    if (poly) {
      const coords = poly.map(([x, y]) => `${fmt(x / width)} ${fmt(y / height)}`);
      lines.push(`${d.classId} ${coords.join(" ")}`);
      continue;
    }
    const x1 = Math.max(0, Math.min(width, d.box.x1));
    const x2 = Math.max(0, Math.min(width, d.box.x2));
    const y1 = Math.max(0, Math.min(height, d.box.y1));
    const y2 = Math.max(0, Math.min(height, d.box.y2));
    const cx = (x1 + x2) / 2 / width;
    const cy = (y1 + y2) / 2 / height;
    lines.push(`${d.classId} ${fmt(cx)} ${fmt(cy)} ${fmt((x2 - x1) / width)} ${fmt((y2 - y1) / height)}`);
  }
  return lines.length > 0 ? `${lines.join("\n")}\n` : "";
}

/** `classes.txt` companion: one class name per line, index = class id. */
export function formatClassesTxt(names: readonly string[]): string {
  return names.length > 0 ? `${names.join("\n")}\n` : "";
}

export type ParsedYoloLine =
  | { kind: "box"; classId: number; cx: number; cy: number; w: number; h: number }
  | { kind: "polygon"; classId: number; points: Array<[number, number]> };

/** Parses YOLO TXT (box and polygon lines). Throws `SyntaxError` with the 1-based line number on bad input. */
export function parseYoloTxt(text: string): ParsedYoloLine[] {
  const out: ParsedYoloLine[] = [];
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = (lines[i] as string).trim();
    if (line === "") continue;
    const parts = line.split(/\s+/);
    const classId = Number(parts[0]);
    const nums = parts.slice(1).map(Number);
    if (!Number.isInteger(classId) || classId < 0 || nums.some((n) => !Number.isFinite(n))) {
      throw new SyntaxError(`YOLO TXT line ${i + 1}: not numeric`);
    }
    if (nums.length === 4) {
      const [cx, cy, w, h] = nums as [number, number, number, number];
      out.push({ kind: "box", classId, cx, cy, w, h });
    } else if (nums.length >= 6 && nums.length % 2 === 0) {
      const points: Array<[number, number]> = [];
      for (let k = 0; k < nums.length; k += 2) points.push([nums[k] as number, nums[k + 1] as number]);
      out.push({ kind: "polygon", classId, points });
    } else {
      throw new SyntaxError(`YOLO TXT line ${i + 1}: expected 4 box values or an even number (>= 6) of polygon values`);
    }
  }
  return out;
}
