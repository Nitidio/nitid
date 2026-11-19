import type { Box } from "../api/types";
import type { ResultDetection } from "./results-model";

/** Maps image pixels to canvas pixels. */
export interface Transform {
  scale: number;
}

/** Scale that fits an image into a maximum drawing width without upscaling past 1:1 by default. */
export function fitScale(imageWidth: number, maxWidth: number, allowUpscale = false): number {
  if (imageWidth <= 0 || maxWidth <= 0) return 1;
  const s = maxWidth / imageWidth;
  return allowUpscale ? s : Math.min(1, s);
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Convert a box in original-image pixels to a canvas rectangle, clamped to the image
 * (the server may return boxes slightly outside it). Returns null for a degenerate box.
 */
export function boxToRect(box: Box, imageWidth: number, imageHeight: number, scale: number): Rect | null {
  const x1 = Math.max(0, Math.min(imageWidth, Math.min(box.x1, box.x2)));
  const x2 = Math.max(0, Math.min(imageWidth, Math.max(box.x1, box.x2)));
  const y1 = Math.max(0, Math.min(imageHeight, Math.min(box.y1, box.y2)));
  const y2 = Math.max(0, Math.min(imageHeight, Math.max(box.y1, box.y2)));
  if (![x1, x2, y1, y2].every(Number.isFinite) || x2 - x1 <= 0 || y2 - y1 <= 0) return null;
  return { x: x1 * scale, y: y1 * scale, w: (x2 - x1) * scale, h: (y2 - y1) * scale };
}

export function labelText(d: Pick<ResultDetection, "name" | "confidence">): string {
  return `${d.name} ${Math.round(d.confidence * 100)}%`;
}

/**
 * Place a label of the given size above the rectangle, or inside its top edge when
 * there is no room above, and keep it inside the canvas horizontally.
 */
export function labelPosition(
  rect: Rect,
  labelWidth: number,
  labelHeight: number,
  canvasWidth: number,
): { x: number; y: number } {
  const y = rect.y - labelHeight >= 0 ? rect.y - labelHeight : rect.y;
  const x = Math.max(0, Math.min(rect.x, canvasWidth - labelWidth));
  return { x, y };
}

/** Number of colours in the class palette. */
export const PALETTE_SIZE = 10;

/**
 * Stable class colour. Hues are spread by the golden angle so neighbouring ids differ; lightness is
 * mid-range so a white label stays readable. Colour is never the only cue: labels name the class.
 */
export function classColor(classId: number): string {
  const hue = Math.round((((classId % 360) * 137.508) % 360 + 360) % 360);
  return `hsl(${hue} 70% 38%)`;
}

/** Flatten polygon coordinates into canvas space. */
export function scalePolygon(polygon: readonly number[], scale: number): number[] {
  return polygon.map((v) => v * scale);
}
