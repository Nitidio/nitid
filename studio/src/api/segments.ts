import type { Segments } from "./types";

/**
 * Turns the `segments` outline of a detection into a clean ring of `[x, y]` vertices.
 *
 * The server data is untrusted input, so this is deliberately forgiving and never throws:
 * - `x` and `y` of different length: the extra entries of the longer one are ignored;
 * - vertices with a non-finite coordinate are dropped;
 * - a repeated closing vertex (last equal to first) is removed, the ring is implicitly closed;
 * - fewer than 3 vertices remain: returns null (a line or a point is not an area).
 *
 * Coordinates are NOT clamped to the image here; the consumers that need it (YOLO) clamp.
 */
export function segmentsToRing(segments: Segments | undefined | null): Array<[number, number]> | null {
  if (!segments || !Array.isArray(segments.x) || !Array.isArray(segments.y)) return null;
  const n = Math.min(segments.x.length, segments.y.length);
  const ring: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const x = segments.x[i];
    const y = segments.y[i];
    if (typeof x === "number" && typeof y === "number" && Number.isFinite(x) && Number.isFinite(y)) ring.push([x, y]);
  }
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (ring.length > 1 && first && last && first[0] === last[0] && first[1] === last[1]) ring.pop();
  return ring.length >= 3 ? ring : null;
}

/** Same ring as a flat `[x0, y0, x1, y1, ...]` array (the shape of `DetectionMask.polygons`). */
export function ringToFlat(ring: ReadonlyArray<readonly [number, number]>): number[] {
  return ring.flatMap(([x, y]) => [x, y]);
}
