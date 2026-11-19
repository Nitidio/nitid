/**
 * Mask -> polygon.
 *
 * Algorithm (documented in docs/studio-development.md, "Exports"):
 *  1. Label the 8-connected components of the binary mask and keep the LARGEST
 *     one (a YOLO segmentation line holds exactly one polygon; tiny specks are
 *     dropped). Holes are ignored: only the outer boundary is traced.
 *  2. Trace that component's outer boundary by crack following: walk along the
 *     pixel EDGES (grid corners, so coordinates are 0..width / 0..height) with
 *     the foreground always on the right-hand side, starting at the top-left
 *     corner of the first foreground pixel in raster order. At every corner
 *     look at the two pixels ahead: ahead-left foreground -> turn left, else
 *     ahead-right foreground -> go straight, else turn right. The tracing
 *     rule joins diagonal neighbours, matching the 8-connected labelling. The
 *     loop ends when it returns to the start corner with the start direction.
 *  3. Drop collinear vertices, then simplify the closed ring with
 *     Ramer-Douglas-Peucker (tolerance in mask pixels, default 1).
 * The result encloses every pixel of the component, so its area is the mask
 * area up to the simplification tolerance. Vertices are in mask-local pixel
 * corners; add the mask origin to get image coordinates.
 */
import type { ExportMask } from "./model";

export type Point = readonly [number, number];

/** Labels 8-connected foreground components; returns the label raster and the largest label (0 if none). */
function largestComponent(data: Uint8Array, w: number, h: number): { labels: Int32Array; best: number } {
  const labels = new Int32Array(w * h);
  let next = 0;
  let best = 0;
  let bestSize = 0;
  const stack: number[] = [];
  for (let start = 0; start < w * h; start++) {
    if (!data[start] || labels[start]) continue;
    next++;
    let size = 0;
    labels[start] = next;
    stack.push(start);
    while (stack.length > 0) {
      const p = stack.pop() as number;
      size++;
      const px = p % w;
      const py = (p - px) / w;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if (nx < 0 || nx >= w || (dx === 0 && dy === 0)) continue;
          const q = ny * w + nx;
          if (data[q] && !labels[q]) {
            labels[q] = next;
            stack.push(q);
          }
        }
      }
    }
    if (size > bestSize) {
      bestSize = size;
      best = next;
    }
  }
  return { labels, best };
}

/** Outer boundary of the component `label`, as grid-corner vertices (consecutive collinear ones removed). */
function traceBoundary(labels: Int32Array, label: number, w: number, h: number): Point[] {
  const fg = (x: number, y: number): boolean => x >= 0 && y >= 0 && x < w && y < h && labels[y * w + x] === label;
  let first = -1;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === label) {
      first = i;
      break;
    }
  }
  if (first < 0) return [];
  const sx = first % w;
  const sy = (first - sx) / w;

  // Directions: 0 east, 1 south, 2 west, 3 north (y grows downwards).
  const DX = [1, 0, -1, 0];
  const DY = [0, 1, 0, -1];
  let x = sx;
  let y = sy;
  let dir = 0;
  const ring: Point[] = [];
  const limit = 4 * (w + 1) * (h + 1) + 8; // a simple boundary cannot be longer; guards against bugs
  for (let steps = 0; steps < limit; steps++) {
    // Pixels ahead of corner (x, y) when heading `dir`.
    let leftAhead: boolean;
    let rightAhead: boolean;
    switch (dir) {
      case 0:
        leftAhead = fg(x, y - 1);
        rightAhead = fg(x, y);
        break;
      case 1:
        leftAhead = fg(x, y);
        rightAhead = fg(x - 1, y);
        break;
      case 2:
        leftAhead = fg(x - 1, y);
        rightAhead = fg(x - 1, y - 1);
        break;
      default:
        leftAhead = fg(x - 1, y - 1);
        rightAhead = fg(x, y - 1);
        break;
    }
    let ndir = dir;
    if (leftAhead) ndir = (dir + 3) % 4;
    else if (!rightAhead) ndir = (dir + 1) % 4;
    // Back at the start corner, leaving the way we left it first: the loop is closed.
    if (steps > 0 && x === sx && y === sy && ndir === 0) break;
    if (ndir !== dir || ring.length === 0) ring.push([x, y]);
    dir = ndir;
    x += DX[dir] as number;
    y += DY[dir] as number;
  }
  return ring;
}

function perpendicularDistance(p: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** Ramer-Douglas-Peucker on an open polyline (first and last point are kept). Iterative. */
function rdp(points: readonly Point[], eps: number): Point[] {
  const n = points.length;
  if (n <= 2) return points.slice();
  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;
  const stack: Array<[number, number]> = [[0, n - 1]];
  while (stack.length > 0) {
    const [lo, hi] = stack.pop() as [number, number];
    let maxD = -1;
    let idx = -1;
    for (let i = lo + 1; i < hi; i++) {
      const d = perpendicularDistance(points[i] as Point, points[lo] as Point, points[hi] as Point);
      if (d > maxD) {
        maxD = d;
        idx = i;
      }
    }
    if (idx >= 0 && maxD > eps) {
      keep[idx] = 1;
      stack.push([lo, idx], [idx, hi]);
    }
  }
  return points.filter((_, i) => keep[i] === 1);
}

/** Simplifies a closed ring: split it at the vertex farthest from vertex 0 and run RDP on both halves. */
export function simplifyRing(ring: readonly Point[], eps: number): Point[] {
  if (ring.length <= 3 || eps <= 0) return ring.slice();
  let far = 0;
  let farD = -1;
  const p0 = ring[0] as Point;
  for (let i = 1; i < ring.length; i++) {
    const p = ring[i] as Point;
    const d = (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2;
    if (d > farD) {
      farD = d;
      far = i;
    }
  }
  const a = rdp(ring.slice(0, far + 1), eps);
  const b = rdp([...ring.slice(far), p0], eps);
  return [...a.slice(0, -1), ...b.slice(0, -1)];
}

/**
 * Polygon (vertices in mask-local pixel corners) of the largest component of
 * `mask`, or `null` when the mask is empty or the polygon degenerates below
 * 3 vertices.
 */
export function maskToPolygonLocal(mask: ExportMask, tolerance = 1): Point[] | null {
  const { width: w, height: h, data } = mask;
  if (w <= 0 || h <= 0 || data.length < w * h) return null;
  const { labels, best } = largestComponent(data, w, h);
  if (best === 0) return null;
  const ring = traceBoundary(labels, best, w, h);
  // Never simplify by more than a quarter of the mask's thinner side: a 50x1 strip must stay a strip.
  const poly = simplifyRing(ring, Math.min(tolerance, 0.25 * Math.min(w, h)));
  return poly.length >= 3 ? poly : ring.length >= 3 ? ring : null;
}

/** Same polygon in pixels of the ORIGINAL image (mask origin added). */
export function maskToPolygon(mask: ExportMask, tolerance = 1): Point[] | null {
  const local = maskToPolygonLocal(mask, tolerance);
  return local ? local.map(([x, y]) => [x + mask.x, y + mask.y] as const) : null;
}

/** Shoelace area of a polygon (absolute value). */
export function polygonArea(poly: readonly Point[]): number {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i] as Point;
    const [x2, y2] = poly[(i + 1) % poly.length] as Point;
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}
