/** Shared builders for the export tests (not part of the runtime bundle). */
import type { ExportDetection, ExportMask, ExportResults } from "./model";
import type { RasterBackend } from "./raster";

/** 16x12 RGB gradient JPEG (made with Pillow, quality 60). */
export const TINY_JPEG_BASE64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdBRkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wAARCAAMABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDnbXSuny1r2uldPlratbaLj5a17W2i4+WlXx0jPKsxnof/2Q==";

export function tinyJpeg(): Uint8Array {
  return Uint8Array.from(atob(TINY_JPEG_BASE64), (c) => c.charCodeAt(0));
}

export function rectMask(x: number, y: number, width: number, height: number): ExportMask {
  return { x, y, width, height, data: new Uint8Array(width * height).fill(1) };
}

/** Mask from rows of "#" / "." characters. */
export function maskFromRows(x: number, y: number, rows: string[]): ExportMask {
  const width = rows[0]?.length ?? 0;
  const data = new Uint8Array(width * rows.length);
  rows.forEach((r, j) => [...r].forEach((c, i) => (data[j * width + i] = c === "#" ? 1 : 0)));
  return { x, y, width, height: rows.length, data };
}

export function det(over: Partial<ExportDetection> = {}): ExportDetection {
  return {
    box: { x1: 10, y1: 20, x2: 110, y2: 70 },
    confidence: 0.9,
    classId: 0,
    name: "person",
    hasMask: false,
    ...over,
  };
}

export function results(over: Partial<ExportResults> = {}): ExportResults {
  return {
    source: "browser",
    image: { name: "street view.JPG", width: 200, height: 100 },
    model: "nitid1s",
    task: "detect",
    backend: "wasm",
    device: null,
    timings: { preprocessMs: 3.1, inferenceMs: 28.4, postprocessMs: 1.2 },
    createdAt: "2026-11-19T10:00:00.000Z",
    detections: [det(), det({ box: { x1: 100, y1: 0, x2: 200, y2: 50 }, confidence: 0.5, classId: 2, name: "car" })],
    ...over,
  };
}

/** Recording fake of the canvas backend (jsdom has no canvas). `toBlob` returns PNG/JPEG magic bytes. */
export interface FakeCanvas {
  width: number;
  height: number;
  calls: Array<[string, ...unknown[]]>;
  getContext(kind: "2d"): FakeCtx;
}
export interface FakeCtx {
  calls: Array<[string, ...unknown[]]>;
  [k: string]: unknown;
}

export function fakeRaster() {
  const canvases: FakeCanvas[] = [];
  const backend = {
    createCanvas(width: number, height: number): FakeCanvas {
      const calls: Array<[string, ...unknown[]]> = [];
      const ctx = {
        calls,
        fillStyle: "",
        strokeStyle: "",
        lineWidth: 1,
        font: "",
        textBaseline: "",
        drawImage: (...a: unknown[]) => calls.push(["drawImage", ...a]),
        beginPath: () => calls.push(["beginPath"]),
        moveTo: (...a: unknown[]) => calls.push(["moveTo", ...a]),
        lineTo: (...a: unknown[]) => calls.push(["lineTo", ...a]),
        closePath: () => calls.push(["closePath"]),
        fill: () => calls.push(["fill", ctx.fillStyle]),
        stroke: () => calls.push(["stroke", ctx.strokeStyle]),
        fillRect: (...a: unknown[]) => calls.push(["fillRect", ...a]),
        strokeRect: (...a: unknown[]) => calls.push(["strokeRect", ctx.strokeStyle, ...a]),
        fillText: (...a: unknown[]) => calls.push(["fillText", ...a]),
        putImageData: (...a: unknown[]) => calls.push(["putImageData", ...a]),
        measureText: (t: string) => ({ width: t.length * 7 }),
      };
      const c: FakeCanvas = { width, height, calls, getContext: () => ctx as unknown as FakeCtx };
      canvases.push(c);
      return c;
    },
    toBlob(_c: unknown, type: "image/png" | "image/jpeg"): Promise<Blob> {
      const bytes = type === "image/png" ? new Uint8Array([0x89, 0x50, 0x4e, 0x47]) : tinyJpeg();
      return Promise.resolve(new Blob([bytes as BlobPart], { type }));
    },
    makeImageData: (rgba: Uint8ClampedArray, width: number, height: number) => ({ rgba, width, height }),
  };
  return { backend: backend as unknown as RasterBackend, canvases };
}
