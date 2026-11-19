/**
 * Raster exports: annotated image and crops.
 *
 * Canvas access goes through the small {@link RasterBackend} interface so the
 * drawing logic is unit-tested with a recording fake (jsdom has no canvas).
 * The default backend uses `OffscreenCanvas` when present, else a detached
 * `<canvas>`.
 */
import type { ExportDetection, ExportMask, ExportResults } from "./model";
import { baseName } from "./model";

export interface Ctx2DLike {
  drawImage(image: unknown, ...args: number[]): void;
  fillRect(x: number, y: number, w: number, h: number): void;
  strokeRect(x: number, y: number, w: number, h: number): void;
  fillText(text: string, x: number, y: number): void;
  measureText(text: string): { width: number };
  putImageData(data: unknown, x: number, y: number): void;
  getImageData?(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray; width: number; height: number };
  fillStyle: string | unknown;
  strokeStyle: string | unknown;
  lineWidth: number;
  font: string;
  textBaseline: string;
}

export interface CanvasLike {
  width: number;
  height: number;
  getContext(kind: "2d"): Ctx2DLike | null;
}

export interface RasterBackend {
  createCanvas(width: number, height: number): CanvasLike;
  /** Encodes the canvas. */
  toBlob(canvas: CanvasLike, type: "image/png" | "image/jpeg", quality?: number): Promise<Blob>;
  /** Wraps raw RGBA pixels as something `drawImage` accepts. */
  makeImageData(rgba: Uint8ClampedArray, width: number, height: number): unknown;
}

export const browserRaster: RasterBackend = {
  createCanvas(width, height) {
    if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height) as unknown as CanvasLike;
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    return c as unknown as CanvasLike;
  },
  toBlob(canvas, type, quality) {
    if (typeof OffscreenCanvas !== "undefined" && canvas instanceof OffscreenCanvas) {
      return canvas.convertToBlob({ type, quality });
    }
    return new Promise<Blob>((resolve, reject) => {
      (canvas as unknown as HTMLCanvasElement).toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Canvas could not be encoded"))),
        type,
        quality,
      );
    });
  },
  makeImageData(rgba, width, height) {
    return new ImageData(new Uint8ClampedArray(rgba), width, height);
  },
};

/** Deterministic colour per class id (golden-angle hue steps). */
export function classColor(classId: number): { h: number; css: string; rgb: [number, number, number] } {
  const h = Math.round((classId * 137.508) % 360);
  const s = 0.7;
  const l = 0.4;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number): number => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  const rgb: [number, number, number] = [f(0), f(8), f(4)];
  return { h, css: `hsl(${h} 70% 40%)`, rgb };
}

/** RGBA overlay for a mask: `alpha` (0..255) where the mask is set, 0 elsewhere. */
export function maskOverlay(mask: ExportMask, rgb: readonly [number, number, number], alpha: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(mask.width * mask.height * 4);
  for (let i = 0; i < mask.width * mask.height; i++) {
    if (mask.data[i]) {
      out[i * 4] = rgb[0];
      out[i * 4 + 1] = rgb[1];
      out[i * 4 + 2] = rgb[2];
      out[i * 4 + 3] = alpha;
    }
  }
  return out;
}

export function labelText(d: ExportDetection): string {
  return `${d.name} ${Math.round(d.confidence * 100)}%`;
}

export interface AnnotateOptions {
  /** Mask overlay opacity 0..1 (default 0.4). */
  maskAlpha?: number;
  showLabels?: boolean;
}

/** Draws `image` and the detections of `results` on a new canvas of the ORIGINAL image size. */
export function renderAnnotated(
  results: ExportResults,
  image: unknown,
  backend: RasterBackend = browserRaster,
  options: AnnotateOptions = {},
): CanvasLike {
  const { width, height } = results.image;
  const canvas = backend.createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas is not available");
  ctx.drawImage(image, 0, 0, width, height);

  const unit = Math.max(1, Math.min(width, height));
  const lineWidth = Math.max(2, Math.round(unit / 300));
  const fontPx = Math.max(12, Math.round(unit / 45));
  const alpha = Math.round(255 * (options.maskAlpha ?? 0.4));
  ctx.lineWidth = lineWidth;
  ctx.font = `${fontPx}px sans-serif`;
  ctx.textBaseline = "top";

  // Masks first so boxes and labels stay readable on top.
  for (const d of results.detections) {
    if (!d.mask) continue;
    const overlay = maskOverlay(d.mask, classColor(d.classId).rgb, alpha);
    const tmp = backend.createCanvas(d.mask.width, d.mask.height);
    const tctx = tmp.getContext("2d");
    if (!tctx) continue;
    tctx.putImageData(backend.makeImageData(overlay, d.mask.width, d.mask.height), 0, 0);
    ctx.drawImage(tmp, d.mask.x, d.mask.y);
  }

  for (const d of results.detections) {
    const { css } = classColor(d.classId);
    const x = d.box.x1;
    const y = d.box.y1;
    ctx.strokeStyle = css;
    ctx.strokeRect(x, y, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1);
    if (options.showLabels === false) continue;
    const text = labelText(d);
    const pad = Math.round(fontPx / 4);
    const boxH = fontPx + pad * 2;
    const boxW = ctx.measureText(text).width + pad * 2;
    // Label above the box, or inside its top edge when there is no room above.
    const ly = y - boxH >= 0 ? y - boxH : y;
    const lx = Math.max(0, Math.min(x, width - boxW));
    ctx.fillStyle = css;
    ctx.fillRect(lx, ly, boxW, boxH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, lx + pad, ly + pad);
  }
  return canvas;
}

export async function annotatedBlob(
  results: ExportResults,
  image: unknown,
  backend: RasterBackend = browserRaster,
  type: "image/png" | "image/jpeg" = "image/png",
  options: AnnotateOptions = {},
): Promise<Blob> {
  return backend.toBlob(renderAnnotated(results, image, backend, options), type, type === "image/jpeg" ? 0.9 : undefined);
}

/** Integer pixel rectangle of a box, clamped to the image; at least 1x1. */
export function cropRect(
  d: ExportDetection,
  width: number,
  height: number,
): { sx: number; sy: number; sw: number; sh: number } {
  const sx = Math.max(0, Math.min(width - 1, Math.floor(d.box.x1)));
  const sy = Math.max(0, Math.min(height - 1, Math.floor(d.box.y1)));
  const ex = Math.max(sx + 1, Math.min(width, Math.ceil(d.box.x2)));
  const ey = Math.max(sy + 1, Math.min(height, Math.ceil(d.box.y2)));
  return { sx, sy, sw: ex - sx, sh: ey - sy };
}

export interface CropFile {
  name: string;
  blob: Blob;
}

export function cropFileName(results: ExportResults, d: ExportDetection, index: number): string {
  const cls = d.name.replace(/[^\p{L}\p{N}_-]+/gu, "_").replace(/^_+|_+$/g, "") || `class_${d.classId}`;
  return `${baseName(results.image.name)}_${String(index).padStart(4, "0")}_${cls}.png`;
}

/** One PNG per detection (box region of the ORIGINAL image, no overlays). */
export async function renderCrops(
  results: ExportResults,
  image: unknown,
  backend: RasterBackend = browserRaster,
): Promise<CropFile[]> {
  const out: CropFile[] = [];
  for (let i = 0; i < results.detections.length; i++) {
    const d = results.detections[i] as ExportDetection;
    const { sx, sy, sw, sh } = cropRect(d, results.image.width, results.image.height);
    const canvas = backend.createCanvas(sw, sh);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas is not available");
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
    out.push({ name: cropFileName(results, d, i), blob: await backend.toBlob(canvas, "image/png") });
  }
  return out;
}
