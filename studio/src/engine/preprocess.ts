/**
 * Preprocessing: RGBA pixels -> the float32 `[1, 3, S, S]` tensor the ONNX expects.
 *
 * What Python does (`dfine/utils/sources.py:331-341`, `_preprocess`): the BGR
 * frame is converted to RGB, wrapped in a PIL image, then
 * `T.Resize((S, S))` followed by `T.ToTensor()`:
 *
 * - `Resize((S, S))` with a pair does NOT keep the aspect ratio ("squash", no
 *   letterbox) and, for a PIL image, calls `PIL.Image.resize(..., BILINEAR)`.
 * - `ToTensor` divides uint8 by 255 (range 0-1), no mean/std, layout CHW.
 * - Alpha is dropped (`cv2.imread` returns 3 channels), not composited.
 *
 * PIL's bilinear is NOT the plain 2x2 bilinear of a GPU/canvas: when
 * shrinking it widens the triangle filter by the scale factor (antialiasing)
 * and works on 8-bit intermediates with 22-bit fixed-point weights. This file
 * re-implements that algorithm (`Pillow/src/libImaging/Resample.c`,
 * `precompute_coeffs`, `normalize_coeffs_8bpc`, `ImagingResampleHorizontal_8bpc`
 * / `Vertical_8bpc`), so the default `"pil"` mode is bit-exact with Python
 * (checked against fixtures generated with Pillow 12.1.1).
 *
 * A `"canvas"` mode (browser `drawImage` scaling) is also offered because it
 * is GPU-accelerated and needs no pixel copy in JS, but it is NOT the same
 * resampler: browsers do not antialias consistently, so results differ from
 * Python by a few 8-bit levels on average and by a lot on sharp edges when
 * shrinking. The tolerance is documented in `docs/studio-engine.md`.
 */
import { EngineError } from "./errors";
import { MAX_IMAGE_PIXELS, type RgbaImage } from "./types";

export type ResizeMode = "pil" | "canvas";

const PRECISION_BITS = 22; // 32 - 8 - 2, Pillow Resample.c
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);

/** Throws `EngineError` when the dimensions are invalid or above the pixel limit. */
export function validateDimensions(width: number, height: number, maxPixels: number = MAX_IMAGE_PIXELS): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new EngineError("invalid_image", `Image size must be positive integers, got ${width}x${height}`);
  }
  if (width * height > maxPixels) {
    throw new EngineError(
      "image_too_large",
      `Image has ${((width * height) / 1e6).toFixed(1)} Mpx; the limit is ${(maxPixels / 1e6).toFixed(0)} Mpx. Use a smaller image.`,
    );
  }
}

/** Throws `EngineError` when the image cannot be processed. */
export function validateImage(image: RgbaImage, maxPixels: number = MAX_IMAGE_PIXELS): void {
  const { width, height, data } = image;
  validateDimensions(width, height, maxPixels);
  if (data.length !== width * height * 4) {
    throw new EngineError(
      "invalid_image",
      `RGBA buffer has ${data.length} bytes but ${width}x${height} needs ${width * height * 4}`,
    );
  }
}

/** Drops the alpha channel: RGBA -> packed RGB (`Uint8Array`, length w*h*3). */
export function rgbaToRgb(image: RgbaImage): Uint8Array {
  const { data, width, height } = image;
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < width * height * 4; i += 4, j += 3) {
    out[j] = data[i]!;
    out[j + 1] = data[i + 1]!;
    out[j + 2] = data[i + 2]!;
  }
  return out;
}

interface Coefficients {
  /** First source index per output pixel. */
  bounds: Int32Array;
  /** Number of taps per output pixel. */
  counts: Int32Array;
  /** `ksize` fixed-point weights per output pixel. */
  weights: Int32Array;
  ksize: number;
}

function triangle(x: number): number {
  const a = Math.abs(x);
  return a < 1 ? 1 - a : 0;
}

/** Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` for BILINEAR (support 1). */
function precomputeCoefficients(inSize: number, outSize: number): Coefficients {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = 1 * filterScale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize);
  const counts = new Int32Array(outSize);
  const weights = new Int32Array(outSize * ksize);
  const tmp = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterScale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    tmp.fill(0);
    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const w = triangle((x + xmin - center + 0.5) * ss);
      tmp[x] = w;
      total += w;
    }
    for (let x = 0; x < xmax; x++) {
      const k = total !== 0 ? tmp[x]! / total : tmp[x]!;
      // C cast `(int)(0.5 + k * (1 << 22))` truncates toward zero.
      weights[xx * ksize + x] = Math.trunc(k < 0 ? -0.5 + k * ONE : 0.5 + k * ONE);
    }
    bounds[xx] = xmin;
    counts[xx] = xmax;
  }
  return { bounds, counts, weights, ksize };
}

function clip8(accumulator: number): number {
  const v = accumulator >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/**
 * Bit-exact re-implementation of `PIL.Image.resize(size, Image.BILINEAR)` for an
 * 8-bit RGB image (packed, 3 channels). Horizontal pass first, then vertical,
 * each rounding to 8 bits, like Pillow. A dimension that already has the target
 * size is left alone (Pillow skips that pass too).
 */
export function resizeRgbPil(
  rgb: Uint8Array,
  inWidth: number,
  inHeight: number,
  outWidth: number,
  outHeight: number,
): Uint8Array {
  let current = rgb;
  let width = inWidth;
  if (inWidth !== outWidth) {
    const { bounds, counts, weights, ksize } = precomputeCoefficients(inWidth, outWidth);
    const out = new Uint8Array(outWidth * inHeight * 3);
    for (let y = 0; y < inHeight; y++) {
      const rowIn = y * inWidth * 3;
      const rowOut = y * outWidth * 3;
      for (let x = 0; x < outWidth; x++) {
        const start = bounds[x]!;
        const n = counts[x]!;
        for (let c = 0; c < 3; c++) {
          let acc = HALF;
          for (let k = 0; k < n; k++) {
            acc += current[rowIn + (start + k) * 3 + c]! * weights[x * ksize + k]!;
          }
          out[rowOut + x * 3 + c] = clip8(acc);
        }
      }
    }
    current = out;
    width = outWidth;
  }
  if (inHeight !== outHeight) {
    const { bounds, counts, weights, ksize } = precomputeCoefficients(inHeight, outHeight);
    const out = new Uint8Array(width * outHeight * 3);
    for (let y = 0; y < outHeight; y++) {
      const start = bounds[y]!;
      const n = counts[y]!;
      for (let x = 0; x < width * 3; x++) {
        let acc = HALF;
        for (let k = 0; k < n; k++) {
          acc += current[(start + k) * width * 3 + x]! * weights[y * ksize + k]!;
        }
        out[y * width * 3 + x] = clip8(acc);
      }
    }
    current = out;
  } else if (current === rgb) {
    current = rgb.slice();
  }
  return current;
}

/**
 * Plain (non-antialiased) 2x2 bilinear sampling with half-pixel centres, what
 * a browser does for `drawImage` without good downscaling. NOT used by the
 * engine; it is the reference model the tests use to quantify how far a
 * canvas-style resampler can drift from PIL (see `preprocess.test.ts`).
 */
export function resizeRgbBilinearNoAntialias(
  rgb: Uint8Array,
  inWidth: number,
  inHeight: number,
  outWidth: number,
  outHeight: number,
): Uint8Array {
  const out = new Uint8Array(outWidth * outHeight * 3);
  for (let y = 0; y < outHeight; y++) {
    const sy = Math.max(((y + 0.5) * inHeight) / outHeight - 0.5, 0);
    const y0 = Math.min(Math.floor(sy), inHeight - 1);
    const y1 = Math.min(y0 + 1, inHeight - 1);
    const fy = sy - y0;
    for (let x = 0; x < outWidth; x++) {
      const sx = Math.max(((x + 0.5) * inWidth) / outWidth - 0.5, 0);
      const x0 = Math.min(Math.floor(sx), inWidth - 1);
      const x1 = Math.min(x0 + 1, inWidth - 1);
      const fx = sx - x0;
      for (let c = 0; c < 3; c++) {
        const p = (yy: number, xx: number) => rgb[(yy * inWidth + xx) * 3 + c]!;
        const top = p(y0, x0) * (1 - fx) + p(y0, x1) * fx;
        const bottom = p(y1, x0) * (1 - fx) + p(y1, x1) * fx;
        out[(y * outWidth + x) * 3 + c] = Math.round(top * (1 - fy) + bottom * fy);
      }
    }
  }
  return out;
}

/** Packed RGB uint8 `[S, S, 3]` -> float32 CHW `[3, S, S]` in 0-1 (`ToTensor`). */
export function rgbToChwFloat(rgb: Uint8Array, width: number, height: number): Float32Array {
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = rgb[i * 3]! / 255;
    out[plane + i] = rgb[i * 3 + 1]! / 255;
    out[2 * plane + i] = rgb[i * 3 + 2]! / 255;
  }
  return out;
}

/** RGBA image -> float32 CHW tensor data of length `3 * size * size` (PIL-exact resize). */
export function preprocessRgba(image: RgbaImage, size: number): Float32Array {
  validateImage(image);
  const rgb = rgbaToRgb(image);
  const resized = resizeRgbPil(rgb, image.width, image.height, size, size);
  return rgbToChwFloat(resized, size, size);
}

/** Minimal 2D-canvas surface used by the canvas mode (injectable for tests). */
export interface CanvasLike {
  width: number;
  height: number;
  getContext(type: "2d", options?: unknown): CanvasContextLike | null;
}
export interface CanvasContextLike {
  imageSmoothingEnabled: boolean;
  imageSmoothingQuality: "low" | "medium" | "high";
  drawImage(source: unknown, dx: number, dy: number, dw: number, dh: number): void;
  getImageData(x: number, y: number, w: number, h: number): { data: Uint8ClampedArray };
}
export type CanvasFactory = (width: number, height: number) => CanvasLike;

const defaultCanvasFactory: CanvasFactory = (width, height) => {
  if (typeof OffscreenCanvas === "undefined") {
    throw new EngineError("backend_unavailable", "OffscreenCanvas is not available in this browser");
  }
  return new OffscreenCanvas(width, height) as unknown as CanvasLike;
};

/**
 * Canvas mode: lets the browser scale `source` (an `ImageBitmap`) to S x S with
 * high-quality smoothing, then reads the pixels back. NOT bit-exact with Python
 * (see the file header).
 */
export function preprocessCanvas(
  source: { width: number; height: number },
  size: number,
  makeCanvas: CanvasFactory = defaultCanvasFactory,
): Float32Array {
  validateDimensions(source.width, source.height);
  const canvas = makeCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new EngineError("backend_unavailable", "Could not create a 2D canvas context");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  return preprocessRgba({ data, width: size, height: size }, size);
}

/** Reads the pixels of an `ImageBitmap` at native resolution (no scaling). */
export function bitmapToRgba(
  bitmap: { width: number; height: number },
  makeCanvas: CanvasFactory = defaultCanvasFactory,
): RgbaImage {
  validateDimensions(bitmap.width, bitmap.height);
  const canvas = makeCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new EngineError("backend_unavailable", "Could not create a 2D canvas context");
  ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return { data, width: bitmap.width, height: bitmap.height };
}
