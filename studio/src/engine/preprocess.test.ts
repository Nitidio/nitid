import { describe, expect, it } from "vitest";
import {
  bitmapToRgba,
  preprocessCanvas,
  preprocessRgba,
  resizeRgbBilinearNoAntialias,
  resizeRgbPil,
  rgbaToRgb,
  rgbToChwFloat,
  validateImage,
  type CanvasContextLike,
  type CanvasLike,
} from "./preprocess";
import { loadFixture, synthImage } from "./test-utils";

const levels = (tensor: Float32Array, size: number) => {
  const plane = size * size;
  return (c: number, y: number, x: number) => Math.round(tensor[c * plane + y * size + x]! * 255);
};

describe("preprocess: parity with Pillow/torchvision (Resize((S,S)) + ToTensor)", () => {
  for (const task of ["detect", "segment"] as const) {
    const fixture = loadFixture(task);
    for (const expected of fixture.preprocess) {
      const { width, height } = expected.image;
      it(`${task}: ${width}x${height} -> ${fixture.imgsz}x${fixture.imgsz} is bit-exact`, () => {
        const tensor = preprocessRgba(synthImage(width, height), fixture.imgsz);
        const at = levels(tensor, fixture.imgsz);
        // Exact equality: the TS port of Pillow's fixed-point bilinear must not drift at all.
        for (const sample of expected.samples) {
          expect(at(sample.c, sample.y, sample.x), JSON.stringify(sample)).toBe(sample.level);
        }
        const plane = fixture.imgsz * fixture.imgsz;
        const sums = [0, 1, 2].map((c) => {
          let sum = 0;
          for (let i = 0; i < plane; i++) sum += Math.round(tensor[c * plane + i]! * 255);
          return sum;
        });
        expect(sums).toEqual(expected.channel_level_sums);
      });
    }
  }
});

describe("preprocess: tensor contract", () => {
  it("returns float32 CHW in [0, 1] with length 3*S*S", () => {
    const tensor = preprocessRgba(synthImage(33, 17), 64);
    expect(tensor).toBeInstanceOf(Float32Array);
    expect(tensor.length).toBe(3 * 64 * 64);
    expect(Math.min(...tensor)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...tensor)).toBeLessThanOrEqual(1);
  });

  it("is RGB planar: channel 0 is red, 1 green, 2 blue", () => {
    const image = { width: 2, height: 2, data: new Uint8ClampedArray([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]) };
    const tensor = rgbToChwFloat(rgbaToRgb(image), 2, 2);
    expect(Array.from(tensor)).toEqual([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("squashes: does not keep the aspect ratio and does not pad", () => {
    // A 100x10 all-white image fills the whole 64x64 tensor (a letterbox would leave black bars).
    const data = new Uint8ClampedArray(100 * 10 * 4).fill(255);
    const tensor = preprocessRgba({ width: 100, height: 10, data }, 64);
    expect(Math.min(...tensor)).toBe(1);
  });

  it("ignores alpha (Python drops it, it does not composite)", () => {
    const opaque = synthImage(8, 8);
    const translucent = { ...opaque, data: opaque.data.map((v, i) => (i % 4 === 3 ? 10 : v)) };
    expect(preprocessRgba(translucent, 32)).toEqual(preprocessRgba(opaque, 32));
  });

  it("a same-size image is copied unchanged (Pillow skips both passes)", () => {
    const image = synthImage(32, 32);
    const rgb = rgbaToRgb(image);
    const out = resizeRgbPil(rgb, 32, 32, 32, 32);
    expect(out).toEqual(rgb);
    expect(out).not.toBe(rgb);
  });
});

describe("preprocess: interpolation difference vs a canvas-style resampler (documented tolerance)", () => {
  // A browser canvas does plain bilinear sampling, without PIL's antialiasing when shrinking.
  // `resizeRgbBilinearNoAntialias` models that. The image is deliberately high-frequency
  // (the blue channel is `(7x + 13y) mod 256`), which is a worst case; natural photos drift less.
  const drift = (w: number, h: number) => {
    const rgb = rgbaToRgb(synthImage(w, h));
    const pil = resizeRgbPil(rgb, w, h, 64, 64);
    const naive = resizeRgbBilinearNoAntialias(rgb, w, h, 64, 64);
    let sum = 0;
    let max = 0;
    for (let i = 0; i < pil.length; i++) {
      const d = Math.abs(pil[i]! - naive[i]!);
      sum += d;
      max = Math.max(max, d);
    }
    return { mean: sum / pil.length, max };
  };

  it("upscaling: both agree to within 1 level", () => {
    expect(drift(40, 30).max).toBeLessThanOrEqual(1);
  });

  it("downscaling: they differ, and the difference grows with the shrink factor", () => {
    const small = drift(150, 110);
    const large = drift(640, 480);
    // Measured (2026-11): mean 2.6 / max 95 at 150x110; mean 10.6 / max 122 at 640x480 (levels of 255).
    expect(small.mean).toBeGreaterThan(1);
    expect(large.mean).toBeGreaterThan(small.mean);
    expect(large.mean).toBeLessThan(40);
  });
});

describe("preprocess: limits and errors", () => {
  it("rejects images above the pixel limit with image_too_large", () => {
    const fake = { width: 5001, height: 4001, data: new Uint8ClampedArray(4) };
    expect(() => validateImage(fake)).toThrowError(/20 Mpx/);
    try {
      validateImage(fake);
    } catch (error) {
      expect((error as { code: string }).code).toBe("image_too_large");
    }
  });

  it("accepts exactly 20 Mpx dimensions", () => {
    expect(() => validateImage({ width: 5000, height: 4000, data: new Uint8ClampedArray(5000 * 4000 * 4) })).not.toThrow();
  });

  it("rejects a buffer whose length does not match the size", () => {
    expect(() => validateImage({ width: 4, height: 4, data: new Uint8ClampedArray(10) })).toThrowError(/needs 64/);
  });

  it.each([
    [0, 5],
    [5, 0],
    [-1, 5],
    [2.5, 5],
  ])("rejects non-positive or fractional size %s x %s", (width, height) => {
    expect(() => validateImage({ width, height, data: new Uint8ClampedArray(0) })).toThrowError(/positive integers/);
  });
});

function fakeCanvas() {
  const calls: unknown[][] = [];
  const state: { smoothing?: boolean; quality?: string; size?: [number, number] } = {};
  const makeCanvas = (width: number, height: number): CanvasLike => {
    state.size = [width, height];
    const ctx: CanvasContextLike = {
      get imageSmoothingEnabled() {
        return state.smoothing ?? false;
      },
      set imageSmoothingEnabled(v: boolean) {
        state.smoothing = v;
      },
      get imageSmoothingQuality() {
        return (state.quality ?? "low") as "low" | "medium" | "high";
      },
      set imageSmoothingQuality(v) {
        state.quality = v;
      },
      drawImage: (...args: unknown[]) => void calls.push(args),
      getImageData: (_x, _y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4).fill(128) }),
    };
    return { width, height, getContext: () => ctx };
  };
  return { makeCanvas, calls, state };
}

describe("preprocess: canvas mode (approximate, opt-in)", () => {
  it("lets the canvas scale to SxS with high-quality smoothing and returns a CHW tensor", () => {
    const { makeCanvas, calls, state } = fakeCanvas();
    const bitmap = { width: 300, height: 200 };
    const tensor = preprocessCanvas(bitmap, 64, makeCanvas);
    expect(state.size).toEqual([64, 64]);
    expect(state.smoothing).toBe(true);
    expect(state.quality).toBe("high");
    expect(calls[0]).toEqual([bitmap, 0, 0, 64, 64]);
    expect(tensor.length).toBe(3 * 64 * 64);
    expect(tensor[0]).toBeCloseTo(128 / 255, 6);
  });

  it("bitmapToRgba reads the bitmap at its native size without scaling", () => {
    const { makeCanvas, calls, state } = fakeCanvas();
    const rgba = bitmapToRgba({ width: 7, height: 5 }, makeCanvas);
    expect(state.size).toEqual([7, 5]);
    expect(calls[0]).toEqual([{ width: 7, height: 5 }, 0, 0, 7, 5]);
    expect(rgba.data.length).toBe(7 * 5 * 4);
  });

  it("reports backend_unavailable when OffscreenCanvas is missing", () => {
    expect(() => bitmapToRgba({ width: 2, height: 2 })).toThrowError(/OffscreenCanvas/);
  });

  it("enforces the pixel limit before allocating a canvas", () => {
    const { makeCanvas } = fakeCanvas();
    expect(() => preprocessCanvas({ width: 6000, height: 4000 }, 64, makeCanvas)).toThrowError(/Mpx/);
  });
});
