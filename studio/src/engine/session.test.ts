import { describe, expect, it, vi } from "vitest";
import { EngineSession, assertAbsoluteWasmBase, chooseNumThreads, type OrtModule, type OrtSessionLike, type SessionDeps } from "./session";
import { boxIou, compareWithPython, loadFixture, postprocessedTensors, synthImage } from "./test-utils";

const fixture = loadFixture("segment");
const meta = fixture.meta;

interface FakeOptions {
  inputNames?: string[];
  outputNames?: string[];
  createFails?: (providers: string[]) => boolean;
  runFails?: boolean;
  delayMs?: number;
}

function fakeOrt(options: FakeOptions = {}) {
  const log = {
    created: [] as string[][],
    feeds: [] as Record<string, { dims: readonly number[]; data: Float32Array }>[],
    released: 0,
    disposed: 0,
    concurrent: 0,
    maxConcurrent: 0,
  };
  const env = { wasm: {} as { wasmPaths?: string | Record<string, string>; numThreads?: number; proxy?: boolean } };
  class Tensor {
    constructor(
      readonly type: string,
      readonly data: Float32Array,
      readonly dims: readonly number[],
    ) {}
  }
  const outputs = postprocessedTensors(fixture);
  const session: OrtSessionLike = {
    inputNames: options.inputNames ?? ["images"],
    outputNames: options.outputNames ?? ["labels", "boxes", "scores", "masks"],
    run: async (feeds) => {
      log.concurrent++;
      log.maxConcurrent = Math.max(log.maxConcurrent, log.concurrent);
      log.feeds.push(feeds as never);
      if (options.delayMs) await new Promise((r) => setTimeout(r, options.delayMs));
      log.concurrent--;
      if (options.runFails) throw new Error("ORT blew up");
      return Object.fromEntries(
        Object.entries(outputs).map(([name, t]) => [name, { data: t.data, dims: t.dims, dispose: () => void log.disposed++ }]),
      ) as never;
    },
    release: async () => void log.released++,
  };
  const ort: OrtModule = {
    env,
    InferenceSession: {
      create: async (_bytes, opts) => {
        log.created.push(opts.executionProviders);
        if (options.createFails?.(opts.executionProviders)) throw new Error(`no ${opts.executionProviders[0]}`);
        return session;
      },
    },
    Tensor: Tensor as never,
  };
  return { ort, env, log };
}

function deps(ort: OrtModule, env: Partial<SessionDeps["env"]> = {}, loadOrt = vi.fn(async () => ort)): SessionDeps {
  return {
    loadOrt,
    env: { crossOriginIsolated: false, hardwareConcurrency: 8, hasWebGpu: async () => false, ...env },
  };
}

const modelBytes = new Uint8Array([1, 2, 3]).buffer;
const image = synthImage(40, 30);

describe("session: environment decisions", () => {
  it("numThreads is 1 without cross-origin isolation, up to 4 with it", () => {
    expect(chooseNumThreads({ crossOriginIsolated: false, hardwareConcurrency: 16 })).toBe(1);
    expect(chooseNumThreads({ crossOriginIsolated: true, hardwareConcurrency: 16 })).toBe(4);
    expect(chooseNumThreads({ crossOriginIsolated: true, hardwareConcurrency: 2 })).toBe(2);
    expect(chooseNumThreads({ crossOriginIsolated: true, hardwareConcurrency: 0 })).toBe(1);
  });

  it("wasmBase must be absolute", () => {
    expect(assertAbsoluteWasmBase("/ort/")).toBe("/ort/");
    expect(assertAbsoluteWasmBase("/ort")).toBe("/ort/");
    expect(assertAbsoluteWasmBase("https://studio.test/ort/")).toBe("https://studio.test/ort/");
    for (const bad of ["ort/", "./ort/", "../ort/"]) {
      expect(() => assertAbsoluteWasmBase(bad)).toThrowError(/absolute/);
    }
  });
});

describe("session: creation", () => {
  it("configures ORT (absolute wasmPaths, threads, no proxy) and uses the wasm provider by default", async () => {
    const { ort, env, log } = fakeOrt();
    const session = await EngineSession.create(deps(ort, { crossOriginIsolated: true, hardwareConcurrency: 8 }), { modelBytes, meta });
    expect(env.wasm).toEqual({ wasmPaths: "/ort/", numThreads: 4, proxy: false });
    expect(log.created).toEqual([["wasm"]]);
    expect(session.info).toMatchObject({ backend: "wasm", numThreads: 4 });
    expect(session.info.fallbackReason).toBeUndefined();
  });

  it("is single-threaded when the page is not cross-origin isolated", async () => {
    const { ort, env } = fakeOrt();
    await EngineSession.create(deps(ort), { modelBytes, meta });
    expect(env.wasm.numThreads).toBe(1);
  });

  it("rejects a relative wasmBase before loading anything", async () => {
    const { ort } = fakeOrt();
    const loadOrt = vi.fn(async () => ort);
    await expect(EngineSession.create(deps(ort, {}, loadOrt), { modelBytes, meta, wasmBase: "ort/" })).rejects.toMatchObject({
      code: "invalid_options",
    });
    expect(loadOrt).not.toHaveBeenCalled();
  });

  it("requires a meta: explicit or embedded", async () => {
    const { ort } = fakeOrt();
    const protobufWithoutMeta = Uint8Array.from([0x08, 0x08]).buffer; // ir_version only
    await expect(EngineSession.create(deps(ort), { modelBytes: protobufWithoutMeta })).rejects.toMatchObject({ code: "invalid_meta" });
  });

  it("rejects an invalid explicit meta", async () => {
    const { ort } = fakeOrt();
    await expect(EngineSession.create(deps(ort), { modelBytes, meta: { nope: true } })).rejects.toMatchObject({ code: "invalid_meta" });
  });

  it("fails clearly when the ONNX does not expose what the meta promises, and releases the session", async () => {
    const { ort, log } = fakeOrt({ outputNames: ["labels", "boxes", "scores"] });
    await expect(EngineSession.create(deps(ort), { modelBytes, meta })).rejects.toMatchObject({
      code: "unsupported_output",
      message: expect.stringContaining("masks"),
    });
    expect(log.released).toBe(1);
  });

  it("reports ORT start-up failures as backend_unavailable", async () => {
    const { ort } = fakeOrt({ createFails: () => true });
    await expect(EngineSession.create(deps(ort), { modelBytes, meta })).rejects.toMatchObject({ code: "backend_unavailable" });
  });

  describe("webgpu is optional", () => {
    it("is not even loaded unless requested", async () => {
      const { ort } = fakeOrt();
      const loadOrt = vi.fn(async () => ort);
      await EngineSession.create(deps(ort, { hasWebGpu: async () => true }, loadOrt), { modelBytes, meta });
      expect(loadOrt.mock.calls.map((c) => (c as unknown[])[0])).toEqual(["wasm"]);
    });

    it("backend 'webgpu' with an adapter uses it (with wasm as the EP fallback list)", async () => {
      const { ort, log } = fakeOrt();
      const loadOrt = vi.fn(async () => ort);
      const session = await EngineSession.create(deps(ort, { hasWebGpu: async () => true }, loadOrt), { modelBytes, meta, backend: "webgpu" });
      expect(session.info.backend).toBe("webgpu");
      expect(log.created).toEqual([["webgpu", "wasm"]]);
      expect(loadOrt.mock.calls.map((c) => (c as unknown[])[0])).toEqual(["webgpu"]);
    });

    it("backend 'webgpu' without an adapter falls back to wasm and says why", async () => {
      const { ort } = fakeOrt();
      const session = await EngineSession.create(deps(ort), { modelBytes, meta, backend: "webgpu" });
      expect(session.info).toMatchObject({ backend: "wasm", fallbackReason: expect.stringContaining("not available") });
    });

    it("backend 'auto' without an adapter silently uses wasm", async () => {
      const { ort } = fakeOrt();
      const session = await EngineSession.create(deps(ort), { modelBytes, meta, backend: "auto" });
      expect(session.info.backend).toBe("wasm");
      expect(session.info.fallbackReason).toBeUndefined();
    });

    it("falls back to wasm when the webgpu session cannot be created", async () => {
      const { ort, log } = fakeOrt({ createFails: (eps) => eps[0] === "webgpu" });
      const session = await EngineSession.create(deps(ort, { hasWebGpu: async () => true }), { modelBytes, meta, backend: "auto" });
      expect(session.info).toMatchObject({ backend: "wasm", fallbackReason: expect.stringContaining("no webgpu") });
      expect(log.created).toEqual([["webgpu", "wasm"], ["wasm"]]);
    });

    it("a throwing adapter probe counts as no WebGPU (never breaks)", async () => {
      const { ort } = fakeOrt();
      const session = await EngineSession.create(
        deps(ort, {
          hasWebGpu: async () => {
            throw new Error("gpu blew up");
          },
        }),
        { modelBytes, meta, backend: "webgpu" },
      );
      expect(session.info.backend).toBe("wasm");
    });
  });
});

describe("session: run", () => {
  it("feeds a float32 [1,3,S,S] tensor and returns detections equal to Python's (fake ORT replays the fixture)", async () => {
    const { ort, log } = fakeOrt();
    const session = await EngineSession.create(deps(ort), { modelBytes, meta });
    const testCase = fixture.cases.find((c) => c.name === "default")!;
    const result = await session.run({ kind: "rgba", ...image }, { conf: testCase.conf, maskThreshold: testCase.mask_threshold });

    const feed = log.feeds[0]!.images!;
    expect(feed.dims).toEqual([1, 3, 64, 64]);
    expect(feed.data).toBeInstanceOf(Float32Array);
    expect(feed.data.length).toBe(3 * 64 * 64);

    const report = compareWithPython(result.detections, testCase.detections, image);
    expect(report.minBoxIou).toBeGreaterThanOrEqual(0.98);
    expect(report.maxScoreDelta).toBeLessThanOrEqual(0.01);
    expect(report.minMaskIou).toBeGreaterThanOrEqual(0.95);
    expect(result.imageSize).toEqual({ width: 40, height: 30 });
    expect(result.timings.inferenceMs).toBeGreaterThanOrEqual(0);
    expect(boxIou([0, 0, 1, 1], [0, 0, 1, 1])).toBe(1);
  });

  it("disposes the ORT output tensors after decoding", async () => {
    const { ort, log } = fakeOrt();
    const session = await EngineSession.create(deps(ort), { modelBytes, meta });
    await session.run({ kind: "rgba", ...image });
    expect(log.disposed).toBe(4);
  });

  it("serialises concurrent runs (an ORT session is not re-entrant)", async () => {
    const { ort, log } = fakeOrt({ delayMs: 5 });
    const session = await EngineSession.create(deps(ort), { modelBytes, meta });
    await Promise.all([1, 2, 3].map(() => session.run({ kind: "rgba", ...image })));
    expect(log.feeds.length).toBe(3);
    expect(log.maxConcurrent).toBe(1);
  });

  it("wraps ORT failures as inference_failed and stays usable", async () => {
    const failing = fakeOrt({ runFails: true });
    const bad = await EngineSession.create(deps(failing.ort), { modelBytes, meta });
    await expect(bad.run({ kind: "rgba", ...image })).rejects.toMatchObject({ code: "inference_failed" });
    await expect(bad.run({ kind: "rgba", ...image })).rejects.toMatchObject({ code: "inference_failed" });
  });

  it("rejects oversize and malformed images before inference", async () => {
    const { ort, log } = fakeOrt();
    const session = await EngineSession.create(deps(ort), { modelBytes, meta });
    await expect(session.run({ kind: "rgba", width: 6000, height: 4000, data: new Uint8ClampedArray(4) })).rejects.toMatchObject({
      code: "image_too_large",
    });
    await expect(session.run({ kind: "rgba", width: 4, height: 4, data: new Uint8ClampedArray(3) })).rejects.toMatchObject({
      code: "invalid_image",
    });
    await expect(session.run({ kind: "bitmap", bitmap: { width: 6000, height: 4000 } })).rejects.toMatchObject({ code: "image_too_large" });
    expect(log.feeds.length).toBe(0);
  });

  it("an ImageBitmap goes through the canvas at native size, then the exact PIL path", async () => {
    const { ort, log } = fakeOrt();
    const drawn: unknown[][] = [];
    const base = deps(ort);
    const session = await EngineSession.create(
      {
        ...base,
        makeCanvas: (w, h) => ({
          width: w,
          height: h,
          getContext: () => ({
            imageSmoothingEnabled: false,
            imageSmoothingQuality: "low",
            drawImage: (...a: unknown[]) => void drawn.push(a),
            getImageData: () => ({ data: new Uint8ClampedArray(image.data) }),
          }),
        }),
      },
      { modelBytes, meta },
    );
    await session.run({ kind: "bitmap", bitmap: { width: 40, height: 30 } });
    expect(drawn[0]!.slice(1)).toEqual([0, 0, 40, 30]); // native size, not 64x64
    expect(log.feeds[0]!.images!.dims).toEqual([1, 3, 64, 64]);
  });

  it("run after dispose is session_not_ready; dispose releases once and waits for pending runs", async () => {
    const { ort, log } = fakeOrt({ delayMs: 5 });
    const session = await EngineSession.create(deps(ort), { modelBytes, meta });
    const pending = session.run({ kind: "rgba", ...image });
    await session.dispose();
    await session.dispose();
    await expect(pending).resolves.toBeDefined();
    expect(log.released).toBe(1);
    await expect(session.run({ kind: "rgba", ...image })).rejects.toMatchObject({ code: "session_not_ready" });
  });
});
