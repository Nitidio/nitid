import { describe, expect, it, vi } from "vitest";
import type { InitResult } from "../workers/protocol";
import type { RunResult } from "../engine/session";
import type { Detection } from "../engine/types";
import { buildBrowserResults } from "./results";
import { BrowserRunner, type EngineLike } from "./runner";
import type { BrowserModel } from "./catalog";

const serverModel: BrowserModel = {
  key: "detect:nitid1s", id: "nitid1s", task: "detect", label: "nitid1s (detect)", origin: "server",
  url: "/api/v1/onnx/nitid1s?task=detect",
};
const hostedModel: BrowserModel = {
  key: "detect:h", id: "h", task: "detect", label: "h", origin: "hosted", url: "https://m.test/h.onnx", sha256: "b".repeat(64),
};

const initResult = (task: "detect" | "segment" = "detect"): InitResult =>
  ({
    backend: "wasm", numThreads: 1, modelSource: "cache", modelCached: false,
    meta: { task, imgsz: 640 },
  }) as unknown as InitResult;

function det(over: Partial<Detection> = {}): Detection {
  return { box: { x1: 10, y1: 10, x2: 50, y2: 40 }, confidence: 0.9, class_id: 0, name: "person", has_mask: false, ...over };
}
const runResult = (detections: Detection[] = [det()]): RunResult => ({
  detections,
  imageSize: { width: 100, height: 80 },
  timings: { preprocessMs: 1, inferenceMs: 2, postprocessMs: 3 },
});

function fakeEngine(task: "detect" | "segment" = "detect") {
  const engine = {
    init: vi.fn(async () => initResult(task)),
    run: vi.fn(async () => runResult()),
    dispose: vi.fn(async () => undefined),
    terminate: vi.fn(),
  };
  return engine satisfies EngineLike;
}

const file = (name = "a.png") => new File([new Uint8Array([1])], name, { type: "image/png" });
const decode = async () => ({ width: 100, height: 80 }) as unknown as ImageBitmap;

describe("BrowserRunner", () => {
  it("server model: fetches the meta, passes sha256 + bearer token to the engine", async () => {
    const engine = fakeEngine();
    const meta = { task: "detect", sha256: "c".repeat(64) };
    const runner = new BrowserRunner({
      createEngine: () => engine, getToken: () => "tok", fetchServerMeta: async () => meta, decode,
    });
    const info = await runner.load(serverModel);
    expect(engine.init).toHaveBeenCalledWith(
      { modelUrl: serverModel.url, meta, sha256: "c".repeat(64), headers: { Authorization: "Bearer tok" } },
      undefined,
    );
    expect(info).toMatchObject({ backend: "wasm", source: "cache", imgsz: 640 });
  });

  it("hosted model: no token, no meta request, catalogue sha256", async () => {
    const engine = fakeEngine();
    const fetchServerMeta = vi.fn();
    const runner = new BrowserRunner({ createEngine: () => engine, getToken: () => "tok", fetchServerMeta, decode });
    await runner.load(hostedModel);
    expect(fetchServerMeta).not.toHaveBeenCalled();
    expect(engine.init).toHaveBeenCalledWith({ modelUrl: "https://m.test/h.onnx", sha256: "b".repeat(64) }, undefined);
  });

  it("refuses a model whose task differs from the catalogue", async () => {
    const runner = new BrowserRunner({ createEngine: () => fakeEngine("segment"), getToken: () => null, fetchServerMeta: async () => ({}), decode });
    await expect(runner.load(hostedModel)).rejects.toMatchObject({ code: "invalid_meta" });
    expect(runner.current).toBeNull();
  });

  it("runs files in order and builds RunResults", async () => {
    const engine = fakeEngine();
    let t = 0;
    const runner = new BrowserRunner({ createEngine: () => engine, getToken: () => null, fetchServerMeta: async () => ({}), decode, now: () => (t += 10) });
    await runner.load(hostedModel);
    const results = await runner.runFiles([file("a.png"), file("b.png")]);
    expect(engine.run).toHaveBeenCalledTimes(2);
    expect(engine.run).toHaveBeenCalledWith({ kind: "bitmap", bitmap: expect.anything() }, { conf: 0.1 });
    expect(results).toMatchObject({ source: "browser", model: "h", task: "detect", backend: "wasm", device: null });
    expect(results.items.map((i) => i.name)).toEqual(["a.png", "b.png"]);
    expect(results.wallClockMs).toBe(10);
  });

  it("run before load is an error; close terminates the worker", async () => {
    const engine = fakeEngine();
    const runner = new BrowserRunner({ createEngine: () => engine, getToken: () => null, fetchServerMeta: async () => ({}), decode });
    await expect(runner.runFiles([file()])).rejects.toMatchObject({ code: "session_not_ready" });
    await runner.load(hostedModel);
    runner.close();
    expect(engine.terminate).toHaveBeenCalled();
    expect(runner.current).toBeNull();
  });

  it("propagates engine errors", async () => {
    const engine = fakeEngine();
    engine.run.mockRejectedValueOnce(Object.assign(new Error("too big"), { code: "image_too_large" }));
    const runner = new BrowserRunner({ createEngine: () => engine, getToken: () => null, fetchServerMeta: async () => ({}), decode });
    await runner.load(hostedModel);
    await expect(runner.runFiles([file()])).rejects.toThrow("too big");
  });
});

describe("buildBrowserResults", () => {
  it("maps detections and timings, polygons from masks, and exports through fromEngineResult", () => {
    const mask = { x: 10, y: 10, width: 4, height: 4, data: new Uint8Array(16).fill(1) };
    const results = buildBrowserResults({
      runId: "r", model: "m", task: "segment", backend: "wasm",
      items: [{ name: "street view.png", result: runResult([det({ has_mask: true, mask })]) }],
    });
    const d = results.items[0]!.detections[0]!;
    expect(d.hasMask).toBe(true);
    expect(d.mask?.polygons[0]?.length).toBeGreaterThanOrEqual(6);
    expect(results.items[0]!.speed).toEqual({ preprocess: 1, inference: 2, postprocess: 3 });
    const exp = results.toExport!(0);
    expect(exp).toMatchObject({ source: "browser", model: "m", task: "segment", backend: "wasm", image: { name: "street view.png", width: 100, height: 80 } });
    expect(exp.detections[0]?.mask?.data).toHaveLength(16);
    expect(() => results.toExport!(3)).toThrow(RangeError);
  });
});
