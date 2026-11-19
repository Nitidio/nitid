import { describe, expect, it, vi } from "vitest";
import type { LocalApi } from "../api/local";
import type { Run } from "../api/types";
import type { BrowserModel } from "../browser/catalog";
import { BrowserRunner, type EngineLike } from "../browser/runner";
import { laneTitle, runLane, type LaneSpec } from "./lanes";

const run: Run = {
  id: "r1", model: "nitid1s", task: "detect", backend: "torch", device: "cpu", status: "done", error_msg: null, item_count: 1,
  items: [{ id: 0, name: "a.png", width: 100, height: 80, speed: { preprocess: 1, inference: 2, postprocess: 3 }, detections: [] }],
};
const file = new File([new Uint8Array([1])], "a.png", { type: "image/png" });
const serverSpec: LaneSpec = {
  kind: "server",
  model: { id: "nitid1s", task: "detect", size: "s", downloaded: true, source: "registry" },
  choice: { backend: "openvino", device: "GPU" },
};
const bModel: BrowserModel = { key: "detect:nitid1s", id: "nitid1s", task: "detect", label: "x", origin: "hosted", url: "/m.onnx" };

function api(over: Partial<LocalApi> = {}): LocalApi {
  return {
    createRun: vi.fn(async () => "r1"),
    getRun: vi.fn(async () => run),
    ...over,
  } as unknown as LocalApi;
}

function runnerWith(engine: Partial<EngineLike>) {
  const full: EngineLike = {
    init: vi.fn(async () => ({ backend: "wasm", numThreads: 1, modelSource: "network", modelCached: true, meta: { task: "detect", imgsz: 64 } }) as never),
    run: vi.fn(async () => ({ detections: [], imageSize: { width: 100, height: 80 }, timings: { preprocessMs: 1, inferenceMs: 2, postprocessMs: 3 } })),
    dispose: vi.fn(async () => undefined),
    terminate: vi.fn(),
    ...engine,
  };
  const runner = new BrowserRunner({
    createEngine: () => full, getToken: () => null, fetchServerMeta: async () => ({}),
    decode: async () => ({}) as ImageBitmap,
  });
  return { runner, full };
}

describe("runLane", () => {
  it("server lane: posts the run with the chosen backend/device and converts it", async () => {
    const a = api();
    const out = await runLane(serverSpec, file, { api: a, createRunner: () => runnerWith({}).runner });
    expect(a.createRun).toHaveBeenCalledWith(
      [file],
      { model: "nitid1s", task: "detect", backend: "openvino", device: "GPU", conf: 0.1 },
      undefined,
    );
    expect(out).toMatchObject({ status: "ok", loadMs: null, results: { source: "local", backend: "torch" } });
  });

  it("browser lane: loads, runs and always closes the worker", async () => {
    const { runner, full } = runnerWith({});
    const out = await runLane({ kind: "browser", model: bModel }, file, { api: api(), createRunner: () => runner });
    expect(out).toMatchObject({ status: "ok", results: { source: "browser", backend: "wasm" } });
    expect(full.terminate).toHaveBeenCalled();
  });

  it("a failing lane becomes an error outcome and the worker is still closed", async () => {
    const { runner, full } = runnerWith({ init: vi.fn(async () => { throw Object.assign(new Error("offline"), { name: "EngineError", code: "model_fetch_failed" }); }) });
    const out = await runLane({ kind: "browser", model: bModel }, file, { api: api(), createRunner: () => runner });
    expect(out.status).toBe("error");
    expect(full.terminate).toHaveBeenCalled();
  });

  it("a failing server lane does not throw", async () => {
    const out = await runLane(serverSpec, file, { api: api({ createRun: vi.fn(async () => { throw new Error("boom"); }) }), createRunner: () => runnerWith({}).runner });
    expect(out).toMatchObject({ status: "error", error: { message: "boom" } });
  });
});

describe("laneTitle", () => {
  it("names runtime and device", () => {
    expect(laneTitle(serverSpec)).toBe("nitid1s (detect), server: OpenVINO / GPU");
    expect(laneTitle({ kind: "browser", model: bModel })).toBe("nitid1s (detect), in the browser");
  });
});
