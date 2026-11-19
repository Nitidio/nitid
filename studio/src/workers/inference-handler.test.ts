import { describe, expect, it, vi } from "vitest";
import { InferenceClient, type WorkerLike } from "../engine/client";
import { EngineError } from "../engine/errors";
import type { LoadedModel } from "../engine/model-cache";
import type { ImageInput, InitInfo, RunResult } from "../engine/session";
import { loadFixture, synthImage } from "../engine/test-utils";
import { InferenceHandler, type HandlerDeps, type SessionPort } from "./inference-handler";
import type { WorkerRequest, WorkerResponse } from "./protocol";

const fixture = loadFixture("segment");
const info: InitInfo = { backend: "wasm", numThreads: 1, meta: fixture.meta };

function result(masks = 1): RunResult {
  return {
    detections: Array.from({ length: masks }, (_, i) => ({
      box: { x1: 0, y1: 0, x2: 2, y2: 2 },
      confidence: 0.9,
      class_id: i,
      name: "cat",
      has_mask: true,
      mask: { x: 0, y: 0, width: 2, height: 2, data: new Uint8Array([1, 0, 0, 1]) },
    })),
    imageSize: { width: 2, height: 2 },
    timings: { preprocessMs: 1, inferenceMs: 2, postprocessMs: 3 },
  };
}

function setup(overrides: Partial<HandlerDeps> = {}) {
  const posted: { message: WorkerResponse; transfer: Transferable[] }[] = [];
  const sessions: (SessionPort & { run: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn> })[] = [];
  const deps: HandlerDeps = {
    post: (message, transfer = []) => void posted.push({ message, transfer }),
    loadModel: vi.fn(
      async (args): Promise<LoadedModel> => {
        args.onProgress(5, 10);
        args.onProgress(10, 10);
        return { bytes: new ArrayBuffer(10), source: "network", sha256: undefined, cached: true };
      },
    ),
    createSession: vi.fn(async () => {
      const session = {
        info,
        run: vi.fn<(image: ImageInput) => Promise<RunResult>>(async () => result(2)),
        dispose: vi.fn(async () => undefined),
      };
      sessions.push(session);
      return session;
    }),
    ...overrides,
  };
  return { handler: new InferenceHandler(deps), deps, posted, sessions };
}

const rgba = () => ({ kind: "rgba" as const, data: new ArrayBuffer(16), width: 2, height: 2 });

describe("inference handler", () => {
  it("init from a URL: loads (with progress), creates the session and replies with the info", async () => {
    const { handler, deps, posted } = setup();
    await handler.handle({ id: 1, type: "init", modelUrl: "/api/v1/onnx/nitid1s", sha256: "a".repeat(64), backend: "auto", wasmBase: "/ort/" });
    expect(posted.map((p) => p.message)).toEqual([
      { id: 1, type: "progress", loaded: 5, total: 10 },
      { id: 1, type: "progress", loaded: 10, total: 10 },
      { id: 1, type: "ok", result: { ...info, modelSource: "network", modelCached: true } },
    ]);
    expect(deps.loadModel).toHaveBeenCalledWith(expect.objectContaining({ url: "/api/v1/onnx/nitid1s", sha256: "a".repeat(64) }));
    expect(deps.createSession).toHaveBeenCalledWith({ modelBytes: expect.any(ArrayBuffer), backend: "auto", wasmBase: "/ort/" });
  });

  it("forwards request headers (the bearer token) to the loader", async () => {
    const { handler, deps } = setup();
    await handler.handle({ id: 1, type: "init", modelUrl: "/api/v1/onnx/m", headers: { Authorization: "Bearer t" } });
    expect(deps.loadModel).toHaveBeenCalledWith(expect.objectContaining({ headers: { Authorization: "Bearer t" } }));
  });

  it("init from bytes (a user file) skips the loader and passes the meta through", async () => {
    const { handler, deps, posted } = setup();
    await handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    expect(deps.loadModel).not.toHaveBeenCalled();
    expect(deps.createSession).toHaveBeenCalledWith({ modelBytes: expect.any(ArrayBuffer), meta: fixture.meta });
    expect(posted[0]!.message).toMatchObject({ type: "ok", result: { modelSource: "bytes", modelCached: false } });
  });

  it("init needs exactly one of modelUrl / modelBytes", async () => {
    const { handler, posted } = setup();
    await handler.handle({ id: 1, type: "init" });
    await handler.handle({ id: 2, type: "init", modelUrl: "/m", modelBytes: new ArrayBuffer(1) });
    expect(posted.map((p) => p.message)).toEqual([
      { id: 1, type: "error", error: { code: "invalid_options", message: expect.any(String) } },
      { id: 2, type: "error", error: { code: "invalid_options", message: expect.any(String) } },
    ]);
  });

  it("run before init is session_not_ready, and closes a transferred bitmap", async () => {
    const { handler, posted } = setup();
    const close = vi.fn();
    await handler.handle({ id: 1, type: "run", image: { kind: "bitmap", bitmap: { close } as unknown as ImageBitmap } });
    expect(posted[0]!.message).toMatchObject({ id: 1, type: "error", error: { code: "session_not_ready" } });
    expect(close).toHaveBeenCalledOnce();
  });

  it("run forwards the image and options, and transfers the mask buffers", async () => {
    const { handler, posted, sessions } = setup();
    await handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    await handler.handle({ id: 2, type: "run", image: rgba(), options: { conf: 0.3, classes: [1] } });
    const call = sessions[0]!.run.mock.calls[0]!;
    expect(call[0]).toMatchObject({ kind: "rgba", width: 2, height: 2 });
    expect(call[0].data).toBeInstanceOf(Uint8ClampedArray);
    expect(call[1]).toEqual({ conf: 0.3, classes: [1] });
    const reply = posted.at(-1)!;
    expect(reply.message).toMatchObject({ id: 2, type: "ok" });
    expect(reply.transfer.length).toBe(2); // one ArrayBuffer per mask
  });

  it("a failing run replies with a serialised EngineError; the handler stays usable", async () => {
    const { handler, posted, sessions } = setup();
    await handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    sessions[0]!.run.mockRejectedValueOnce(new EngineError("image_too_large", "too big"));
    await handler.handle({ id: 2, type: "run", image: rgba() });
    await handler.handle({ id: 3, type: "run", image: rgba() });
    expect(posted.at(-2)!.message).toEqual({ id: 2, type: "error", error: { code: "image_too_large", message: "too big" } });
    expect(posted.at(-1)!.message).toMatchObject({ id: 3, type: "ok" });
  });

  it("unknown errors become worker_error", async () => {
    const { handler, posted } = setup({ createSession: vi.fn(async () => Promise.reject(new TypeError("boom"))) });
    await handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    expect(posted[0]!.message).toEqual({ id: 1, type: "error", error: { code: "worker_error", message: "boom" } });
  });

  it("a second init disposes the previous session first; dispose releases it", async () => {
    const { handler, sessions, posted } = setup();
    await handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    await handler.handle({ id: 2, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    expect(sessions[0]!.dispose).toHaveBeenCalledOnce();
    await handler.handle({ id: 3, type: "dispose" });
    expect(sessions[1]!.dispose).toHaveBeenCalledOnce();
    expect(posted.at(-1)!.message).toEqual({ id: 3, type: "ok", result: null });
    await handler.handle({ id: 4, type: "run", image: rgba() });
    expect(posted.at(-1)!.message).toMatchObject({ type: "error", error: { code: "session_not_ready" } });
  });

  it("processes requests in arrival order even when init is slow", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    const { handler, posted } = setup({
      createSession: vi.fn(async () => {
        await gate;
        return { info, run: vi.fn(async () => result(0)), dispose: vi.fn(async () => undefined) };
      }),
    });
    const init = handler.handle({ id: 1, type: "init", modelBytes: new ArrayBuffer(4), meta: fixture.meta });
    const run = handler.handle({ id: 2, type: "run", image: rgba() });
    release();
    await Promise.all([init, run]);
    expect(posted.map((p) => `${p.message.id}:${p.message.type}`)).toEqual(["1:ok", "2:ok"]);
  });

  it("rejects an unknown request type", async () => {
    const { handler, posted } = setup();
    await handler.handle({ id: 9, type: "nope" } as unknown as WorkerRequest);
    expect(posted[0]!.message).toMatchObject({ id: 9, type: "error", error: { code: "worker_error" } });
  });
});

/** A fake Worker that runs the real handler in-process (messages are structured-cloned like postMessage). */
function loopbackWorker(handler: InferenceHandler, posted: { message: WorkerResponse }[]) {
  const listeners: Record<string, ((event: never) => void)[]> = {};
  const worker: WorkerLike = {
    postMessage: (message: WorkerRequest) => void handler.handle(structuredClone(message)),
    addEventListener: ((type: string, listener: (event: never) => void) => void (listeners[type] ??= []).push(listener)) as WorkerLike["addEventListener"],
    terminate: vi.fn(),
  };
  const emit = (type: string, event: unknown) => listeners[type]?.forEach((l) => l(event as never));
  return { worker, emit, posted };
}

describe("inference client over the protocol", () => {
  function build() {
    const ctx = setup();
    const loop = loopbackWorker(ctx.handler, ctx.posted);
    // Wire the handler's output back to the client's "message" listeners.
    const post = (ctx.handler as unknown as { deps: HandlerDeps }).deps.post;
    (ctx.handler as unknown as { deps: HandlerDeps }).deps.post = (message, transfer) => {
      post(message, transfer);
      loop.emit("message", { data: message });
    };
    return { ...ctx, ...loop, client: new InferenceClient(loop.worker) };
  }

  it("init -> run -> dispose round trip with progress callbacks", async () => {
    const { client } = build();
    const progress = vi.fn();
    const initInfo = await client.init({ modelUrl: "/m", backend: "wasm" }, progress);
    expect(initInfo.backend).toBe("wasm");
    expect(progress.mock.calls).toEqual([[5, 10], [10, 10]]);
    const out = await client.run({ kind: "rgba", ...(({ data, width, height }) => ({ data: data.buffer as ArrayBuffer, width, height }))(synthImage(2, 2)) });
    expect(out.detections.length).toBe(2);
    await expect(client.dispose()).resolves.toBeUndefined();
  });

  it("rejects with a typed EngineError carrying the worker's code", async () => {
    const { client } = build();
    await expect(client.run({ kind: "rgba", data: new ArrayBuffer(16), width: 2, height: 2 })).rejects.toMatchObject({
      name: "EngineError",
      code: "session_not_ready",
    });
  });

  it("a crashed worker rejects every pending request with worker_error", async () => {
    const posted: WorkerRequest[] = [];
    const listeners: Record<string, ((event: never) => void)[]> = {};
    const worker: WorkerLike = {
      postMessage: (message) => void posted.push(message),
      addEventListener: ((type: string, l: (event: never) => void) => void (listeners[type] ??= []).push(l)) as WorkerLike["addEventListener"],
      terminate: vi.fn(),
    };
    const client = new InferenceClient(worker);
    const a = client.init({ modelUrl: "/m" });
    const b = client.dispose();
    listeners.error![0]!({ message: "out of memory" } as never);
    await expect(a).rejects.toMatchObject({ code: "worker_error", message: expect.stringContaining("out of memory") });
    await expect(b).rejects.toMatchObject({ code: "worker_error" });
  });

  it("terminate stops the worker and later calls fail fast", async () => {
    const worker: WorkerLike = { postMessage: vi.fn(), addEventListener: vi.fn(), terminate: vi.fn() };
    const client = new InferenceClient(worker);
    const pending = client.dispose();
    client.terminate();
    await expect(pending).rejects.toMatchObject({ code: "worker_error" });
    expect(worker.terminate).toHaveBeenCalledOnce();
    await expect(client.dispose()).rejects.toMatchObject({ code: "worker_error" });
  });

  it("reports postMessage failures (e.g. a detached buffer) as worker_error", async () => {
    const worker: WorkerLike = {
      postMessage: () => {
        throw new DOMException("detached", "DataCloneError");
      },
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    };
    await expect(new InferenceClient(worker).dispose()).rejects.toMatchObject({ code: "worker_error" });
  });

  it("transfers the RGBA buffer and the model bytes instead of copying them", async () => {
    const calls: Transferable[][] = [];
    const worker: WorkerLike = {
      postMessage: (_m, transfer = []) => void calls.push(transfer),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    };
    const client = new InferenceClient(worker);
    const bytes = new ArrayBuffer(8);
    const pixels = new ArrayBuffer(16);
    void client.init({ modelBytes: bytes }).catch(() => undefined);
    void client.run({ kind: "rgba", data: pixels, width: 2, height: 2 }).catch(() => undefined);
    expect(calls).toEqual([[bytes], [pixels]]);
  });
});
