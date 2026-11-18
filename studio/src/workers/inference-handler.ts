/**
 * Request handler of the inference worker, separated from the `self.onmessage`
 * glue so it can be tested with a fake session and no real Worker.
 *
 * Requests are processed strictly in arrival order (a queue), so `init` always
 * completes before a following `run`, and a new `init` disposes the previous
 * session first.
 */
import { EngineError, serializeError } from "../engine/errors";
import type { LoadedModel } from "../engine/model-cache";
import type { ImageInput, RunOptions, RunResult, SessionInit } from "../engine/session";
import type { InitResult, RunImage, WorkerRequest, WorkerResponse } from "./protocol";
import type { InitInfo } from "../engine/session";

/** What the handler needs from an engine session (satisfied by `EngineSession`). */
export interface SessionPort {
  readonly info: InitInfo;
  run(image: ImageInput, options?: RunOptions): Promise<RunResult>;
  dispose(): Promise<void>;
}

export interface HandlerDeps {
  post(message: WorkerResponse, transfer?: Transferable[]): void;
  /** Downloads (or reads from cache) the ONNX bytes. */
  loadModel(args: {
    url: string;
    sha256?: string;
    onProgress: (loaded: number, total: number | undefined) => void;
  }): Promise<LoadedModel>;
  createSession(init: SessionInit): Promise<SessionPort>;
}

export class InferenceHandler {
  private session: SessionPort | null = null;
  private queue: Promise<void> = Promise.resolve();

  constructor(private readonly deps: HandlerDeps) {}

  /** Enqueue a request; resolves when its response has been posted. */
  handle(request: WorkerRequest): Promise<void> {
    const task = this.queue.then(() => this.process(request));
    this.queue = task.catch(() => undefined);
    return task;
  }

  private async process(request: WorkerRequest): Promise<void> {
    try {
      switch (request.type) {
        case "init":
          return this.reply(request.id, await this.init(request));
        case "run":
          // `await` matters: without it a rejection would escape this try/catch.
          await this.runRequest(request.id, request.image, request.options);
          return;
        case "dispose":
          await this.disposeSession();
          return this.reply(request.id, null);
        default:
          throw new EngineError("worker_error", `Unknown request type "${String((request as { type?: unknown }).type)}"`);
      }
    } catch (error) {
      this.deps.post({ id: (request as { id: number }).id, type: "error", error: serializeError(error) });
    }
  }

  private reply(id: number, result: InitResult | RunResult | null, transfer: Transferable[] = []): void {
    this.deps.post({ id, type: "ok", result }, transfer);
  }

  private async disposeSession(): Promise<void> {
    const current = this.session;
    this.session = null;
    if (current) await current.dispose();
  }

  private async init(request: Extract<WorkerRequest, { type: "init" }>): Promise<InitResult> {
    if ((request.modelUrl === undefined) === (request.modelBytes === undefined)) {
      throw new EngineError("invalid_options", "init needs exactly one of modelUrl or modelBytes");
    }
    await this.disposeSession();

    let bytes: ArrayBuffer;
    let source: InitResult["modelSource"] = "bytes";
    let cached = false;
    if (request.modelBytes !== undefined) {
      bytes = request.modelBytes;
    } else {
      const loaded = await this.deps.loadModel({
        url: request.modelUrl!,
        ...(request.sha256 !== undefined ? { sha256: request.sha256 } : {}),
        onProgress: (loadedBytes, total) =>
          this.deps.post({ id: request.id, type: "progress", loaded: loadedBytes, total }),
      });
      bytes = loaded.bytes;
      source = loaded.source;
      cached = loaded.source === "cache" || loaded.cached;
    }

    const init: SessionInit = { modelBytes: bytes };
    if (request.meta !== undefined) init.meta = request.meta;
    if (request.backend !== undefined) init.backend = request.backend;
    if (request.wasmBase !== undefined) init.wasmBase = request.wasmBase;
    this.session = await this.deps.createSession(init);
    return { ...this.session.info, modelSource: source, modelCached: cached };
  }

  private async runRequest(id: number, image: RunImage, options?: RunOptions): Promise<void> {
    try {
      if (!this.session) throw new EngineError("session_not_ready", "Call init before run");
      const input: ImageInput =
        image.kind === "rgba"
          ? { kind: "rgba", data: new Uint8ClampedArray(image.data), width: image.width, height: image.height }
          : { kind: "bitmap", bitmap: image.bitmap };
      const result = await this.session.run(input, options);
      // Transfer the mask buffers instead of copying them.
      const transfer: Transferable[] = [];
      for (const detection of result.detections) {
        if (detection.mask) transfer.push(detection.mask.data.buffer);
      }
      this.reply(id, result, transfer);
    } finally {
      if (image.kind === "bitmap") (image.bitmap as { close?: () => void }).close?.();
    }
  }
}
