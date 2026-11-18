/**
 * Page-side client of the inference worker: a promise API over the message
 * protocol in `src/workers/protocol.ts`.
 *
 *     const engine = createInferenceClient();
 *     const info = await engine.init({ modelUrl: "/api/v1/onnx/nitid1s?task=detect" });
 *     const { detections } = await engine.run({ kind: "bitmap", bitmap }, { conf: 0.5 });
 *     await engine.dispose();
 */
import type { InitRequest, InitResult, RunImage, WorkerRequest, WorkerResponse } from "../workers/protocol";
import { deserializeError, EngineError } from "./errors";
import type { RunOptions, RunResult } from "./session";

/** The subset of `Worker` the client needs (so tests can pass a fake). */
export interface WorkerLike {
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: { data: WorkerResponse }) => void): void;
  addEventListener(type: "error" | "messageerror", listener: (event: { message?: string }) => void): void;
  terminate(): void;
}

export type InitParams = Omit<InitRequest, "id" | "type">;

interface Pending {
  resolve(value: InitResult | RunResult | null): void;
  reject(error: unknown): void;
  onProgress?: (loaded: number, total: number | undefined) => void;
}

export class InferenceClient {
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private closed = false;

  constructor(private readonly worker: WorkerLike) {
    worker.addEventListener("message", (event) => this.onMessage(event.data));
    worker.addEventListener("error", (event) =>
      this.failAll(new EngineError("worker_error", `The inference worker crashed: ${event.message ?? "unknown error"}`)),
    );
    worker.addEventListener("messageerror", () =>
      this.failAll(new EngineError("worker_error", "The inference worker sent a message that could not be read")),
    );
  }

  init(params: InitParams, onProgress?: Pending["onProgress"]): Promise<InitResult> {
    const transfer = params.modelBytes ? [params.modelBytes] : [];
    return this.request((id) => ({ id, type: "init", ...params }), transfer, onProgress) as Promise<InitResult>;
  }

  run(image: RunImage, options?: RunOptions): Promise<RunResult> {
    const transfer: Transferable[] = image.kind === "rgba" ? [image.data] : [image.bitmap];
    return this.request(
      (id) => ({ id, type: "run", image, ...(options ? { options } : {}) }),
      transfer,
    ) as Promise<RunResult>;
  }

  async dispose(): Promise<void> {
    await this.request((id) => ({ id, type: "dispose" }), []);
  }

  /** Stops the worker; pending requests are rejected. */
  terminate(): void {
    this.closed = true;
    this.worker.terminate();
    this.failAll(new EngineError("worker_error", "The inference worker was terminated"));
  }

  private request(
    build: (id: number) => WorkerRequest,
    transfer: Transferable[],
    onProgress?: Pending["onProgress"],
  ): Promise<InitResult | RunResult | null> {
    if (this.closed) return Promise.reject(new EngineError("worker_error", "The inference worker was terminated"));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, ...(onProgress ? { onProgress } : {}) });
      try {
        this.worker.postMessage(build(id), transfer);
      } catch (error) {
        this.pending.delete(id);
        reject(new EngineError("worker_error", `Could not send the request to the worker: ${String(error)}`));
      }
    });
  }

  private onMessage(message: WorkerResponse): void {
    const entry = this.pending.get(message.id);
    if (!entry) return;
    if (message.type === "progress") {
      entry.onProgress?.(message.loaded, message.total);
      return;
    }
    this.pending.delete(message.id);
    if (message.type === "ok") entry.resolve(message.result);
    else entry.reject(deserializeError(message.error));
  }

  private failAll(error: EngineError): void {
    for (const entry of this.pending.values()) entry.reject(error);
    this.pending.clear();
  }
}

/** Starts the real worker (bundled by Vite, same-origin). */
export function createInferenceClient(): InferenceClient {
  const worker = new Worker(new URL("../workers/inference.worker.ts", import.meta.url), { type: "module" });
  return new InferenceClient(worker as unknown as WorkerLike);
}
