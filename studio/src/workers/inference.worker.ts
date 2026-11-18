/**
 * Web Worker entry point of the inference engine.
 *
 * Create it from the page with `createInferenceClient()` (`src/engine/client.ts`),
 * which uses `new Worker(new URL("../workers/inference.worker.ts", import.meta.url),
 * { type: "module" })` so Vite bundles it same-origin.
 */
import { loadModelBytes } from "../engine/model-cache";
import { EngineSession } from "../engine/session";
import { InferenceHandler } from "./inference-handler";
import { loadOrt } from "./ort-loader";
import type { WorkerRequest, WorkerResponse } from "./protocol";

interface WorkerScope {
  onmessage: ((event: { data: WorkerRequest }) => void) | null;
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
  crossOriginIsolated?: boolean;
  navigator: {
    hardwareConcurrency?: number;
    gpu?: { requestAdapter(): Promise<unknown | null> };
  };
}

const scope = self as unknown as WorkerScope;

const handler = new InferenceHandler({
  post: (message, transfer) => scope.postMessage(message, transfer ?? []),
  loadModel: (args) => loadModelBytes(args),
  createSession: (init) =>
    EngineSession.create(
      {
        loadOrt,
        env: {
          crossOriginIsolated: scope.crossOriginIsolated === true,
          hardwareConcurrency: scope.navigator.hardwareConcurrency ?? 1,
          hasWebGpu: async () => {
            const gpu = scope.navigator.gpu;
            return gpu !== undefined && (await gpu.requestAdapter()) !== null;
          },
        },
      },
      init,
    ),
});

scope.onmessage = (event) => {
  void handler.handle(event.data);
};
