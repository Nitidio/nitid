import { createApiClient, type ApiClient } from "./client";
import type { Devices, Job, Model, Run, RunParams, Task } from "./types";

/** Uploads of several 25 MB images take longer than a JSON call. */
export const UPLOAD_TIMEOUT_MS = 120_000;

/** Typed wrappers around the local endpoints of the contract. */
export interface LocalApi {
  listDevices(signal?: AbortSignal): Promise<Devices>;
  listModels(signal?: AbortSignal): Promise<Model[]>;
  /** `POST /models/{id}/download?task=`; resolves with the job id. */
  startDownload(id: string, task: Task, signal?: AbortSignal): Promise<string>;
  getJob(jobId: string, signal?: AbortSignal): Promise<Job>;
  /** `POST /runs` (multipart: `files` + JSON `params`); resolves with the run id. */
  createRun(files: File[], params: RunParams, signal?: AbortSignal): Promise<string>;
  getRun(runId: string, signal?: AbortSignal): Promise<Run>;
  /** Original image of an item; fetched with the bearer token, so it cannot be an `<img src>`. */
  getItemImage(runId: string, itemId: number, signal?: AbortSignal): Promise<Blob>;
}

export function createLocalApi(client: ApiClient = createApiClient()): LocalApi {
  const sig = (signal?: AbortSignal) => (signal ? { signal } : {});
  return {
    listDevices: (signal) => client.request<Devices>("/devices", sig(signal)),
    listModels: (signal) => client.request<Model[]>("/models", sig(signal)),
    async startDownload(id, task, signal) {
      const body = await client.request<{ job_id: string }>(
        `/models/${encodeURIComponent(id)}/download?task=${encodeURIComponent(task)}`,
        { method: "POST", ...sig(signal) },
      );
      return body.job_id;
    },
    getJob: (jobId, signal) => client.request<Job>(`/jobs/${encodeURIComponent(jobId)}`, sig(signal)),
    async createRun(files, params, signal) {
      const form = new FormData();
      for (const file of files) form.append("files", file, file.name);
      // A plain string part (not a Blob): a Blob would be sent as a file part with a filename,
      // which a `Form(str)` field does not accept.
      form.append("params", JSON.stringify(params));
      const body = await client.request<{ id: string }>("/runs", {
        method: "POST",
        formData: form,
        timeoutMs: UPLOAD_TIMEOUT_MS,
        ...sig(signal),
      });
      return body.id;
    },
    getRun: (runId, signal) => client.request<Run>(`/runs/${encodeURIComponent(runId)}`, sig(signal)),
    getItemImage: (runId, itemId, signal) =>
      client.request<Blob>(`/runs/${encodeURIComponent(runId)}/items/${itemId}/image`, {
        responseType: "blob",
        timeoutMs: 30_000,
        ...sig(signal),
      }),
  };
}
