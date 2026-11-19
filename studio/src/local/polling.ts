import { ApiError } from "../api/client";
import type { Job, Run } from "../api/types";

/** Delay before poll number `attempt` (0-based): 400 ms growing 1.5x up to 2 s. */
export function pollDelayMs(attempt: number, base = 400, max = 2000): number {
  return Math.min(max, Math.round(base * 1.5 ** Math.max(0, attempt)));
}

export class PollTimeoutError extends Error {
  constructor(what: string, ms: number) {
    super(`${what} did not finish within ${Math.round(ms / 1000)} s`);
    this.name = "PollTimeoutError";
  }
}

/** The server finished the run/job with status "failed". */
export class RemoteFailure extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RemoteFailure";
  }
}

export interface PollOptions<T> {
  fetch: () => Promise<T>;
  /** Returns the final value when finished, "continue" to poll again; throws to fail. */
  step: (value: T) => "continue" | "done";
  onUpdate?: (value: T) => void;
  signal?: AbortSignal;
  timeoutMs?: number;
  what?: string;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  now?: () => number;
  /** Consecutive network/timeouts tolerated before giving up. */
  maxTransientErrors?: number;
}

export function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Generic polling loop: backoff, overall timeout, abort, tolerance for blips. */
export async function pollUntil<T>(opts: PollOptions<T>): Promise<T> {
  const sleep = opts.sleep ?? abortableSleep;
  const now = opts.now ?? Date.now;
  const timeoutMs = opts.timeoutMs ?? 10 * 60_000;
  const maxTransient = opts.maxTransientErrors ?? 3;
  const start = now();
  let transient = 0;

  for (let attempt = 0; ; attempt += 1) {
    if (opts.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    try {
      const value = await opts.fetch();
      transient = 0;
      opts.onUpdate?.(value);
      if (opts.step(value) === "done") return value;
    } catch (err) {
      const retryable = err instanceof ApiError && (err.kind === "network" || err.kind === "timeout");
      if (!retryable || ++transient > maxTransient) throw err;
    }
    if (now() - start >= timeoutMs) throw new PollTimeoutError(opts.what ?? "The operation", timeoutMs);
    await sleep(pollDelayMs(attempt), opts.signal);
  }
}

export type RunPollOptions = Omit<PollOptions<Run>, "step" | "fetch"> & {
  getRun: () => Promise<Run>;
};

/** Poll `GET /runs/{id}` until `done`; `failed` becomes a `RemoteFailure` carrying `error_msg`. */
export function pollRun({ getRun, ...rest }: RunPollOptions): Promise<Run> {
  return pollUntil<Run>({
    ...rest,
    fetch: getRun,
    what: rest.what ?? "The run",
    step(run) {
      if (run.status === "failed") throw new RemoteFailure(run.error_msg || "The run failed.");
      return run.status === "done" ? "done" : "continue";
    },
  });
}

export type JobPollOptions = Omit<PollOptions<Job>, "step" | "fetch"> & {
  getJob: () => Promise<Job>;
};

/** Poll `GET /jobs/{id}` until `done`; reports progress through `onUpdate`. */
export function pollJob({ getJob, ...rest }: JobPollOptions): Promise<Job> {
  return pollUntil<Job>({
    ...rest,
    fetch: getJob,
    what: rest.what ?? "The download",
    step(job) {
      if (job.status === "failed") throw new RemoteFailure(job.error || "The download failed.");
      return job.status === "done" ? "done" : "continue";
    },
  });
}

/* ---------------- UI state machine for one run ---------------- */

export type RunState =
  | { phase: "idle" }
  | { phase: "uploading"; startedAt: number }
  | { phase: "processing"; runId: string; startedAt: number; status: Run["status"] }
  | { phase: "done"; runId: string; run: Run; elapsedMs: number }
  | { phase: "failed"; message: string; code?: string; hint?: string };

export type RunAction =
  | { type: "start"; at: number }
  | { type: "created"; runId: string }
  | { type: "status"; status: Run["status"] }
  | { type: "finished"; run: Run; at: number }
  | { type: "error"; message: string; code?: string; hint?: string }
  | { type: "reset" };

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "start":
      return state.phase === "uploading" || state.phase === "processing"
        ? state
        : { phase: "uploading", startedAt: action.at };
    case "created":
      return state.phase === "uploading"
        ? { phase: "processing", runId: action.runId, startedAt: state.startedAt, status: "pending" }
        : state;
    case "status":
      return state.phase === "processing" ? { ...state, status: action.status } : state;
    case "finished":
      return state.phase === "processing"
        ? { phase: "done", runId: state.runId, run: action.run, elapsedMs: action.at - state.startedAt }
        : state;
    case "error":
      return state.phase === "uploading" || state.phase === "processing"
        ? {
            phase: "failed",
            message: action.message,
            ...(action.code ? { code: action.code } : {}),
            ...(action.hint ? { hint: action.hint } : {}),
          }
        : state;
    case "reset":
      return { phase: "idle" };
  }
}
