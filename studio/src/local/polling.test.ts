import { describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { Job, Run } from "../api/types";
import { pollDelayMs, pollJob, PollTimeoutError, pollRun, RemoteFailure, runReducer, type RunState } from "./polling";

const run = (status: Run["status"], extra: Partial<Run> = {}): Run => ({
  id: "r_1",
  model: "nitid1s",
  task: "detect",
  backend: "torch",
  device: "cpu",
  status,
  error_msg: null,
  item_count: 1,
  items: [],
  ...extra,
});

const noSleep = () => Promise.resolve();

describe("pollDelayMs", () => {
  it("grows geometrically and is capped", () => {
    expect([0, 1, 2, 10].map((n) => pollDelayMs(n))).toEqual([400, 600, 900, 2000]);
  });
});

describe("pollRun", () => {
  it("polls until done and reports each update", async () => {
    const seq = [run("pending"), run("running"), run("done")];
    const getRun = vi.fn(async () => seq.shift() as Run);
    const seen: string[] = [];
    const out = await pollRun({ getRun, sleep: noSleep, onUpdate: (r) => seen.push(r.status) });
    expect(out.status).toBe("done");
    expect(seen).toEqual(["pending", "running", "done"]);
    expect(getRun).toHaveBeenCalledTimes(3);
  });

  it("turns a failed run into RemoteFailure with error_msg", async () => {
    const getRun = async () => run("failed", { error_msg: "out of memory" });
    await expect(pollRun({ getRun, sleep: noSleep })).rejects.toThrow(new RemoteFailure("out of memory"));
  });

  it("times out", async () => {
    let t = 0;
    const err = await pollRun({
      getRun: async () => run("running"),
      sleep: noSleep,
      now: () => (t += 1000),
      timeoutMs: 3000,
    }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(PollTimeoutError);
  });

  it("tolerates a few network blips but not endless ones", async () => {
    let n = 0;
    const flaky = async () => {
      n += 1;
      if (n <= 2) throw new ApiError("network", "down");
      return run("done");
    };
    await expect(pollRun({ getRun: flaky, sleep: noSleep })).resolves.toMatchObject({ status: "done" });

    const dead = async (): Promise<Run> => {
      throw new ApiError("network", "down");
    };
    await expect(pollRun({ getRun: dead, sleep: noSleep, maxTransientErrors: 2 })).rejects.toMatchObject({
      kind: "network",
    });
  });

  it("does not retry HTTP errors", async () => {
    const getRun = vi.fn(async (): Promise<Run> => {
      throw new ApiError("http", "gone", 404, { code: "not_found" });
    });
    await expect(pollRun({ getRun, sleep: noSleep })).rejects.toMatchObject({ code: "not_found" });
    expect(getRun).toHaveBeenCalledTimes(1);
  });

  it("stops when aborted", async () => {
    const ctl = new AbortController();
    ctl.abort();
    await expect(pollRun({ getRun: async () => run("running"), signal: ctl.signal })).rejects.toMatchObject({
      name: "AbortError",
    });
  });
});

describe("pollJob", () => {
  it("reports progress and finishes on done", async () => {
    const seq: Job[] = [
      { status: "running", progress: 0.2 },
      { status: "running", progress: 0.8 },
      { status: "done", progress: 1 },
    ];
    const progress: number[] = [];
    await pollJob({ getJob: async () => seq.shift() as Job, sleep: noSleep, onUpdate: (j) => progress.push(j.progress) });
    expect(progress).toEqual([0.2, 0.8, 1]);
  });

  it("fails with the job error", async () => {
    await expect(
      pollJob({ getJob: async () => ({ status: "failed", progress: 0.1, error: "network unreachable" }), sleep: noSleep }),
    ).rejects.toThrow("network unreachable");
  });
});

describe("runReducer", () => {
  const idle: RunState = { phase: "idle" };

  it("walks idle -> uploading -> processing -> done", () => {
    let s = runReducer(idle, { type: "start", at: 100 });
    expect(s).toEqual({ phase: "uploading", startedAt: 100 });
    s = runReducer(s, { type: "created", runId: "r_1" });
    expect(s).toMatchObject({ phase: "processing", runId: "r_1", status: "pending" });
    s = runReducer(s, { type: "status", status: "running" });
    expect(s).toMatchObject({ status: "running" });
    s = runReducer(s, { type: "finished", run: run("done"), at: 1600 });
    expect(s).toMatchObject({ phase: "done", elapsedMs: 1500, runId: "r_1" });
  });

  it("ignores start while busy and events in the wrong phase", () => {
    const busy = runReducer(idle, { type: "start", at: 1 });
    expect(runReducer(busy, { type: "start", at: 2 })).toBe(busy);
    expect(runReducer(idle, { type: "created", runId: "x" })).toBe(idle);
    expect(runReducer(idle, { type: "finished", run: run("done"), at: 5 })).toBe(idle);
    expect(runReducer(idle, { type: "error", message: "x" })).toBe(idle);
  });

  it("moves to failed with code and hint, and reset returns to idle", () => {
    const busy = runReducer(idle, { type: "start", at: 1 });
    const failed = runReducer(busy, { type: "error", message: "bad", code: "invalid_params", hint: "check" });
    expect(failed).toEqual({ phase: "failed", message: "bad", code: "invalid_params", hint: "check" });
    expect(runReducer(failed, { type: "reset" })).toEqual(idle);
    expect(runReducer(failed, { type: "start", at: 9 })).toMatchObject({ phase: "uploading" });
  });
});
