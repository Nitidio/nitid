import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";
import type { LocalApi } from "../api/local";
import type { RunParams } from "../api/types";
import { describeError, isAbort } from "./errors";
import { pollRun, runReducer, type RunState } from "./polling";
import { fromRun, type RunResults } from "./results-model";

const INITIAL: RunState = { phase: "idle" };

/** Drive one run: upload, poll, and expose the state machine plus the converted results. */
export function useRun(api: LocalApi, now: () => number = Date.now) {
  const [state, dispatch] = useReducer(runReducer, INITIAL);
  const ctl = useRef<AbortController | null>(null);

  useEffect(() => () => ctl.current?.abort(), []);

  const start = useCallback(
    async (files: File[], params: RunParams) => {
      ctl.current?.abort();
      const c = new AbortController();
      ctl.current = c;
      dispatch({ type: "start", at: now() });
      try {
        const runId = await api.createRun(files, params, c.signal);
        dispatch({ type: "created", runId });
        const run = await pollRun({
          getRun: () => api.getRun(runId, c.signal),
          signal: c.signal,
          onUpdate: (r) => dispatch({ type: "status", status: r.status }),
        });
        dispatch({ type: "finished", run, at: now() });
      } catch (e) {
        if (isAbort(e)) return;
        const d = describeError(e);
        dispatch({
          type: "error",
          message: d.message,
          ...(d.code ? { code: d.code } : {}),
          ...(d.hint ? { hint: d.hint } : {}),
        });
      }
    },
    [api, now],
  );

  const reset = useCallback(() => {
    ctl.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  const results = useMemo<RunResults | null>(
    () => (state.phase === "done" ? fromRun(state.run, state.elapsedMs) : null),
    [state],
  );

  return { state, start, reset, results };
}
