import { useCallback, useEffect, useRef, useState } from "react";
import type { LocalApi } from "../api/local";
import type { Model } from "../api/types";
import { describeError, isAbort, type DescribedError } from "./errors";
import { modelKey } from "./models";
import { pollJob } from "./polling";

export type DownloadState =
  | { status: "running"; progress: number }
  | { status: "failed"; error: DescribedError };

/** Track registry downloads: `POST /models/{id}/download` then `GET /jobs/{id}` until done. */
export function useDownloads(api: LocalApi, onDone: () => void | Promise<void>) {
  const [downloads, setDownloads] = useState<Record<string, DownloadState>>({});
  const ctl = useRef(new AbortController());
  const onDoneRef = useRef(onDone);

  useEffect(() => {
    onDoneRef.current = onDone;
  });
  useEffect(() => {
    const c = new AbortController();
    ctl.current = c;
    return () => c.abort();
  }, []);

  const start = useCallback(
    async (model: Model) => {
      const key = modelKey(model);
      const signal = ctl.current.signal;
      setDownloads((d) => ({ ...d, [key]: { status: "running", progress: 0 } }));
      try {
        const jobId = await api.startDownload(model.id, model.task, signal);
        await pollJob({
          getJob: () => api.getJob(jobId, signal),
          signal,
          onUpdate: (job) =>
            setDownloads((d) => ({ ...d, [key]: { status: "running", progress: job.progress } })),
        });
        setDownloads((d) => {
          const rest = { ...d };
          delete rest[key];
          return rest;
        });
        await onDoneRef.current();
      } catch (e) {
        if (isAbort(e)) return;
        setDownloads((d) => ({ ...d, [key]: { status: "failed", error: describeError(e) } }));
      }
    },
    [api],
  );

  return { downloads, start };
}
