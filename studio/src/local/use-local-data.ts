import { useCallback, useEffect, useState } from "react";
import type { LocalApi } from "../api/local";
import type { Devices, Model } from "../api/types";
import { describeError, isAbort, type DescribedError } from "./errors";

export type Loadable<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: DescribedError };

/** Load `GET /devices` and `GET /models`; `reloadModels` refreshes only the model list. */
export function useLocalData(api: LocalApi) {
  const [devices, setDevices] = useState<Loadable<Devices>>({ status: "loading" });
  const [models, setModels] = useState<Loadable<Model[]>>({ status: "loading" });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const ctl = new AbortController();
    api
      .listDevices(ctl.signal)
      .then((data) => setDevices({ status: "ready", data }))
      .catch((e: unknown) => {
        if (!isAbort(e)) setDevices({ status: "error", error: describeError(e) });
      });
    api
      .listModels(ctl.signal)
      .then((data) => setModels({ status: "ready", data }))
      .catch((e: unknown) => {
        if (!isAbort(e)) setModels({ status: "error", error: describeError(e) });
      });
    return () => ctl.abort();
  }, [api, nonce]);

  const reloadModels = useCallback(async () => {
    try {
      setModels({ status: "ready", data: await api.listModels() });
    } catch (e) {
      setModels({ status: "error", error: describeError(e) });
    }
  }, [api]);

  const reloadAll = useCallback(() => {
    setDevices({ status: "loading" });
    setModels({ status: "loading" });
    setNonce((n) => n + 1);
  }, []);

  return { devices, models, reloadModels, reloadAll };
}
