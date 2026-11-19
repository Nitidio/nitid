import { useEffect, useState } from "react";
import { createApiClient } from "../api/client";
import { createLocalApi, type LocalApi } from "../api/local";
import type { BackendInfo } from "../api/backend";
import { hostedModelsBaseUrl } from "../config";
import { isAbort, type DescribedError } from "../local/errors";
import { describeBrowserError } from "./errors";
import { fromServerModels, loadHostedCatalog, type BrowserModel } from "./catalog";

export type BrowserModels =
  | { status: "loading" }
  /** Hosted site without a configured catalogue. */
  | { status: "unconfigured" }
  | { status: "ready"; models: BrowserModel[]; origin: "server" | "hosted" }
  | { status: "error"; error: DescribedError };

export interface BrowserModelsDeps {
  api?: LocalApi;
  /** Defaults to the build-time constant. */
  hostedBaseUrl?: string | null;
  fetchImpl?: typeof fetch;
}

/**
 * Local deployment: the server's downloaded models. Hosted: the catalogue under the configured
 * base URL, or "unconfigured" (a clear empty state; nothing is requested).
 */
export function useBrowserModels(backend: BackendInfo | null, deps: BrowserModelsDeps = {}): BrowserModels {
  const [state, setState] = useState<BrowserModels>({ status: "loading" });
  const { api: apiDep, fetchImpl } = deps;
  const base = deps.hostedBaseUrl === undefined ? hostedModelsBaseUrl() : deps.hostedBaseUrl;
  const mode = backend?.mode;

  useEffect(() => {
    if (!mode) return;
    const ctl = new AbortController();
    const fail = (e: unknown) => {
      if (!isAbort(e)) setState({ status: "error", error: describeBrowserError(e) });
    };
    if (mode === "local") {
      const api = apiDep ?? createLocalApi(createApiClient());
      api
        .listModels(ctl.signal)
        .then((models) => setState({ status: "ready", models: fromServerModels(models), origin: "server" }))
        .catch(fail);
    } else if (base !== null) {
      loadHostedCatalog(base, fetchImpl)
        .then((models) => !ctl.signal.aborted && setState({ status: "ready", models, origin: "hosted" }))
        .catch(fail);
    }
    return () => ctl.abort();
  }, [mode, apiDep, base, fetchImpl]);

  if (!mode) return { status: "loading" };
  // Derived, not stored: nothing is requested when no catalogue is configured.
  if (mode === "hosted" && base === null) return { status: "unconfigured" };
  return state;
}
