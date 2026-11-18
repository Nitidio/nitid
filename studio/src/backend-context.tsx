import { createContext, useContext } from "react";
import type { BackendInfo } from "./api/backend";

/** `null` while the start-up probe is still running. */
export const BackendContext = createContext<BackendInfo | null>(null);

export function useBackend(): BackendInfo | null {
  return useContext(BackendContext);
}
