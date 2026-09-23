import type { RunStatus } from "../types";

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return <span className={`status-badge status-${status}`}>{status}</span>;
}
