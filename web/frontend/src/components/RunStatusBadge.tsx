import type { RunStatus } from "../types";

const colors: Record<RunStatus, string> = {
  pending: "#f59e0b",
  running: "#3b82f6",
  done: "#10b981",
  failed: "#ef4444",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <span
      style={{
        background: colors[status],
        color: "#fff",
        padding: "2px 8px",
        borderRadius: 4,
        fontSize: 12,
        fontWeight: 600,
        textTransform: "uppercase",
      }}
    >
      {status}
    </span>
  );
}
