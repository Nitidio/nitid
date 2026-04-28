import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { getRun } from "../api/runs";
import { AnnotatedImage } from "../components/AnnotatedImage";
import { DetectionTable } from "../components/DetectionTable";
import { RunStatusBadge } from "../components/RunStatusBadge";
import type { RunItem } from "../types";

export function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const runId = Number(id);
  const [selected, setSelected] = useState<RunItem | null>(null);

  const { data: run, isLoading } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRun(runId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "pending" || s === "running" ? 2000 : false;
    },
  });

  if (isLoading || !run) return <div className="loading">Loading...</div>;

  return (
    <div className="page">
      <div className="page-header">
        <Link to="/runs">← Runs</Link>
        <h2>Run #{run.id}</h2>
        <RunStatusBadge status={run.status} />
      </div>

      <div className="run-meta">
        <span>Model: <strong>{run.model_name}</strong></span>
        <span>Conf: <strong>{run.conf}</strong></span>
        <span>Imgsz: <strong>{run.imgsz}</strong></span>
        {run.classes && <span>Classes: <strong>{run.classes.join(", ")}</strong></span>}
        <span>Input: <strong>{run.input_type}</strong></span>
        <span>Items: <strong>{run.item_count}</strong></span>
      </div>

      {run.status === "failed" && (
        <pre className="error-msg">{run.error_msg}</pre>
      )}

      {(run.status === "pending" || run.status === "running") && (
        <p className="loading">Processing... results will appear automatically.</p>
      )}

      {run.items.length > 0 && (
        <>
          <div className="item-grid">
            {run.items.map((item) => (
              item.result_snapshot_path ? (
                <div
                  key={item.id}
                  className={`item-card${selected?.id === item.id ? " selected" : ""}`}
                  onClick={() => setSelected(selected?.id === item.id ? null : item)}
                >
                  <AnnotatedImage runId={runId} filename={item.result_snapshot_path} />
                  <div className="item-card-footer">
                    {item.frame_idx != null && <span>Frame {item.frame_idx}</span>}
                    <span>{item.detections.length} detection{item.detections.length !== 1 ? "s" : ""}</span>
                  </div>
                </div>
              ) : null
            ))}
          </div>

          {selected && (
            <div className="detail-panel">
              <div className="detail-panel-header">
                <h3>
                  {selected.frame_idx != null ? `Frame ${selected.frame_idx}` : "Detections"}
                </h3>
                <button onClick={() => setSelected(null)}>✕</button>
              </div>
              <div className="detail-panel-body">
                <div className="detail-image">
                  <AnnotatedImage runId={runId} filename={selected.result_snapshot_path!} />
                </div>
                <div className="detail-table">
                  <DetectionTable detections={selected.detections} />
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
