import { useId } from "react";
import type { Model } from "../api/types";
import type { DownloadState } from "./use-downloads";
import { ErrorNotice } from "./ErrorNotice";
import { findModel, groupByTask, modelKey, modelLabel, TASK_LABEL } from "./models";

export interface ModelPickerProps {
  models: readonly Model[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  downloads: Record<string, DownloadState>;
  onDownload: (model: Model) => void;
  disabled?: boolean;
}

export function ModelPicker({ models, selectedKey, onSelect, downloads, onDownload, disabled }: ModelPickerProps) {
  const id = useId();
  const selected = selectedKey ? findModel(models, selectedKey) : undefined;
  const dl = selected ? downloads[modelKey(selected)] : undefined;

  if (models.length === 0) {
    return <p>No models are available on this server.</p>;
  }

  return (
    <div className="field">
      <label htmlFor={`${id}-model`}>Model and task</label>
      <select
        id={`${id}-model`}
        value={selectedKey ?? ""}
        disabled={disabled}
        aria-describedby={`${id}-state`}
        onChange={(e) => onSelect(e.target.value)}
      >
        {groupByTask(models).map((g) => (
          <optgroup key={g.task} label={TASK_LABEL[g.task]}>
            {g.models.map((m) => (
              <option key={modelKey(m)} value={modelKey(m)}>
                {modelLabel(m)}
                {m.downloaded ? "" : " - not downloaded"}
              </option>
            ))}
          </optgroup>
        ))}
      </select>

      <div id={`${id}-state`} className="field-state">
        {selected && selected.downloaded && <span>Ready on this computer.</span>}
        {selected && !selected.downloaded && dl?.status !== "running" && (
          <>
            <span>Not downloaded yet.</span>{" "}
            <button type="button" className="secondary" onClick={() => onDownload(selected)} disabled={disabled}>
              Download {selected.id}
            </button>
          </>
        )}
        {dl?.status === "running" && (
          <label className="progress-row">
            <span>Downloading {selected?.id}</span>
            <progress max={1} value={dl.progress} aria-valuetext={`${Math.round(dl.progress * 100)}%`} />
            <span aria-hidden="true">{Math.round(dl.progress * 100)}%</span>
          </label>
        )}
      </div>
      {dl?.status === "failed" && <ErrorNotice error={dl.error} title="Download failed." />}
    </div>
  );
}
