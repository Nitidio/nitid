import { useId } from "react";
import type { Backend } from "../api/types";
import type { BackendOption, DeviceChoice } from "./devices";

export interface DevicePickerProps {
  options: readonly BackendOption[];
  choice: DeviceChoice;
  onChange: (choice: DeviceChoice) => void;
  disabled?: boolean;
}

export function DevicePicker({ options, choice, onChange, disabled }: DevicePickerProps) {
  const id = useId();
  const current = options.find((b) => b.backend === choice.backend);

  function changeBackend(backend: Backend) {
    const next = options.find((b) => b.backend === backend);
    const device = next?.devices.find((d) => d.available)?.value;
    if (device) onChange({ backend, device });
  }

  return (
    <div className="field-row">
      <div className="field">
        <label htmlFor={`${id}-backend`}>Backend</label>
        <select
          id={`${id}-backend`}
          value={choice.backend}
          disabled={disabled}
          onChange={(e) => changeBackend(e.target.value as Backend)}
        >
          {options.map((b) => (
            <option key={b.backend} value={b.backend} disabled={!b.available}>
              {b.label}
              {b.available ? "" : " (not available)"}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor={`${id}-device`}>Device</label>
        <select
          id={`${id}-device`}
          value={choice.device}
          disabled={disabled}
          onChange={(e) => onChange({ backend: choice.backend, device: e.target.value })}
        >
          {current?.devices.map((d) => (
            <option key={d.value} value={d.value} disabled={!d.available}>
              {d.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
