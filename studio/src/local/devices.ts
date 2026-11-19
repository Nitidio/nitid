import type { Backend, Devices } from "../api/types";

export interface DeviceOption {
  /** Value sent as `device` in `RunParams`. */
  value: string;
  label: string;
  available: boolean;
}

export interface BackendOption {
  backend: Backend;
  label: string;
  available: boolean;
  devices: DeviceOption[];
}

function openvinoLabel(name: string): string {
  const upper = name.toUpperCase();
  if (upper === "CPU") return "CPU";
  if (upper.startsWith("GPU")) return `${name} (integrated or Arc GPU)`;
  if (upper.startsWith("NPU")) return `${name} (neural processor)`;
  return name;
}

/**
 * Turn `GET /devices` into pickable options. Known device families the machine lacks are
 * still listed (disabled, "not detected") so the user sees what is missing; only reported
 * devices are enabled. OpenVINO device names are passed through exactly as reported.
 */
export function buildBackendOptions(devices: Devices): BackendOption[] {
  const torchDevices: DeviceOption[] = [
    { value: "cpu", label: "CPU", available: devices.torch.cpu },
    ...(devices.torch.cuda.length > 0
      ? devices.torch.cuda.map((c) => ({
          value: `cuda:${c.index}`,
          label: `CUDA ${c.index}: ${c.name}`,
          available: true,
        }))
      : [{ value: "cuda:0", label: "CUDA GPU (not detected)", available: false }]),
  ];

  const reported = devices.openvino.available ? devices.openvino.devices : [];
  const ovDevices: DeviceOption[] = reported.map((name) => ({
    value: name,
    label: openvinoLabel(name),
    available: true,
  }));
  for (const family of ["CPU", "GPU", "NPU"]) {
    if (!reported.some((n) => n.toUpperCase().startsWith(family))) {
      ovDevices.push({ value: family, label: `${family} (not detected)`, available: false });
    }
  }

  return [
    {
      backend: "torch",
      label: "PyTorch",
      available: torchDevices.some((d) => d.available),
      devices: torchDevices,
    },
    {
      backend: "openvino",
      label: "OpenVINO",
      available: ovDevices.some((d) => d.available),
      devices: ovDevices,
    },
  ];
}

export interface DeviceChoice {
  backend: Backend;
  device: string;
}

/** First available backend/device: PyTorch CPU if possible, else the first real device. */
export function defaultDeviceChoice(options: readonly BackendOption[]): DeviceChoice | null {
  for (const b of options) {
    const d = b.devices.find((x) => x.available);
    if (b.available && d) return { backend: b.backend, device: d.value };
  }
  return null;
}

/** Is the choice still valid for the (possibly refreshed) option list? */
export function isChoiceAvailable(options: readonly BackendOption[], choice: DeviceChoice): boolean {
  return (
    options.find((b) => b.backend === choice.backend)?.devices.some((d) => d.value === choice.device && d.available) ??
    false
  );
}
