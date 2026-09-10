"""
Best-effort hardware telemetry for a run item.

True %busy for Intel GPU/NPU is not reliably available on this stack: the
`intel_vpu` (NPU) driver exposes no engine busy-time counters via
`/proc/<pid>/fdinfo` at all, and the `i915` (GPU) driver's counters read 0
even immediately after real inference work in local testing. Rather than
fabricate a number, this module reports what *is* real and available from
the same fdinfo interface: this process's Intel GPU/NPU memory footprint.
"""

from __future__ import annotations

import os
from pathlib import Path

_DRIVER_BY_DEVICE = {"npu": "intel_vpu", "gpu": "i915"}


def sample_cpu_percent() -> float | None:
    """System-wide CPU utilization since the previous call. None if `psutil`
    isn't installed (it's an optional dependency of the `web` extra)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.cpu_percent(interval=None)


def sample_device_memory_kib(
    backend: str, device: str | None, *, proc_root: Path = Path("/proc")
) -> int | None:
    """This process's resident memory (KiB) on the OpenVINO GPU/NPU device
    currently selected, read from this process's own `/proc/<pid>/fdinfo`
    DRM client entries. None for backend="torch", CPU, or when unavailable."""
    if backend != "openvino" or not device:
        return None
    driver = _DRIVER_BY_DEVICE.get(device.lower())
    if driver is None:
        return None

    fd_dir = proc_root / str(os.getpid()) / "fd"
    fdinfo_dir = proc_root / str(os.getpid()) / "fdinfo"
    try:
        fd_names = [entry.name for entry in fd_dir.iterdir()]
    except OSError:
        return None

    total_kib = 0
    found = False
    for fd_name in fd_names:
        try:
            text = (fdinfo_dir / fd_name).read_text()
        except OSError:
            continue
        if f"drm-driver:\t{driver}" not in text:
            continue
        found = True
        for line in text.splitlines():
            if not line.startswith("drm-resident"):
                continue
            _, _, value = line.partition(":")
            digits = value.strip().split()[0]
            if digits.isdigit():
                total_kib += int(digits)
    return total_kib if found else None
