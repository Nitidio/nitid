"""Unit tests for web.api.services.telemetry."""

from __future__ import annotations

import os

from web.api.services.telemetry import sample_cpu_percent, sample_device_memory_kib


def _write_fake_proc(tmp_path, fd_contents: dict[str, str]):
    pid_dir = tmp_path / str(os.getpid())
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "fdinfo").mkdir(parents=True)
    for name, text in fd_contents.items():
        (pid_dir / "fd" / name).write_text("")
        (pid_dir / "fdinfo" / name).write_text(text)
    return tmp_path


def test_sample_device_memory_kib_sums_resident_regions_for_matching_driver(tmp_path):
    proc_root = _write_fake_proc(
        tmp_path,
        {
            "3": (
                "pos:\t0\nflags:\t02100002\nmnt_id:\t30\nino:\t1119\n"
                "drm-driver:\ti915\n"
                "drm-total-system0:\t8476 KiB\n"
                "drm-resident-system0:\t24 KiB\n"
                "drm-resident-stolen-local0:\t100 KiB\n"
            ),
            "4": (
                "pos:\t0\nflags:\t02500002\nmnt_id:\t30\nino:\t719\n"
                "drm-driver:\tintel_vpu\n"
                "drm-total-memory:\t65168 KiB\n"
                "drm-resident-memory:\t65168 KiB\n"
            ),
        },
    )

    assert sample_device_memory_kib("openvino", "GPU", proc_root=proc_root) == 124
    assert sample_device_memory_kib("openvino", "NPU", proc_root=proc_root) == 65168
    assert sample_device_memory_kib("openvino", "gpu", proc_root=proc_root) == 124


def test_sample_device_memory_kib_returns_none_when_not_openvino_gpu_npu(tmp_path):
    proc_root = _write_fake_proc(tmp_path, {})
    assert sample_device_memory_kib("torch", "cpu", proc_root=proc_root) is None
    assert sample_device_memory_kib("openvino", None, proc_root=proc_root) is None
    assert sample_device_memory_kib("openvino", "CPU", proc_root=proc_root) is None


def test_sample_device_memory_kib_returns_none_when_driver_not_found(tmp_path):
    proc_root = _write_fake_proc(
        tmp_path, {"3": "drm-driver:\ti915\ndrm-resident-system0:\t1 KiB\n"}
    )
    assert sample_device_memory_kib("openvino", "NPU", proc_root=proc_root) is None


def test_sample_cpu_percent_returns_a_float():
    value = sample_cpu_percent()
    assert value is None or isinstance(value, float)
