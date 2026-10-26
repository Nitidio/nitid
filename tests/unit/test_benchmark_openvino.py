import os
import subprocess
import sys
import time

import numpy as np
import pytest

from tools.benchmark_openvino import (
    RunResult,
    Telemetry,
    background_cpu_cores,
    benchmark_app_commands,
    is_reliable,
    parse_matrix,
    parse_weights,
    render_markdown,
    summarize_latency,
    to_coco_detections,
)


def test_summarize_latency_reports_mean_percentiles_and_fps():
    stats = summarize_latency([10.0, 20.0, 30.0, 40.0], warmup=5)

    assert stats.mean_ms == 25.0
    assert stats.p50_ms == 25.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 40.0
    assert stats.fps == pytest.approx(40.0)
    assert (stats.runs, stats.warmup) == (4, 5)


def test_summarize_latency_rejects_empty_runs():
    with pytest.raises(ValueError):
        summarize_latency([], warmup=0)


def test_parse_matrix_normalizes_case():
    assert parse_matrix(["gpu:fp16", "CPU:FP32"]) == [("GPU", "FP16"), ("CPU", "FP32")]


@pytest.mark.parametrize("entry", ["GPU", "GPU:INT4", ":FP16"])
def test_parse_matrix_rejects_invalid_entries(entry):
    with pytest.raises(ValueError):
        parse_matrix([entry])


def test_parse_weights():
    assert parse_weights(["model1n=weights/n.pth"]) == {"model1n": "weights/n.pth"}
    with pytest.raises(ValueError):
        parse_weights(["model1n"])


def test_to_coco_detections_rescales_clamps_and_maps_categories():
    detections = to_coco_detections(
        image_id=7,
        labels=np.array([0, 1]),
        boxes=np.array([[0.0, 0.0, 320.0, 640.0], [600.0, 10.0, 700.0, 20.0]]),
        scores=np.array([0.9, 0.5]),
        original_size=(1280, 320),
        imgsz=640,
        label_to_cat_id={0: 1, 1: 3},
    )

    assert detections[0] == {
        "image_id": 7,
        "category_id": 1,
        "bbox": [0.0, 0.0, 640.0, 320.0],
        "score": pytest.approx(0.9),
    }
    # x2 = 700 * 2 = 1400 is clamped to the image width.
    assert detections[1]["category_id"] == 3
    assert detections[1]["bbox"][2] == pytest.approx(1280 - 1200)


def _environment():
    return {
        "cpu": "Test CPU",
        "memory_gb": 16.0,
        "devices": {"CPU": {"FULL_DEVICE_NAME": "Test CPU"}, "GPU": {"FULL_DEVICE_NAME": "iGPU"}},
        "driver_packages": {},
        "os": "Linux",
        "openvino": "2026.2.1",
        "nitid": "0.9.2",
        "torch": "2.5.1",
        "on_ac_power": True,
        "platform_profile": "performance",
        "cpu_governor": "performance",
    }


def test_render_markdown_has_one_row_per_model_and_reports_failures():
    ok = RunResult("model1s", "GPU", "FP16", effective_precision="f16")
    ok.latency = summarize_latency([5.0, 7.0], warmup=1)
    ok.coco_map50_95, ok.val_images = 50.1, 5000
    failed = RunResult("model1s", "NPU", "FP16", status="failed", error="compile error")

    report = render_markdown(_environment(), [ok, failed], imgsz=640)

    assert "| `model1s` | 50.7 | 50.1 | 6.00 ms | — | failed | 3.49 ms |" in report
    assert "- `model1s` NPU FP16: compile error" in report
    assert "mAP evaluated on 5000 images" in report


def test_benchmark_app_commands_cover_every_combination(tmp_path):
    xml = tmp_path / "model1s_640.xml"
    commands = benchmark_app_commands({"model1s": xml}, [("CPU", "FP32"), ("GPU", "FP16")])

    assert commands == [
        f"benchmark_app -m {xml} -d CPU -hint latency -infer_precision f32 -niter 1000",
        f"benchmark_app -m {xml} -d GPU -hint latency -infer_precision f16 -niter 1000",
    ]


def test_background_cpu_cores_subtracts_own_usage():
    # 400 busy jiffies system-wide, 100 of them ours, over 2 s at 100 Hz: 1.5 cores elsewhere.
    assert background_cpu_cores(400, 100, 2.0, 100) == pytest.approx(1.5)
    assert background_cpu_cores(100, 150, 1.0, 100) == 0.0  # sampling skew never goes negative
    assert background_cpu_cores(100, 0, 0.0, 100) == 0.0


@pytest.mark.parametrize(
    ("mean", "expected"), [(0.2, True), (0.5, True), (3.0, False), (None, None)]
)
def test_is_reliable(mean, expected):
    assert is_reliable({"background_cpu_mean_cores": mean}, 0.5) is expected


def test_render_markdown_flags_unreliable_runs():
    noisy = RunResult("model1s", "CPU", "FP32", effective_precision="f32", reliable=False)
    noisy.latency = summarize_latency([150.0, 170.0], warmup=1)
    noisy.telemetry = {"background_cpu_mean_cores": 9.4, "background_cpu_max_cores": 14.0}

    report = render_markdown(_environment(), [noisy], imgsz=640)

    assert "160.00 ms ⚠" in report
    assert "## Unreliable runs" in report
    assert "9.4 cores on average, 14.0 at peak" in report


@pytest.mark.skipif(not os.path.exists("/proc/stat"), reason="needs Linux /proc")
def test_telemetry_detects_a_competing_process():
    busy = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        with Telemetry(interval_s=0.2) as telemetry:
            time.sleep(1.5)
    finally:
        busy.kill()
        busy.wait()

    assert telemetry.result["background_cpu_mean_cores"] > 0.5
