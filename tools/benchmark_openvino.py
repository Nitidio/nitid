"""Benchmark nitid detection models with OpenVINO Runtime on CPU, GPU and NPU.

The latency method follows the one behind the D-FINE paper's T4 numbers (the
"Speed T4 TRT10 FP16" column in the README), so results on Intel hardware can
be read next to them:

- the exported model includes the postprocessor, batch 1, 640x640 input;
- 1000 warm-up runs on a constant input, then one timed run per COCO val2017
  image; preprocessing and input upload happen outside the timed region;
- the headline figure is the mean latency, reported with p50/p90/p99.

Accuracy is COCO mAP50-95 of the same OpenVINO IR on the same device and
precision, so a precision or device that loses accuracy shows up next to its
speed. Every run writes a JSON report stamped with the hardware, drivers,
library versions and IR hashes, plus a Markdown summary.

COCO val2017 layout expected under ``--coco``::

    val2017/*.jpg
    annotations/instances_val2017.json

Example::

    uv sync --extra openvino --extra train
    uv run python -m tools.benchmark_openvino --coco datasets/coco --output runs/benchmark/ptl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODELS = ["model1s", "model1m", "model1l", "model1x"]
DEFAULT_MATRIX = ["CPU:FP32", "GPU:FP16", "NPU:FP16"]
# Mean CPU, in cores, that other processes may use during a run before it is flagged unreliable.
DEFAULT_MAX_BACKGROUND_CPU = 1.0
PRECISION_HINTS = {"FP32": "f32", "FP16": "f16"}

# Published D-FINE reference figures, as listed in the README.
REFERENCE_T4_TRT10_FP16_MS = {
    "model1n": 2.12,
    "model1s": 3.49,
    "model1m": 5.62,
    "model1l": 8.07,
    "model1x": 12.89,
}
REFERENCE_COCO_MAP = {
    "model1n": 42.8,
    "model1s": 50.7,
    "model1m": 55.1,
    "model1l": 57.3,
    "model1x": 59.3,
}


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps: float
    runs: int
    warmup: int


@dataclass
class RunResult:
    model: str
    device: str
    precision: str
    status: str = "ok"
    error: str | None = None
    effective_precision: str | None = None
    compile_s: float | None = None
    first_inference_ms: float | None = None
    latency: LatencyStats | None = None
    coco_map50_95: float | None = None
    coco_map50: float | None = None
    val_images: int | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    reliable: bool | None = None


# ── Statistics ────────────────────────────────────────────────────────────────


def summarize_latency(times_ms: list[float], warmup: int) -> LatencyStats:
    """Summarize per-run latencies in milliseconds."""
    if not times_ms:
        raise ValueError("No timed runs to summarize")
    arr = np.asarray(times_ms, dtype=np.float64)
    mean = float(arr.mean())
    return LatencyStats(
        mean_ms=mean,
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        fps=1000.0 / mean,
        runs=len(times_ms),
        warmup=warmup,
    )


def parse_matrix(entries: list[str]) -> list[tuple[str, str]]:
    """Parse ``DEVICE:PRECISION`` entries such as ``GPU:FP16``."""
    combos: list[tuple[str, str]] = []
    for entry in entries:
        device, sep, precision = entry.partition(":")
        precision = precision.upper()
        if not sep or not device or precision not in PRECISION_HINTS:
            raise ValueError(
                f"Invalid matrix entry {entry!r}: expected DEVICE:PRECISION with "
                f"PRECISION in {sorted(PRECISION_HINTS)}"
            )
        combos.append((device.upper(), precision))
    return combos


def parse_weights(entries: list[str]) -> dict[str, str]:
    """Parse ``MODEL=PATH`` checkpoint overrides."""
    weights: dict[str, str] = {}
    for entry in entries:
        name, sep, path = entry.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"Invalid --weights entry {entry!r}: expected MODEL=PATH")
        weights[name] = path
    return weights


# ── Data ──────────────────────────────────────────────────────────────────────


def preprocess(image_path: Path, imgsz: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image the way nitid does (PIL resize, then ToTensor) as NCHW float32."""
    import torchvision.transforms as T
    from PIL import Image

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        size = rgb.size  # (width, height)
        tensor = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])(rgb)
    return tensor.unsqueeze(0).numpy(), size


def to_coco_detections(
    image_id: int,
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    original_size: tuple[int, int],
    imgsz: int,
    label_to_cat_id: dict[int, int],
) -> list[dict[str, Any]]:
    """Convert postprocessed outputs at ``imgsz`` to COCO results in original pixels."""
    width, height = original_size
    scaled = boxes.astype(np.float64).copy()
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]] * width / imgsz, 0, width)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]] * height / imgsz, 0, height)
    detections = []
    for label, (x1, y1, x2, y2), score in zip(labels.tolist(), scaled.tolist(), scores.tolist()):
        detections.append(
            {
                "image_id": image_id,
                "category_id": label_to_cat_id.get(int(label), int(label) + 1),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return detections


# ── Environment and telemetry ─────────────────────────────────────────────────


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_model() -> str:
    text = _read("/proc/cpuinfo") or ""
    for line in text.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_properties(core: Any, device: str) -> dict[str, str]:
    """Collect the identifying properties a device exposes."""
    wanted = (
        "FULL_DEVICE_NAME",
        "DEVICE_ARCHITECTURE",
        "DEVICE_TYPE",
        "OPTIMIZATION_CAPABILITIES",
        "GPU_DEVICE_TOTAL_MEM_SIZE",
        "NPU_DRIVER_VERSION",
        "NPU_COMPILER_VERSION",
        "DRIVER_VERSION",
    )
    try:
        supported = set(core.get_property(device, "SUPPORTED_PROPERTIES"))
    except Exception:
        supported = set(wanted)
    props: dict[str, str] = {}
    for name in wanted:
        if name in supported:
            try:
                props[name] = str(core.get_property(device, name))
            except Exception:
                continue
    return props


def collect_environment(core: Any) -> dict[str, Any]:
    """Stamp the run with hardware, OS, drivers and library versions."""
    import openvino as ov
    import torch

    import dfine

    mem_kb = None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal"):
            mem_kb = int(line.split()[1])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(),
        "nitid": dfine.__version__,
        "openvino": ov.get_version(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": _cpu_model(),
        "memory_gb": round(mem_kb / 1024 / 1024, 1) if mem_kb else None,
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "on_ac_power": _ac_online(),
        "devices": {d: device_properties(core, d) for d in core.available_devices},
        "driver_packages": driver_packages(),
    }


# Debian/Ubuntu packages that carry the Intel GPU and NPU user-space drivers.
DRIVER_PACKAGES = (
    "intel-opencl-icd",
    "libze-intel-gpu1",
    "intel-level-zero-gpu",
    "libze1",
    "intel-driver-compiler-npu",
    "intel-level-zero-npu",
)


def driver_packages() -> dict[str, str]:
    """Installed versions of the Intel GPU/NPU driver packages, where dpkg is available."""
    try:
        completed = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Package} ${Version}\\n", *DRIVER_PACKAGES],
            capture_output=True,
            text=True,
        )
    except OSError:
        return {}
    versions = {}
    for line in completed.stdout.splitlines():
        name, _, version = line.partition(" ")
        if version:
            versions[name] = version
    return versions


def _ac_online() -> bool | None:
    for supply in Path("/sys/class/power_supply").glob("*"):
        if (_read(str(supply / "type")) or "") == "Mains":
            return _read(str(supply / "online")) == "1"
    return None


def _package_temp_c() -> float | None:
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (_read(str(zone / "type")) or "") == "x86_pkg_temp":
            raw = _read(str(zone / "temp"))
            return int(raw) / 1000 if raw else None
    return None


def _rapl_energy_uj() -> int | None:
    raw = _read("/sys/class/powercap/intel-rapl:0/energy_uj")
    return int(raw) if raw else None


def _cpu_busy_jiffies() -> int | None:
    """System-wide busy CPU time from the aggregate ``cpu`` line of /proc/stat."""
    line = (_read("/proc/stat") or "").splitlines()[:1]
    if not line or not line[0].startswith("cpu "):
        return None
    values = [int(v) for v in line[0].split()[1:9]]  # user .. steal
    idle, iowait = values[3], values[4]
    return sum(values) - idle - iowait


def _own_jiffies() -> int | None:
    """CPU time (user + system, all threads) of this process from /proc/self/stat."""
    raw = _read("/proc/self/stat")
    if not raw:
        return None
    fields = raw.rsplit(")", 1)[1].split()  # skip "pid (comm)", which may contain spaces
    return int(fields[11]) + int(fields[12])  # utime, stime


def background_cpu_cores(busy_delta: int, own_delta: int, elapsed_s: float, hz: int) -> float:
    """CPU used by other processes over an interval, in cores (1.0 = one busy core)."""
    if elapsed_s <= 0 or hz <= 0:
        return 0.0
    return max(0, busy_delta - own_delta) / hz / elapsed_s


class Telemetry:
    """Sample temperature, RAPL package energy and other processes' CPU while a block runs.

    The CPU used by other processes is the system's busy time minus this process's own, so a
    run can be flagged when something else was competing for the machine. Readings the platform
    does not expose (RAPL often needs root) stay ``None``.
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.temps: list[float] = []
        self.background: list[float] = []
        self._hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._energy_start: int | None = None
        self._t0 = 0.0
        self.result: dict[str, Any] = {}

    def _sample(self) -> None:
        last = (_cpu_busy_jiffies(), _own_jiffies(), time.perf_counter())
        while not self._stop.wait(self.interval_s):
            temp = _package_temp_c()
            if temp is not None:
                self.temps.append(temp)
            now = (_cpu_busy_jiffies(), _own_jiffies(), time.perf_counter())
            if None not in (last[0], last[1], now[0], now[1]):
                self.background.append(
                    background_cpu_cores(
                        now[0] - last[0],  # type: ignore[operator]
                        now[1] - last[1],  # type: ignore[operator]
                        now[2] - last[2],
                        self._hz,
                    )
                )
            last = now

    def __enter__(self) -> Telemetry:
        self._energy_start = _rapl_energy_uj()
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        elapsed = time.perf_counter() - self._t0
        energy_end = _rapl_energy_uj()
        power = None
        if self._energy_start is not None and energy_end is not None and elapsed > 0:
            delta = energy_end - self._energy_start
            if delta >= 0:  # the counter wraps; drop the reading if it did
                power = round(delta / 1e6 / elapsed, 2)
        self.result = {
            "package_power_w": power,
            "package_temp_start_c": self.temps[0] if self.temps else None,
            "package_temp_max_c": max(self.temps) if self.temps else None,
            "package_temp_end_c": self.temps[-1] if self.temps else None,
            "background_cpu_mean_cores": (
                round(statistics.fmean(self.background), 2) if self.background else None
            ),
            "background_cpu_max_cores": round(max(self.background), 2) if self.background else None,
            "load_avg_1m_end": float((_read("/proc/loadavg") or "nan").split()[0]),
            "duration_s": round(elapsed, 2),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_reliable(telemetry: dict[str, Any], max_background_cpu: float) -> bool | None:
    """Whether other processes stayed under ``max_background_cpu`` cores; ``None`` if unknown."""
    mean = telemetry.get("background_cpu_mean_cores")
    if mean is None:
        return None
    return bool(mean <= max_background_cpu)


# ── Benchmark steps ───────────────────────────────────────────────────────────


def export_ir(model: str, weights: str | None, imgsz: int, ir_dir: Path) -> Path:
    """Export ``model`` to an FP32 OpenVINO IR with the postprocessor, reusing a cached one."""
    xml = ir_dir / f"{model}_{imgsz}.xml"
    if xml.is_file() and xml.with_suffix(".bin").is_file():
        return xml
    from nitid import NITID

    detector = NITID(weights or model, task="detect", device="cpu", verbose=False)
    ir_dir.mkdir(parents=True, exist_ok=True)
    detector.export(format="openvino", imgsz=imgsz, half=False, output=xml, verbose=False)
    return xml


def compile_for(core: Any, xml: Path, device: str, precision: str) -> tuple[Any, str | None]:
    """Compile with the latency hint and the requested inference precision."""
    config: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
    try:
        supported = set(core.get_property(device, "SUPPORTED_PROPERTIES"))
    except Exception:
        supported = set()
    if "INFERENCE_PRECISION_HINT" in supported:
        config["INFERENCE_PRECISION_HINT"] = PRECISION_HINTS[precision]
    compiled = core.compile_model(core.read_model(xml), device, config)
    try:
        hint = compiled.get_property("INFERENCE_PRECISION_HINT")
        effective = hint.to_string() if hasattr(hint, "to_string") else str(hint)
    except Exception:
        effective = None
    return compiled, effective


def measure_latency(
    compiled: Any, images: list[Path], imgsz: int, runs: int, warmup: int
) -> tuple[list[float], float]:
    """Return per-image latencies (ms) and the first-inference time (ms)."""
    import openvino as ov

    request = compiled.create_infer_request()
    constant = np.ones((1, 3, imgsz, imgsz), dtype=np.float32)
    request.set_input_tensor(ov.Tensor(constant))
    t0 = time.perf_counter()
    request.infer()
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        request.infer()

    times: list[float] = []
    for index in range(runs):
        array, _ = preprocess(images[index % len(images)], imgsz)
        request.set_input_tensor(ov.Tensor(array))
        t0 = time.perf_counter()
        request.infer()
        times.append((time.perf_counter() - t0) * 1e3)
    return times, first_ms


def evaluate_coco(
    compiled: Any, coco: Any, image_dir: Path, image_ids: list[int], imgsz: int
) -> tuple[float, float]:
    """COCO bbox mAP50-95 and mAP50 of the compiled IR over ``image_ids``."""
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval

    label_to_cat_id = {i: c for i, c in enumerate(sorted(coco.cats))}
    request = compiled.create_infer_request()
    names = [output.get_any_name() for output in compiled.outputs]
    detections: list[dict[str, Any]] = []
    for image_id in image_ids:
        info = coco.imgs[image_id]
        array, size = preprocess(image_dir / info["file_name"], imgsz)
        request.infer({0: array})
        out = {name: request.get_tensor(name).data for name in names}
        detections.extend(
            to_coco_detections(
                image_id,
                out["labels"][0],
                out["boxes"][0],
                out["scores"][0],
                size,
                imgsz,
                label_to_cat_id,
            )
        )
    if not detections:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        predictions = coco.loadRes(detections)
        evaluator = COCOeval(coco, predictions, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0]) * 100, float(evaluator.stats[1]) * 100


# ── Reporting ─────────────────────────────────────────────────────────────────


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(environment: dict[str, Any], results: list[RunResult], imgsz: int) -> str:
    """Render a README-style table: one row per model, one column pair per device."""
    combos = list(dict.fromkeys((r.device, r.precision) for r in results))
    models = list(dict.fromkeys(r.model for r in results))
    by_key = {(r.model, r.device, r.precision): r for r in results}

    header = ["Model", "COCO mAP<sup>50-95</sup> (paper)"]
    for device, precision in combos:
        header += [f"{device} {precision} mAP", f"{device} {precision} latency"]
    header.append("T4 TRT10 FP16 (paper)")
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join([" :-- "] + [" :-: "] * (len(header) - 1)) + "|",
    ]
    for model in models:
        row = [f"`{model}`", _fmt(REFERENCE_COCO_MAP.get(model), 1)]
        for device, precision in combos:
            result = by_key.get((model, device, precision))
            if result is None or result.status != "ok" or result.latency is None:
                row += ["—", "failed" if result is not None else "—"]
                continue
            flag = " ⚠" if result.reliable is False else ""
            row += [_fmt(result.coco_map50_95, 1), f"{result.latency.mean_ms:.2f} ms{flag}"]
        reference = REFERENCE_T4_TRT10_FP16_MS.get(model)
        row.append(f"{reference:.2f} ms" if reference is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    devices = environment["devices"]
    env_lines = [
        f"- CPU: {environment['cpu']}, {environment['memory_gb']} GB RAM",
        *(
            f"- {name}: {props.get('FULL_DEVICE_NAME', name)}"
            for name, props in devices.items()
            if name != "CPU"
        ),
        "- Drivers: "
        + (
            ", ".join(f"{k} {v}" for k, v in environment["driver_packages"].items())
            or "not detected"
        ),
        f"- OS: {environment['os']}",
        f"- OpenVINO {environment['openvino']}, nitid {environment['nitid']}, "
        f"torch {environment['torch']}",
        f"- Power: AC={environment['on_ac_power']}, "
        f"platform profile={environment['platform_profile']}, "
        f"governor={environment['cpu_governor']}",
    ]
    failures = [r for r in results if r.status != "ok"]
    failure_lines = [f"- `{r.model}` {r.device} {r.precision}: {r.error}" for r in failures]
    detail = [
        "| Model | Device | Precision | Effective | mean | p50 | p90 | p99 | FPS | "
        "compile | first run | power | temp max | other CPU |",
        "| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for r in results:
        if r.latency is None:
            continue
        lat = r.latency
        detail.append(
            f"| `{r.model}` | {r.device} | {r.precision} | {r.effective_precision or '—'} | "
            f"{lat.mean_ms:.2f} | {lat.p50_ms:.2f} | {lat.p90_ms:.2f} | {lat.p99_ms:.2f} | "
            f"{lat.fps:.1f} | {_fmt(r.compile_s)} s | {_fmt(r.first_inference_ms, 1)} ms | "
            f"{_fmt(r.telemetry.get('package_power_w'), 1)} W | "
            f"{_fmt(r.telemetry.get('package_temp_max_c'), 0)} °C | "
            f"{_fmt(r.telemetry.get('background_cpu_mean_cores'))} cores |"
        )
    val_note = next((r.val_images for r in results if r.val_images), None)
    sections = [
        "# OpenVINO benchmark",
        f"_COCO val2017 at {imgsz}x{imgsz}, batch 1, postprocessor included. Latency is the "
        "mean over one run per image after warm-up, as in the D-FINE paper; the T4 column is "
        f"the paper's TensorRT 10 FP16 figure. mAP evaluated on {val_note or 'no'} images._",
        "",
        *lines,
        "",
        "## Environment",
        "",
        *env_lines,
        "",
        "## Details",
        "",
        *detail,
    ]
    unreliable = [r for r in results if r.reliable is False]
    if unreliable:
        sections += [
            "",
            "## Unreliable runs",
            "",
            "Other processes competed for the machine during these runs (marked ⚠ above). "
            "Rerun them on an idle machine before using the numbers.",
            "",
            *(
                f"- `{r.model}` {r.device} {r.precision}: other processes used "
                f"{r.telemetry.get('background_cpu_mean_cores')} cores on average, "
                f"{r.telemetry.get('background_cpu_max_cores')} at peak"
                for r in unreliable
            ),
        ]
    if failure_lines:
        sections += ["", "## Failures", "", *failure_lines]
    return "\n".join(sections) + "\n"


def benchmark_app_commands(xmls: dict[str, Path], combos: list[tuple[str, str]]) -> list[str]:
    """Equivalent ``benchmark_app`` invocations, for an independent cross-check.

    ``benchmark_app`` repeats one input, which flatters D-FINE on GPUs: its
    data-dependent sampling is faster when every run reads the same memory.
    Expect lower latencies there than in this script's per-image runs.
    """
    return [
        f"benchmark_app -m {xml} -d {device} -hint latency "
        f"-infer_precision {PRECISION_HINTS[precision]} -niter 1000"
        for xml in xmls.values()
        for device, precision in combos
    ]


# ── Main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coco", type=Path, required=True, help="COCO root with val2017/")
    parser.add_argument("--annotations", type=Path, help="defaults to <coco>/annotations/...")
    parser.add_argument("--images", type=Path, help="image directory, defaults to <coco>/val2017")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--weights", nargs="*", default=[], help="MODEL=PATH checkpoint overrides")
    parser.add_argument(
        "--matrix", nargs="+", default=DEFAULT_MATRIX, help="DEVICE:PRECISION entries"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=1000, help="timed runs (one per image)")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--val-images", type=int, help="limit mAP evaluation to N images")
    parser.add_argument("--skip-val", action="store_true", help="latency only")
    parser.add_argument("--cooldown", type=float, default=10.0, help="seconds between runs")
    parser.add_argument(
        "--max-background-cpu",
        type=float,
        default=DEFAULT_MAX_BACKGROUND_CPU,
        help="flag a run when other processes use more CPU than this, in cores, on average",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark/openvino"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    combos = parse_matrix(args.matrix)
    weights = parse_weights(args.weights)

    import openvino as ov
    from pycocotools.coco import COCO

    annotations = args.annotations or args.coco / "annotations" / "instances_val2017.json"
    image_dir = args.images or args.coco / "val2017"
    with open(os.devnull, "w") as devnull, _redirect(devnull):
        coco = COCO(str(annotations))
    image_ids = sorted(coco.imgs)
    latency_images = [image_dir / coco.imgs[i]["file_name"] for i in image_ids]
    val_ids = image_ids[: args.val_images] if args.val_images else image_ids

    core = ov.Core()
    available = set(core.available_devices)
    environment = collect_environment(core)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"OpenVINO {environment['openvino']} on {environment['cpu']}; devices: {sorted(available)}"
    )

    results: list[RunResult] = []
    xmls: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for model in args.models:
        try:
            xmls[model] = export_ir(model, weights.get(model), args.imgsz, args.output / "ir")
            hashes[model] = sha256(xmls[model].with_suffix(".bin"))
        except Exception as exc:  # noqa: BLE001 — record and continue with other models
            for device, precision in combos:
                results.append(RunResult(model, device, precision, "failed", f"export: {exc}"))
            print(f"[{model}] export failed: {exc}")
            continue
        for device, precision in combos:
            result = RunResult(model, device, precision)
            results.append(result)
            if device not in available:
                result.status, result.error = "skipped", f"{device} not available"
                continue
            label = f"[{model} {device} {precision}]"
            try:
                t0 = time.perf_counter()
                compiled, result.effective_precision = compile_for(
                    core, xmls[model], device, precision
                )
                result.compile_s = time.perf_counter() - t0
                with Telemetry() as telemetry:
                    times, result.first_inference_ms = measure_latency(
                        compiled, latency_images, args.imgsz, args.runs, args.warmup
                    )
                result.telemetry = telemetry.result
                result.reliable = is_reliable(result.telemetry, args.max_background_cpu)
                result.latency = summarize_latency(times, args.warmup)
                print(
                    f"{label} mean {result.latency.mean_ms:.2f} ms, "
                    f"p99 {result.latency.p99_ms:.2f} ms"
                )
                if result.reliable is False:
                    print(
                        f"{label} UNRELIABLE: other processes used "
                        f"{result.telemetry['background_cpu_mean_cores']} CPU cores on average"
                    )
                if not args.skip_val:
                    result.coco_map50_95, result.coco_map50 = evaluate_coco(
                        compiled, coco, image_dir, val_ids, args.imgsz
                    )
                    result.val_images = len(val_ids)
                    print(f"{label} mAP50-95 {result.coco_map50_95:.1f}")
                del compiled
            except Exception as exc:  # noqa: BLE001 — unsupported combos are a result
                result.status, result.error = "failed", f"{type(exc).__name__}: {exc}"
                print(f"{label} failed: {result.error}")
            _write_reports(args.output, environment, results, xmls, hashes, combos, args.imgsz)
            time.sleep(args.cooldown)

    _write_reports(args.output, environment, results, xmls, hashes, combos, args.imgsz)
    print(f"Report: {args.output / 'results.md'}")
    return 0 if all(r.status == "ok" for r in results) else 1


def _redirect(stream: Any) -> Any:
    import contextlib

    return contextlib.redirect_stdout(stream)


def _write_reports(
    output: Path,
    environment: dict[str, Any],
    results: list[RunResult],
    xmls: dict[str, Path],
    hashes: dict[str, str],
    combos: list[tuple[str, str]],
    imgsz: int,
) -> None:
    """Rewrite the reports after every run so a crash keeps what was measured."""
    payload = {
        "environment": environment,
        "models": {m: {"ir": str(x), "weights_sha256": hashes.get(m)} for m, x in xmls.items()},
        "results": [asdict(r) for r in results],
        "statistics_note": "latency in milliseconds; mAP in percent",
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    (output / "results.md").write_text(render_markdown(environment, results, imgsz))
    (output / "benchmark_app.sh").write_text(
        "#!/bin/sh\n" + "\n".join(benchmark_app_commands(xmls, combos)) + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
