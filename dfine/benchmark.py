"""Benchmark exported nitid models on the current machine."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from dfine.utils.runs import resolve_run_dir, write_run_metadata

SUPPORTED_FORMATS = ("onnx", "torchscript", "tensorrt")


@dataclass
class BenchmarkResult:
    format: str
    latency_ms: float | None
    throughput_batches_s: float | None
    fps: float | None
    status: str = "ok"


def _measure(
    run: Callable[[], Any], synchronize: Callable[[], None], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        run()
    synchronize()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        run()
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _summarize(format_name: str, samples: list[float], batch: int) -> BenchmarkResult:
    latency = statistics.mean(samples)
    batches_per_second = 1000.0 / latency
    return BenchmarkResult(
        format=format_name,
        latency_ms=latency,
        throughput_batches_s=batches_per_second,
        fps=batches_per_second * batch,
    )


def _format_table(results: list[BenchmarkResult]) -> str:
    headers = ("Format", "Latency (ms)", "Throughput (batch/s)", "FPS", "Status")
    rows = []
    for result in results:
        rows.append(
            (
                result.format,
                f"{result.latency_ms:.3f}" if result.latency_ms is not None else "-",
                (
                    f"{result.throughput_batches_s:.2f}"
                    if result.throughput_batches_s is not None
                    else "-"
                ),
                f"{result.fps:.2f}" if result.fps is not None else "-",
                result.status,
            )
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    return "\n".join(
        [separator, render(headers), separator, *(render(row) for row in rows), separator]
    )


def _torchscript_runner(path: Path, batch: int, imgsz: int, device: str):
    target = torch.device(device)
    model = torch.jit.load(str(path), map_location=target).eval()
    images = torch.zeros(batch, 3, imgsz, imgsz, device=target)

    def run() -> Any:
        with torch.inference_mode():
            return model(images)

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elif target.type == "mps":
            torch.mps.synchronize()

    return run, synchronize


def _onnx_runner(path: Path, batch: int, imgsz: int, device: str):
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime is not installed") from None

    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(path), providers=providers)
    input_name = session.get_inputs()[0].name
    images = np.zeros((batch, 3, imgsz, imgsz), dtype=np.float32)
    return lambda: session.run(None, {input_name: images}), lambda: None


def _tensorrt_runner(path: Path, batch: int, imgsz: int, device: str):
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("TensorRT requires a CUDA device; use device=cuda")
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError("tensorrt is not installed") from None

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the engine")
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()

    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        input_name = next(
            name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        context.set_input_shape(input_name, (batch, 3, imgsz, imgsz))
        named_tensors: dict[str, torch.Tensor] = {}
        for name in names:
            shape = tuple(context.get_tensor_shape(name))
            dtype = torch.from_numpy(
                np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))
            ).dtype
            named_tensors[name] = torch.empty(shape, dtype=dtype, device=device)
            context.set_tensor_address(name, named_tensors[name].data_ptr())

        def run() -> Any:
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            return named_tensors

    else:
        input_index = next(
            index for index in range(engine.num_bindings) if engine.binding_is_input(index)
        )
        context.set_binding_shape(input_index, (batch, 3, imgsz, imgsz))
        binding_tensors: list[torch.Tensor] = []
        for index in range(engine.num_bindings):
            shape = tuple(context.get_binding_shape(index))
            dtype = torch.from_numpy(
                np.empty((), dtype=trt.nptype(engine.get_binding_dtype(index)))
            ).dtype
            binding_tensors.append(torch.empty(shape, dtype=dtype, device=device))
        bindings = [tensor.data_ptr() for tensor in binding_tensors]

        def run() -> Any:
            if not context.execute_async_v2(bindings, stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            return binding_tensors

    return run, torch.cuda.synchronize


def benchmark_model(
    model: Any,
    *,
    formats: list[str] | tuple[str, ...] = SUPPORTED_FORMATS,
    imgsz: int = 640,
    batch: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    device: str | None = None,
    half: bool = False,
    project: str = "runs/benchmark",
    name: str = "exp",
    save_dir: str | Path | None = None,
    exist_ok: bool = False,
) -> tuple[list[BenchmarkResult], Path]:
    """Export and benchmark the requested runtime formats."""
    selected = [str(item).lower() for item in formats]
    unsupported = sorted(set(selected) - set(SUPPORTED_FORMATS))
    if unsupported:
        raise ValueError(f"Unsupported benchmark formats: {', '.join(unsupported)}")
    if batch < 1 or warmup < 0 or iterations < 1 or imgsz < 1:
        raise ValueError(
            "batch, iterations, and imgsz must be positive; warmup must be non-negative"
        )

    target = device or str(model.device)
    run_dir = resolve_run_dir(project=project, name=name, save_dir=save_dir, exist_ok=exist_ok)
    write_run_metadata(
        run_dir,
        {
            "mode": "benchmark",
            "formats": selected,
            "imgsz": imgsz,
            "batch": batch,
            "warmup": warmup,
            "iterations": iterations,
            "device": target,
            "half": half,
        },
    )

    factories = {
        "onnx": _onnx_runner,
        "torchscript": _torchscript_runner,
        "tensorrt": _tensorrt_runner,
    }
    suffixes = {"onnx": ".onnx", "torchscript": ".torchscript", "tensorrt": ".engine"}
    results = []
    for format_name in selected:
        artifact = run_dir / "artifacts" / f"model{suffixes[format_name]}"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        try:
            if format_name == "tensorrt" and not target.startswith("cuda"):
                raise RuntimeError("TensorRT requires a CUDA device; use device=cuda")
            model.export(
                format=format_name,
                imgsz=imgsz,
                batch=batch,
                dynamic=False,
                simplify=False,
                half=half and format_name == "tensorrt",
                device=target,
                output=artifact,
                exist_ok=True,
                verbose=False,
            )
            run, synchronize = factories[format_name](artifact, batch, imgsz, target)
            results.append(
                _summarize(format_name, _measure(run, synchronize, warmup, iterations), batch)
            )
        except (ImportError, RuntimeError, ValueError) as error:
            results.append(BenchmarkResult(format_name, None, None, None, f"skipped: {error}"))

    report = run_dir / "benchmark.json"
    report.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")
    return results, report
