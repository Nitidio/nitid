"""OpenVINO Runtime inference backend — Intel CPU, integrated GPU, and NPU.

Used internally by ``DFINE``/``NITID`` when constructed with
``backend="openvino"``. Not part of the public API.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from dfine.exporter import DFINEExporter

_OPENVINO_INSTALL_HINT = "OpenVINO is not installed. Install it with: uv sync --extra openvino"


def _import_openvino():
    try:
        import openvino as ov
    except ImportError:
        raise ImportError(_OPENVINO_INSTALL_HINT) from None
    return ov


def list_openvino_devices() -> list[str]:
    """Return the OpenVINO device identifiers available on this machine."""
    ov = _import_openvino()
    return list(ov.Core().available_devices)


def resolve_openvino_device(device: str | None) -> str:
    """
    Validate/select an OpenVINO device string.

    ``None`` or ``"auto"`` selects the first available of NPU, then GPU (Intel
    integrated GPU), then CPU. Any other value must match (case-insensitively)
    one of ``ov.Core().available_devices``, or a ``ValueError`` naming the
    devices that are actually available is raised.
    """
    available = list_openvino_devices()
    if device is None or device.strip().lower() == "auto":
        for preferred in ("NPU", "GPU", "CPU"):
            if preferred in available:
                return preferred
        raise ValueError(f"No OpenVINO devices are available on this machine: {available}")

    requested = device.strip().upper()
    for candidate in available:
        if candidate.upper() == requested:
            return candidate
    raise ValueError(
        f"OpenVINO device {device!r} is not available on this machine. "
        f"Available devices: {available}"
    )


class OpenVINORawModel:
    """
    Adapts a compiled OpenVINO model to the raw dict-output calling
    convention ``DFINEPredictor`` expects from ``self.model(tensor)``.

    Exposes a no-op ``deploy()`` and ``_deployed = True`` so it satisfies the
    deployed-model guard in ``DFINEPredictor.__init__`` unchanged.

    ``ov.CompiledModel.__call__`` reuses a single internal infer request and
    raises "Infer Request is busy" under concurrent calls from multiple
    threads (confirmed by testing, not just docs) — a real scenario once
    several jobs can target the same device at once. Each thread gets its
    own ``InferRequest``, created once and reused for that thread's later
    calls, per OpenVINO's documented multi-threaded serving pattern.
    """

    _deployed = True

    def __init__(self, compiled_model, output_names: list[str]) -> None:
        self._compiled = compiled_model
        self._output_names = output_names
        self._local = threading.local()

    def deploy(self) -> "OpenVINORawModel":
        return self

    def _infer_request(self):
        request = getattr(self._local, "request", None)
        if request is None:
            request = self._compiled.create_infer_request()
            self._local.request = request
        return request

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_outputs = self._infer_request().infer([images.detach().cpu().numpy()])
        return {
            name: torch.from_numpy(np.asarray(raw_outputs[index]))
            for index, name in enumerate(self._output_names)
        }


def compile_raw_openvino(
    exporter: "DFINEExporter", *, imgsz: int, ov_device: str
) -> OpenVINORawModel:
    """
    Trace ``exporter``'s deployed model to a raw (non-postprocessed) OpenVINO
    IR at ``imgsz`` and compile it for ``ov_device``.

    ``exporter.model`` must already be a deployed inference copy (as built by
    ``DFINE._get_deployed_model()``) — this mirrors how ``DFINE.export()``
    constructs a ``DFINEExporter``.
    """
    ov = _import_openvino()
    model, output_names = exporter.prepare_raw_trace(imgsz)

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=exporter.device)
    with tempfile.TemporaryDirectory(prefix=".nitid-openvino-raw-") as directory:
        onnx_path = Path(directory) / "model.onnx"
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            dynamo=False,
            opset_version=17,
            input_names=["images"],
            output_names=output_names,
        )
        ov_model = ov.convert_model(str(onnx_path))

    compiled = ov.Core().compile_model(ov_model, ov_device)
    return OpenVINORawModel(compiled, output_names)
