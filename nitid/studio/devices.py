"""Inference hardware discovery for ``GET /api/v1/devices``.

Everything here is *tolerant*: a missing optional dependency (OpenVINO) or a
broken driver stack must never turn the endpoint into a 500. The worst outcome
is that a backend is reported as unavailable.

PyTorch and OpenVINO are imported lazily, inside the functions, so importing
:mod:`nitid.studio` stays cheap.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def torch_devices() -> dict[str, Any]:
    """Return the PyTorch devices: CPU always, CUDA devices if the runtime sees any.

    ``torch.cuda.is_available()`` is the gate. Per-device name lookups are
    wrapped so one failing device does not hide the others.
    """
    cuda: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    name = str(torch.cuda.get_device_name(index))
                except Exception:  # driver hiccup: keep the device, drop the name
                    logger.warning("Could not read the name of CUDA device %d", index, exc_info=True)
                    name = f"CUDA device {index}"
                cuda.append({"index": index, "name": name})
    except Exception:
        logger.warning("CUDA discovery failed; reporting no CUDA devices", exc_info=True)
        cuda = []
    return {"cpu": True, "cuda": cuda}


def openvino_devices() -> dict[str, Any]:
    """Return ``{available, devices}`` for the OpenVINO runtime.

    ``available`` is true when OpenVINO imports *and* reports at least one
    device; if it is not installed (the usual case) or fails to initialise, the
    result is ``{"available": False, "devices": []}`` and no error is raised.
    """
    try:
        from dfine.nn.openvino_runtime import list_openvino_devices

        devices = [str(device) for device in list_openvino_devices()]
    except ImportError:
        return {"available": False, "devices": []}
    except Exception:
        logger.warning("OpenVINO discovery failed; reporting it as unavailable", exc_info=True)
        return {"available": False, "devices": []}
    return {"available": bool(devices), "devices": devices}


def collect_devices() -> dict[str, Any]:
    """Payload of ``GET /api/v1/devices`` (schema ``Devices`` in the API contract)."""
    return {"torch": torch_devices(), "openvino": openvino_devices()}
