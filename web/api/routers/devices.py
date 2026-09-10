from __future__ import annotations

from fastapi import APIRouter

from web.api.schemas.devices import DeviceReport, OpenVINOReport, TorchReport

router = APIRouter()


@router.get("/devices", response_model=DeviceReport)
def get_devices() -> DeviceReport:
    """Report the compute devices actually available on this server, so a
    caller only offers choices that will really work here. Unauthenticated:
    it reveals hardware capability only, no run/user data, so a client can
    show it before login."""
    import torch

    torch_report = TorchReport(cpu=True, cuda=torch.cuda.is_available())

    try:
        from dfine.nn.openvino_runtime import list_openvino_devices

        openvino_report = OpenVINOReport(available=True, devices=list_openvino_devices())
    except ImportError:
        openvino_report = OpenVINOReport(available=False, devices=[])

    return DeviceReport(torch=torch_report, openvino=openvino_report)
