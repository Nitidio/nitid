from __future__ import annotations

from pydantic import BaseModel


class TorchReport(BaseModel):
    cpu: bool
    cuda: bool


class OpenVINOReport(BaseModel):
    available: bool
    devices: list[str]


class DeviceReport(BaseModel):
    torch: TorchReport
    openvino: OpenVINOReport
