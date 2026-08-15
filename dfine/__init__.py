"""
nitid: Ultralytics-style D-FINE detection and instance segmentation.

Usage:
    from dfine import DFINE

    model = DFINE("dfine_l")
    results = model("image.jpg")
    model.train(data="coco.yaml", epochs=50)
    model.export(format="onnx")
"""

from dfine.gstreamer import (
    GStreamerHardwareProfile,
    get_hardware_profile,
    inspect_gstreamer_capabilities,
)
from dfine.media import (
    Frame,
    FrameMetadata,
    FrameSink,
    FrameSource,
    GStreamerVideoSink,
    build_gstreamer_output_pipeline,
)
from dfine.model import DFINE
from dfine.onvif import (
    ONVIFCamera,
    ONVIFDevice,
    ONVIFError,
    ONVIFMediaProfile,
    discover_onvif_devices,
)
from dfine.utils.reporting import BugReport, bugreport
from dfine.utils.sources import GStreamerFrameSource

__version__ = "0.1.0"
__all__ = [
    "BugReport",
    "DFINE",
    "Frame",
    "FrameMetadata",
    "FrameSink",
    "FrameSource",
    "GStreamerVideoSink",
    "ONVIFCamera",
    "ONVIFDevice",
    "ONVIFError",
    "ONVIFMediaProfile",
    "GStreamerFrameSource",
    "GStreamerHardwareProfile",
    "build_gstreamer_output_pipeline",
    "get_hardware_profile",
    "inspect_gstreamer_capabilities",
    "discover_onvif_devices",
    "bugreport",
]
