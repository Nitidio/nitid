"""Compatibility namespace for existing ``dfine`` imports.

New applications should import :class:`nitid.NITID` from :mod:`nitid`.
"""

from dfine.gstreamer import inspect_gstreamer_capabilities
from dfine.media import (
    Frame,
    FrameMetadata,
    FrameSink,
    FrameSource,
    GStreamerVideoSink,
    build_gstreamer_output_pipeline,
)
from dfine.model import DFINE
from dfine.nitid import NITID
from dfine.results import SemanticMask
from dfine.utils.reporting import BugReport, bugreport
from dfine.utils.sources import GStreamerFrameSource

__version__ = "0.1.0"
__all__ = [
    "BugReport",
    "DFINE",
    "NITID",
    "Frame",
    "FrameMetadata",
    "FrameSink",
    "FrameSource",
    "GStreamerVideoSink",
    "SemanticMask",
    "GStreamerFrameSource",
    "build_gstreamer_output_pipeline",
    "inspect_gstreamer_capabilities",
    "bugreport",
]
