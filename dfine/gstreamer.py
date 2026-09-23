"""GStreamer availability checks for the OpenCV backend."""

from __future__ import annotations

import shutil

import cv2


def gstreamer_available() -> bool:
    """Return whether the active OpenCV build has GStreamer support."""
    return any(
        line.strip().startswith("GStreamer:") and "YES" in line.upper()
        for line in cv2.getBuildInformation().splitlines()
    )


def inspect_gstreamer_capabilities() -> dict[str, bool]:
    """Return host GStreamer availability for diagnostics and CLI output."""
    return {
        "opencv_gstreamer": gstreamer_available(),
        "gst_inspect": shutil.which("gst-inspect-1.0") is not None,
    }
