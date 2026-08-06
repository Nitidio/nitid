"""Named GStreamer codec profiles and host capability inspection."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2

Direction = Literal["decode", "encode"]
ElementChecker = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class GStreamerHardwareProfile:
    """Required elements and pipeline fragment for one acceleration family."""

    name: str
    description: str
    decoder_candidates: tuple[str, ...]
    decoder_dependencies: tuple[str, ...]
    decoder_template: str
    encoder_candidates: tuple[str, ...]
    encoder_dependencies: tuple[str, ...]
    encoder_template: str


HARDWARE_PROFILES: dict[str, GStreamerHardwareProfile] = {
    "software": GStreamerHardwareProfile(
        name="software",
        description="Portable libav/x264 software codec path",
        decoder_candidates=("avdec_h264",),
        decoder_dependencies=("videoconvert",),
        decoder_template="{codec} ! videoconvert",
        encoder_candidates=("x264enc",),
        encoder_dependencies=("videoconvert",),
        encoder_template=(
            "videoconvert ! video/x-raw,format=I420 "
            "! {codec} tune=zerolatency speed-preset=veryfast key-int-max=30"
        ),
    ),
    "vaapi": GStreamerHardwareProfile(
        name="vaapi",
        description="Intel/AMD VA-API H.264 acceleration",
        decoder_candidates=("vah264dec", "vaapih264dec"),
        decoder_dependencies=("videoconvert",),
        decoder_template="{codec} ! videoconvert",
        encoder_candidates=("vah264enc", "vaapih264enc"),
        encoder_dependencies=("videoconvert",),
        encoder_template="videoconvert ! video/x-raw,format=NV12 ! {codec}",
    ),
    "v4l2": GStreamerHardwareProfile(
        name="v4l2",
        description="Linux V4L2 memory-to-memory H.264 acceleration",
        decoder_candidates=("v4l2h264dec",),
        decoder_dependencies=("videoconvert",),
        decoder_template="{codec} ! videoconvert",
        encoder_candidates=("v4l2h264enc",),
        encoder_dependencies=("videoconvert",),
        encoder_template="videoconvert ! video/x-raw,format=NV12 ! {codec}",
    ),
    "nvidia": GStreamerHardwareProfile(
        name="nvidia",
        description="NVIDIA desktop GStreamer CUDA/NVENC path",
        decoder_candidates=("nvh264dec",),
        decoder_dependencies=("cudadownload", "videoconvert"),
        decoder_template="{codec} ! cudadownload ! videoconvert",
        encoder_candidates=("nvh264enc",),
        encoder_dependencies=("cudaupload", "videoconvert"),
        encoder_template=(
            "videoconvert ! video/x-raw,format=NV12 ! cudaupload ! {codec} zerolatency=true"
        ),
    ),
    "jetson": GStreamerHardwareProfile(
        name="jetson",
        description="NVIDIA Jetson V4L2/NVMM H.264 path",
        decoder_candidates=("nvv4l2decoder",),
        decoder_dependencies=("nvvidconv", "videoconvert"),
        decoder_template=("{codec} ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert"),
        encoder_candidates=("nvv4l2h264enc",),
        encoder_dependencies=("nvvidconv",),
        encoder_template=("nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! {codec}"),
    ),
}

PROFILE_ALIASES = {
    "cpu": "software",
    "va-api": "vaapi",
    "v4l2m2m": "v4l2",
    "nvenc": "nvidia",
}


def gstreamer_available() -> bool:
    """Return whether the active OpenCV build has GStreamer support."""
    return any(
        line.strip().startswith("GStreamer:") and "YES" in line.upper()
        for line in cv2.getBuildInformation().splitlines()
    )


def gstreamer_element_available(element: str) -> bool:
    """Return whether gst-inspect can resolve one element without printing output."""
    executable = shutil.which("gst-inspect-1.0")
    if executable is None:
        return False
    result = subprocess.run(
        [executable, element],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def get_hardware_profile(name: str) -> GStreamerHardwareProfile:
    """Resolve a canonical profile name or documented alias."""
    normalized = name.strip().lower().replace("_", "-")
    normalized = PROFILE_ALIASES.get(normalized, normalized)
    try:
        return HARDWARE_PROFILES[normalized]
    except KeyError as exc:
        choices = ", ".join(HARDWARE_PROFILES)
        raise ValueError(f"unknown GStreamer hardware profile '{name}'; choose: {choices}") from exc


def resolve_hardware_fragment(
    name: str,
    direction: Direction,
    *,
    element_available: ElementChecker = gstreamer_element_available,
    require_opencv: bool = True,
) -> str:
    """Validate a profile on this host and return its decode/encode fragment."""
    if require_opencv and not gstreamer_available():
        raise RuntimeError("OpenCV was built without GStreamer support")
    profile = get_hardware_profile(name)
    if direction == "decode":
        candidates = profile.decoder_candidates
        dependencies = profile.decoder_dependencies
        template = profile.decoder_template
    elif direction == "encode":
        candidates = profile.encoder_candidates
        dependencies = profile.encoder_dependencies
        template = profile.encoder_template
    else:  # pragma: no cover - protected by Literal in typed callers
        raise ValueError("direction must be 'decode' or 'encode'")

    missing_dependencies = [item for item in dependencies if not element_available(item)]
    codec = next((item for item in candidates if element_available(item)), None)
    if codec is None or missing_dependencies:
        missing = list(missing_dependencies)
        if codec is None:
            missing.append("one of " + "/".join(candidates))
        raise RuntimeError(
            f"GStreamer profile '{profile.name}' cannot {direction} on this host; "
            f"missing: {', '.join(missing)}"
        )
    return template.format(codec=codec)


def inspect_gstreamer_capabilities(
    *,
    element_available: ElementChecker = gstreamer_element_available,
) -> dict[str, object]:
    """Return deterministic host/profile availability for diagnostics and CLI output."""
    opencv_enabled = gstreamer_available()
    profiles: dict[str, dict[str, object]] = {}
    for name, profile in HARDWARE_PROFILES.items():
        decode_elements = (*profile.decoder_dependencies, *profile.decoder_candidates)
        encode_elements = (*profile.encoder_dependencies, *profile.encoder_candidates)
        decode_elements_available = all(
            element_available(item) for item in profile.decoder_dependencies
        ) and any(element_available(item) for item in profile.decoder_candidates)
        encode_elements_available = all(
            element_available(item) for item in profile.encoder_dependencies
        ) and any(element_available(item) for item in profile.encoder_candidates)
        profiles[name] = {
            "description": profile.description,
            "decode": opencv_enabled and decode_elements_available,
            "encode": opencv_enabled and encode_elements_available,
            "decode_elements_available": decode_elements_available,
            "encode_elements_available": encode_elements_available,
            "decode_elements": list(decode_elements),
            "encode_elements": list(encode_elements),
        }
    return {
        "opencv_gstreamer": opencv_enabled,
        "gst_inspect": shutil.which("gst-inspect-1.0") is not None,
        "profiles": profiles,
    }
