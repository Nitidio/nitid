"""Tests for named GStreamer hardware profiles and capability inspection."""

from __future__ import annotations

import pytest

from dfine.gstreamer import (
    get_hardware_profile,
    inspect_gstreamer_capabilities,
    resolve_hardware_fragment,
)
from dfine.media import build_gstreamer_output_pipeline
from dfine.utils.sources import build_gstreamer_pipeline


def test_profile_names_aliases_and_unknown_values():
    assert get_hardware_profile("cpu").name == "software"
    assert get_hardware_profile("VA_API").name == "vaapi"
    assert get_hardware_profile("v4l2m2m").name == "v4l2"
    assert get_hardware_profile("nvenc").name == "nvidia"
    with pytest.raises(ValueError, match="choose: software, vaapi, v4l2, nvidia, jetson"):
        get_hardware_profile("mystery")


def test_profile_resolution_selects_available_candidate_and_checks_dependencies():
    available = {"vaapih264dec", "videoconvert"}
    fragment = resolve_hardware_fragment(
        "vaapi",
        "decode",
        element_available=available.__contains__,
        require_opencv=False,
    )
    assert fragment == "vaapih264dec ! videoconvert"

    with pytest.raises(RuntimeError, match="missing: videoconvert"):
        resolve_hardware_fragment(
            "vaapi",
            "decode",
            element_available={"vah264dec"}.__contains__,
            require_opencv=False,
        )

    with pytest.raises(RuntimeError, match="one of vah264enc/vaapih264enc"):
        resolve_hardware_fragment(
            "vaapi",
            "encode",
            element_available={"videoconvert"}.__contains__,
            require_opencv=False,
        )


def test_profile_resolution_rejects_opencv_without_gstreamer(monkeypatch):
    monkeypatch.setattr("dfine.gstreamer.gstreamer_available", lambda: False)
    with pytest.raises(RuntimeError, match="OpenCV was built without GStreamer"):
        resolve_hardware_fragment("software", "decode")


def test_capability_report_is_deterministic(monkeypatch):
    available = {"videoconvert", "avdec_h264", "x264enc"}
    monkeypatch.setattr("dfine.gstreamer.gstreamer_available", lambda: True)
    monkeypatch.setattr("dfine.gstreamer.shutil.which", lambda name: "/usr/bin/gst-inspect-1.0")

    report = inspect_gstreamer_capabilities(element_available=available.__contains__)

    assert report["opencv_gstreamer"] is True
    assert report["gst_inspect"] is True
    profiles = report["profiles"]
    assert isinstance(profiles, dict)
    assert profiles["software"]["decode"] is True
    assert profiles["software"]["encode"] is True
    assert profiles["vaapi"]["decode"] is False


def test_hardware_profile_builds_explicit_h264_rtsp_decode_pipeline():
    available = {"vah264dec", "videoconvert"}
    pipeline = build_gstreamer_pipeline(
        "rtsp://camera/live",
        hardware_profile="vaapi",
        _element_available=available.__contains__,
    )

    assert "rtph264depay ! h264parse ! vah264dec" in pipeline
    assert "decodebin" not in pipeline
    assert pipeline.endswith("appsink drop=true max-buffers=1 sync=false")


def test_rtsp_credentials_use_source_properties_not_uri_userinfo():
    pipeline = build_gstreamer_pipeline(
        "rtsp://camera/live",
        rtsp_username='op"erator',
        rtsp_password="s/ecret",
    )

    assert 'location="rtsp://camera/live"' in pipeline
    assert 'user-id="op\\"erator"' in pipeline
    assert 'user-pw="s/ecret"' in pipeline
    assert "@camera" not in pipeline
    with pytest.raises(ValueError, match="requires rtsp_username"):
        build_gstreamer_pipeline("rtsp://camera/live", rtsp_password="secret")


def test_input_hardware_profiles_reject_ambiguous_sources(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    with pytest.raises(ValueError, match="H.264 RTSP"):
        build_gstreamer_pipeline(video, hardware_profile="vaapi")
    with pytest.raises(ValueError, match="cannot be combined"):
        build_gstreamer_pipeline(
            "camera",
            pipeline="videotestsrc ! appsink",
            hardware_profile="vaapi",
        )


def test_hardware_profile_builds_output_encoder_pipeline(tmp_path):
    available = {"videoconvert", "v4l2h264enc"}
    pipeline = build_gstreamer_output_pipeline(
        tmp_path / "segments",
        segment_duration=30,
        hardware_profile="v4l2",
        _element_available=available.__contains__,
    )

    assert "video/x-raw,format=NV12 ! v4l2h264enc" in pipeline
    assert "splitmuxsink" in pipeline
    with pytest.raises(ValueError, match="cannot be combined"):
        build_gstreamer_output_pipeline(
            tmp_path / "out.mp4",
            encoder="x264enc",
            hardware_profile="vaapi",
            _element_available=lambda element: True,
        )
