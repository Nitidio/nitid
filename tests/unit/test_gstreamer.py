"""Tests for GStreamer capability inspection and RTSP input pipelines."""

from __future__ import annotations

import pytest

from dfine.gstreamer import inspect_gstreamer_capabilities
from dfine.utils.sources import build_gstreamer_pipeline


def test_capability_report_is_deterministic(monkeypatch):
    monkeypatch.setattr("dfine.gstreamer.gstreamer_available", lambda: True)
    monkeypatch.setattr("dfine.gstreamer.shutil.which", lambda name: None)

    assert inspect_gstreamer_capabilities() == {"opencv_gstreamer": True, "gst_inspect": False}


def test_rtsp_source_decodes_with_decodebin():
    pipeline = build_gstreamer_pipeline("rtsp://camera/live")

    assert pipeline.startswith('rtspsrc location="rtsp://camera/live" latency=200 protocols=tcp')
    assert " ! decodebin ! " in pipeline
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
