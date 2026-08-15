"""Unit tests for command-line help output."""

import sys
import types

import pytest

from tools.dfine_cli import COMMAND_HELP, COMMANDS, _configure_output_sink, main, parse_args


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("predict", ("source=SOURCE", "conf=FLOAT", "save=BOOL", "dfine predict")),
        (
            "track",
            (
                "source=SOURCE",
                "tracker=NAME",
                "backend=NAME",
                "output=DEST",
                "track_activation_threshold=FLOAT",
                "enable_cmc=BOOL",
                "direction_consistency_weight=FLOAT",
                "dfine track",
            ),
        ),
        ("download", ("model=NAME", "weights=NAME", "force=BOOL", "dfine download")),
        ("train", ("data=PATH", "epochs=INT", "dfine train")),
        ("val", ("data=PATH", "split=NAME", "dfine val")),
        ("export", ("format=FORMAT", "opset=INT", "dfine export")),
        ("info", ("detailed=BOOL", "dfine info")),
        ("gstreamer-info", ("named decode/encode profiles", "software", "jetson")),
        ("onvif", ("action=discover", "password_env=NAME", "action=uri")),
        ("bugreport", ("environment-only", "dfine bugreport")),
    ],
)
def test_command_help_exits_successfully_without_loading_model(
    command, expected_text, capsys, monkeypatch
):
    monkeypatch.setitem(sys.modules, "dfine", None)

    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for text in expected_text:
        assert text in output


def test_every_command_has_help_text():
    assert set(COMMAND_HELP) == COMMANDS


def test_short_help_flag_is_supported(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "predict", "-h"])

    assert exc_info.value.code == 0
    assert "Usage:\n  dfine predict" in capsys.readouterr().out


def test_general_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "nitid D-FINE CLI" in output
    assert "dfine COMMAND --help" in output


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "unknown"])

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "ERROR: unknown command 'unknown'" in output
    assert "Commands:" in output


def test_train_cli_parses_list_controls():
    command, args = parse_args(
        ["dfine", "train", "data=data.yml", "classes=[0,2]", "freeze=[backbone,decoder]"]
    )

    assert command == "train"
    assert args["classes"] == [0, 2]
    assert args["freeze"] == ["backbone", "decoder"]


def test_cli_passes_weights_to_model_constructor(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            observed.update(model=model, task=task, weights=weights)

        def predict(self, source, **kwargs):
            observed["source"] = source
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(["dfine", "predict", "model=dfine_s", "weights=coco", "source=image.jpg"])

    assert observed == {
        "model": "dfine_s",
        "task": "detect",
        "weights": "coco",
        "source": "image.jpg",
    }


def test_track_cli_streams_and_groups_tracker_options(monkeypatch, capsys):
    observed = {}

    class Result:
        def __init__(self, save_path=None):
            self.save_path = save_path

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            observed.update(model=model, task=task, weights=weights)

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return iter([Result("runs/track/exp/video.mp4"), Result("runs/track/exp/video.mp4")])

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(
        [
            "dfine",
            "track",
            "model=dfine_s",
            "source=video.mp4",
            "conf=0.5",
            "save=true",
            "lost_track_buffer=60",
            "track_activation_threshold=0.4",
        ]
    )

    assert observed == {
        "model": "dfine_s",
        "task": "detect",
        "weights": "default",
        "source": "video.mp4",
        "kwargs": {
            "conf": 0.5,
            "save": True,
            "stream": True,
            "tracker_kwargs": {
                "lost_track_buffer": 60,
                "track_activation_threshold": 0.4,
            },
        },
    }
    output = capsys.readouterr().out
    assert "Tracked 2 frames" in output
    assert output.count("Saved runs/track/exp/video.mp4") == 1


def test_cli_passes_segment_task_to_model(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            observed.update(model=model, task=task, weights=weights)

        def predict(self, source, **kwargs):
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(["dfine", "predict", "model=dfine_s", "task=segment", "source=image.jpg"])

    assert observed["task"] == "segment"


def test_track_cli_respects_explicit_stream_and_tracker_selection(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(
        [
            "dfine",
            "track",
            "source=0",
            "tracker=botsort",
            "enable_cmc=true",
            "cmc_method=orb",
            "cmc_downscale=4",
            "stream=false",
        ]
    )

    assert observed == {
        "source": 0,
        "kwargs": {
            "tracker": "botsort",
            "stream": False,
            "tracker_kwargs": {
                "enable_cmc": True,
                "cmc_method": "orb",
                "cmc_downscale": 4,
            },
        },
    }


def test_track_cli_forwards_gstreamer_reconnect_options(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(
        [
            "dfine",
            "track",
            "source=rtsp://camera/live",
            "backend=gstreamer",
            "reconnect=true",
            "reconnect_initial_delay=0.5",
            "reconnect_max_delay=10",
            "reconnect_attempts=4",
            "rtsp_latency=300",
            "rtsp_transport=udp",
            "hardware_profile=vaapi",
        ]
    )

    assert observed == {
        "source": "rtsp://camera/live",
        "kwargs": {
            "backend": "gstreamer",
            "reconnect": True,
            "reconnect_initial_delay": 0.5,
            "reconnect_max_delay": 10,
            "reconnect_attempts": 4,
            "rtsp_latency": 300,
            "rtsp_transport": "udp",
            "hardware_profile": "vaapi",
            "stream": True,
        },
    }


def test_track_cli_builds_gstreamer_segment_sink(monkeypatch, capsys):
    observed = {}

    class FakeSink:
        def __init__(self, destination, **kwargs):
            observed.update(destination=destination, sink_kwargs=kwargs)

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, model_kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    fake_module.GStreamerVideoSink = FakeSink
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(
        [
            "dfine",
            "track",
            "source=video.mp4",
            "output=runs/segments",
            "segment_duration=60",
            "output_fps=15",
            "output_hardware_profile=v4l2",
        ]
    )

    assert observed["destination"] == "runs/segments"
    assert observed["sink_kwargs"] == {
        "pipeline": None,
        "fps": 15,
        "segment_duration": 60,
        "hardware_profile": "v4l2",
    }
    assert observed["source"] == "video.mp4"
    assert observed["model_kwargs"] == {"sink": observed["model_kwargs"]["sink"], "stream": True}
    assert "Wrote annotated output to runs/segments" in capsys.readouterr().out


def test_output_options_require_a_destination():
    with pytest.raises(ValueError, match="require output="):
        _configure_output_sink({"segment_duration": 60})


def test_gstreamer_info_does_not_load_a_model(monkeypatch, capsys):
    monkeypatch.setattr(
        "dfine.gstreamer.inspect_gstreamer_capabilities",
        lambda: {
            "opencv_gstreamer": True,
            "gst_inspect": True,
            "profiles": {
                "software": {
                    "description": "Software path",
                    "decode": True,
                    "encode": True,
                },
                "vaapi": {
                    "description": "VA-API path",
                    "decode": True,
                    "encode": False,
                },
            },
        },
    )

    main(["dfine", "gstreamer-info"])

    output = capsys.readouterr().out
    assert "OpenCV GStreamer: yes" in output
    assert "software decode=yes encode=yes" in output
    assert "vaapi" in output and "encode=no" in output


def test_onvif_discovery_cli(monkeypatch, capsys):
    from dfine.onvif import ONVIFDevice

    observed = {}

    def discover(timeout, interface=None):
        observed.update(timeout=timeout, interface=interface)
        return [
            ONVIFDevice(
                "urn:uuid:camera",
                ("http://192.0.2.10/onvif/device_service",),
            )
        ]

    monkeypatch.setattr("dfine.onvif.discover_onvif_devices", discover)
    main(["dfine", "onvif", "action=discover", "timeout=1.5", "interface=192.0.2.20"])

    assert observed == {"timeout": 1.5, "interface": "192.0.2.20"}
    output = capsys.readouterr().out
    assert "Discovered 1 ONVIF device" in output
    assert "http://192.0.2.10/onvif/device_service" in output


def test_onvif_profiles_and_uri_cli_use_environment_password(monkeypatch, capsys):
    from dfine.onvif import ONVIFMediaProfile

    observed = []

    class FakeCamera:
        def __init__(self, host, username=None, password=None, **kwargs):
            observed.append((host, username, password, kwargs))

        def get_profiles(self):
            return [ONVIFMediaProfile("main", "Main Stream", "H264", 1920, 1080, 25)]

        def get_stream_uri(self, profile=None):
            observed.append(("profile", profile))
            return "rtsp://camera/live"

    monkeypatch.setattr("dfine.onvif.ONVIFCamera", FakeCamera)
    monkeypatch.setenv("CAMERA_SECRET", "not-printed")

    main(
        [
            "dfine",
            "onvif",
            "action=profiles",
            "host=camera.local",
            "username=operator",
            "password_env=CAMERA_SECRET",
        ]
    )
    assert observed[0][1:3] == ("operator", "not-printed")
    assert "Main Stream — H264 1920x1080 25fps" in capsys.readouterr().out

    main(
        [
            "dfine",
            "onvif",
            "action=uri",
            "host=camera.local",
            "profile=main",
            "password_env=CAMERA_SECRET",
        ]
    )
    assert observed[-1] == ("profile", "main")
    assert capsys.readouterr().out.strip() == "rtsp://camera/live"


def test_onvif_cli_rejects_password_argument_and_missing_host(capsys):
    with pytest.raises(SystemExit) as password_error:
        main(["dfine", "onvif", "action=profiles", "host=camera", "password=secret"])
    assert password_error.value.code == 1
    assert "password_env" in capsys.readouterr().out

    with pytest.raises(SystemExit) as host_error:
        main(["dfine", "onvif", "action=uri"])
    assert host_error.value.code == 1
    assert "host= is required" in capsys.readouterr().out


def test_onvif_cli_reports_protocol_errors_without_traceback(monkeypatch, capsys):
    from dfine.onvif import ONVIFError

    monkeypatch.setattr(
        "dfine.onvif.discover_onvif_devices",
        lambda **kwargs: (_ for _ in ()).throw(ONVIFError("multicast blocked")),
    )
    with pytest.raises(SystemExit) as error:
        main(["dfine", "onvif", "action=discover"])

    assert error.value.code == 1
    assert capsys.readouterr().out.strip() == "ERROR: multicast blocked"


def test_track_cli_reads_rtsp_password_from_environment(monkeypatch, capsys):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)
    monkeypatch.setenv("CAMERA_RTSP_PASSWORD", "secret-value")

    main(
        [
            "dfine",
            "track",
            "source=rtsp://camera/live",
            "backend=gstreamer",
            "rtsp_username=operator",
            "rtsp_password_env=CAMERA_RTSP_PASSWORD",
        ]
    )

    assert observed["kwargs"]["rtsp_username"] == "operator"
    assert observed["kwargs"]["rtsp_password"] == "secret-value"
    assert "secret-value" not in capsys.readouterr().out

    with pytest.raises(SystemExit) as error:
        main(
            [
                "dfine",
                "track",
                "source=rtsp://camera/live",
                "rtsp_password=visible",
            ]
        )
    assert error.value.code == 1
    assert "rtsp_password_env" in capsys.readouterr().out


def test_track_cli_requires_source(monkeypatch, capsys):
    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    with pytest.raises(SystemExit) as error:
        main(["dfine", "track", "model=dfine_s"])

    assert error.value.code == 1
    assert "source= is required for track" in capsys.readouterr().out
