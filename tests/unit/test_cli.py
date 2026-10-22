"""Unit tests for command-line help output."""

import sys
import types

import pytest

from nitid.cli import COMMAND_HELP, COMMANDS, main, parse_args


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("predict", ("source=SOURCE", "conf=FLOAT", "save=BOOL", "nitid predict")),
        (
            "track",
            (
                "source=SOURCE",
                "tracker=NAME",
                "rtsp_username=USER",
                "rtsp_password_env=NAME",
                "track_activation_threshold=FLOAT",
                "enable_cmc=BOOL",
                "direction_consistency_weight=FLOAT",
                "nitid track",
            ),
        ),
        (
            "download",
            ("model=NAME", "task=TASK", "weights=NAME", "force=BOOL", "nitid download"),
        ),
        ("train", ("data=PATH", "task=TASK", "epochs=INT", "nitid train")),
        ("val", ("data=PATH", "task=TASK", "split=NAME", "nitid val")),
        ("export", ("task=TASK", "format=FORMAT", "opset=INT", "nitid export")),
        ("convert", ("data=DATA", "target=FORMAT", "output=PATH", "nitid convert")),
        ("info", ("task=TASK", "detailed=BOOL", "nitid info")),
        ("bugreport", ("environment-only", "nitid bugreport")),
    ],
)
def test_command_help_exits_successfully_without_loading_model(
    command, expected_text, capsys, monkeypatch
):
    monkeypatch.setitem(sys.modules, "nitid", None)

    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for text in expected_text:
        assert text in output


def test_every_command_has_help_text():
    assert set(COMMAND_HELP) == COMMANDS


def test_short_help_flag_is_supported(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "predict", "-h"])

    assert exc_info.value.code == 0
    assert "Usage:\n  nitid predict" in capsys.readouterr().out


def test_general_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Nitid CLI" in output
    assert "nitid COMMAND --help" in output


def test_legacy_dfine_command_uses_canonical_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["dfine", "--help"])

    assert exc_info.value.code == 0
    assert "Usage:\n  nitid COMMAND" in capsys.readouterr().out


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["nitid", "unknown"])

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "ERROR: unknown command 'unknown'" in output
    assert "Commands:" in output


def test_train_cli_parses_list_controls():
    command, args = parse_args(
        ["nitid", "train", "data=data.yml", "classes=[0,2]", "freeze=[backbone,decoder]"]
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
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    main(["nitid", "predict", "model=model1s", "weights=coco", "source=image.jpg"])

    assert observed == {
        "model": "model1s",
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
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    main(
        [
            "nitid",
            "track",
            "model=model1s",
            "source=video.mp4",
            "conf=0.5",
            "save=true",
            "lost_track_buffer=60",
            "track_activation_threshold=0.4",
        ]
    )

    assert observed == {
        "model": "model1s",
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
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    main(["nitid", "predict", "model=model1s", "task=segment", "source=image.jpg"])

    assert observed["task"] == "segment"


def test_cli_passes_semantic_prediction_options(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            observed.update(model=model, task=task, weights=weights)

        def predict(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    main(
        [
            "nitid",
            "predict",
            "model=semantic.pth",
            "task=semantic",
            "source=image.jpg",
            "return_probs=true",
        ]
    )

    assert observed["task"] == "semantic"
    assert observed["kwargs"]["return_probs"] is True


def test_track_cli_respects_explicit_stream_and_tracker_selection(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    main(
        [
            "nitid",
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


def test_removed_video_io_options_are_not_documented():
    for command in ("predict", "track"):
        for option in ("backend=", "output=", "segment_duration=", "rtsp_latency="):
            assert option not in COMMAND_HELP[command]


def test_track_cli_reads_rtsp_password_from_environment(monkeypatch, capsys):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)
    monkeypatch.setenv("CAMERA_RTSP_PASSWORD", "secret-value")

    main(
        [
            "nitid",
            "track",
            "source=rtsp://camera/live",
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
                "nitid",
                "track",
                "source=rtsp://camera/live",
                "rtsp_password=visible",
            ]
        )
    assert error.value.code == 1
    assert "rtsp_password_env" in capsys.readouterr().out

    with pytest.raises(SystemExit) as error:
        main(
            [
                "nitid",
                "track",
                "source=rtsp://camera/live",
                "rtsp_password_env=CAMERA_RTSP_PASSWORD",
            ]
        )
    assert error.value.code == 1
    assert "rtsp_username= is required" in capsys.readouterr().out


def test_track_cli_requires_source(monkeypatch, capsys):
    class FakeDFINE:
        def __init__(self, model, *, task="detect", weights="default"):
            pass

    fake_module = types.ModuleType("dfine")
    fake_module.NITID = FakeDFINE
    monkeypatch.setitem(sys.modules, "nitid", fake_module)

    with pytest.raises(SystemExit) as error:
        main(["nitid", "track", "model=model1s"])

    assert error.value.code == 1
    assert "source= is required for track" in capsys.readouterr().out


@pytest.fixture()
def captured_download(monkeypatch, tmp_path):
    from dfine.utils import downloads

    calls: list[dict] = []

    def fake_download_model(model, **kwargs):
        calls.append({"model": model, **kwargs})
        return tmp_path / "wrapped.pth"

    monkeypatch.setattr(downloads, "download_model", fake_download_model)
    return calls


@pytest.mark.parametrize(
    ("args", "model", "task"),
    [
        ([], "dfine_l", "detect"),
        (["model=model1s"], "dfine_s", "detect"),
        (["model=MODEL1X", "task=segment"], "dfine_x", "segment"),
        (["model=model1n", "task=segment"], "dfine_n", "segment"),
        (["model=dfine_m"], "dfine_m", "detect"),
    ],
)
def test_download_cli_maps_public_model_names(captured_download, capsys, args, model, task):
    main(["nitid", "download", *args])

    assert len(captured_download) == 1
    assert captured_download[0]["model"] == model
    assert captured_download[0]["task"] == task
    assert "Downloaded wrapped checkpoint" in capsys.readouterr().out


def test_download_cli_rejects_model1n_detection(captured_download):
    with pytest.raises(ValueError, match="model1n"):
        main(["nitid", "download", "model=model1n"])

    assert captured_download == []
