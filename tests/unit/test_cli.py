"""Unit tests for command-line help output."""

import sys
import types

import pytest

from tools.dfine_cli import COMMAND_HELP, COMMANDS, main, parse_args


@pytest.mark.parametrize(
    ("command", "expected_text"),
    [
        ("predict", ("source=SOURCE", "conf=FLOAT", "save=BOOL", "dfine predict")),
        (
            "track",
            (
                "source=SOURCE",
                "tracker=NAME",
                "track_activation_threshold=FLOAT",
                "dfine track",
            ),
        ),
        ("download", ("model=NAME", "weights=NAME", "force=BOOL", "dfine download")),
        ("train", ("data=PATH", "epochs=INT", "dfine train")),
        ("val", ("data=PATH", "split=NAME", "dfine val")),
        ("export", ("format=FORMAT", "opset=INT", "dfine export")),
        ("info", ("detailed=BOOL", "dfine info")),
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
        def __init__(self, model, *, weights="default"):
            observed.update(model=model, weights=weights)

        def predict(self, source, **kwargs):
            observed["source"] = source
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(["dfine", "predict", "model=dfine_s", "weights=coco", "source=image.jpg"])

    assert observed == {"model": "dfine_s", "weights": "coco", "source": "image.jpg"}


def test_track_cli_streams_and_groups_tracker_options(monkeypatch, capsys):
    observed = {}

    class Result:
        def __init__(self, save_path=None):
            self.save_path = save_path

    class FakeDFINE:
        def __init__(self, model, *, weights="default"):
            observed.update(model=model, weights=weights)

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


def test_track_cli_respects_explicit_stream_and_tracker_selection(monkeypatch):
    observed = {}

    class FakeDFINE:
        def __init__(self, model, *, weights="default"):
            pass

        def track(self, source, **kwargs):
            observed.update(source=source, kwargs=kwargs)
            return []

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    main(["dfine", "track", "source=0", "tracker=byte-track", "stream=false"])

    assert observed == {
        "source": 0,
        "kwargs": {"tracker": "byte-track", "stream": False},
    }


def test_track_cli_requires_source(monkeypatch, capsys):
    class FakeDFINE:
        def __init__(self, model, *, weights="default"):
            pass

    fake_module = types.ModuleType("dfine")
    fake_module.DFINE = FakeDFINE
    monkeypatch.setitem(sys.modules, "dfine", fake_module)

    with pytest.raises(SystemExit) as error:
        main(["dfine", "track", "model=dfine_s"])

    assert error.value.code == 1
    assert "source= is required for track" in capsys.readouterr().out
