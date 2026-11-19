from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_studio", REPO_ROOT / "scripts/build_studio.py"
)
assert _spec is not None and _spec.loader is not None
build_studio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_studio)


class FakeRunner:
    """Simulates subprocess.run for node/pnpm; records pnpm calls."""

    def __init__(self, node="v22.23.5", pnpm="10.28.0", fail_on=None, create_index=None):
        self.node, self.pnpm, self.fail_on = node, pnpm, fail_on
        self.create_index = create_index
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if cmd[-1] == "--version":
            out = self.node if cmd[0].endswith("node") else self.pnpm
            return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")
        step = " ".join(cmd[1:])
        if self.fail_on == step:
            return subprocess.CompletedProcess(cmd, 7)
        if step == "run build" and self.create_index is not None:
            self.create_index.mkdir(parents=True, exist_ok=True)
            (self.create_index / "index.html").write_text("<div id='root'></div>")
        return subprocess.CompletedProcess(cmd, 0)


def _which(found=("node", "pnpm")):
    return lambda name: f"/bin/{name}" if name in found else None


@pytest.fixture
def studio(tmp_path):
    folder = tmp_path / "studio"
    folder.mkdir()
    (folder / "package.json").write_text(json.dumps({"packageManager": "pnpm@10.28.0"}))
    return folder


def test_parse_version():
    assert build_studio.parse_version("v22.13.1") == (22, 13, 1)
    assert build_studio.parse_version("10.28.0") == (10, 28, 0)
    with pytest.raises(ValueError):
        build_studio.parse_version("nope")


def test_build_runs_frozen_install_then_build(studio, tmp_path):
    static = tmp_path / "static"
    runner = FakeRunner(create_index=static)
    build_studio.build(studio, static, runner, _which(), env={})
    pnpm_calls = [(c, k) for c, k in runner.calls if c[-1] != "--version"]
    assert [c[1:] for c, _ in pnpm_calls] == [["install", "--frozen-lockfile"], ["run", "build"]]
    for _, kwargs in pnpm_calls:
        assert kwargs["cwd"] == studio
        assert kwargs["env"]["STUDIO_OUT_DIR"] == str(static)


@pytest.mark.parametrize("node", ["v22.12.0", "v20.11.1", "v18.0.0"])
def test_old_node_is_rejected(studio, tmp_path, node):
    with pytest.raises(build_studio.BuildError) as error:
        build_studio.build(studio, tmp_path / "s", FakeRunner(node=node), _which(), env={})
    assert error.value.code == build_studio.EXIT_ENVIRONMENT
    assert "22.13" in str(error.value)


def test_node_22_13_exactly_is_accepted(studio):
    assert build_studio.check_node(FakeRunner(node="v22.13.0"), _which()) == "v22.13.0"


def test_missing_node_and_pnpm(studio, tmp_path):
    for found, word in ((("pnpm",), "Node.js"), (("node",), "pnpm")):
        with pytest.raises(build_studio.BuildError) as error:
            build_studio.build(studio, tmp_path / "s", FakeRunner(), _which(found), env={})
        assert error.value.code == build_studio.EXIT_ENVIRONMENT
        assert word in str(error.value)


def test_missing_studio_folder(tmp_path):
    with pytest.raises(build_studio.BuildError) as error:
        build_studio.build(tmp_path / "nope", tmp_path / "s", FakeRunner(), _which(), env={})
    assert error.value.code == build_studio.EXIT_ENVIRONMENT


@pytest.mark.parametrize("step", ["install --frozen-lockfile", "run build"])
def test_failing_step_gives_build_failed(studio, tmp_path, step):
    runner = FakeRunner(fail_on=step, create_index=tmp_path / "s")
    with pytest.raises(build_studio.BuildError) as error:
        build_studio.build(studio, tmp_path / "s", runner, _which(), env={})
    assert error.value.code == build_studio.EXIT_BUILD_FAILED
    assert step in str(error.value)


def test_install_failure_skips_build(studio, tmp_path):
    runner = FakeRunner(fail_on="install --frozen-lockfile")
    with pytest.raises(build_studio.BuildError):
        build_studio.build(studio, tmp_path / "s", runner, _which(), env={})
    assert not any(c[1:] == ["run", "build"] for c, _ in runner.calls)


def test_build_without_index_fails(studio, tmp_path):
    with pytest.raises(build_studio.BuildError) as error:
        build_studio.build(studio, tmp_path / "s", FakeRunner(), _which(), env={})
    assert error.value.code == build_studio.EXIT_BUILD_FAILED
    assert "index.html" in str(error.value)


def test_pnpm_version_mismatch_only_warns(studio, capsys):
    build_studio.check_pnpm(studio, FakeRunner(pnpm="9.0.0"), _which())
    assert "pins pnpm 10.28.0" in capsys.readouterr().err


def test_check_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_studio, "STATIC_DIR", tmp_path)
    assert build_studio.main(["--check-only"]) == build_studio.EXIT_MISSING_STATIC
    assert "index.html is missing" in capsys.readouterr().err
    (tmp_path / "index.html").write_text("x")
    assert build_studio.main(["--check-only"]) == build_studio.EXIT_OK


def test_main_maps_build_error_to_exit_code(monkeypatch, capsys):
    def boom(*_args):
        raise build_studio.BuildError("nope", 2)

    monkeypatch.setattr(build_studio, "build", boom)
    assert build_studio.main([]) == 2
    assert "ERROR: nope" in capsys.readouterr().err


def test_expected_pnpm_version_matches_repo_package_json():
    assert build_studio.expected_pnpm_version(REPO_ROOT / "studio") is not None
