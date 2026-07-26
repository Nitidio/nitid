import pytest
import yaml

from dfine.utils.runs import (
    atomic_output_path,
    collect_environment,
    increment_path,
    resolve_run_dir,
    write_run_metadata,
)


def test_increment_path_claims_numbered_directories(tmp_path):
    assert increment_path(tmp_path / "exp") == tmp_path / "exp"
    assert increment_path(tmp_path / "exp") == tmp_path / "exp2"
    assert increment_path(tmp_path / "exp") == tmp_path / "exp3"


def test_exist_ok_reuses_directory(tmp_path):
    path = increment_path(tmp_path / "exp")
    assert increment_path(path, exist_ok=True) == path


def test_resume_requires_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory was not found"):
        resolve_run_dir(project=tmp_path, name="exp", resume=True)


def test_atomic_output_preserves_destination_on_failure(tmp_path):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"complete")
    with pytest.raises(RuntimeError):
        with atomic_output_path(destination) as temporary:
            temporary.write_bytes(b"partial")
            raise RuntimeError("interrupted")
    assert destination.read_bytes() == b"complete"
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == []


def test_metadata_contains_resolved_args_and_environment(tmp_path):
    write_run_metadata(tmp_path, {"save_dir": tmp_path, "epochs": 1})
    args = yaml.safe_load((tmp_path / "args.yaml").read_text())
    environment = yaml.safe_load((tmp_path / "environment.yaml").read_text())
    assert args == {"save_dir": str(tmp_path), "epochs": 1}
    assert set(environment) >= {"python", "platform", "pytorch", "gpus", "packages"}


def test_collect_environment_contains_diagnostic_fields():
    environment = collect_environment()

    assert environment["platform"]
    assert environment["python"]
    assert "pytorch" in environment
    assert "cuda_version" in environment
    assert "cudnn_version" in environment
    assert "gpus" in environment
    assert "packages" in environment


def test_run_metadata_uses_shared_environment_collector(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dfine.utils.runs.collect_environment", lambda: {"snapshot_marker": "shared"}
    )

    write_run_metadata(tmp_path, {})

    environment = yaml.safe_load((tmp_path / "environment.yaml").read_text())
    assert environment == {"snapshot_marker": "shared"}
