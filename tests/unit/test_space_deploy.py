"""Deployment files of the public demo (Dockerfile and Cloud Run script in ``spaces/``).

These tests do not build the image or call Google Cloud. They keep the pieces that are easy to break
silently in step with each other: the port and host Cloud Run needs, the Gradio version, the weights
fetched at build time, and the cost limits of the deploy command.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "spaces" / "demo"
SCRIPT = REPO_ROOT / "spaces" / "deploy_cloud_run.sh"

_spec = importlib.util.spec_from_file_location("space_demo_app", DEMO / "app.py")
assert _spec is not None and _spec.loader is not None
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (DEMO / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_binds_gradio_to_the_port_cloud_run_sends_requests_to(dockerfile: str) -> None:
    assert "GRADIO_SERVER_NAME=0.0.0.0" in dockerfile
    assert "GRADIO_SERVER_PORT=8080" in dockerfile
    # The deploy command must send traffic to that same port.
    assert "--port 8080" in SCRIPT.read_text(encoding="utf-8")


def test_dockerfile_starts_the_app_from_the_directory_that_holds_the_weights(
    dockerfile: str,
) -> None:
    assert "WORKDIR /app" in dockerfile
    assert "COPY . ." in dockerfile
    assert 'CMD ["python", "app.py"]' in dockerfile


def test_dockerfile_gradio_pin_matches_the_space_card(dockerfile: str) -> None:
    pinned = re.search(r'"gradio==([0-9.]+)"', dockerfile)
    card = re.search(r"^sdk_version:\s*([0-9.]+)\s*$", (DEMO / "README.md").read_text(), re.M)
    assert pinned is not None and card is not None
    assert pinned.group(1) == card.group(1)


def test_dockerfile_installs_the_demo_requirements(dockerfile: str) -> None:
    assert "COPY requirements.txt ." in dockerfile
    assert "pip install -r requirements.txt" in dockerfile
    requirements = (DEMO / "requirements.txt").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert re.search(r"^nitid>=", requirements, re.M)


def test_prefetched_weights_are_combinations_the_demo_offers(dockerfile: str) -> None:
    run = re.search(r'RUN python -c "(.+)"', dockerfile)
    assert run is not None, "the Dockerfile must download weights while building"
    pairs = ast.literal_eval(re.search(r"in (\(\(.*\)\))\]", run.group(1)).group(1))  # type: ignore[union-attr]
    assert pairs, "at least one model must be prefetched"
    for task, size in pairs:
        sizes, _ = app.sizes_for_task(task)
        assert size in sizes, f"({task!r}, {size!r}) is not offered by the demo"


def test_dockerignore_keeps_caches_out_of_the_image() -> None:
    lines = (DEMO / ".dockerignore").read_text(encoding="utf-8").split()
    assert "__pycache__/" in lines


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")


def _run_script(**env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if k not in {"PROJECT", "DRY_RUN"}}
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=False,
    )


@needs_bash
def test_deploy_script_requires_a_project() -> None:
    result = _run_script(DRY_RUN="1")
    assert result.returncode != 0
    assert "PROJECT" in result.stderr


@needs_bash
def test_deploy_script_dry_run_prints_the_command_with_the_cost_limits() -> None:
    result = _run_script(PROJECT="my-project", DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.startswith("gcloud run deploy nitid-demo ")
    assert "--project my-project" in out
    assert "--source " in out and out.split("--source ")[1].split()[0].endswith("/spaces/demo")
    assert "--region europe-west1" in out
    assert "--allow-unauthenticated" in out
    # The spend limits. A budget only sends alerts, so these are the real brake.
    assert "--min-instances 0" in out
    assert "--max-instances 1" in out


@needs_bash
def test_deploy_script_lets_size_and_region_change_but_not_the_instance_limits() -> None:
    result = _run_script(
        PROJECT="p",
        DRY_RUN="1",
        REGION="europe-west4",
        CPU="1",
        MEMORY="2Gi",
        SERVICE="demo-test",
        MAX_INSTANCES="10",
        MIN_INSTANCES="3",
    )
    out = result.stdout
    assert result.returncode == 0, result.stderr
    assert out.startswith("gcloud run deploy demo-test ")
    assert "--region europe-west4" in out
    assert "--cpu 1" in out and "--memory 2Gi" in out
    assert "--max-instances 1" in out and "--min-instances 0" in out
    assert "10" not in out.split("--max-instances")[1].split()[0]


@needs_bash
def test_deploy_script_dry_run_does_not_need_gcloud() -> None:
    path = os.pathsep.join(
        p for p in os.environ.get("PATH", "").split(os.pathsep) if not (Path(p) / "gcloud").exists()
    )
    clean = {k: v for k, v in os.environ.items() if k != "PATH"}
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**clean, "PATH": path, "PROJECT": "p", "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
