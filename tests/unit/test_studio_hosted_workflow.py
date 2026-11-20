"""Lint of .github/workflows/studio-hosted.yml: manual only, deploy gated, no secret in the file."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TEXT = (WORKFLOWS / "studio-hosted.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(TEXT)


def triggers() -> dict:
    return WORKFLOW.get("on", WORKFLOW.get(True))  # YAML 1.1 reads a bare `on` as True


def test_runs_only_when_started_by_hand():
    assert set(triggers()) == {"workflow_dispatch"}


def test_deploy_is_an_explicit_boolean_input_defaulting_to_off():
    deploy = triggers()["workflow_dispatch"]["inputs"]["deploy"]
    assert deploy["type"] == "boolean" and deploy["default"] is False


def test_deploy_job_is_gated_and_last():
    jobs = WORKFLOW["jobs"]
    gate = jobs["deploy"]["if"]
    assert "inputs.deploy == true" in gate and "workflow_dispatch" in gate
    assert jobs["deploy"]["needs"] == "assemble"
    assert set(jobs["assemble"]["needs"]) == {"build-ui", "publish-bundle"}
    others = [name for name in jobs if name != "deploy"]
    assert all("if" not in jobs[name] or "deploy" not in str(jobs[name]["if"]) for name in others)


def test_only_the_deploy_job_uses_cloudflare_secrets():
    for name, job in WORKFLOW["jobs"].items():
        uses_secret = "secrets.CLOUDFLARE" in str(job)
        assert uses_secret == (name == "deploy"), name
    assert "secrets.CLOUDFLARE_API_TOKEN" in TEXT and "secrets.CLOUDFLARE_ACCOUNT_ID" in TEXT


def test_smoke_test_runs_before_any_deploy():
    steps = WORKFLOW["jobs"]["assemble"]["steps"]
    commands = " ".join(str(s.get("run", "")) for s in steps)
    assert "pnpm run smoke:hosted" in commands
    assert commands.index("assemble_hosted.py") < commands.index("smoke:hosted")


def test_permissions_are_read_only_and_no_literal_credentials():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert not re.search(
        r"(?i)(api[_-]?token|account[_-]?id)\s*[:=]\s*['\"]?[A-Za-z0-9]{20,}", TEXT
    )


def test_models_are_served_from_the_same_origin():
    assert WORKFLOW["env"]["STUDIO_MODELS_BASE_URL"] == "/models/"
