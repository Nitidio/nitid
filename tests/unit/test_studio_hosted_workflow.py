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


def test_models_url_is_a_repository_variable_not_a_secret_and_not_hardcoded():
    assert "STUDIO_MODELS_BASE_URL" not in (WORKFLOW.get("env") or {}), "no workflow-wide value"
    assert "vars.STUDIO_MODELS_BASE_URL" in TEXT
    assert "secrets.STUDIO_MODELS_BASE_URL" not in TEXT
    assert not re.search(r"STUDIO_MODELS_BASE_URL:\s*['\"]?(https?://|/)", TEXT)
    assert not re.search(
        r"https://models\.[a-z0-9.-]+/",
        "\n".join(
            ln
            for ln in TEXT.splitlines()
            if not ln.lstrip().startswith("#") and "::error" not in ln
        ),
    )


def test_ui_is_built_for_the_models_variable():
    steps = WORKFLOW["jobs"]["build-ui"]["steps"]
    build = next(s for s in steps if s.get("run") == "pnpm run build:hosted")
    assert build["env"]["STUDIO_MODELS_BASE_URL"] == "${{ vars.STUDIO_MODELS_BASE_URL }}"


def test_deploy_fails_clearly_when_the_variable_is_empty():
    jobs = WORKFLOW["jobs"]
    for job, needs_gate in (("build-ui", True), ("deploy", False)):
        guard = next(
            s
            for s in jobs[job]["steps"]
            if "STUDIO_MODELS_BASE_URL is empty" in str(s.get("run", ""))
        )
        assert '[ -z "$STUDIO_MODELS_BASE_URL" ]' in guard["run"] and "exit 1" in guard["run"]
        assert "::error::" in guard["run"] and "docs/studio-hosted.md" in guard["run"]
        assert guard["env"]["STUDIO_MODELS_BASE_URL"] == "${{ vars.STUDIO_MODELS_BASE_URL }}"
        if needs_gate:  # the dry run must not need the variable
            assert "inputs.deploy == true" in guard["if"]
        build_index = [s.get("run") for s in jobs[job]["steps"]].index(guard["run"])
        assert build_index >= 0


def test_the_guard_runs_before_the_ui_is_built():
    runs = [str(s.get("run", "")) for s in WORKFLOW["jobs"]["build-ui"]["steps"]]
    guard = next(i for i, r in enumerate(runs) if "is empty" in r)
    build = runs.index("pnpm run build:hosted")
    assert guard < build


def test_bundle_is_published_with_default_sizes_as_an_artifact_only():
    jobs = WORKFLOW["jobs"]
    publish = next(
        s["run"]
        for s in jobs["publish-bundle"]["steps"]
        if "--out" in str(s.get("run", "")) and "publish_models" in str(s.get("run", ""))
    )
    # Run as a module: `python tools/publish_models.py` puts tools/ (not the repo root) on sys.path and the
    # script's `from tools.check_models_lock import ...` fails with ModuleNotFoundError (seen on the first run).
    assert "python -m tools.publish_models" in publish
    assert "tools/publish_models.py" not in publish
    assert "--sizes" not in publish, "default sizes (n, s): the 25 MiB limit no longer applies"
    uploads = [
        s
        for s in jobs["publish-bundle"]["steps"]
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads and uploads[0]["with"]["name"] == "hosted-bundle"
    assert "tools.check_hosted_bundle" in " ".join(
        str(s.get("run", "")) for s in jobs["publish-bundle"]["steps"]
    )


def test_site_is_assembled_cross_origin_and_never_with_the_bundle_inside():
    steps = WORKFLOW["jobs"]["assemble"]["steps"]
    commands = [
        str(s.get("run", "")) for s in steps if "assemble_hosted.py" in str(s.get("run", ""))
    ]
    assert len(commands) == 1
    command = commands[0]
    assert '--models-base-url "$STUDIO_MODELS_BASE_URL"' in command
    assert "--validate-bundle bundle" in command
    assert "--bundle bundle" not in command.replace("--validate-bundle bundle", "")
    assert "--allow-http-loopback" not in TEXT, "plain http is for the local smoke test only"


def test_wrangler_is_pinned_to_a_version_that_applies_headers():
    deploy = WORKFLOW["jobs"]["deploy"]["steps"]
    step = next(
        s for s in deploy if str(s.get("uses", "")).startswith("cloudflare/wrangler-action")
    )
    version = str(step["with"].get("wranglerVersion", ""))
    # The action installs wrangler 3.90.0 when it finds none; the first real deploy then lacked COOP/COEP/CSP.
    assert version and version[0].isdigit() and int(version.split(".")[0]) >= 4
