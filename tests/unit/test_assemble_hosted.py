"""Unit tests for scripts/assemble_hosted.py (the gate before a hosted deploy)."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "assemble_hosted", REPO_ROOT / "scripts/assemble_hosted.py"
)
assert _spec is not None and _spec.loader is not None
assemble_hosted = importlib.util.module_from_spec(_spec)
sys.modules["assemble_hosted"] = assemble_hosted  # dataclasses need the module registered
_spec.loader.exec_module(assemble_hosted)

MODELS_BASE = "https://models.example.test/"
TEMPLATE = "/*\n  X-Test: 1\n  Content-Security-Policy: default-src 'self'; connect-src 'self'{{MODELS_CONNECT_SRC}}; x\n"

ONNX = b"not really onnx, but bytes are bytes"
SHA = hashlib.sha256(ONNX).hexdigest()


def make_ui(root: Path) -> Path:
    (root / "assets").mkdir(parents=True)
    (root / "ort").mkdir()
    (root / "index.html").write_text("<!doctype html>", encoding="utf-8")
    # The built UI has its models base URL baked in (vite `define`); the assembler looks for it.
    (root / "assets" / "index-abc.js").write_text(f'const base="{MODELS_BASE}";', encoding="utf-8")
    (root / "ort" / "ort-wasm-simd-threaded.wasm").write_bytes(b"\0asm")
    return root


def make_bundle(
    root: Path, *, sha: str = SHA, entries: list | None = None, onnx: bytes = ONNX
) -> Path:
    root.mkdir(parents=True)
    (root / "nitid1s-detect.onnx").write_bytes(onnx)
    catalog = {
        "models": entries
        or [{"id": "nitid1s", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": sha}]
    }
    (root / "models.json").write_text(json.dumps(catalog), encoding="utf-8")
    digest = hashlib.sha256(onnx).hexdigest()
    (root / "SHA256SUMS").write_text(f"{digest}  nitid1s-detect.onnx\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    return root


@pytest.fixture
def parts(tmp_path):
    headers = tmp_path / "_headers"
    headers.write_text(TEMPLATE, encoding="utf-8")
    return {
        "ui": make_ui(tmp_path / "ui"),
        "bundle": make_bundle(tmp_path / "bundle"),
        "headers": headers,
        "out": tmp_path / "site",
    }


def run(parts, **kwargs):
    """Same-origin mode: the bundle is copied into the site."""
    return assemble_hosted.assemble(
        parts["ui"], parts["out"], parts["headers"], bundle=parts["bundle"], **kwargs
    )


def run_cross(parts, **kwargs):
    """Cross-origin mode: the models live elsewhere, nothing of the bundle is copied."""
    kwargs.setdefault("models_base_url", MODELS_BASE)
    return assemble_hosted.assemble(parts["ui"], parts["out"], parts["headers"], **kwargs)


def csp_of(out: Path) -> str:
    line = next(
        ln
        for ln in (out / "_headers").read_text(encoding="utf-8").splitlines()
        if "Content-Sec" in ln
    )
    return line.split(":", 1)[1].strip()


def problems_of(parts) -> list[str]:
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run(parts)
    return info.value.problems


def test_assembles_ui_bundle_and_headers_into_one_folder(parts):
    report = run(parts)
    out = parts["out"]
    assert (out / "index.html").is_file()
    assert (out / "ort" / "ort-wasm-simd-threaded.wasm").is_file()
    assert (
        csp_of(out) == "default-src 'self'; connect-src 'self'; x"
    )  # same origin: no extra source
    assert (out / "models" / "models.json").is_file()
    assert (out / "models" / "nitid1s-detect.onnx").read_bytes() == ONNX
    assert (out / "models" / "LICENSE").is_file()  # licence files travel with the models
    assert report.models == ["nitid1s"]
    assert report.files == 8
    assert report.warnings == []
    # The inputs are never modified.
    assert not (parts["ui"] / "models").exists()
    # No staging leftovers next to the output.
    assert [p.name for p in out.parent.iterdir() if p.name.startswith(".site-")] == []


def test_missing_models_json(parts):
    (parts["bundle"] / "models.json").unlink()
    assert any("no models.json" in p for p in problems_of(parts))
    assert not parts["out"].exists()


@pytest.mark.parametrize("body", ["not json", "[]", '{"models": []}', '{"models": {}}'])
def test_unusable_catalogue(parts, body):
    (parts["bundle"] / "models.json").write_text(body, encoding="utf-8")
    assert problems_of(parts)


def test_sha256_mismatch_in_catalogue(parts):
    make_bundle_again = parts["bundle"] / "models.json"
    make_bundle_again.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "a", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": "0" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert any("does not match models.json" in p for p in problems_of(parts))


def test_sha256_mismatch_in_sums(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(
        f"{'1' * 64}  nitid1s-detect.onnx\n", encoding="utf-8"
    )
    assert any("SHA256SUMS: sha256 of nitid1s-detect.onnx" in p for p in problems_of(parts))


def test_sums_may_name_missing_or_unparseable_files(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(f"{SHA}  gone.onnx\nnonsense\n", encoding="utf-8")
    problems = problems_of(parts)
    assert any("'gone.onnx' is not a file" in p for p in problems)
    assert any("cannot parse" in p for p in problems)


def test_binary_mode_marker_in_sums_is_accepted(parts):
    (parts["bundle"] / "SHA256SUMS").write_text(f"{SHA} *nitid1s-detect.onnx\n", encoding="utf-8")
    run(parts)


def test_referenced_file_must_exist(parts):
    (parts["bundle"] / "nitid1s-detect.onnx").unlink()
    (parts["bundle"] / "SHA256SUMS").unlink()
    assert any("file does not exist" in p for p in problems_of(parts))


@pytest.mark.parametrize(
    "url", ["../secret.onnx", "/etc/passwd", "https://example.com/m.onnx", "a\\b.onnx", ""]
)
def test_urls_must_stay_inside_the_bundle(parts, url):
    entries = [{"id": "a", "task": "detect", "url": url, "sha256": SHA}]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    assert any(".url" in p for p in problems_of(parts))


@pytest.mark.parametrize("name", ["w.pth", "w.PT", "w.ckpt", "w.safetensors", "w.pkl"])
def test_training_checkpoints_are_refused_anywhere_in_the_bundle(parts, name):
    sub = parts["bundle"] / "extra"
    sub.mkdir()
    (sub / name).write_bytes(b"weights")
    assert any(name in p and "only ONNX" in p for p in problems_of(parts))
    assert not parts["out"].exists()


def test_symlinks_are_refused(parts):
    (parts["bundle"] / "link.onnx").symlink_to(parts["bundle"] / "nitid1s-detect.onnx")
    assert any("symlink" in p for p in problems_of(parts))


def test_cloudflare_file_size_limit(parts, monkeypatch):
    monkeypatch.setattr(assemble_hosted, "MAX_FILE_BYTES", 10)
    problems = problems_of(parts)
    assert any("allows 25 MiB per file" in p for p in problems)


def test_cloudflare_file_count_limit(parts, monkeypatch):
    monkeypatch.setattr(assemble_hosted, "MAX_FILES", 2)
    assert any("allows 2 per version" in p for p in problems_of(parts))


def test_missing_sha256_and_sums_are_warnings_not_errors(parts):
    entries = [{"id": "a", "task": "detect", "url": "nitid1s-detect.onnx"}]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    (parts["bundle"] / "SHA256SUMS").unlink()
    (parts["bundle"] / "stray.onnx").write_bytes(b"x")
    report = run(parts)
    text = " | ".join(report.warnings)
    assert "no sha256" in text and "no SHA256SUMS" in text and "stray.onnx is not listed" in text


def test_duplicate_and_malformed_entries(parts):
    entry = {"id": "a", "task": "detect", "url": "nitid1s-detect.onnx", "sha256": SHA}
    entries = [
        entry,
        entry,
        {"id": "b", "task": "pose", "url": "nitid1s-detect.onnx"},
        "x",
        {"id": 3},
    ]
    (parts["bundle"] / "models.json").write_text(json.dumps({"models": entries}), encoding="utf-8")
    problems = " | ".join(problems_of(parts))
    for fragment in (
        "duplicate model",
        '.task must be "detect"',
        "is not an object",
        ".id must be",
    ):
        assert fragment in problems


def test_ui_must_be_built(parts):
    (parts["ui"] / "index.html").unlink()
    (parts["ui"] / "ort" / "ort-wasm-simd-threaded.wasm").unlink()
    problems = " | ".join(problems_of(parts))
    assert "no index.html" in problems and "no ort/*.wasm" in problems


def test_ui_must_not_already_have_models(parts):
    (parts["ui"] / "models").mkdir()
    assert any("reserved for the bundle" in p for p in problems_of(parts))


def test_headers_template_must_exist(parts):
    parts["headers"].unlink()
    assert any("headers template" in p for p in problems_of(parts))


def test_output_must_be_empty_or_cleaned(parts):
    parts["out"].mkdir()
    (parts["out"] / "old.txt").write_text("old", encoding="utf-8")
    assert any("not empty" in p for p in problems_of(parts))
    assert (parts["out"] / "old.txt").exists()  # a refused run touches nothing
    run(parts, clean=True)
    assert not (parts["out"] / "old.txt").exists()
    assert (parts["out"] / "index.html").is_file()


def test_output_must_not_overlap_inputs(parts):
    parts["out"] = parts["bundle"]
    assert any("overlaps" in p for p in problems_of(parts))
    parts["out"] = parts["ui"] / "sub"
    assert any("overlaps" in p for p in problems_of(parts))


def test_failed_run_leaves_no_staging_folder(parts, monkeypatch):
    monkeypatch.setattr(
        assemble_hosted.shutil, "copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        run(parts)
    assert not parts["out"].exists()
    assert [p.name for p in parts["out"].parent.iterdir() if p.name.startswith(".site-")] == []


def test_main_exit_codes_and_messages(parts, capsys):
    argv = [
        "--ui",
        str(parts["ui"]),
        "--bundle",
        str(parts["bundle"]),
        "--headers-template",
        str(parts["headers"]),
        "--out",
        str(parts["out"]),
    ]
    assert assemble_hosted.main(argv) == 0
    assert "assembled" in capsys.readouterr().out
    assert assemble_hosted.main([*argv, "--clean"]) == 0
    assert assemble_hosted.main(argv) == 1  # not empty any more, no --clean
    assert "not empty" in capsys.readouterr().err


def test_real_headers_template_is_the_default():
    assert assemble_hosted.DEFAULT_HEADERS.is_file()
    assert (
        assemble_hosted.DEFAULT_HEADERS == REPO_ROOT / "studio/hosting/cloudflare/_headers.template"
    )
    assert not (REPO_ROOT / "studio/hosting/cloudflare/_headers").exists(), (
        "a committed _headers would be deployed without the models origin"
    )


# --------------------------------------------------------------------------- cross-origin mode


def test_cross_origin_copies_no_bundle_and_allows_exactly_the_models_origin(parts):
    report = run_cross(parts)
    out = parts["out"]
    assert report.mode == "cross-origin" and report.models_origin == "https://models.example.test"
    assert not (out / "models").exists()
    assert not list(out.rglob("*.onnx"))
    assert (out / "index.html").is_file() and (
        out / "ort" / "ort-wasm-simd-threaded.wasm"
    ).is_file()
    assert csp_of(out) == "default-src 'self'; connect-src 'self' https://models.example.test; x"
    assert "{{" not in (out / "_headers").read_text(encoding="utf-8")


def test_cross_origin_origin_comes_from_the_base_url_even_with_a_path(parts):
    parts["ui"].joinpath("assets", "index-abc.js").write_text(
        'const b="https://Models.Example.test:8443/nitid/v1/";', encoding="utf-8"
    )
    report = run_cross(parts, models_base_url="https://Models.Example.test:8443/nitid/v1/")
    assert report.models_origin == "https://models.example.test:8443"
    assert csp_of(parts["out"]).endswith("https://models.example.test:8443; x")


def test_cross_origin_with_models_origin_instead_of_base_url(parts):
    run_cross(parts, models_base_url=None, models_origin="https://models.example.test")
    assert "connect-src 'self' https://models.example.test;" in csp_of(parts["out"])


def test_cross_origin_applies_no_size_limit_to_a_validated_bundle(parts, tmp_path, monkeypatch):
    """The bundle never enters the site: a 'large' model is fine there (nitid1s is 39.7 MiB)."""
    big = b"x" * 500  # "large" for this test: the UI files are all smaller than the limit below
    entries = [
        {
            "id": "nitid1s",
            "task": "detect",
            "url": "nitid1s-detect.onnx",
            "sha256": hashlib.sha256(big).hexdigest(),
        }
    ]
    large = make_bundle(tmp_path / "large", entries=entries, onnx=big)
    monkeypatch.setattr(assemble_hosted, "MAX_FILE_BYTES", 300)
    report = run_cross(parts, validate_bundle_dir=large, bundle_check="basic")
    assert report.models == ["nitid1s"]
    assert not (parts["out"] / "models").exists()
    # ... whereas the same files copied into the site are refused.
    parts["bundle"] = large
    parts["out"] = parts["out"].parent / "site2"
    assert any("allows 25 MiB per file" in p for p in problems_of(parts))


def test_cross_origin_still_limits_the_files_of_the_site(parts, monkeypatch):
    monkeypatch.setattr(assemble_hosted, "MAX_FILE_BYTES", 2)  # the UI's own files are bigger
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts)
    assert any("ui:" in p and "25 MiB" in p for p in info.value.problems)


def test_cross_origin_validate_bundle_reports_the_same_problems(parts):
    bad = make_bundle(parts["out"].parent / "bad", sha="0" * 64)
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, validate_bundle_dir=bad, bundle_check="basic")
    assert any(p.startswith("--validate-bundle:") and "sha256" in p for p in info.value.problems)
    assert not parts["out"].exists()


def test_cross_origin_validate_bundle_refuses_checkpoints(parts):
    (parts["bundle"] / "dfine_l.pth").write_bytes(b"x")
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, validate_bundle_dir=parts["bundle"], bundle_check="basic")
    assert any(".pth" in p for p in info.value.problems)


def test_cross_origin_full_check_runs_tools_check_hosted_bundle(parts):
    """The fixture is not a real bundle (its ONNX is garbage): the strict tool must object, not crash.

    The positive case (a real bundle passes) is test_check_hosted_bundle.py::test_assemble_hosted_full_check.
    """
    pytest.importorskip("jsonschema")
    pytest.importorskip("onnx")
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, validate_bundle_dir=parts["bundle"])
    text = " | ".join(info.value.problems)
    assert "--validate-bundle:" in text and "could not read the bundle" in text


def test_full_check_without_dev_dependencies_is_a_clear_problem(parts, monkeypatch):
    real_import = builtins.__import__

    def no_tools(name, *args, **kwargs):
        if name == "tools":
            raise ImportError("No module named 'onnx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tools)
    problems = assemble_hosted.full_bundle_problems(parts["bundle"])
    assert problems and "--bundle-check basic" in problems[0]


def test_cross_origin_ui_must_be_built_for_the_same_models_url(parts):
    parts["ui"].joinpath("assets", "index-abc.js").write_text(
        'const b="/models/";', encoding="utf-8"
    )
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts)
    message = " ".join(info.value.problems)
    assert "was not built for https://models.example.test/" in message
    assert "STUDIO_MODELS_BASE_URL=https://models.example.test/ pnpm run build:hosted" in message


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"models_base_url": "http://models.example.test/"}, "must use https"),
        ({"models_base_url": "https://*.example.test/"}, "wildcards"),
        ({"models_base_url": "https://models.example.test;script-src *"}, "not allowed"),
        ({"models_base_url": "https://user:pw@models.example.test/"}, "credentials"),
        ({"models_base_url": "https://models.example.test/?a=1"}, "query"),
        ({"models_base_url": "ftp://models.example.test/"}, "https"),
        ({"models_base_url": "models.example.test"}, "https"),
        ({"models_base_url": ""}, "empty"),
        (
            {"models_base_url": None, "models_origin": "https://models.example.test/models/"},
            "no path",
        ),
        ({"models_base_url": None, "models_origin": "https://*"}, "wildcards"),
    ],
)
def test_cross_origin_refuses_an_unsafe_origin(parts, kwargs, fragment):
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, **kwargs)
    assert fragment in " ".join(info.value.problems)
    assert not parts["out"].exists()


def test_exactly_one_models_location_is_required(parts):
    for kwargs in (
        {},  # none
        {"bundle": parts["bundle"], "models_base_url": MODELS_BASE},
        {"models_base_url": MODELS_BASE, "models_origin": "https://models.example.test"},
    ):
        with pytest.raises(assemble_hosted.AssembleError) as info:
            assemble_hosted.assemble(parts["ui"], parts["out"], parts["headers"], **kwargs)
        assert "exactly one of" in " ".join(info.value.problems)
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run(parts, validate_bundle_dir=parts["bundle"])
    assert "--validate-bundle" in " ".join(info.value.problems)


def test_loopback_http_needs_the_explicit_test_flag(parts):
    parts["ui"].joinpath("assets", "index-abc.js").write_text(
        'const b="http://127.0.0.1:9000/";', encoding="utf-8"
    )
    kwargs = {"models_base_url": "http://127.0.0.1:9000/"}
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, **kwargs)
    assert "must use https" in " ".join(info.value.problems)
    run_cross(parts, allow_http_loopback=True, **kwargs)
    assert "connect-src 'self' http://127.0.0.1:9000;" in csp_of(parts["out"])


def test_loopback_flag_does_not_open_other_http_hosts(parts):
    with pytest.raises(assemble_hosted.AssembleError) as info:
        run_cross(parts, models_base_url="http://models.example.test/", allow_http_loopback=True)
    assert "localhost only" in " ".join(info.value.problems)


def test_main_cross_origin_and_usage_errors(parts, capsys):
    base = ["--ui", str(parts["ui"]), "--headers-template", str(parts["headers"])]
    argv = [*base, "--out", str(parts["out"]), "--models-base-url", MODELS_BASE]
    assert assemble_hosted.main(argv) == 0
    out = capsys.readouterr().out
    assert "models hosted at https://models.example.test" in out and "bundle not validated" in out
    assert (
        assemble_hosted.main(
            [*argv, "--clean", "--validate-bundle", str(parts["bundle"]), "--bundle-check", "basic"]
        )
        == 0
    )
    assert "validated: nitid1s" in capsys.readouterr().out
    assert (
        assemble_hosted.main(
            [*base, "--out", str(parts["out"]), "--models-base-url", "http://x.test/", "--clean"]
        )
        == 1
    )
    assert "must use https" in capsys.readouterr().err
    for bad in ([], ["--bundle", "b", "--models-base-url", MODELS_BASE]):
        with pytest.raises(SystemExit) as exit_info:
            assemble_hosted.main([*base, *bad])
        assert exit_info.value.code == 2


# --------------------------------------------------------------------------- the headers generator


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://models.nitid.io", "https://models.nitid.io"),
        ("https://models.nitid.io/", "https://models.nitid.io"),
        ("HTTPS://Models.Nitid.IO", "https://models.nitid.io"),
        ("https://models.nitid.io:443", "https://models.nitid.io"),
        ("https://models.nitid.io:8443", "https://models.nitid.io:8443"),
        ("https://a.b.c.example.org", "https://a.b.c.example.org"),
    ],
)
def test_parse_models_origin_accepts_and_normalises(value, expected):
    assert assemble_hosted.parse_models_origin(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "http://models.nitid.io",
        "https://*.nitid.io",
        "*",
        "https://models.nitid.io/models/",
        "https://models.nitid.io/x",
        "https://models.nitid.io?x=1",
        "https://models.nitid.io#f",
        "https://u:p@models.nitid.io",
        "https://models.nitid.io;img-src *",
        "https://models.nitid.io 'unsafe-eval'",
        "https://models.nitid.io, https://evil.test",
        "models.nitid.io",
        "//models.nitid.io",
        "https://",
        "https://-bad.example.org",
        "https://models.nitid.io:99999",
        "https://models.nitid.io:abc",
        "https://[::1]",
        "file:///etc/passwd",
        "javascript:alert(1)",
    ],
)
def test_parse_models_origin_refuses(value):
    with pytest.raises(assemble_hosted.OriginError):
        assemble_hosted.parse_models_origin(value)


def test_origin_of_base_url_keeps_only_the_origin():
    assert (
        assemble_hosted.origin_of_base_url("https://m.example.org/a/b/") == "https://m.example.org"
    )
    with pytest.raises(assemble_hosted.OriginError):
        assemble_hosted.origin_of_base_url("http://m.example.org/a/")


def test_render_headers_fills_only_the_connect_src_placeholder():
    template = (REPO_ROOT / "studio/hosting/cloudflare/_headers.template").read_text(
        encoding="utf-8"
    )
    same = assemble_hosted.render_headers(template, None)
    cross = assemble_hosted.render_headers(template, "https://models.nitid.io")
    assert "{{" not in "".join(ln for ln in same.splitlines() if not ln.startswith("#"))
    same_lines, cross_lines = same.splitlines(), cross.splitlines()
    assert len(same_lines) == len(cross_lines)
    changed = [(a, b) for a, b in zip(same_lines, cross_lines, strict=True) if a != b]
    assert len(changed) == 1
    assert "connect-src 'self';" in changed[0][0]
    assert "connect-src 'self' https://models.nitid.io;" in changed[0][1]
    assert changed[0][1] == changed[0][0].replace(
        "connect-src 'self';", "connect-src 'self' https://models.nitid.io;", 1
    )


@pytest.mark.parametrize(
    "template",
    [
        "/*\n  Content-Security-Policy: connect-src 'self'\n",  # no placeholder
        "/*\n  A: connect-src 'self'{{MODELS_CONNECT_SRC}}\n  B: connect-src 'self'{{MODELS_CONNECT_SRC}}\n",
        "/*\n  A: x {{MODELS_CONNECT_SRC}}\n",  # not after connect-src 'self'
        "/*\n  A: connect-src 'self'{{MODELS_CONNECT_SRC}} {{OTHER}}\n",
    ],
)
def test_render_headers_requires_one_well_placed_placeholder(template):
    with pytest.raises(assemble_hosted.AssembleError):
        assemble_hosted.render_headers(template, "https://models.nitid.io")


def test_render_headers_checks_the_origin_again():
    template = "/*\n  A: connect-src 'self'{{MODELS_CONNECT_SRC}}\n"
    for bad in (
        "http://models.nitid.io",
        "https://*.nitid.io",
        "https://models.nitid.io/x",
        "https://MODELS.nitid.io",
    ):
        with pytest.raises((assemble_hosted.OriginError, assemble_hosted.AssembleError)):
            assemble_hosted.render_headers(template, bad)
