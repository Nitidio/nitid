"""The hosted site's Cloudflare configuration (studio/hosting/cloudflare) against the local app.

Cheap and dependency-free (no Studio extra): it parses the committed ``_headers`` and ``wrangler.jsonc``
and compares the security policy with ``nitid/studio/security.py``, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nitid.studio.security import CONTENT_SECURITY_POLICY, SECURITY_HEADERS

REPO_ROOT = Path(__file__).resolve().parents[3]
CF_DIR = REPO_ROOT / "studio" / "hosting" / "cloudflare"
HEADERS_FILE = CF_DIR / "_headers"

MAX_RULES = 100  # https://developers.cloudflare.com/workers/static-assets/headers/
MAX_LINE = 2000


def parse_headers(text: str) -> list[tuple[str, list[tuple[str, str | None]]]]:
    """``[(pattern, [(name, value | None)])]``; ``None`` is a detach (``! Name``)."""
    rules: list[tuple[str, list[tuple[str, str | None]]]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        assert len(raw) <= MAX_LINE, f"_headers line {number} exceeds {MAX_LINE} characters"
        if not raw[0].isspace():
            assert raw.startswith("/"), f"line {number}: pattern must start with /"
            rules.append((raw.strip(), []))
            continue
        assert rules, f"line {number}: header before a pattern"
        body = raw.strip()
        if body.startswith("! "):
            rules[-1][1].append((body[2:].strip(), None))
        else:
            name, sep, value = body.partition(":")
            assert sep and name.strip(), f"line {number}: expected 'Name: value'"
            rules[-1][1].append((name.strip(), value.strip()))
    return rules


def headers_for(rules, path: str) -> dict[str, str]:
    """Cloudflare's documented combination: file order, join with ', ', ``! Name`` detaches."""
    out: dict[str, str] = {}
    for pattern, entries in rules:
        regex = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$"
        if not re.match(regex, path):
            continue
        for name, value in entries:
            key = name.lower()
            if value is None:
                out.pop(key, None)
            elif key in out:
                out[key] = f"{out[key]}, {value}"
            else:
                out[key] = value
    return out


def parse_csp(policy: str) -> dict[str, frozenset[str]]:
    directives: dict[str, frozenset[str]] = {}
    for part in policy.split(";"):
        tokens = part.split()
        if tokens:
            assert tokens[0] not in directives, f"duplicate CSP directive {tokens[0]}"
            directives[tokens[0]] = frozenset(tokens[1:])
    return directives


@pytest.fixture(scope="module")
def rules():
    return parse_headers(HEADERS_FILE.read_text(encoding="utf-8"))


PATHS = [
    "/",
    "/index.html",
    "/run/browser",  # a client route: answered with index.html by the SPA fallback
    "/assets/index-AbC123.js",
    "/ort/ort-wasm-simd-threaded.wasm",
    "/ort/ort-wasm-simd-threaded.mjs",
    "/models/models.json",
    "/models/nitid1s-detect.onnx",
]


def test_file_respects_cloudflare_limits(rules):
    assert 0 < len(rules) <= MAX_RULES
    assert all(pattern.count("*") <= 1 for pattern, _ in rules), "one splat per pattern"


@pytest.mark.parametrize("path", PATHS)
def test_every_security_header_of_the_local_app_is_present_everywhere(rules, path):
    served = headers_for(rules, path)
    for name, expected in SECURITY_HEADERS.items():
        if name == "Content-Security-Policy":
            continue  # compared directive by directive below
        assert served.get(name.lower()) == expected, f"{name} on {path}"
    assert "content-security-policy" in served


@pytest.mark.parametrize("path", PATHS)
def test_required_isolation_and_hardening_headers(rules, path):
    served = headers_for(rules, path)
    assert served["cross-origin-opener-policy"] == "same-origin"
    assert served["cross-origin-embedder-policy"] == "require-corp"
    assert served["cross-origin-resource-policy"] == "same-origin"
    assert served["x-content-type-options"] == "nosniff"
    assert served["referrer-policy"] == "no-referrer"
    permissions = served["permissions-policy"]
    for feature in ("camera", "microphone", "geolocation"):
        assert f"{feature}=()" in permissions


@pytest.mark.parametrize("path", PATHS)
def test_csp_allows_everything_the_local_app_needs(rules, path):
    """Hosted must allow every source the local policy allows (the same app runs under both)."""
    hosted = parse_csp(headers_for(rules, path)["content-security-policy"])
    local = parse_csp(CONTENT_SECURITY_POLICY)
    for directive, sources in local.items():
        assert directive in hosted, f"hosted CSP lacks {directive}"
        assert sources <= hosted[directive], (
            f"hosted {directive} lacks {sources - hosted[directive]}"
        )


@pytest.mark.parametrize("path", PATHS)
def test_csp_is_not_looser_than_the_local_one(rules, path):
    """... and nothing more: no extra directive, no extra source (a CDN, 'unsafe-eval', a wildcard)."""
    hosted = parse_csp(headers_for(rules, path)["content-security-policy"])
    local = parse_csp(CONTENT_SECURITY_POLICY)
    assert hosted == local


def test_csp_has_no_dangerous_sources(rules):
    csp = headers_for(rules, "/")["content-security-policy"]
    assert "'unsafe-eval'" not in csp
    assert "*" not in csp
    assert "http:" not in csp and "https:" not in csp
    parsed = parse_csp(csp)
    assert parsed["script-src"] == {"'self'", "'wasm-unsafe-eval'"}
    assert parsed["connect-src"] == {"'self'"}
    assert parsed["frame-ancestors"] == {"'none'"}


def test_caching_policy(rules):
    immutable = "public, max-age=31536000, immutable"
    assert headers_for(rules, "/assets/index-AbC123.js")["cache-control"] == immutable
    for path in ("/", "/index.html", "/run/browser", "/models/models.json"):
        assert headers_for(rules, path)["cache-control"] == "no-cache", path
    # The ORT files keep fixed names, so they are revalidated (ETag), never frozen for a year.
    assert (
        "immutable" not in headers_for(rules, "/ort/ort-wasm-simd-threaded.wasm")["cache-control"]
    )


def test_cache_control_is_never_joined_from_two_rules(rules):
    """Cloudflare joins repeated headers with commas; for Cache-Control that would be contradictory."""
    for path in PATHS:
        value = headers_for(rules, path)["cache-control"]
        assert not ("no-cache" in value and "immutable" in value), path


def test_content_type_is_left_to_the_file_extension(rules):
    """Setting Content-Type in _headers could be joined with the inferred one: not done on purpose."""
    names = {name.lower() for _, entries in rules for name, _ in entries}
    assert "content-type" not in names


def test_file_holds_no_secret_or_vendor_url():
    lines = HEADERS_FILE.read_text(encoding="utf-8").splitlines()
    text = "\n".join(
        ln for ln in lines if not ln.lstrip().startswith("#")
    )  # comments may mention them
    assert not re.search(r"https?://", text)
    assert not re.search(r"(?i)token|secret|password|api[_-]?key", text)


def load_jsonc(path: Path) -> dict:
    """JSONC with whole-line ``//`` comments and no trailing commas (that is all the file uses)."""
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("//")
    ]
    return json.loads("\n".join(lines))


def test_wrangler_config_serves_the_assembled_directory_as_a_spa():
    config = load_jsonc(CF_DIR / "wrangler.jsonc")
    assert config["name"] == "nitid-app"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", config["compatibility_date"])
    assets = config["assets"]
    assert assets["not_found_handling"] == "single-page-application"
    assert "main" not in config, (
        "assets-only: no Worker script, so _headers applies to every response"
    )
    assert "run_worker_first" not in assets
    # The directory is the one scripts/assemble_hosted.py writes by default.
    resolved = (CF_DIR / assets["directory"]).resolve()
    assert resolved == REPO_ROOT / "studio" / "dist-hosted-site"
    assert not re.search(
        r"(?i)token|secret|account_id", json.dumps(config)
    )  # secrets live in GitHub
