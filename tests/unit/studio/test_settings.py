"""Unit tests for Studio settings validation and argument parsing (no FastAPI needed)."""

import dataclasses

import pytest

from nitid.studio.settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MIN_TOKEN_LENGTH,
    SettingsError,
    StudioSettings,
    is_loopback_host,
    normalize_host,
    parse_args,
)


def test_defaults_are_loopback_with_random_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = StudioSettings()

    assert settings.host == DEFAULT_HOST == "127.0.0.1"
    assert settings.port == DEFAULT_PORT == 8000
    assert settings.models_dir == (tmp_path / "models").resolve()
    assert settings.open_browser is True
    assert settings.is_loopback
    assert settings.api_token and len(settings.api_token) >= 32


def test_random_token_differs_between_instances():
    assert StudioSettings().api_token != StudioSettings().api_token


def test_settings_are_immutable():
    settings = StudioSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.port = 9999  # type: ignore[misc]


def test_replace_keeps_generated_token():
    settings = StudioSettings()
    replaced = dataclasses.replace(settings, port=1234)
    assert replaced.port == 1234
    assert replaced.api_token == settings.api_token


def test_repr_does_not_leak_token():
    settings = StudioSettings(token="supersecret-token")
    assert "supersecret-token" not in repr(settings)


def test_named_limits():
    assert MAX_IMAGE_BYTES == 25 * 1024 * 1024
    assert MAX_IMAGE_PIXELS == 20_000_000


@pytest.mark.parametrize("port", [-1, 65536, 100000])
def test_port_out_of_range_is_rejected(port):
    with pytest.raises(SettingsError, match="port"):
        StudioSettings(port=port)


@pytest.mark.parametrize("port", [0, 1, 8000, 65535])
def test_port_bounds_are_accepted(port):
    assert StudioSettings(port=port).port == port


@pytest.mark.parametrize("port", ["80", 80.0, None, True])
def test_port_must_be_an_int(port):
    with pytest.raises(SettingsError, match="integer"):
        StudioSettings(port=port)  # type: ignore[arg-type]


@pytest.mark.parametrize("host", ["", "  ", "bad host", "http://x", "a/b", "a@b", "-x"])
def test_invalid_hosts_are_rejected(host):
    with pytest.raises(SettingsError):
        StudioSettings(host=host, token="long-enough-token")


def test_ipv6_brackets_are_normalised():
    assert normalize_host("[::1]") == "::1"
    assert StudioSettings(host="[::1]").host == "::1"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.5", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("studio.example.com", False),
    ],
)
def test_is_loopback_host(host, expected):
    assert is_loopback_host(host) is expected


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "studio.example.com"])
def test_non_loopback_host_requires_explicit_token(host):
    with pytest.raises(SettingsError, match="requires an explicit token"):
        StudioSettings(host=host)
    settings = StudioSettings(host=host, token="long-enough-token")
    assert not settings.is_loopback


@pytest.mark.parametrize("token", ["short", "has space in it", "tab\tcharacter-x", "a" * 7])
def test_bad_tokens_are_rejected(token):
    with pytest.raises(SettingsError, match="token"):
        StudioSettings(token=token)


def test_minimum_token_length_is_accepted():
    assert StudioSettings(token="x" * MIN_TOKEN_LENGTH).api_token == "x" * MIN_TOKEN_LENGTH


def test_models_dir_may_not_be_a_file(tmp_path):
    file_path = tmp_path / "afile"
    file_path.write_text("x")
    with pytest.raises(SettingsError, match="not a directory"):
        StudioSettings(models_dir=file_path)


def test_models_dir_need_not_exist_yet(tmp_path):
    target = tmp_path / "later" / "models"
    assert StudioSettings(models_dir=target).models_dir == target.resolve()
    assert not target.exists()  # validation never creates directories


# --- command-line parsing -------------------------------------------------------------------


def test_parse_args_empty_uses_defaults():
    settings = parse_args([])
    assert (settings.host, settings.port, settings.open_browser) == ("127.0.0.1", 8000, True)


def test_parse_args_all_options(tmp_path):
    settings = parse_args(
        [
            "host=0.0.0.0",
            "port=0",
            f"models={tmp_path}",
            "open=false",
            "token=abcdefgh1234",
        ]
    )
    assert settings.host == "0.0.0.0"
    assert settings.port == 0
    assert settings.models_dir == tmp_path.resolve()
    assert settings.open_browser is False
    assert settings.api_token == "abcdefgh1234"


def test_parse_args_no_browser_flag():
    assert parse_args(["--no-browser"]).open_browser is False
    assert parse_args(["open=true", "--no-browser"]).open_browser is False


@pytest.mark.parametrize(
    ("raw", "expected"), [("true", True), ("1", True), ("False", False), ("no", False)]
)
def test_parse_args_open_values(raw, expected):
    assert parse_args([f"open={raw}"]).open_browser is expected


def test_parse_args_keeps_token_as_raw_string():
    # No numeric coercion: leading zeros survive.
    assert parse_args(["token=00112233"]).api_token == "00112233"


def test_parse_args_host_wildcard_without_token_fails():
    with pytest.raises(SettingsError, match="explicit token"):
        parse_args(["host=0.0.0.0"])


@pytest.mark.parametrize(
    "tokens",
    [
        ["port=abc"],
        ["port=99999"],
        ["open=maybe"],
        ["unknown=1"],
        ["justaword"],
        ["models="],
        ["token="],
        ["--bogus"],
    ],
)
def test_parse_args_rejects_bad_input(tokens):
    with pytest.raises(SettingsError):
        parse_args(tokens)
