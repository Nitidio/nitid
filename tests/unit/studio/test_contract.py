"""Executable checks for the Nitid Studio API contract (``studio/api-contract``).

What is verified:

* ``openapi.json`` is a valid OpenAPI 3.1 document and every internal ``$ref`` resolves.
* ``onnx-meta.schema.json`` is a valid JSON Schema 2020-12 document.
* Every ``example`` embedded in the OpenAPI document validates against its own schema.
* Every valid fixture passes and every invalid fixture fails, for the schema it is named after.
* The ``OnnxMeta`` component inside the OpenAPI document and the standalone meta schema agree.
* The endpoints and rules promised by ADR-0005 are really present in the document.

Fixture naming: ``<kind>.<name>.valid.json`` must validate, ``<kind>.invalid.<rule>.json`` must
fail JSON Schema validation, and ``<kind>.invalid-semantic.<rule>.json`` must pass the schema but
fail the cross-field check in :func:`meta_semantic_errors` (equalities JSON Schema cannot express).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")
openapi_spec_validator = pytest.importorskip("openapi_spec_validator")
referencing = pytest.importorskip("referencing")

from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402
from referencing.jsonschema import DRAFT202012  # noqa: E402

CONTRACT = Path(__file__).resolve().parents[3] / "studio" / "api-contract"
FIXTURES = CONTRACT / "fixtures"
URN = "urn:nitid-studio:openapi"

# fixture kind (file name prefix) -> component schema in openapi.json, or None for the meta schema file
KINDS: dict[str, str | None] = {
    "meta": None,
    "health": "Health",
    "devices": "Devices",
    "models": "ModelList",
    "job": "Job",
    "run-params": "RunParams",
    "run": "Run",
    "error": "ErrorResponse",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


OPENAPI = _load(CONTRACT / "openapi.json")
META_SCHEMA = _load(CONTRACT / "onnx-meta.schema.json")
REGISTRY = Registry().with_resource(URN, Resource.from_contents(OPENAPI, DRAFT202012))


def _validator_for_component(name: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"{URN}#/components/schemas/{name}"}, registry=REGISTRY)


def _validator_for_inline(schema: dict[str, Any]) -> Draft202012Validator:
    # Point "#/components/..." refs at the OpenAPI document registered under URN.
    absolute = json.loads(json.dumps(schema).replace('"#/components/', f'"{URN}#/components/'))
    return Draft202012Validator(absolute, registry=REGISTRY)


META_VALIDATOR = Draft202012Validator(META_SCHEMA)


def meta_semantic_errors(meta: dict[str, Any]) -> list[str]:
    """Cross-field rules that JSON Schema 2020-12 cannot express."""
    errors = []
    shape = meta["input"]["shape"]
    if shape[2] != shape[3]:
        errors.append(f"input shape is not square: {shape}")
    if shape[2] != meta["imgsz"] or shape[3] != meta["imgsz"]:
        errors.append(f"input shape {shape} does not match imgsz={meta['imgsz']}")
    return errors


def _validator_for_kind(kind: str) -> Draft202012Validator:
    component = KINDS[kind]
    return META_VALIDATOR if component is None else _validator_for_component(component)


def _fixtures(marker: str) -> list[Path]:
    return sorted(p for p in FIXTURES.glob("*.json") if f".{marker}" in p.name)


def _kind(path: Path) -> str:
    return path.name.split(".", 1)[0]


# --------------------------------------------------------------------------- documents


def test_openapi_is_valid_openapi_31() -> None:
    assert OPENAPI["openapi"].startswith("3.1")
    openapi_spec_validator.validate(OPENAPI)


def test_meta_schema_is_valid_2020_12() -> None:
    assert META_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(META_SCHEMA)


def test_all_component_schemas_are_valid_2020_12() -> None:
    for name, schema in OPENAPI["components"]["schemas"].items():
        Draft202012Validator.check_schema(schema)
        assert schema is not None, name


def _walk_refs(node: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                found.append((path, value))
            else:
                found.extend(_walk_refs(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_walk_refs(value, f"{path}/{i}"))
    return found


def _resolve_pointer(doc: Any, ref: str) -> Any:
    assert ref.startswith("#/"), f"external or non-pointer $ref not allowed: {ref}"
    node = doc
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@pytest.mark.parametrize("doc_name", ["openapi", "meta_schema"])
def test_every_ref_resolves(doc_name: str) -> None:
    doc = OPENAPI if doc_name == "openapi" else META_SCHEMA
    refs = _walk_refs(doc)
    assert refs, "expected at least one $ref"
    for where, ref in refs:
        _resolve_pointer(doc, ref)  # raises KeyError/IndexError when dangling


def test_no_component_is_orphaned() -> None:
    """Every component schema is reachable from a path, another schema, or is a documented root."""
    text = json.dumps(OPENAPI)
    for name in OPENAPI["components"]["schemas"]:
        assert f"#/components/schemas/{name}" in text, f"schema {name} is never referenced"


# --------------------------------------------------------------------------- embedded examples


def _media_examples() -> list[tuple[str, dict[str, Any], Any]]:
    out = []
    for path, item in OPENAPI["paths"].items():
        for method, op in item.items():
            responses = dict(op.get("responses", {}))
            for status, resp in responses.items():
                for mime, media in resp.get("content", {}).items():
                    if "example" in media:
                        out.append(
                            (
                                f"{method.upper()} {path} {status} {mime}",
                                media["schema"],
                                media["example"],
                            )
                        )
    for name, resp in OPENAPI["components"]["responses"].items():
        for mime, media in resp["content"].items():
            out.append((f"components.responses.{name} {mime}", media["schema"], media["example"]))
    return out


def test_there_are_embedded_examples() -> None:
    assert len(_media_examples()) >= 15


@pytest.mark.parametrize(
    "label,schema,example", _media_examples(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_media_type_examples_validate(label: str, schema: dict[str, Any], example: Any) -> None:
    _validator_for_inline(schema).validate(example)


def _schema_examples() -> list[tuple[str, int, Any]]:
    return [
        (name, i, ex)
        for name, schema in OPENAPI["components"]["schemas"].items()
        for i, ex in enumerate(schema.get("examples", []))
    ]


@pytest.mark.parametrize("name,index,example", _schema_examples(), ids=lambda v: str(v))
def test_component_schema_examples_validate(name: str, index: int, example: Any) -> None:
    _validator_for_component(name).validate(example)


# --------------------------------------------------------------------------- fixtures


def test_fixture_directory_is_complete_and_well_named() -> None:
    files = sorted(FIXTURES.glob("*.json"))
    assert files, "no fixtures found"
    for path in files:
        assert _kind(path) in KINDS, f"unknown fixture kind: {path.name}"
        assert (
            ".valid." in path.name or ".invalid." in path.name or ".invalid-semantic." in path.name
        ), f"fixture name must say valid/invalid/invalid-semantic: {path.name}"
    for kind in KINDS:
        kinds = [p for p in files if _kind(p) == kind]
        assert any(".valid." in p.name for p in kinds), f"no valid fixture for {kind}"
        assert any(".invalid" in p.name for p in kinds), f"no invalid fixture for {kind}"


@pytest.mark.parametrize("path", _fixtures("valid."), ids=lambda p: p.name)
def test_valid_fixtures_pass(path: Path) -> None:
    data = _load(path)
    _validator_for_kind(_kind(path)).validate(data)
    if _kind(path) == "meta":
        assert meta_semantic_errors(data) == []


_SCHEMA_INVALID = [p for p in _fixtures("invalid.") if ".invalid-semantic." not in p.name]


@pytest.mark.parametrize("path", _SCHEMA_INVALID, ids=lambda p: p.name)
def test_invalid_fixtures_fail(path: Path) -> None:
    validator = _validator_for_kind(_kind(path))
    assert not validator.is_valid(_load(path)), f"{path.name} unexpectedly validates"


@pytest.mark.parametrize("path", _fixtures("invalid-semantic."), ids=lambda p: p.name)
def test_semantic_invalid_meta_passes_schema_but_fails_cross_checks(path: Path) -> None:
    data = _load(path)
    META_VALIDATOR.validate(data)
    assert meta_semantic_errors(data), f"{path.name} should violate a cross-field rule"


# --------------------------------------------------------------------------- meta rules (direct)


def _base(task: str) -> dict[str, Any]:
    return _load(FIXTURES / f"meta.{task}.valid.json")


def test_meta_sha256_is_optional_but_checked_when_present() -> None:
    meta = _base("detect")
    assert "sha256" not in meta
    assert META_VALIDATOR.is_valid(meta)
    meta["sha256"] = "0" * 64
    assert META_VALIDATOR.is_valid(meta)
    meta["sha256"] = "0" * 63
    assert not META_VALIDATOR.is_valid(meta)
    meta["sha256"] = "G" * 64
    assert not META_VALIDATOR.is_valid(meta)


def test_meta_masks_only_in_segment() -> None:
    detect, segment = _base("detect"), _base("segment")
    assert [o["name"] for o in detect["outputs"]] == ["labels", "boxes", "scores"]
    assert [o["name"] for o in segment["outputs"]] == ["labels", "boxes", "scores", "masks"]
    assert segment["outputs"][3]["stride"] == 4
    swapped = copy.deepcopy(detect)
    swapped["task"] = "segment"
    assert not META_VALIDATOR.is_valid(swapped)


def test_meta_names_is_the_only_open_object() -> None:
    def closed(schema: dict[str, Any], where: str) -> None:
        if schema.get("type") == "object" and "properties" in schema:
            assert schema.get("additionalProperties") is False, where
        for key, value in schema.items():
            if isinstance(value, dict):
                closed(value, f"{where}/{key}")

    closed(META_SCHEMA, "")
    names = META_SCHEMA["properties"]["names"]
    assert names["additionalProperties"] == {"type": "string"}


def test_meta_required_keys_match_canonical_form() -> None:
    assert set(META_SCHEMA["required"]) == {
        "schema_version", "task", "model", "imgsz", "input", "outputs",
        "names", "topk", "default_iou", "nitid_version",
    }  # fmt: skip
    assert "sha256" in META_SCHEMA["properties"]
    assert "sha256" not in META_SCHEMA["required"]


# --------------------------------------------------------------------------- OpenAPI <-> meta schema


def test_openapi_onnxmeta_component_matches_standalone_schema() -> None:
    body = {k: v for k, v in copy.deepcopy(META_SCHEMA).items() if k not in ("$schema", "$id")}
    defs = body.pop("$defs")
    expected = json.loads(json.dumps(body).replace("#/$defs/", "#/components/schemas/OnnxMeta_"))
    actual = copy.deepcopy(OPENAPI["components"]["schemas"]["OnnxMeta"])
    actual.pop("examples")
    assert actual == expected
    for name, definition in defs.items():
        assert OPENAPI["components"]["schemas"][f"OnnxMeta_{name}"] == json.loads(
            json.dumps(definition).replace("#/$defs/", "#/components/schemas/OnnxMeta_")
        )


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("meta.*.json")), ids=lambda p: p.name)
def test_embedded_and_standalone_meta_schemas_agree_on_fixtures(path: Path) -> None:
    data = _load(path)
    assert _validator_for_component("OnnxMeta").is_valid(data) == META_VALIDATOR.is_valid(data)


# --------------------------------------------------------------------------- promised surface

EXPECTED_OPERATIONS = {
    ("get", "/health"),
    ("get", "/devices"),
    ("get", "/models"),
    ("post", "/models/{id}/download"),
    ("get", "/jobs/{job_id}"),
    ("post", "/runs"),
    ("get", "/runs"),
    ("get", "/runs/{id}"),
    ("delete", "/runs/{id}"),
    ("get", "/runs/{id}/items/{n}/image"),
    ("get", "/runs/{id}/items/{n}.json"),
    ("get", "/runs/{id}/export.csv"),
    ("get", "/runs/{id}/export.zip"),
    ("get", "/onnx/{model}/meta"),
    ("get", "/onnx/{model}"),
}


def test_all_promised_operations_exist_and_nothing_else() -> None:
    actual = {(m, p) for p, item in OPENAPI["paths"].items() for m in item if m != "parameters"}
    assert actual == EXPECTED_OPERATIONS


def test_operation_ids_are_unique() -> None:
    ids = [op["operationId"] for item in OPENAPI["paths"].values() for op in item.values()]
    assert len(ids) == len(set(ids))


def test_security_is_bearer_with_public_health() -> None:
    assert OPENAPI["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
        "description": OPENAPI["components"]["securitySchemes"]["bearerAuth"]["description"],
    }
    assert OPENAPI["security"] == [{"bearerAuth": []}]
    assert OPENAPI["paths"]["/health"]["get"]["security"] == []
    for path, item in OPENAPI["paths"].items():
        for method, op in item.items():
            if path != "/health":
                assert "security" not in op, f"{method} {path} must inherit global bearerAuth"


def test_every_error_response_uses_the_common_envelope() -> None:
    for path, item in OPENAPI["paths"].items():
        for method, op in item.items():
            for status, resp in op["responses"].items():
                if status.startswith(("4", "5")):
                    assert "$ref" in resp, f"{method} {path} {status}"
                    name = resp["$ref"].rsplit("/", 1)[1]
                    schema = OPENAPI["components"]["responses"][name]["content"][
                        "application/json"
                    ]["schema"]
                    assert schema == {"$ref": "#/components/schemas/ErrorResponse"}


def test_upload_limits_are_documented() -> None:
    op = OPENAPI["paths"]["/runs"]["post"]
    assert op["x-limits"] == {"max_image_bytes": 25 * 1024 * 1024, "max_image_megapixels": 20}
    assert "25 MB" in op["description"] and "20 Mpx" in op["description"]


def test_onnx_binary_documents_etag_and_conditional_get() -> None:
    op = OPENAPI["paths"]["/onnx/{model}"]["get"]
    assert "ETag" in op["responses"]["200"]["headers"]
    assert "304" in op["responses"]
    assert any(p["name"] == "If-None-Match" and p["in"] == "header" for p in op["parameters"])


def test_run_params_cover_the_agreed_fields() -> None:
    props = OPENAPI["components"]["schemas"]["RunParams"]["properties"]
    assert set(props) == {
        "model", "task", "conf", "iou", "imgsz", "classes",
        "mask_threshold", "backend", "device", "frame_step",
    }  # fmt: skip
    assert OPENAPI["components"]["schemas"]["Backend"]["enum"] == ["torch", "openvino"]
    assert OPENAPI["components"]["schemas"]["Task"]["enum"] == ["detect", "segment"]


def test_status_enums() -> None:
    schemas = OPENAPI["components"]["schemas"]
    expected = ["pending", "running", "done", "failed"]
    assert schemas["RunStatus"]["enum"] == expected
    assert schemas["Job"]["properties"]["status"]["enum"] == expected
