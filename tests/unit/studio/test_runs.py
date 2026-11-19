"""Unit tests for ``nitid.studio.runs`` (parameters, images, serialisation, store)."""

import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi", reason="needs the 'studio' extra")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dfine.results import Boxes, Masks, Results  # noqa: E402
from nitid.studio import runs  # noqa: E402
from nitid.studio.runs import (  # noqa: E402
    ParamsError,
    PayloadTooLarge,
    RunParams,
    UnsupportedImage,
    parse_run_params,
)

MINIMAL = {"model": "nitid1s", "task": "detect", "backend": "torch"}


def png(width=8, height=6) -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((height, width, 3), np.uint8))
    assert ok
    return buf.tobytes()


# --- parse_run_params ---------------------------------------------------------------------


def test_minimal_params_get_defaults():
    p = parse_run_params(json.dumps(MINIMAL))
    assert p == RunParams("nitid1s", "detect", "torch")
    assert (p.conf, p.iou, p.imgsz, p.mask_threshold, p.classes, p.device) == (
        0.25,
        0.85,
        640,
        0.5,
        None,
        None,
    )


def test_full_params_roundtrip():
    p = parse_run_params(
        json.dumps(
            {
                **MINIMAL,
                "conf": 1,
                "iou": 0,
                "imgsz": 96,
                "classes": [0, 2],
                "mask_threshold": 0.3,
                "device": " cpu ",
                "frame_step": 3,
            }
        )
    )
    assert (p.conf, p.iou, p.imgsz, p.classes, p.device, p.frame_step) == (
        1.0,
        0.0,
        96,
        (0, 2),
        "cpu",
        3,
    )


@pytest.mark.parametrize(
    "text",
    ["", "nope", "[]", "3", "null", '{"model": ""}', "[" * 100000],
)
def test_non_object_params_are_rejected(text):
    with pytest.raises(ParamsError):
        parse_run_params(text)


@pytest.mark.parametrize(
    "override",
    [
        {"model": 3},
        {"model": ""},
        {"task": "semantic"},
        {"task": None},
        {"backend": "onnx"},
        {"conf": -0.1},
        {"conf": 1.01},
        {"conf": "0.5"},
        {"conf": True},
        {"iou": 2},
        {"mask_threshold": -1},
        {"imgsz": 31},
        {"imgsz": 33},
        {"imgsz": 64.0},
        {"imgsz": True},
        {"classes": "0"},
        {"classes": [0, -1]},
        {"classes": [1.5]},
        {"classes": [True]},
        {"device": 0},
        {"device": "  "},
        {"frame_step": 0},
        {"frame_step": 1.5},
        {"unknown": 1},
    ],
)
def test_invalid_param_values_are_rejected(override):
    with pytest.raises(ParamsError):
        parse_run_params(json.dumps({**MINIMAL, **override}))


def test_classes_null_is_allowed():
    assert parse_run_params(json.dumps({**MINIMAL, "classes": None})).classes is None


# --- validate_device ----------------------------------------------------------------------

DEVICES = {
    "torch": {"cpu": True, "cuda": [{"index": 0, "name": "GPU"}]},
    "openvino": {"available": True, "devices": ["CPU", "GPU"]},
}
NO_OV = {**DEVICES, "openvino": {"available": False, "devices": []}}


def params_for(backend, device=None):
    return RunParams("m", "detect", backend, device)


def test_validate_device_accepts_known_devices():
    for device in (None, "cpu", "cuda:0"):
        runs.validate_device(params_for("torch", device), DEVICES)
    for device in (None, "CPU", "GPU"):
        runs.validate_device(params_for("openvino", device), DEVICES)


@pytest.mark.parametrize(
    ("backend", "device", "devices"),
    [
        ("torch", "cuda:1", DEVICES),
        ("torch", "cuda:0", {"torch": {"cpu": True, "cuda": []}}),
        ("torch", "CPU", DEVICES),
        ("openvino", "NPU", DEVICES),
        ("openvino", "cpu", DEVICES),
        ("openvino", None, NO_OV),
    ],
)
def test_validate_device_rejects_unknown_or_unavailable(backend, device, devices):
    with pytest.raises(ParamsError):
        runs.validate_device(params_for(backend, device), devices)


# --- names, images ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cat.jpg", "cat.jpg"),
        ("../../etc/passwd", "passwd"),
        ("C:\\temp\\dog.png", "dog.png"),
        ("a\x00b\nc.png", "abc.png"),
        ("", "image-3"),
        (None, "image-3"),
        ("///", "image-3"),
        ("x" * 500, "x" * runs.MAX_NAME_LENGTH),
    ],
)
def test_safe_name(raw, expected):
    assert runs.safe_name(raw, 3) == expected


def test_inspect_image_accepts_supported_formats():
    for ext, fmt in ((".png", "PNG"), (".jpg", "JPEG"), (".bmp", "BMP"), (".webp", "WEBP")):
        ok, buf = cv2.imencode(ext, np.zeros((6, 8, 3), np.uint8))
        info = runs.inspect_image(buf.tobytes())
        assert (info.format, info.width, info.height) == (fmt, 8, 6)
        assert info.media_type.startswith("image/")


@pytest.mark.parametrize("data", [b"", b"GIF89a....", b"%PDF-1.7", png()[:20], b"\x00" * 64])
def test_inspect_image_rejects_non_images(data):
    with pytest.raises(UnsupportedImage):
        runs.inspect_image(data)


def test_inspect_image_enforces_byte_limit_before_decoding(monkeypatch):
    monkeypatch.setattr(runs, "MAX_IMAGE_BYTES", 10)
    # Garbage bigger than the limit must give 413, not "unsupported": no decoder ran.
    with pytest.raises(PayloadTooLarge):
        runs.inspect_image(b"\x00" * 11)


def test_inspect_image_enforces_pixel_limit(monkeypatch):
    monkeypatch.setattr(runs, "MAX_IMAGE_PIXELS", 47)
    with pytest.raises(PayloadTooLarge):
        runs.inspect_image(png(8, 6))  # 48 px
    monkeypatch.setattr(runs, "MAX_IMAGE_PIXELS", 48)
    assert runs.inspect_image(png(8, 6)).width == 8


def test_decode_image_returns_bgr_and_rejects_garbage():
    assert runs.decode_image(png(8, 6)).shape == (6, 8, 3)
    with pytest.raises(runs.RunError):
        runs.decode_image(b"not an image")


# --- serialisation ------------------------------------------------------------------------


def make_result(with_masks=False, names=None, empty=False):
    image = np.zeros((20, 30, 3), np.uint8)
    names = names if names is not None else {0: "cat", 1: "dog"}
    if empty:
        return Results(image, "x", names, boxes=None)
    data = torch.tensor([[1.0, 2.0, 11.0, 12.0, 0.9, 0], [3.0, 4.0, 20.0, 18.0, 0.5, 1]])
    boxes = Boxes(data, (20, 30))
    masks = None
    if with_masks:
        mask = torch.zeros((2, 20, 30))
        mask[0, 2:12, 1:11] = 1
        mask[1, 4:18, 3:20] = 1
        masks = Masks(mask, (20, 30))
    return Results(
        image, "x", names, boxes=boxes, masks=masks,
        speed={"preprocess": 1.23456, "inference": 2.0, "postprocess": 0.5},
    )  # fmt: skip


def test_result_to_item_detect():
    item = runs.result_to_item(4, "a.png", make_result())
    assert item["id"] == 4 and item["name"] == "a.png"
    assert (item["width"], item["height"]) == (30, 20)
    first = item["detections"][0]
    assert first == {
        "box": {"x1": 1.0, "y1": 2.0, "x2": 11.0, "y2": 12.0},
        "confidence": 0.9,
        "class_id": 0,
        "name": "cat",
        "has_mask": False,
    }
    assert item["speed"] == {"preprocess": 1.235, "inference": 2.0, "postprocess": 0.5}
    json.dumps(item)  # plain JSON types only


def test_result_to_item_segment_has_polygons():
    item = runs.result_to_item(0, "a.png", make_result(with_masks=True))
    for det in item["detections"]:
        assert det["has_mask"] is True
        assert len(det["segments"]["x"]) == len(det["segments"]["y"]) >= 3
    xs = item["detections"][0]["segments"]["x"]
    assert min(xs) >= 1 and max(xs) <= 11


def test_result_to_item_without_detections_and_unknown_class():
    assert runs.result_to_item(0, "a", make_result(empty=True))["detections"] == []
    item = runs.result_to_item(0, "a", make_result(names={}))
    assert [d["name"] for d in item["detections"]] == ["unknown", "unknown"]
    item = runs.result_to_item(0, "a", make_result(names={"1": "dog"}))
    assert [d["name"] for d in item["detections"]] == ["unknown", "dog"]


def test_confidence_is_clamped_to_unit_interval():
    result = make_result()
    result.boxes.data[0, 4] = 1.0000001
    assert runs.result_to_item(0, "a", result)["detections"][0]["confidence"] == 1.0


def test_items_to_csv_has_header_rows_and_neutralises_formulas():
    item = runs.result_to_item(0, "=cmd|' /C calc'!A0", make_result(names={0: "+x", 1: "dog"}))
    lines = runs.items_to_csv([item]).splitlines()
    assert lines[0] == "item,image,x1,y1,x2,y2,confidence,class_id,name"
    assert len(lines) == 3
    assert lines[1].startswith("0,'=cmd")
    assert lines[1].endswith(",0,'+x")
    assert runs.items_to_csv([]).splitlines() == [lines[0]]


def test_describe_run_error_never_uses_exception_text():
    assert runs.describe_run_error(runs.RunError("safe")) == "safe"
    assert (
        runs.describe_run_error(MemoryError("/secret")) == "Out of memory while running the model."
    )
    assert (
        runs.describe_run_error(FileNotFoundError("/secret/x"))
        == "Could not read the stored image."
    )
    assert runs.describe_run_error(KeyError("/secret")) == "Inference failed (KeyError)."


# --- resolve_model ------------------------------------------------------------------------


def test_resolve_model_official_not_downloaded(tmp_path):
    with pytest.raises(runs.ModelNotDownloaded):
        runs.resolve_model(tmp_path, "nitid1s", "detect")


def test_resolve_model_official_downloaded(tmp_path, monkeypatch):
    weights = tmp_path / "w.pth"
    weights.write_bytes(b"x")
    monkeypatch.setattr(runs.registry, "weights_path", lambda d, e: weights)
    resolved = runs.resolve_model(tmp_path, "nitid1s", "detect")
    assert resolved == runs.ResolvedModel("nitid1s", "detect", weights)


def test_resolve_model_unknown_and_wrong_task(tmp_path, monkeypatch):
    with pytest.raises(ParamsError, match="does not exist"):
        runs.resolve_model(tmp_path, "ghost", "detect")
    local = runs.registry.LocalModel("mine", "segment", None, tmp_path / "mine.pth")
    monkeypatch.setattr(runs.registry, "scan_local_models", lambda d: [local])
    with pytest.raises(ParamsError, match="no weights for task"):
        runs.resolve_model(tmp_path, "mine", "detect")
    assert runs.resolve_model(tmp_path, "mine", "segment").path == local.path
    # nano detection is registered since decision D8: known, but not downloaded yet
    with pytest.raises(runs.ModelNotDownloaded):
        runs.resolve_model(tmp_path, "nitid1n", "detect")


# --- RunStore -----------------------------------------------------------------------------


def image_entry(name="a.png"):
    data = png()
    return name, data, runs.inspect_image(data)


MODEL = runs.ResolvedModel("m", "detect", None)  # type: ignore[arg-type]
PARAMS = RunParams("m", "detect", "torch")


def test_store_create_get_list_delete(tmp_path):
    store = runs.RunStore(root=tmp_path)
    first = store.create(PARAMS, MODEL, [image_entry("one.png")])
    second = store.create(PARAMS, MODEL, [image_entry(), image_entry()])
    assert first.startswith("r_") and first != second

    snap = store.get(second)
    assert snap.status == "pending" and snap.item_count == 2 and snap.items == ()
    assert snap.payload()["items"] == []
    assert (tmp_path / second / "1.png").is_file()
    assert [s.id for s in store.list()] == [second, first]  # newest first

    assert store.delete(first) is True and store.delete(first) is False
    assert store.get(first) is None and not (tmp_path / first).exists()
    store.close()
    assert store.list() == []


def test_store_evicts_oldest_finished_run_only(tmp_path):
    store = runs.RunStore(root=tmp_path, max_runs=2)
    a = store.create(PARAMS, MODEL, [image_entry()])
    b = store.create(PARAMS, MODEL, [image_entry()])
    store.finish(store.record(a), None)
    c = store.create(PARAMS, MODEL, [image_entry()])
    assert store.get(a) is None and store.get(b) and store.get(c)
    # b and c are both active: nothing can be evicted, the table temporarily overshoots
    d = store.create(PARAMS, MODEL, [image_entry()])
    assert {s.id for s in store.list()} == {b, c, d}


def test_store_state_transitions(tmp_path):
    store = runs.RunStore(root=tmp_path)
    rid = store.create(PARAMS, MODEL, [image_entry()])
    record = store.record(rid)
    assert store.mark_running(record) and store.get(rid).status == "running"
    store.add_item(record, {"id": 0})
    store.finish(record, "boom")
    snap = store.get(rid)
    assert (snap.status, snap.error_msg, snap.items) == ("failed", "boom", ({"id": 0},))
    gone = store.create(PARAMS, MODEL, [image_entry()])
    record = store.record(gone)
    store.delete(gone)
    assert store.mark_running(record) is False  # deleted while queued


def test_store_owns_and_removes_its_temp_directory():
    store = runs.RunStore()
    store.create(PARAMS, MODEL, [image_entry()])
    root = store._root
    assert root.is_dir() and root.name.startswith("nitid-studio-runs-")
    store.close()
    assert not root.exists()


def test_build_zip_contents(tmp_path):
    store = runs.RunStore(root=tmp_path)
    rid = store.create(PARAMS, MODEL, [image_entry()])
    store.add_item(store.record(rid), runs.result_to_item(0, "a.png", make_result()))
    archive = zipfile.ZipFile(io.BytesIO(runs.build_zip(store.get(rid))))
    assert sorted(archive.namelist()) == sorted(
        ["run.json", "detections.csv", "items/0.json", "images/0.png"]
    )
    assert json.loads(archive.read("run.json"))["id"] == rid
    assert archive.read("images/0.png") == png()
