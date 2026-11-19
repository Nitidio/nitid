"""Server vs browser parity, with the real stack in real Chrome.

The browser half (``parity.mjs``) loads the engine harness from the real Studio app, downloads the
ONNX from ``GET /api/v1/onnx/{model}`` and runs ORT-web in a worker. The server half is
``POST /api/v1/runs`` (PyTorch, CPU) on the same image. The detections are compared here.

Tolerances are the module constants below; ``docs/studio-e2e.md`` explains how they were chosen.
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import httpx
import numpy as np
import pytest
from support import Skipped, failed_checks, run_driver

# The server rounds boxes to 0.01 px and confidences to 1e-4 before sending them (so a rounding
# alone explains up to 0.005 px and 5e-5); ORT-web (wasm, float32) and PyTorch add float noise on
# top. Measured maxima (0.0017 px, 5.0e-5, mask IoU 1.0) are in docs/studio-e2e.md.
BOX_TOL_PX = 0.02
BOX_MIN_IOU = 0.999
SCORE_TOL = 1e-4
MASK_MIN_IOU = 0.99
IMGSZ = 640
CONFS = [0.5, 0.1, 0.9]
CASES = {
    "detect": {"model": "tiny-det", "task": "detect"},
    "segment": {"model": "tiny-seg", "task": "segment"},
}


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    area = lambda r: max(0.0, r["x2"] - r["x1"]) * max(0.0, r["y2"] - r["y1"])  # noqa: E731
    union = area(a) + area(b) - inter
    return 1.0 if union == 0 else inter / union


def server_run(origin: str, token: str, image: bytes, case: dict, conf: float) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    params = {**case, "backend": "torch", "device": "cpu", "conf": conf, "imgsz": IMGSZ}
    with httpx.Client(base_url=f"{origin}/api/v1", headers=headers, timeout=120) as client:
        posted = client.post(
            "/runs",
            files={"files": ("image.png", image, "image/png")},
            data={"params": json.dumps(params)},
        )
        assert posted.status_code == 202, posted.text
        run_id = posted.json()["id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            run = client.get(f"/runs/{run_id}").json()
            if run["status"] in ("done", "failed"):
                assert run["status"] == "done", run
                return run["items"][0]
            time.sleep(0.05)
    raise AssertionError("server run did not finish")


def server_mask(detection: dict, width: int, height: int) -> np.ndarray:
    """Rasterise the server's outline (largest external contour) like cv2.findContours inverts."""
    canvas = np.zeros((height, width), np.uint8)
    points = np.stack([detection["segments"]["x"], detection["segments"]["y"]], axis=1)
    cv2.fillPoly(canvas, [np.rint(points).astype(np.int32)], 1)
    return canvas


def browser_mask(detection: dict, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    mask = detection["mask"]
    crop = np.frombuffer(base64.b64decode(mask["data"]), np.uint8).reshape(
        mask["height"], mask["width"]
    )
    canvas[mask["y"] : mask["y"] + mask["height"], mask["x"] : mask["x"] + mask["width"]] = crop
    return canvas


def compare(server: dict, browser: dict) -> dict:
    """Pair detections per class by best box IoU and measure the differences."""
    metrics = {
        "server_count": len(server["detections"]),
        "browser_count": len(browser["detections"]),
        "max_box_px": 0.0,
        "min_box_iou": 1.0,
        "max_score": 0.0,
        "min_mask_iou": 1.0,
        "classes_equal": True,
        "names_equal": True,
        "masks_compared": 0,
    }
    remaining = list(browser["detections"])
    width, height = server["width"], server["height"]
    for s in server["detections"]:
        candidates = [b for b in remaining if b["class_id"] == s["class_id"]]
        if not candidates:
            metrics["classes_equal"] = False
            continue
        best = max(
            candidates,
            key=lambda b: iou(s["box"], b["box"]) - abs(s["confidence"] - b["confidence"]),
        )
        remaining.remove(best)
        metrics["names_equal"] &= s["name"] == best["name"]
        metrics["max_box_px"] = max(
            metrics["max_box_px"],
            *(abs(s["box"][k] - best["box"][k]) for k in ("x1", "y1", "x2", "y2")),
        )
        metrics["min_box_iou"] = min(metrics["min_box_iou"], iou(s["box"], best["box"]))
        metrics["max_score"] = max(metrics["max_score"], abs(s["confidence"] - best["confidence"]))
        if s.get("has_mask") or best.get("mask"):
            sm, bm = server_mask(s, width, height), browser_mask(best, width, height)
            union = int(np.logical_or(sm, bm).sum())
            metrics["min_mask_iou"] = min(
                metrics["min_mask_iou"],
                1.0 if union == 0 else float(np.logical_and(sm, bm).sum()) / union,
            )
            metrics["masks_compared"] += 1
    metrics["classes_equal"] &= not remaining
    return metrics


@pytest.fixture(scope="module")
def browser_side(harness, tmp_path_factory):
    out = tmp_path_factory.mktemp("parity") / "browser.json"
    try:
        result = run_driver("parity.mjs", out, "--url", harness.url, timeout=900)
    except Skipped as why:
        pytest.skip(str(why))
    print(result["log"])
    assert not failed_checks(result), "\n".join(failed_checks(result))
    return result


@pytest.mark.parametrize("kind", ["detect", "segment"])
def test_browser_engine_matches_server(kind, harness, browser_side, fixtures):
    image = fixtures["image"].read_bytes()
    report = {}
    for conf in CONFS:
        browser = browser_side["data"]["runs"][kind]["results"][str(conf)]
        server = server_run(harness.origin, harness.token, image, CASES[kind], conf)
        report[conf] = compare(server, {**browser, "detections": browser["detections"]})
    print("PARITY", kind, json.dumps(report, indent=1))
    for conf, m in report.items():
        where = f"{kind} conf={conf}"
        assert m["server_count"] == m["browser_count"], f"{where}: detection count {m}"
        assert m["classes_equal"] and m["names_equal"], f"{where}: classes {m}"
        assert m["max_box_px"] <= BOX_TOL_PX, f"{where}: box {m}"
        assert m["min_box_iou"] >= BOX_MIN_IOU, f"{where}: box iou {m}"
        assert m["max_score"] <= SCORE_TOL, f"{where}: score {m}"
        if kind == "segment":
            assert m["min_mask_iou"] >= MASK_MIN_IOU, f"{where}: mask {m}"
    assert report[0.5]["server_count"] > 0, "the fixture must produce detections at conf 0.5"
    assert report[0.9]["server_count"] == 0, "conf 0.9 must be above every score"


def test_environment_is_recorded(browser_side):
    print("ENVIRONMENT", json.dumps(browser_side["data"]["environment"], indent=1))
    print("WEBGPU", json.dumps(browser_side["data"]["webgpuProbe"], indent=1))
    print("WORKER DOWNLOAD", browser_side["data"]["workerDownloadWithoutToken"])
    print("ENGINE INFO", json.dumps(browser_side["data"]["runs"]["segment"]["info"], indent=1))
