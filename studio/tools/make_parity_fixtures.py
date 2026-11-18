"""Generate the tiny JSON parity fixtures used by the Studio engine tests.

The browser engine (``studio/src/engine``) must reproduce what the Python
predictor produces. This script builds a *tiny* random checkpoint at run time
(in the style of ``scripts/verify_install.py``), exports it to ONNX, runs both
the PyTorch predictor and ONNX Runtime on the same synthetic image and writes
what the TypeScript tests need to compare against. No binary (no ``.onnx``,
no PNG) is ever written to the repository; the image is a deterministic
formula that the TypeScript tests re-implement.

Run from the repository root (or anywhere) with the worktree on the path::

    PYTHONPATH=<repo> <venv>/bin/python studio/tools/make_parity_fixtures.py

Output: ``studio/src/engine/__fixtures__/parity.{detect,segment}.json``.
The files are regenerated, never edited by hand. Total size must stay below
100 KB (checked at the end of the script).

Facts replicated, with their source in the Python code base:

* preprocessing: BGR ndarray -> RGB PIL -> ``T.Resize((S, S))`` -> ``T.ToTensor()``
  (``dfine/utils/sources.py:331-341``). The tensor that really reaches the model
  is captured with a forward pre-hook, so the fixture records the real thing.
* detections: ``DFINEPredictor._postprocess`` (``dfine/predictor.py:377-470``):
  ``scores > conf``, ``isin(labels, classes)``, ``clip_boxes``, masks bilinear to
  the original size (``align_corners=False``), ``>= mask_threshold``, then
  ``crop_masks_to_boxes``. No NMS unless ``augment=True``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "src" / "engine" / "__fixtures__"
MAX_TOTAL_BYTES = 100 * 1024

IMGSZ = 64
CLASS_NAMES = {0: "cat", 1: "dog"}
# Image used for the end-to-end case (smaller than IMGSZ on purpose: upscaling,
# non-square, so the "squash" resize and the box rescaling are both exercised).
MAIN_IMAGE = {"generator": "gradient-v1", "width": 40, "height": 30}
# Extra preprocessing-only cases: a downscale (where Pillow's antialiasing
# matters) and a square no-op.
PREPROCESS_CASES = [
    {"generator": "gradient-v1", "width": 150, "height": 110},
    {"generator": "gradient-v1", "width": 64, "height": 64},
    {"generator": "gradient-v1", "width": 97, "height": 33},
]
N_SAMPLES = 48


def synth_image(width: int, height: int) -> np.ndarray:
    """Deterministic RGB uint8 image, ``gradient-v1``. Mirrored in TypeScript."""
    ys, xs = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), dtype=np.int64)
    img[..., 0] = xs * 255 // max(width - 1, 1)
    img[..., 1] = ys * 255 // max(height - 1, 1)
    img[..., 2] = (xs * 7 + ys * 13) % 256
    # A bright rectangle and a dark one, so that edges exist.
    x0, x1, y0, y1 = width // 5, width // 2, height // 4, height // 2
    img[y0:y1, x0:x1] = (250, 40, 40)
    x0, x1, y0, y1 = width // 2, width * 4 // 5, height // 2, height * 4 // 5
    img[y0:y1, x0:x1] = (20, 30, 220)
    return img.astype(np.uint8)


def make_checkpoint(root: Path, task: str) -> Path:
    """Tiny checkpoint in the style of ``scripts/verify_install.py``.

    Weights are random but seeded; a few head parameters are perturbed so scores
    and boxes are varied (a freshly initialised model gives ~equal scores and
    tiny boxes, which would make the parity test uninformative).
    """
    from dfine.nn.build import build_model
    from dfine.nn.configs import make_model_config

    config = make_model_config("dfine_s", task=task, num_classes=2, image_size=(IMGSZ, IMGSZ))
    config["DFINETransformer"]["num_layers"] = 1
    config["DFINETransformer"]["num_queries"] = 10
    config["DFINETransformer"]["num_denoising"] = 0
    config["HybridEncoder"]["depth_mult"] = 0.1
    model = build_model(config)
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        sd = model.state_dict()
        # An untrained tiny net gives the same decoder feature to every query, so
        # per-query score variety cannot be injected. Class biases make the two
        # classes separable by `conf` (scores tie *within* a class; documented).
        sd["decoder.dec_score_head.0.weight"].copy_(torch.randn(2, 256, generator=gen) * 0.02)
        sd["decoder.dec_score_head.0.bias"].copy_(torch.tensor([-0.5, 0.8]))
        # Large, varied boxes (some clipped by the image border) instead of ~4 px ones.
        sd["decoder.enc_bbox_head.layers.2.bias"].add_(torch.tensor([0.0, 0.0, 2.5, 2.5]))
        if task == "segment":
            # Masks with real spatial structure instead of ~0 everywhere.
            sd["decoder.mask_head.layers.2.bias"].add_(torch.randn(256, generator=gen) + 0.1)
    path = root / f"{task}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "names": CLASS_NAMES,
            "epoch": -1,
            "metrics": {},
        },
        path,
    )
    return path


def r(value: float, digits: int = 5) -> float:
    return round(float(value), digits)


def rle(mask: np.ndarray) -> list[int]:
    """Run lengths of the flattened mask, starting with a run of zeros."""
    flat = mask.reshape(-1).astype(np.uint8)
    runs: list[int] = []
    current = 0
    count = 0
    for v in flat:
        if int(v) == current:
            count += 1
        else:
            runs.append(count)
            current = int(v)
            count = 1
    runs.append(count)
    return runs


def preprocess_expected(width: int, height: int) -> dict[str, Any]:
    """Run the real Pillow/torchvision transform and summarise the tensor."""
    import torchvision.transforms as T
    from PIL import Image

    rgb = synth_image(width, height)
    tensor = T.Compose([T.Resize((IMGSZ, IMGSZ)), T.ToTensor()])(Image.fromarray(rgb))
    levels = torch.round(tensor * 255).to(torch.int64)  # ToTensor divides uint8 by 255
    assert torch.allclose(levels.float() / 255.0, tensor, atol=1e-6)
    rng = np.random.RandomState(7)
    cs = rng.randint(0, 3, N_SAMPLES)
    ys = rng.randint(0, IMGSZ, N_SAMPLES)
    xs = rng.randint(0, IMGSZ, N_SAMPLES)
    # Always include the four corners of channel 0.
    cs[:4], ys[:4], xs[:4] = 0, [0, 0, IMGSZ - 1, IMGSZ - 1], [0, IMGSZ - 1, 0, IMGSZ - 1]
    return {
        "image": {"generator": "gradient-v1", "width": width, "height": height},
        "channel_level_sums": [int(levels[c].sum()) for c in range(3)],
        "samples": [
            {"c": int(c), "y": int(y), "x": int(x), "level": int(levels[c, y, x])}
            for c, y, x in zip(cs, ys, xs)
        ],
    }


def pick_conf(scores: list[float]) -> float:
    """A threshold in the widest gap among the top ~12 scores, away from any score."""
    ordered = sorted(scores, reverse=True)
    best_gap, best_conf = -1.0, 0.5
    for hi, lo in zip(ordered[3:12], ordered[4:13]):
        if hi - lo > best_gap:
            best_gap, best_conf = hi - lo, (hi + lo) / 2
    return round(best_conf, 4)


def run_task(task: str, root: Path) -> dict[str, Any]:
    import onnxruntime as ort

    from dfine import DFINE, __version__
    from dfine.export_meta import read_onnx_meta

    checkpoint = make_checkpoint(root, task)
    model = DFINE(checkpoint, task=task, device="cpu", verbose=False)

    # --- the image and the tensor the predictor really feeds the network -----
    rgb = synth_image(MAIN_IMAGE["width"], MAIN_IMAGE["height"])
    bgr = np.ascontiguousarray(rgb[..., ::-1])  # sources.py:_preprocess expects BGR
    captured: list[torch.Tensor] = []
    deployed = model._get_deployed_model()
    handle = deployed.register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))

    def predict(conf: float, classes=None, mask_threshold: float = 0.5):
        return model.predict(
            bgr,
            conf=conf,
            classes=classes,
            mask_threshold=mask_threshold,
            imgsz=IMGSZ,
            verbose=False,
        )[0]

    # conf=0 keeps every row: gives all scores to choose a safe threshold.
    full = predict(0.0)
    scores_all = [float(s) for s in full.boxes.data[:, 4]]
    conf = pick_conf(scores_all)
    tensor = captured[0].numpy()  # [1,3,S,S] float32, exactly what Python feeds
    handle.remove()
    assert tensor.shape == (1, 3, IMGSZ, IMGSZ)

    # --- ONNX: postprocessed and raw layouts -----------------------------------
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, save_dir=root / f"onnx-{task}", verbose=False)
    meta = read_onnx_meta(onnx_path)
    assert meta is not None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(out_names, session.run(None, {"images": tensor})))

    raw_path = model.export(
        format="onnx", imgsz=IMGSZ, postprocess=False, save_dir=root / f"raw-{task}", verbose=False
    )
    raw_meta = read_onnx_meta(raw_path)
    raw_session = ort.InferenceSession(str(raw_path), providers=["CPUExecutionProvider"])
    raw_outputs = dict(
        zip([o.name for o in raw_session.get_outputs()], raw_session.run(None, {"images": tensor}))
    )

    def det_list(result) -> list[dict[str, Any]]:
        data = result.boxes.data.numpy()
        masks = result.masks.data.numpy() if result.masks is not None else None
        items = []
        for i, row in enumerate(data):
            item: dict[str, Any] = {
                "box": [r(v, 4) for v in row[:4]],
                "score": r(row[4], 6),
                "class_id": int(row[5]),
                "name": CLASS_NAMES[int(row[5])],
            }
            if masks is not None:
                item["mask_rle"] = rle(masks[i])
                item["mask_area"] = int(masks[i].sum())
            items.append(item)
        return items

    cases = []
    for name, kwargs in [
        ("default", {"conf": conf}),  # conf inside the gap: only the high-score class
        ("low_conf_all_classes", {"conf": 0.1}),
        ("class_filter", {"conf": 0.1, "classes": [0]}),
        ("low_mask_threshold", {"conf": conf, "mask_threshold": 0.3}),
        ("no_detections", {"conf": 0.95}),
    ]:
        if task == "detect" and name == "low_mask_threshold":
            continue
        result = predict(**kwargs)
        cases.append(
            {
                "name": name,
                "conf": kwargs["conf"],
                "classes": kwargs.get("classes"),
                "mask_threshold": kwargs.get("mask_threshold", 0.5),
                "detections": det_list(result),
            }
        )

    # Sanity: ONNX-postprocessed agrees with torch on the full (conf=0) list.
    onnx_scores = np.sort(outputs["scores"][0])[::-1]
    torch_scores = np.sort(np.array(scores_all))[::-1]
    assert np.allclose(onnx_scores, torch_scores, atol=1e-4), "ONNX and torch scores diverge"

    def flat(a: np.ndarray, digits: int = 5):
        return [r(v, digits) for v in a.reshape(-1)]

    raw_post = {
        "labels": [int(v) for v in outputs["labels"].reshape(-1)],
        "boxes": flat(outputs["boxes"], 4),
        "scores": flat(outputs["scores"], 6),
    }
    raw_layout = {
        "pred_logits": {"dims": list(raw_outputs["pred_logits"].shape), "data": flat(raw_outputs["pred_logits"], 5)},
        "pred_boxes": {"dims": list(raw_outputs["pred_boxes"].shape), "data": flat(raw_outputs["pred_boxes"], 6)},
    }
    if task == "segment":
        raw_post["masks"] = {"dims": list(outputs["masks"].shape), "data": flat(outputs["masks"], 4)}
        raw_layout["pred_masks"] = {
            "dims": list(raw_outputs["pred_masks"].shape),
            "data": flat(raw_outputs["pred_masks"], 4),
        }

    return {
        "fixture_version": 1,
        "task": task,
        "generated_with": {
            "nitid": __version__,
            "torch": torch.__version__.split("+")[0],
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "imgsz": IMGSZ,
        "image": MAIN_IMAGE,
        "meta": meta,
        "raw_meta": raw_meta,
        "preprocess": [preprocess_expected(MAIN_IMAGE["width"], MAIN_IMAGE["height"])]
        + [preprocess_expected(c["width"], c["height"]) for c in PREPROCESS_CASES],
        "postprocessed_outputs": {**raw_post, "dims": {k: list(v.shape) for k, v in outputs.items()}},
        "raw_outputs": raw_layout,
        "cases": cases,
        "all_scores": [r(s, 6) for s in scores_all],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.TemporaryDirectory(prefix="nitid-parity-") as directory:
        for task in ("detect", "segment"):
            torch.manual_seed(0)
            fixture = run_task(task, Path(directory))
            text = json.dumps(fixture, separators=(",", ":"), sort_keys=True) + "\n"
            path = OUT_DIR / f"parity.{task}.json"
            path.write_text(text, encoding="utf-8")
            total += len(text.encode())
            counts = {c["name"]: len(c["detections"]) for c in fixture["cases"]}
            print(f"{path.relative_to(HERE.parent.parent)}: {len(text)} bytes, detections {counts}")
    print(f"total {total} bytes (limit {MAX_TOTAL_BYTES})")
    if total > MAX_TOTAL_BYTES:
        sys.exit("fixtures exceed the size budget")


if __name__ == "__main__":
    main()
