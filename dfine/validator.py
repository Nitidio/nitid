"""
DFINEValidator — COCO mAP evaluation.
Called internally by DFINE.val(). Not part of the public API.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import torch

from dfine.utils.logging import LOGGER


class DFINEValidator:
    """
    Runs COCO-style bounding-box evaluation against a labelled val split.

    The postprocessor is rebuilt from the checkpoint config so that the
    model stays in its current state (eval or train) without side-effects.
    """

    def __init__(self, model, cfg: dict, device: str, names: dict) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.names = names

    def run(
        self,
        data: str,
        imgsz: int,
        batch: int,
        conf: float,
        split: str,
        verbose: bool,
    ) -> dict:
        """
        Evaluate on a COCO-format dataset split.

        Returns a dict with COCO evaluation metrics::

            {
                "mAP50-95": float,  # AP averaged over IoU thresholds 0.50:0.05:0.95
                "mAP50":    float,  # AP at IoU=0.50
                "AR1":      float,  # Average Recall at maxDets=1
                "AR100":    float,  # Average Recall at maxDets=100
            }
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        from dfine.nn.build import build_postprocessor
        from dfine.utils.data import build_coco_dataloader, load_data_yaml

        # Resolve annotation file path (mirrors build_coco_dataloader logic)
        cfg_data = load_data_yaml(data)
        root = Path(cfg_data["path"])
        ann_key = f"{split}_ann"
        if ann_key in cfg_data:
            ann_file = root / cfg_data[ann_key]
        else:
            split_name = Path(cfg_data[split]).name
            ann_file = root / "annotations" / f"instances_{split_name}.json"

        dataloader = build_coco_dataloader(data, split=split, imgsz=imgsz, batch_size=batch)

        # Fresh postprocessor — non-deploy mode returns [{labels, boxes, scores}]
        postprocessor = build_postprocessor(self.cfg)
        postprocessor.to(self.device)
        postprocessor.eval()

        # Reverse map: 0-based label index → COCO category_id for result formatting
        cat_id_to_label = dataloader.dataset.cat_id_to_label
        label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                # All images were resized to imgsz×imgsz; pass that as orig_size.
                # Postprocessor scales boxes to absolute pixel coords in imgsz space.
                orig_sizes = torch.tensor(
                    [[imgsz, imgsz]] * len(images),
                    dtype=torch.float32,
                    device=self.device,
                )
                raw = self.model(images)
                detections = postprocessor(raw, orig_sizes)

                for det, target in zip(detections, targets):
                    img_id = int(target["image_id"][0])
                    mask = det["scores"] > conf
                    boxes  = det["boxes"][mask]   # xyxy absolute (imgsz space)
                    scores = det["scores"][mask]
                    labels = det["labels"][mask]

                    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                        x1, y1, x2, y2 = box
                        results.append({
                            "image_id":    img_id,
                            "category_id": label_to_cat_id.get(int(label), int(label) + 1),
                            "bbox":  [x1, y1, x2 - x1, y2 - y1],  # COCO format: xywh
                            "score": score,
                        })

        coco_gt = COCO(str(ann_file))

        if not results:
            if verbose:
                LOGGER.info("No detections above conf threshold — mAP=0")
            return {"mAP50-95": 0.0, "mAP50": 0.0, "AR1": 0.0, "AR100": 0.0}

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        # summarize() always runs to populate coco_eval.stats (12 elements).
        # pycocotools prints to stdout unconditionally, so redirect when quiet.
        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            coco_eval.summarize()

        # COCOeval.stats layout (subset used here):
        #   [0] AP  @IoU=0.50:0.95  → mAP50-95
        #   [1] AP  @IoU=0.50       → mAP50
        #   [6] AR  @maxDets=1      → AR1
        #   [8] AR  @maxDets=100    → AR100
        stats = coco_eval.stats if len(coco_eval.stats) >= 12 else [0.0] * 12
        return {
            "mAP50-95": float(stats[0]),
            "mAP50":    float(stats[1]),
            "AR1":      float(stats[6]),
            "AR100":    float(stats[8]),
        }
