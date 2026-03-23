"""
DFINEValidator — COCO mAP evaluation.
Called internally by DFINE.val(). Not part of the public API.
"""
from __future__ import annotations

import torch
from dfine.utils.logging import LOGGER


class DFINEValidator:
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
        Run COCO-style evaluation.
        Returns dict with keys: mAP50, mAP50-95, precision, recall.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        from dfine.nn.build import build_postprocessor
        from dfine.utils.data import build_coco_dataloader, load_data_yaml
        from pathlib import Path
        import yaml

        cfg_data = load_data_yaml(data)
        root = Path(cfg_data["path"])
        ann_key = f"{split}_ann"
        if ann_key in cfg_data:
            ann_file = root / cfg_data[ann_key]
        else:
            split_name = Path(cfg_data[split]).name
            ann_file = root / "annotations" / f"instances_{split_name}.json"

        dataloader = build_coco_dataloader(data, split=split, imgsz=imgsz, batch_size=batch)

        postprocessor = build_postprocessor(self.cfg)
        postprocessor.to(self.device)
        postprocessor.eval()

        # build label → category_id reverse map from the dataset
        cat_id_to_label = dataloader.dataset.cat_id_to_label
        label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}

        self.model.eval()
        results = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                # orig sizes as [w, h] for each image (all resized to imgsz)
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
                            "image_id":   img_id,
                            "category_id": label_to_cat_id.get(int(label), int(label) + 1),
                            "bbox":  [x1, y1, x2 - x1, y2 - y1],  # xywh
                            "score": score,
                        })

        coco_gt = COCO(str(ann_file))

        if not results:
            if verbose:
                LOGGER.info("No detections above conf threshold — mAP=0")
            return {"mAP50-95": 0.0, "mAP50": 0.0, "precision": 0.0, "recall": 0.0}

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        import contextlib, io
        coco_eval.evaluate()
        coco_eval.accumulate()
        # summarize() always runs to populate coco_eval.stats; stdout is
        # suppressed when verbose=False since pycocotools prints unconditionally
        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            coco_eval.summarize()

        stats = coco_eval.stats if len(coco_eval.stats) >= 12 else [0.0] * 12
        return {
            "mAP50-95":  float(stats[0]),
            "mAP50":     float(stats[1]),
            "precision": float(stats[8]),
            "recall":    float(stats[6]),
        }
