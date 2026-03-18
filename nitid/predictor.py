"""
DFINEPredictor — inference engine.
Called internally by DFINE.predict(). Not part of the public API.
"""
from __future__ import annotations
from typing import Generator
import torch
import numpy as np
from dfine.results import Results, Boxes
from dfine.utils.sources import LoadSource
from dfine.utils.ops import scale_boxes, clip_boxes


class DFINEPredictor:
    def __init__(self, model, cfg: dict, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

    def run(
        self,
        source,
        conf: float,
        imgsz: int,
        classes: list[int] | None,
        stream: bool,
        augment: bool,
        verbose: bool,
        names: dict | None = None,
    ) -> list | Generator:
        loader = LoadSource(source, imgsz=imgsz, device=self.device)
        gen = self._infer(loader, conf, classes, imgsz, names or {})
        return gen if stream else list(gen)

    def _infer(self, loader: LoadSource, conf, classes, imgsz, names) -> Generator:
        for tensor, orig_img, path in loader:
            with torch.no_grad():
                raw = self.model(tensor)
            result = self._postprocess(raw, orig_img, path, conf, classes, imgsz, names)
            yield result

    def _postprocess(self, raw, orig_img, path, conf_thr, classes, imgsz, names) -> Results:
        """
        D-FINE outputs (labels, boxes, scores) — no NMS needed.
        raw expected to be a dict or tuple depending on D-FINE version.
        """
        # TODO: align with exact D-FINE forward() output contract
        labels, boxes, scores = raw["labels"], raw["boxes"], raw["scores"]

        # Filter by confidence
        mask = scores > conf_thr
        labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

        # Filter by class
        if classes is not None:
            class_mask = torch.isin(labels, torch.tensor(classes, device=labels.device))
            labels, boxes, scores = labels[class_mask], boxes[class_mask], scores[class_mask]

        # Scale from [imgsz x imgsz] back to original image size
        h, w = orig_img.shape[:2]
        boxes = scale_boxes(boxes, from_shape=(imgsz, imgsz), to_shape=(h, w))
        boxes = clip_boxes(boxes, (h, w))

        # Pack into [N, 6]: xyxy conf cls
        if len(boxes):
            data = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
        else:
            data = torch.zeros((0, 6))

        return Results(
            orig_img=orig_img,
            path=path,
            names=names,
            boxes=Boxes(data, orig_shape=(h, w)),
        )
