"""
DFINEPredictor — inference engine.
Called internally by DFINE.predict(). Not part of the public API.
"""
from __future__ import annotations

from typing import Generator

import torch

from dfine.results import Boxes, Results
from dfine.utils.ops import clip_boxes
from dfine.utils.sources import LoadSource


class DFINEPredictor:
    """
    Runs inference for a single source (image, video, directory, stream, …).

    On first use the model is put into deploy mode (BN fusion + weighting
    function materialisation). The ``_deployed`` flag prevents a second
    deploy() call if the same DFINE instance is used for multiple predict()
    calls, since BN fusion is a one-way operation.
    """

    def __init__(self, model, cfg: dict, device: str, names: dict) -> None:
        if not getattr(model, "_deployed", False):
            model.deploy()  # fuses BN, materialises weighting fn as static tensor
            model._deployed = True
        self.model = model
        self.device = device
        self.names = names

        from dfine.nn.build import build_postprocessor
        self._postprocessor = build_postprocessor(cfg)
        self._postprocessor.to(device)

    def run(
        self,
        source,
        conf: float,
        imgsz: int,
        classes: list[int] | None,
        stream: bool,
        augment: bool,
        verbose: bool,
    ) -> list | Generator:
        """Iterate over source and return results (list or generator if stream=True)."""
        loader = LoadSource(source, imgsz=imgsz, device=self.device)
        gen = self._infer(loader, conf, classes)
        return gen if stream else list(gen)

    def _infer(self, loader: LoadSource, conf, classes) -> Generator:
        """Yield one Results object per frame/image."""
        for tensor, orig_img, path in loader:
            h, w = orig_img.shape[:2]
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                raw = self.model(tensor)
                detections = self._postprocessor(raw, orig_size)
            yield self._postprocess(detections[0], orig_img, path, conf, classes)

    def _postprocess(self, det: dict, orig_img, path, conf_thr, classes) -> Results:
        """
        det is one element from DFINEPostProcessor output:
            {labels: [N], boxes: [N, 4] xyxy in pixel coords, scores: [N]}
        """
        labels = det["labels"]
        boxes  = det["boxes"]
        scores = det["scores"]

        mask = scores > conf_thr
        labels, boxes, scores = labels[mask], boxes[mask], scores[mask]

        if classes is not None:
            cls_tensor = torch.tensor(classes, device=labels.device)
            class_mask = torch.isin(labels, cls_tensor)
            labels, boxes, scores = labels[class_mask], boxes[class_mask], scores[class_mask]

        h, w = orig_img.shape[:2]
        boxes = clip_boxes(boxes, (h, w))

        if len(boxes):
            data = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
        else:
            data = torch.zeros((0, 6), device=boxes.device)

        return Results(
            orig_img=orig_img,
            path=path,
            names=self.names,
            boxes=Boxes(data, orig_shape=(h, w)),
        )
