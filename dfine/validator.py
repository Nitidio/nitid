"""
DFINEValidator — COCO mAP evaluation.
Called internally by DFINE.val(). Not part of the public API.
"""
from __future__ import annotations


class DFINEValidator:
    def __init__(self, model, cfg: dict, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

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
        # TODO: implement COCO evaluator
        raise NotImplementedError
