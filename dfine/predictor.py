"""
DFINEPredictor — inference engine.
Called internally by DFINE.predict(). Not part of the public API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator

import torch

from dfine.results import Boxes, Results
from dfine.utils.ops import clip_boxes
from dfine.utils.sources import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, LoadSource


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
        save: bool,
        project: str,
        name: str,
        verbose: bool,
        iou: float = 0.85,
    ) -> list | Generator:
        """Iterate over source and return results (list or generator if stream=True)."""
        loader = LoadSource(source, imgsz=imgsz, device=self.device)
        save_dir = Path(project) / name if save else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        gen = self._infer(loader, conf, classes, augment=augment, iou=iou, save_dir=save_dir)
        return gen if stream else list(gen)

    def _infer(
        self,
        loader: LoadSource,
        conf,
        classes,
        augment: bool = False,
        iou: float = 0.85,
        save_dir: Path | None = None,
    ) -> Generator:
        """Yield one Results object per frame/image."""
        seen: dict[str, int] = {}
        source_iter = iter(loader)
        index = 0
        while True:
            preprocess_start = time.perf_counter()
            try:
                tensor, orig_img, path = next(source_iter)
            except StopIteration:
                break
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
            index += 1

            h, w = orig_img.shape[:2]
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=self.device)
            inference_start = time.perf_counter()
            with torch.no_grad():
                raw = self.model(tensor)
                detections = self._postprocessor(raw, orig_size)
                merged_det = detections[0]
                merged_det["num_orig"] = len(merged_det["boxes"])

                if augment:
                    # Run prediction on horizontally flipped image
                    tensor_flipped = torch.flip(tensor, dims=[3])
                    raw_flipped = self.model(tensor_flipped)
                    detections_flipped = self._postprocessor(raw_flipped, orig_size)
                    det_flipped = detections_flipped[0]

                    # Flip back xyxy pixel-space coordinates: x1_new = w - x2_old, x2_new = w - x1_old
                    boxes_flipped_back = det_flipped["boxes"].clone()
                    if len(boxes_flipped_back) > 0:
                        x1 = w - boxes_flipped_back[:, 2]
                        x2 = w - boxes_flipped_back[:, 0]
                        boxes_flipped_back[:, 0] = x1
                        boxes_flipped_back[:, 2] = x2

                    # Combine detections
                    merged_det = {
                        "labels": torch.cat([merged_det["labels"], det_flipped["labels"]], dim=0),
                        "boxes": torch.cat([merged_det["boxes"], boxes_flipped_back], dim=0),
                        "scores": torch.cat([merged_det["scores"], det_flipped["scores"]], dim=0),
                        "num_orig": merged_det["num_orig"],
                    }
            inference_ms = (time.perf_counter() - inference_start) * 1000

            postprocess_start = time.perf_counter()
            result = self._postprocess(
                merged_det, orig_img, path, conf, classes, augment=augment, iou=iou
            )
            result.speed = {
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": (time.perf_counter() - postprocess_start) * 1000,
            }
            if save_dir is not None:
                save_path = self._save_result(result, save_dir, index, seen)
                result.save_path = str(save_path)
            yield result

    def _save_result(
        self,
        result: Results,
        save_dir: Path,
        index: int,
        seen: dict[str, int],
    ) -> Path:
        """Save an annotated prediction image and return the output path."""
        out_path = save_dir / self._output_filename(result.path, index)
        count = seen.get(out_path.name, 0)
        seen[out_path.name] = count + 1
        if count:
            out_path = out_path.with_name(f"{out_path.stem}_{count + 1}{out_path.suffix}")

        while out_path.exists():
            count += 1
            out_path = out_path.with_name(f"{out_path.stem}_{count + 1}{out_path.suffix}")

        result.save(str(out_path))
        return out_path

    def _output_filename(self, path: str, index: int) -> str:
        """Choose a stable output filename for file, stream, screen, and array sources."""
        source_path = Path(path)
        suffix = source_path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            return source_path.name
        if suffix in VIDEO_EXTENSIONS:
            return f"{source_path.stem}_{index:06d}.jpg"
        if path == "<screen>":
            return f"screen_{index:06d}.jpg"
        if path == "<ndarray>":
            return f"image_{index:06d}.jpg"
        if source_path.name:
            return f"{source_path.name}_{index:06d}.jpg"
        return f"image_{index:06d}.jpg"

    def _postprocess(
        self, det: dict, orig_img, path, conf_thr, classes, augment: bool = False, iou: float = 0.85
    ) -> Results:
        """
        det is one element from DFINEPostProcessor output:
            {labels: [N], boxes: [N, 4] xyxy in pixel coords, scores: [N]}
        """
        labels = det["labels"]
        boxes = det["boxes"]
        scores = det["scores"]
        num_orig = det.get("num_orig", len(boxes))

        # Assign view tracking labels (0 = original view, 1 = flipped TTA view)
        views = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
        views[num_orig:] = 1

        mask = scores > conf_thr
        labels, boxes, scores, views = labels[mask], boxes[mask], scores[mask], views[mask]

        if classes is not None:
            cls_tensor = torch.tensor(classes, device=labels.device)
            class_mask = torch.isin(labels, cls_tensor)
            labels, boxes, scores, views = (
                labels[class_mask],
                boxes[class_mask],
                scores[class_mask],
                views[class_mask],
            )

        if augment and len(boxes) > 0:
            import torchvision

            # Sort by scores in descending order
            order = scores.argsort(descending=True)
            keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

            # Compute all pairwise IoUs
            ious = torchvision.ops.box_iou(boxes, boxes)

            for i in range(len(order)):
                idx_a = order[i]
                if not keep[idx_a]:
                    continue

                # Suppress boxes of the same class from the OTHER view only
                nms_mask = (
                    keep & (labels == labels[idx_a]) & (views != views[idx_a]) & (ious[idx_a] > iou)
                )
                keep[nms_mask] = False

            keep_indices = torch.where(keep)[0]
            labels, boxes, scores = labels[keep_indices], boxes[keep_indices], scores[keep_indices]

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
