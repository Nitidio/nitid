"""
DFINEValidator — COCO mAP evaluation plus Ultralytics-style detection plots.
Called internally by DFINE.val() and DFINETrainer.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from dfine.utils.logging import LOGGER
from dfine.utils.ops import crop_masks_to_boxes


class CocoApi(Protocol):
    imgs: dict[int, dict[str, Any]]


class CocoLikeDataset(Protocol):
    cat_id_to_label: dict[int, int]
    coco: CocoApi


class GTRecord(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    image_id: int


class PredRecord(TypedDict):
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


PerClassRow = TypedDict(
    "PerClassRow",
    {
        "class_id": int,
        "name": str,
        "instances": int,
        "ap50": float,
        "ap50-95": float,
    },
)


class SemanticConfusionMatrix:
    """Streaming pixel confusion matrix with ignored-label handling."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                "Semantic prediction and target shapes must match; "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        valid = target != self.ignore_index
        target_values = target[valid].long()
        prediction_values = prediction[valid].long()
        if target_values.numel() == 0:
            return
        if target_values.min() < 0 or target_values.max() >= self.num_classes:
            raise ValueError("Semantic target contains a class ID outside the configured taxonomy")
        if prediction_values.min() < 0 or prediction_values.max() >= self.num_classes:
            raise ValueError(
                "Semantic prediction contains a class ID outside the configured taxonomy"
            )
        indices = target_values * self.num_classes + prediction_values
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self, names: dict[int, str]) -> dict[str, object]:
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        ground_truth = matrix.sum(dim=1)
        union = ground_truth + matrix.sum(dim=0) - true_positive
        present = ground_truth > 0
        iou = true_positive / union.clamp(min=1)
        mean_iou = float(iou[present].mean()) if present.any() else 0.0
        pixel_accuracy = float(true_positive.sum() / matrix.sum().clamp(min=1))
        per_class = [
            {
                "class_id": class_id,
                "name": names.get(class_id, str(class_id)),
                "pixels": int(ground_truth[class_id]),
                "iou": float(iou[class_id]),
            }
            for class_id in range(self.num_classes)
            if present[class_id]
        ]
        return {
            "mIoU": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "fitness": mean_iou,
            "pixels": int(matrix.sum()),
            "per_class": per_class,
        }


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def _xywhr_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``cx, cy, w, h, angle_radians`` to four polygon corners."""
    if boxes.numel() == 0:
        return torch.zeros((0, 4, 2), dtype=boxes.dtype, device=boxes.device)
    centers = boxes[:, :2]
    widths = boxes[:, 2:3]
    heights = boxes[:, 3:4]
    angles = boxes[:, 4]
    x_offsets = torch.cat((-widths, widths, widths, -widths), dim=1) / 2
    y_offsets = torch.cat((-heights, -heights, heights, heights), dim=1) / 2
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    x = x_offsets * cos - y_offsets * sin + centers[:, 0:1]
    y = x_offsets * sin + y_offsets * cos + centers[:, 1:2]
    return torch.stack((x, y), dim=2)


def rotated_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for rotated rectangles in ``cx, cy, w, h, angle_radians`` format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    polys1 = cast(
        np.ndarray[Any, np.dtype[np.float32]],
        _xywhr_to_corners(boxes1.float()).detach().cpu().numpy().astype(np.float32),
    )
    polys2 = cast(
        np.ndarray[Any, np.dtype[np.float32]],
        _xywhr_to_corners(boxes2.float()).detach().cpu().numpy().astype(np.float32),
    )
    out = np.zeros((len(polys1), len(polys2)), dtype=np.float32)
    areas1 = [abs(float(cv2.contourArea(poly))) for poly in polys1]
    areas2 = [abs(float(cv2.contourArea(poly))) for poly in polys2]
    for i, poly1 in enumerate(polys1):
        for j, poly2 in enumerate(polys2):
            inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
            union = areas1[i] + areas2[j] - float(inter_area)
            out[i, j] = float(inter_area) / union if union > 0 else 0.0
    return torch.from_numpy(out)


def _restore_original_coordinates(
    boxes: torch.Tensor, original_width: int, original_height: int, imgsz: int
) -> torch.Tensor:
    """Map absolute xyxy boxes from D-FINE's square resize to original pixels."""
    restored = boxes.clone()
    if restored.numel():
        restored[:, [0, 2]] *= original_width / imgsz
        restored[:, [1, 3]] *= original_height / imgsz
        restored[:, [0, 2]].clamp_(0, original_width)
        restored[:, [1, 3]].clamp_(0, original_height)
    return restored


def _restore_original_keypoints(
    keypoints: torch.Tensor, original_width: int, original_height: int, imgsz: int
) -> torch.Tensor:
    """Map absolute xy keypoints from D-FINE's square resize to original pixels."""
    restored = keypoints.clone()
    if restored.numel():
        scale_x = original_width / imgsz
        scale_y = original_height / imgsz
        if restored.dim() == 2 and restored.shape[1] % 3 == 0:
            res_view = restored.view(restored.shape[0], -1, 3)
            res_view[..., 0] *= scale_x
            res_view[..., 1] *= scale_y
            res_view[..., 0].clamp_(0, original_width)
            res_view[..., 1].clamp_(0, original_height)
        elif restored.dim() == 2 and restored.shape[1] % 2 == 0:
            res_view = restored.view(restored.shape[0], -1, 2)
            res_view[..., 0] *= scale_x
            res_view[..., 1] *= scale_y
            res_view[..., 0].clamp_(0, original_width)
            res_view[..., 1].clamp_(0, original_height)
        elif restored.dim() == 3:
            restored[..., 0] *= scale_x
            restored[..., 1] *= scale_y
            restored[..., 0].clamp_(0, original_width)
            restored[..., 1].clamp_(0, original_height)
    return restored


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


@contextlib.contextmanager
def _dynamic_eval_geometry(model, imgsz: int):
    """Use dynamic encoder positions and decoder anchors for non-native eval sizes."""
    changed: list[tuple[torch.nn.Module, object]] = []
    requested_size = (imgsz, imgsz)
    for module in model.modules():
        if not hasattr(module, "eval_spatial_size"):
            continue
        configured_size = getattr(module, "eval_spatial_size")
        if configured_size is not None and tuple(configured_size) != requested_size:
            changed.append((module, configured_size))
            setattr(module, "eval_spatial_size", None)
    try:
        yield
    finally:
        for module, configured_size in changed:
            setattr(module, "eval_spatial_size", configured_size)


class DFINEValidator:
    """
    Runs COCO-style bounding-box evaluation against a labelled split.

    The postprocessor is rebuilt from the checkpoint config so that validation
    can run without mutating the model state. In addition to COCO metrics, the
    validator computes precision/recall/F1 curves, a confusion matrix, and
    per-class AP summaries for Ultralytics-style artifacts.
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
        save_dir: str | Path | None = None,
        plots: bool = True,
        classes: list[int] | None = None,
        single_cls: bool = False,
        show_progress: bool | None = None,
    ) -> dict:
        """
        Evaluate on a COCO-format dataset split.

        Returns a dict with scalar metrics plus `per_class` rows and curve
        metadata. The returned scalars are suitable for CSV logging.
        """
        from dfine.nn.build import build_postprocessor
        from dfine.tasks import normalize_task
        from dfine.utils.data import (
            build_detection_dataloader,
            build_obb_dataloader,
            resolve_detection_split,
        )

        task = normalize_task(str(self.cfg.get("task", "detect")))
        if task == "semantic":
            return self._run_semantic(
                data=data,
                imgsz=imgsz,
                batch=batch,
                split=split,
                verbose=verbose,
                save_dir=save_dir,
                plots=plots,
                classes=classes,
                single_cls=single_cls,
                show_progress=show_progress,
            )
        if task == "obb":
            return self._run_obb(
                data=data,
                imgsz=imgsz,
                batch=batch,
                conf=conf,
                split=split,
                verbose=verbose,
                save_dir=save_dir,
                plots=plots,
                classes=classes,
                single_cls=single_cls,
                show_progress=show_progress,
                build_obb_dataloader=build_obb_dataloader,
            )

        COCO = importlib.import_module("pycocotools.coco").COCO
        COCOeval = importlib.import_module("pycocotools.cocoeval").COCOeval
        mask_utils = importlib.import_module("pycocotools.mask")

        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        spec = resolve_detection_split(data, split)
        ann_file = spec.ann_file
        dataloader = build_detection_dataloader(
            data,
            split=split,
            imgsz=imgsz,
            batch_size=batch,
            spec=spec,
            classes=classes,
            single_cls=single_cls,
            task=task,
        )

        postprocessor = build_postprocessor(self.cfg)
        postprocessor.to(self.device)
        postprocessor.eval()

        dataset = cast(CocoLikeDataset, dataloader.dataset)
        cat_id_to_label = dataset.cat_id_to_label
        label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}

        gt_records: list[GTRecord] = []
        pred_records: list[PredRecord] = []
        coco_results: list[dict[str, object]] = []

        self.model.eval()
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"val:{split}",
            leave=False,
            unit="batch",
            disable=not (verbose if show_progress is None else show_progress),
        )
        with _dynamic_eval_geometry(self.model, imgsz), torch.no_grad():
            for images, targets in progress:
                images = images.to(self.device)
                orig_sizes = torch.tensor(
                    [[imgsz, imgsz]] * len(images),
                    dtype=torch.float32,
                    device=self.device,
                )
                raw = self.model(images)
                detections = postprocessor(raw, orig_sizes)

                for det, target in zip(detections, targets):
                    img_id = int(target["image_id"][0])
                    gt_boxes = _cxcywh_to_xyxy(target["boxes"] * imgsz).cpu()
                    gt_labels = target["labels"].cpu()
                    gt_records.append({"boxes": gt_boxes, "labels": gt_labels, "image_id": img_id})

                    pred_boxes = det["boxes"].detach().cpu()
                    pred_scores = det["scores"].detach().cpu()
                    pred_labels = det["labels"].detach().cpu()
                    pred_masks = det.get("masks")
                    if pred_masks is not None:
                        pred_masks = pred_masks.detach().cpu()
                    pred_keypoints = det.get("keypoints")
                    if pred_keypoints is not None:
                        pred_keypoints = pred_keypoints.detach().cpu()
                    pred_records.append(
                        {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
                    )

                    image_info = dataset.coco.imgs[img_id]
                    coco_boxes = _restore_original_coordinates(
                        pred_boxes,
                        original_width=int(image_info["width"]),
                        original_height=int(image_info["height"]),
                        imgsz=imgsz,
                    )
                    coco_keypoints = None
                    if pred_keypoints is not None:
                        coco_keypoints = _restore_original_keypoints(
                            pred_keypoints,
                            original_width=int(image_info["width"]),
                            original_height=int(image_info["height"]),
                            imgsz=imgsz,
                        )

                    mask = pred_scores > conf
                    selected = torch.where(mask)[0]
                    selected_masks = None
                    if pred_masks is not None and selected.numel():
                        selected_masks = torch.nn.functional.interpolate(
                            pred_masks[selected, None].float(),
                            size=(int(image_info["height"]), int(image_info["width"])),
                            mode="bilinear",
                            align_corners=False,
                        )[:, 0]
                        selected_masks = crop_masks_to_boxes(
                            selected_masks >= 0.5,
                            coco_boxes[selected],
                        )

                    for result_index, pred_index in enumerate(selected.tolist()):
                        box = coco_boxes[pred_index].tolist()
                        score = float(pred_scores[pred_index])
                        label = int(pred_labels[pred_index])
                        x1, y1, x2, y2 = box
                        result: dict[str, object] = {
                            "image_id": img_id,
                            "category_id": label_to_cat_id.get(label, label + 1),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                        if selected_masks is not None:
                            encoded = mask_utils.encode(
                                np.asfortranarray(
                                    selected_masks[result_index].numpy().astype(np.uint8)
                                )
                            )
                            counts = encoded.get("counts")
                            if isinstance(counts, bytes):
                                encoded["counts"] = counts.decode("ascii")
                            result["segmentation"] = encoded
                        if coco_keypoints is not None:
                            kpt_inst = coco_keypoints[pred_index].numpy()
                            if kpt_inst.ndim == 2 and kpt_inst.shape[-1] == 2:
                                vis = np.ones((kpt_inst.shape[0], 1), dtype=np.float32) * 2.0
                                kpt_inst = np.hstack([kpt_inst, vis])
                            result["keypoints"] = kpt_inst.flatten().tolist()
                        coco_results.append(result)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(ann_file))
        coco_eval = None
        coco_metrics = {
            "mAP50-95": 0.0,
            "mAP50": 0.0,
            "AR1": 0.0,
            "AR100": 0.0,
            "AR300": 0.0,
        }
        per_class_rows: list[PerClassRow] = []
        mask_metrics = {"mask_mAP50-95": 0.0, "mask_mAP50": 0.0}
        pose_metrics = {"pose_mAP50-95": 0.0, "pose_mAP50": 0.0}

        if coco_results:
            with contextlib.redirect_stdout(io.StringIO()):
                coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            # D-FINE emits 300 predictions and Ultralytics evaluates detection with
            # max_det=300. COCO's default cap of 100 is particularly inappropriate
            # for dense datasets such as SKU-110K (often >100 objects per image).
            coco_eval.params.maxDets = [1, 10, 100, 300]
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
            precision_values = coco_eval.eval["precision"]
            recall_values = coco_eval.eval["recall"]

            def mean_valid(values: np.ndarray) -> float:
                valid = values[values > -1]
                return float(np.mean(valid)) if valid.size else 0.0

            coco_metrics = {
                # precision: [IoU, recall, class, area, max_detections]
                "mAP50-95": mean_valid(precision_values[:, :, :, 0, -1]),
                "mAP50": mean_valid(precision_values[0, :, :, 0, -1]),
                # recall: [IoU, class, area, max_detections]
                "AR1": mean_valid(recall_values[:, :, 0, 0]),
                "AR100": mean_valid(recall_values[:, :, 0, 2]),
                "AR300": mean_valid(recall_values[:, :, 0, -1]),
            }
            per_class_rows = self._per_class_ap(coco_eval, gt_records, cat_id_to_label)

            if task == "segment" and any("segmentation" in item for item in coco_results):
                coco_mask_eval = COCOeval(coco_gt, coco_dt, "segm")
                coco_mask_eval.params.maxDets = [1, 10, 100, 300]
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_mask_eval.evaluate()
                    coco_mask_eval.accumulate()
                mask_precision = coco_mask_eval.eval["precision"]
                mask_metrics = {
                    "mask_mAP50-95": mean_valid(mask_precision[:, :, :, 0, -1]),
                    "mask_mAP50": mean_valid(mask_precision[0, :, :, 0, -1]),
                }

            if task == "pose" and any("keypoints" in item for item in coco_results):
                coco_pose_eval = COCOeval(coco_gt, coco_dt, "keypoints")
                coco_pose_eval.params.maxDets = [1, 10, 100, 300]
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_pose_eval.evaluate()
                    coco_pose_eval.accumulate()
                pose_precision = coco_pose_eval.eval["precision"]
                pose_metrics = {
                    "pose_mAP50-95": mean_valid(pose_precision[:, :, :, 0, -1]),
                    "pose_mAP50": mean_valid(pose_precision[0, :, :, 0, -1]),
                }
        elif verbose:
            LOGGER.info("No detections above conf threshold — metrics are zero")

        thresholds, precisions, recalls, f1_scores = self._precision_recall_curve(
            gt_records, pred_records
        )
        best_idx = int(np.argmax(f1_scores)) if len(f1_scores) else 0
        best_conf = float(thresholds[best_idx]) if len(thresholds) else float(conf)
        precision = float(precisions[best_idx]) if len(precisions) else 0.0
        recall = float(recalls[best_idx]) if len(recalls) else 0.0
        f1 = float(f1_scores[best_idx]) if len(f1_scores) else 0.0
        fitness = self._fitness(
            precision,
            recall,
            (
                mask_metrics["mask_mAP50"]
                if task == "segment"
                else (pose_metrics["pose_mAP50"] if task == "pose" else coco_metrics["mAP50"])
            ),
            (
                mask_metrics["mask_mAP50-95"]
                if task == "segment"
                else (pose_metrics["pose_mAP50-95"] if task == "pose" else coco_metrics["mAP50-95"])
            ),
        )

        confusion_matrix, class_ids = self._confusion_matrix(gt_records, pred_records, best_conf)

        if save_dir is not None and plots:
            self._save_plots(
                save_dir=save_dir,
                confusion_matrix=confusion_matrix,
                class_ids=class_ids,
                thresholds=thresholds,
                precisions=precisions,
                recalls=recalls,
                f1_scores=f1_scores,
            )

        metrics = {
            **coco_metrics,
            **(mask_metrics if task == "segment" else {}),
            **(pose_metrics if task == "pose" else {}),
            "images": len(gt_records),
            "instances": int(sum(gt["labels"].numel() for gt in gt_records)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fitness": fitness,
            "best_conf": best_conf,
            "per_class": per_class_rows,
        }

        if verbose:
            self._print_summary(metrics, per_class_rows)

        return metrics

    def _run_obb(
        self,
        *,
        data: str,
        imgsz: int,
        batch: int,
        conf: float,
        split: str,
        verbose: bool,
        save_dir: str | Path | None,
        plots: bool,
        classes: list[int] | None,
        single_cls: bool,
        show_progress: bool | None,
        build_obb_dataloader,
    ) -> dict[str, object]:
        """Evaluate oriented boxes with class-aware rotated-IoU matching."""
        from dfine.nn.build import build_postprocessor

        del save_dir, plots
        dataloader = build_obb_dataloader(
            data,
            split=split,
            imgsz=imgsz,
            batch_size=batch,
            classes=classes,
            single_cls=single_cls,
        )
        postprocessor = build_postprocessor(self.cfg)
        postprocessor.to(self.device)
        postprocessor.eval()

        gt_records: list[GTRecord] = []
        pred_records: list[PredRecord] = []
        self.model.eval()
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"val:{split}",
            leave=False,
            unit="batch",
            disable=not (verbose if show_progress is None else show_progress),
        )
        with _dynamic_eval_geometry(self.model, imgsz), torch.no_grad():
            for images, targets in progress:
                images = images.to(self.device)
                orig_sizes = torch.tensor(
                    [[imgsz, imgsz]] * len(images), dtype=torch.float32, device=self.device
                )
                detections = postprocessor(self.model(images), orig_sizes)
                for det, target in zip(detections, targets):
                    gt_boxes = target["boxes"].float().clone()
                    if gt_boxes.numel():
                        scale = torch.tensor([imgsz, imgsz, imgsz, imgsz, torch.pi])
                        gt_boxes *= scale
                    gt_records.append(
                        {
                            "boxes": gt_boxes.cpu(),
                            "labels": target["labels"].cpu(),
                            "image_id": int(target["image_id"][0]),
                        }
                    )
                    pred_records.append(
                        {
                            "boxes": det["boxes"].detach().cpu(),
                            "scores": det["scores"].detach().cpu(),
                            "labels": det["labels"].detach().cpu(),
                        }
                    )

        map50, map5095, per_class_rows = self._obb_map(gt_records, pred_records)
        thresholds, precisions, recalls, f1_scores = self._obb_precision_recall_curve(
            gt_records, pred_records
        )
        best_idx = int(np.argmax(f1_scores)) if len(f1_scores) else 0
        best_conf = float(thresholds[best_idx]) if len(thresholds) else float(conf)
        precision = float(precisions[best_idx]) if len(precisions) else 0.0
        recall = float(recalls[best_idx]) if len(recalls) else 0.0
        f1 = float(f1_scores[best_idx]) if len(f1_scores) else 0.0
        metrics: dict[str, object] = {
            "mAP50-95": map5095,
            "mAP50": map50,
            "AR1": recall,
            "AR100": recall,
            "AR300": recall,
            "images": len(gt_records),
            "instances": int(sum(gt["labels"].numel() for gt in gt_records)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fitness": self._fitness(precision, recall, map50, map5095),
            "best_conf": best_conf,
            "per_class": per_class_rows,
        }
        if verbose:
            self._print_summary(metrics, per_class_rows)
        return metrics

    def _run_semantic(
        self,
        *,
        data: str,
        imgsz: int,
        batch: int,
        split: str,
        verbose: bool,
        save_dir: str | Path | None,
        plots: bool,
        classes: list[int] | None,
        single_cls: bool,
        show_progress: bool | None,
    ) -> dict[str, object]:
        """Evaluate dense logits at each image's original resolution."""
        from dfine.utils.data import build_semantic_dataloader, load_data_yaml, normalize_names

        if classes is not None or single_cls:
            raise ValueError("Semantic validation does not support classes or single_cls")
        data_config = load_data_yaml(data)
        data_names = normalize_names(data_config)
        if data_names != self.names:
            raise ValueError(
                "Semantic dataset taxonomy does not match the model checkpoint names: "
                f"dataset={data_names}, model={self.names}"
            )
        ignore_index = data_config.get("ignore_index", 255)
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise ValueError("Data YAML ignore_index must be an integer")
        if ignore_index < 0:
            raise ValueError("Data YAML ignore_index must be non-negative")
        dataloader = build_semantic_dataloader(
            data,
            split=split,
            imgsz=imgsz,
            batch_size=batch,
        )
        confusion = SemanticConfusionMatrix(len(self.names), ignore_index=ignore_index)
        image_count = 0
        self.model.eval()
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"val:{split}",
            leave=False,
            unit="batch",
            disable=not (verbose if show_progress is None else show_progress),
        )
        with _dynamic_eval_geometry(self.model, imgsz), torch.no_grad():
            for images, targets in progress:
                outputs = self.model(images.to(self.device))
                logits = outputs.get("sem_seg_logits")
                if not isinstance(logits, torch.Tensor):
                    raise RuntimeError("Semantic model did not return 'sem_seg_logits'")
                for batch_index, target in enumerate(targets):
                    original_mask = target["orig_mask"].long()
                    original_size = tuple(int(value) for value in target["orig_size"].tolist())
                    resized_logits = torch.nn.functional.interpolate(
                        logits[batch_index : batch_index + 1].float(),
                        size=original_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    prediction = resized_logits.argmax(dim=0).cpu()
                    confusion.update(prediction, original_mask)
                    image_count += 1

        metrics = confusion.compute(self.names)
        metrics["images"] = image_count
        resolved_save_dir = Path(save_dir) if save_dir is not None else None
        if resolved_save_dir is not None and plots:
            resolved_save_dir.mkdir(parents=True, exist_ok=True)
            labels = [self.names.get(index, str(index)) for index in range(len(self.names))]
            matrix = confusion.matrix.numpy()
            self._plot_confusion_matrix(
                resolved_save_dir / "confusion_matrix.png",
                matrix,
                labels,
            )
            self._plot_confusion_matrix(
                resolved_save_dir / "confusion_matrix_normalized.png",
                self._normalize_confusion_matrix(matrix),
                labels,
                normalized=True,
            )
        if verbose:
            LOGGER.info(
                "Semantic validation: %d images, mIoU %.4f, pixel accuracy %.4f",
                image_count,
                _as_float(metrics["mIoU"]),
                _as_float(metrics["pixel_accuracy"]),
            )
        return metrics

    def compact_metrics(self, metrics: dict[str, object]) -> dict[str, float]:
        """Return the scalar validation fields that belong in the epoch row."""
        if "mIoU" in metrics:
            return {
                key: _as_float(metrics.get(key, 0.0))
                for key in ("mIoU", "pixel_accuracy", "fitness")
            }
        keys = ["precision", "recall", "mAP50", "mAP50-95", "fitness"]
        if "mask_mAP50" in metrics:
            keys.extend(["mask_mAP50", "mask_mAP50-95"])
        return {key: _as_float(metrics.get(key, 0.0)) for key in keys}

    def _per_class_ap(
        self,
        coco_eval,
        gt_records: list[GTRecord],
        cat_id_to_label: dict[int, int],
    ) -> list[PerClassRow]:
        precision = coco_eval.eval.get("precision") if coco_eval.eval else None
        if precision is None:
            return []

        gt_counts: defaultdict[int, int] = defaultdict(int)
        for gt in gt_records:
            for label in gt["labels"].tolist():
                gt_counts[int(label)] += 1

        rows: list[PerClassRow] = []
        cat_ids = coco_eval.params.catIds
        for class_idx, cat_id in enumerate(cat_ids):
            label_idx = int(cat_id_to_label.get(int(cat_id), int(cat_id)))
            name = self.names.get(label_idx, str(label_idx))

            class_precision = precision[:, :, class_idx, 0, -1]
            class_precision = class_precision[class_precision > -1]
            ap50_95 = float(np.mean(class_precision)) if class_precision.size else 0.0

            ap50_slice = precision[0, :, class_idx, 0, -1]
            ap50_slice = ap50_slice[ap50_slice > -1]
            ap50 = float(np.mean(ap50_slice)) if ap50_slice.size else 0.0

            instances = int(gt_counts.get(label_idx, 0))
            if instances == 0 and ap50_95 == 0.0 and ap50 == 0.0:
                continue

            rows.append(
                {
                    "class_id": label_idx,
                    "name": name,
                    "instances": instances,
                    "ap50": ap50,
                    "ap50-95": ap50_95,
                }
            )

        return rows

    def _precision_recall_curve(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.linspace(0.0, 0.99, 100)
        precisions = np.zeros_like(thresholds)
        recalls = np.zeros_like(thresholds)
        f1_scores = np.zeros_like(thresholds)

        # Greedily match each image once in descending confidence order. Lower-confidence
        # predictions cannot change matches already assigned to higher-confidence ones, so
        # threshold metrics are just cumulative slices of this single matching pass.
        prediction_scores: list[float] = []
        prediction_is_tp: list[bool] = []
        total_gt = sum(len(gt["boxes"]) for gt in gt_records)
        for gt, pred in zip(gt_records, pred_records):
            gt_boxes, gt_labels = gt["boxes"], gt["labels"]
            pred_order = torch.argsort(pred["scores"], descending=True)
            matched_gt: set[int] = set()
            for pred_idx in pred_order.tolist():
                score = float(pred["scores"][pred_idx])
                label = int(pred["labels"][pred_idx])
                candidate_indices = [
                    gt_idx
                    for gt_idx, gt_label in enumerate(gt_labels.tolist())
                    if int(gt_label) == label and gt_idx not in matched_gt
                ]
                is_tp = False
                if candidate_indices:
                    candidate_boxes = gt_boxes[candidate_indices]
                    ious = box_iou(pred["boxes"][pred_idx : pred_idx + 1], candidate_boxes)[0]
                    best_iou, best_relative_idx = torch.max(ious, dim=0)
                    if float(best_iou) >= 0.5:
                        matched_gt.add(candidate_indices[int(best_relative_idx)])
                        is_tp = True
                prediction_scores.append(score)
                prediction_is_tp.append(is_tp)

        scores = np.asarray(prediction_scores, dtype=np.float32)
        true_positives = np.asarray(prediction_is_tp, dtype=np.bool_)
        for i, threshold in enumerate(thresholds):
            selected = scores >= threshold
            tp = int(np.count_nonzero(true_positives & selected))
            fp = int(np.count_nonzero(~true_positives & selected))
            fn = total_gt - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            precisions[i] = precision
            recalls[i] = recall
            f1_scores[i] = f1

        return thresholds, precisions, recalls, f1_scores

    def _obb_map(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
    ) -> tuple[float, float, list[PerClassRow]]:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)
        class_ids = sorted(
            {
                int(label)
                for record in [*gt_records, *pred_records]
                for label in record["labels"].tolist()
            }
        )
        per_class_rows: list[PerClassRow] = []
        ap50_values: list[float] = []
        ap5095_values: list[float] = []

        for class_id in class_ids:
            instances = sum(
                int((record["labels"] == class_id).sum().item()) for record in gt_records
            )
            if instances == 0:
                continue
            class_ap = [
                self._obb_average_precision(gt_records, pred_records, class_id, float(iou_thr))
                for iou_thr in iou_thresholds
            ]
            ap50 = float(class_ap[0])
            ap5095 = float(np.mean(class_ap))
            ap50_values.append(ap50)
            ap5095_values.append(ap5095)
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "name": self.names.get(class_id, str(class_id)),
                    "instances": instances,
                    "ap50": ap50,
                    "ap50-95": ap5095,
                }
            )

        map50 = float(np.mean(ap50_values)) if ap50_values else 0.0
        map5095 = float(np.mean(ap5095_values)) if ap5095_values else 0.0
        return map50, map5095, per_class_rows

    def _obb_average_precision(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
        class_id: int,
        iou_thresh: float,
    ) -> float:
        predictions: list[tuple[float, int, int]] = []
        total_gt = 0
        for image_index, (gt, pred) in enumerate(zip(gt_records, pred_records)):
            total_gt += int((gt["labels"] == class_id).sum().item())
            pred_indices = torch.where(pred["labels"] == class_id)[0]
            for pred_idx in pred_indices.tolist():
                predictions.append((float(pred["scores"][pred_idx]), image_index, pred_idx))
        if total_gt == 0:
            return 0.0
        if not predictions:
            return 0.0

        predictions.sort(key=lambda item: item[0], reverse=True)
        matched: dict[int, set[int]] = defaultdict(set)
        tp = np.zeros(len(predictions), dtype=np.float32)
        fp = np.zeros(len(predictions), dtype=np.float32)

        for pred_rank, (_, image_index, pred_idx) in enumerate(predictions):
            gt = gt_records[image_index]
            pred = pred_records[image_index]
            gt_indices = torch.where(gt["labels"] == class_id)[0]
            if gt_indices.numel() == 0:
                fp[pred_rank] = 1.0
                continue
            ious = rotated_box_iou(
                pred["boxes"][pred_idx : pred_idx + 1],
                gt["boxes"][gt_indices],
            ).squeeze(0)
            best_iou, best_relative_idx = torch.max(ious, dim=0)
            gt_idx = int(gt_indices[int(best_relative_idx)])
            if float(best_iou) >= iou_thresh and gt_idx not in matched[image_index]:
                matched[image_index].add(gt_idx)
                tp[pred_rank] = 1.0
            else:
                fp[pred_rank] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        return self._compute_ap(recalls, precisions)

    @staticmethod
    def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        recall_curve = np.concatenate(([0.0], recalls, [1.0]))
        precision_curve = np.concatenate(([1.0], precisions, [0.0]))
        for index in range(len(precision_curve) - 1, 0, -1):
            precision_curve[index - 1] = max(precision_curve[index - 1], precision_curve[index])
        indices = np.where(recall_curve[1:] != recall_curve[:-1])[0]
        return float(
            np.sum(
                (recall_curve[indices + 1] - recall_curve[indices]) * precision_curve[indices + 1]
            )
        )

    def _obb_precision_recall_curve(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.linspace(0.0, 0.99, 100)
        precisions = np.zeros_like(thresholds)
        recalls = np.zeros_like(thresholds)
        f1_scores = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            tp, fp, fn = self._obb_count_matches(
                gt_records, pred_records, conf_thresh=float(threshold)
            )
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            precisions[i] = precision
            recalls[i] = recall
            f1_scores[i] = f1
        return thresholds, precisions, recalls, f1_scores

    def _obb_count_matches(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
        conf_thresh: float,
        iou_thresh: float = 0.5,
    ) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for gt, pred in zip(gt_records, pred_records):
            pred_mask = pred["scores"] >= conf_thresh
            img_tp, img_fp, img_fn = self._match_obb_class_aware(
                pred_boxes=pred["boxes"][pred_mask],
                pred_labels=pred["labels"][pred_mask],
                pred_scores=pred["scores"][pred_mask],
                gt_boxes=gt["boxes"],
                gt_labels=gt["labels"],
                iou_thresh=iou_thresh,
            )
            tp += img_tp
            fp += img_fp
            fn += img_fn
        return tp, fp, fn

    def _match_obb_class_aware(
        self,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        iou_thresh: float = 0.5,
    ) -> tuple[int, int, int]:
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return 0, 0, 0
        if len(pred_boxes) == 0:
            return 0, 0, int(len(gt_boxes))
        if len(gt_boxes) == 0:
            return 0, int(len(pred_boxes)), 0
        tp = fp = 0
        matched_gt: set[int] = set()
        pred_order = torch.argsort(pred_scores, descending=True)
        for pred_idx in pred_order.tolist():
            pred_label = int(pred_labels[pred_idx])
            candidate_indices = [
                gt_idx
                for gt_idx, gt_label in enumerate(gt_labels.tolist())
                if int(gt_label) == pred_label and gt_idx not in matched_gt
            ]
            if not candidate_indices:
                fp += 1
                continue
            ious = rotated_box_iou(pred_boxes[pred_idx : pred_idx + 1], gt_boxes[candidate_indices])
            best_iou, best_rel_idx = torch.max(ious.squeeze(0), dim=0)
            if float(best_iou) >= iou_thresh:
                matched_gt.add(candidate_indices[int(best_rel_idx)])
                tp += 1
            else:
                fp += 1
        return tp, fp, len(gt_boxes) - len(matched_gt)

    def _count_matches(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
        conf_thresh: float,
        iou_thresh: float = 0.5,
    ) -> tuple[int, int, int]:
        tp = fp = fn = 0
        for gt, pred in zip(gt_records, pred_records):
            pred_mask = pred["scores"] >= conf_thresh
            pred_boxes = pred["boxes"][pred_mask]
            pred_labels = pred["labels"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            img_tp, img_fp, img_fn = self._match_class_aware(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                iou_thresh=iou_thresh,
            )
            tp += img_tp
            fp += img_fp
            fn += img_fn
        return tp, fp, fn

    def _match_class_aware(
        self,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        iou_thresh: float = 0.5,
    ) -> tuple[int, int, int]:
        tp = fp = fn = 0
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return tp, fp, fn
        if len(pred_boxes) == 0:
            return 0, 0, int(len(gt_boxes))
        if len(gt_boxes) == 0:
            return 0, int(len(pred_boxes)), 0

        matched_gt: set[int] = set()
        pred_order = torch.argsort(pred_scores, descending=True)
        for pred_idx in pred_order.tolist():
            pred_box = pred_boxes[pred_idx : pred_idx + 1]
            pred_label = int(pred_labels[pred_idx])

            candidate_indices = [
                gt_idx
                for gt_idx, gt_label in enumerate(gt_labels.tolist())
                if gt_label == pred_label and gt_idx not in matched_gt
            ]
            if not candidate_indices:
                fp += 1
                continue

            candidate_boxes = gt_boxes[candidate_indices]
            ious = box_iou(pred_box, candidate_boxes).squeeze(0)
            best_iou, best_rel_idx = torch.max(ious, dim=0)
            if float(best_iou) >= iou_thresh:
                matched_gt.add(candidate_indices[int(best_rel_idx)])
                tp += 1
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        return tp, fp, fn

    def _confusion_matrix(
        self,
        gt_records: list[GTRecord],
        pred_records: list[PredRecord],
        conf_thresh: float,
        iou_thresh: float = 0.5,
    ) -> tuple[np.ndarray, list[int]]:
        all_classes: set[int] = set()
        for gt in gt_records:
            all_classes.update(int(x) for x in gt["labels"].tolist())
        for pred in pred_records:
            all_classes.update(int(x) for x in pred["labels"].tolist())
        class_ids = sorted(all_classes)
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
        matrix = np.zeros((len(class_ids) + 1, len(class_ids) + 1), dtype=np.int64)

        for gt, pred in zip(gt_records, pred_records):
            pred_mask = pred["scores"] >= conf_thresh
            pred_boxes = pred["boxes"][pred_mask]
            pred_labels = pred["labels"][pred_mask]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue

            if len(pred_boxes) == 0:
                for gt_label in gt_labels.tolist():
                    matrix[class_to_idx[int(gt_label)], -1] += 1
                continue

            if len(gt_boxes) == 0:
                for pred_label in pred_labels.tolist():
                    matrix[-1, class_to_idx[int(pred_label)]] += 1
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            matched_pred: set[int] = set()
            matched_gt: set[int] = set()

            pred_indices, gt_indices = torch.nonzero(ious >= iou_thresh, as_tuple=True)
            if pred_indices.numel():
                iou_values = ious[pred_indices, gt_indices]
                order = torch.argsort(-iou_values)
                pred_indices = pred_indices[order]
                gt_indices = gt_indices[order]

                for pred_idx, gt_idx in zip(pred_indices.tolist(), gt_indices.tolist()):
                    if pred_idx in matched_pred or gt_idx in matched_gt:
                        continue
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)
                    pred_label = int(pred_labels[pred_idx])
                    gt_label = int(gt_labels[gt_idx])
                    matrix[class_to_idx[gt_label], class_to_idx[pred_label]] += 1

            for pred_idx, pred_label in enumerate(pred_labels.tolist()):
                if pred_idx not in matched_pred:
                    matrix[-1, class_to_idx[int(pred_label)]] += 1

            for gt_idx, gt_label in enumerate(gt_labels.tolist()):
                if gt_idx not in matched_gt:
                    matrix[class_to_idx[int(gt_label)], -1] += 1

        return matrix, class_ids

    def _save_plots(
        self,
        save_dir: Path,
        confusion_matrix: np.ndarray,
        class_ids: list[int],
        thresholds: np.ndarray,
        precisions: np.ndarray,
        recalls: np.ndarray,
        f1_scores: np.ndarray,
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)

        plt = importlib.import_module("matplotlib.pyplot")

        class_names = [self.names.get(cls_id, str(cls_id)) for cls_id in class_ids] + ["background"]
        self._plot_confusion_matrix(
            save_dir / "confusion_matrix.png", confusion_matrix, class_names
        )
        self._plot_confusion_matrix(
            save_dir / "confusion_matrix_normalized.png",
            self._normalize_confusion_matrix(confusion_matrix),
            class_names,
            normalized=True,
        )

        plt.figure(figsize=(6, 5))
        plt.plot(recalls, precisions, color="tab:blue")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "pr_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(thresholds, f1_scores, color="tab:orange")
        plt.xlabel("Confidence")
        plt.ylabel("F1")
        plt.title("F1 Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "f1_curve.png", dpi=200)
        plt.close()

    def _plot_confusion_matrix(
        self,
        path: Path,
        matrix: np.ndarray,
        class_labels: list[str],
        normalized: bool = False,
    ) -> None:
        plt = importlib.import_module("matplotlib.pyplot")
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix" + (" (Normalized)" if normalized else ""))
        plt.colorbar()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45, ha="right")
        plt.yticks(tick_marks, class_labels)

        thresh = float(matrix.max()) / 2.0 if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = f"{value:.2f}" if normalized else f"{int(value)}"
                plt.text(
                    j,
                    i,
                    text,
                    horizontalalignment="center",
                    color="white" if value > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    def _normalize_confusion_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def _fitness(self, precision: float, recall: float, map50: float, map5095: float) -> float:
        return 0.1 * precision + 0.1 * recall + 0.4 * map50 + 0.4 * map5095

    def _print_summary(self, metrics: dict[str, object], per_class_rows: list[PerClassRow]) -> None:
        LOGGER.info(
            "%22s %10s %10s %10s %10s %10s %10s",
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )
        LOGGER.info(
            "%22s %10d %10d %10.3f %10.3f %10.3f %10.3f",
            "all",
            int(_as_float(metrics.get("images", 0))),
            int(_as_float(metrics.get("instances", 0))),
            _as_float(metrics["precision"]),
            _as_float(metrics["recall"]),
            _as_float(metrics["mAP50"]),
            _as_float(metrics["mAP50-95"]),
        )
        if "mask_mAP50" in metrics:
            LOGGER.info(
                "%22s %10.3f %10.3f",
                "Mask",
                _as_float(metrics["mask_mAP50"]),
                _as_float(metrics["mask_mAP50-95"]),
            )
        if "pose_mAP50" in metrics:
            LOGGER.info(
                "%22s %10.3f %10.3f",
                "Pose",
                _as_float(metrics["pose_mAP50"]),
                _as_float(metrics["pose_mAP50-95"]),
            )
        if not per_class_rows:
            return

        LOGGER.info("%22s %10s %10s", "Class", "Instances", "mAP50-95")
        for row in per_class_rows:
            LOGGER.info(
                "%22s %10d %10.3f",
                row["name"],
                row["instances"],
                row["ap50-95"],
            )
