"""
DFINEPredictor — inference engine.
Called internally by DFINE.predict(). Not part of the public API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Generator

import torch

from dfine.media import Frame, FrameMetadata, FrameSink, OpenCVVideoSink
from dfine.results import Boxes, Masks, Results, SemanticMask
from dfine.tasks import normalize_task
from dfine.utils.ops import clip_boxes, crop_masks_to_boxes
from dfine.utils.sources import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, LoadSource


class DFINEPredictor:
    """
    Runs inference for a single source (image, video, directory, stream, …).

    The public ``DFINE`` wrapper passes a deployed inference copy here. The
    predictor must not deploy or otherwise structurally mutate the authoritative
    training model.
    """

    def __init__(self, model, cfg: dict, device: str, names: dict) -> None:
        deploy_fn = getattr(model, "deploy", None)
        if callable(deploy_fn) and not bool(getattr(model, "_deployed", False)):
            raise RuntimeError(
                "DFINEPredictor expects a deployed inference model. "
                "Use DFINE.predict() or DFINE.predictor so the trainable model "
                "is preserved."
            )
        self.model = model
        self.device = device
        self.names = names
        self.task = normalize_task(str(cfg.get("task", "detect")))

        from dfine.nn.build import build_postprocessor

        self._postprocessor = build_postprocessor(cfg)
        self._postprocessor.to(device)

    def run(
        self,
        source,
        conf: float,
        mask_threshold: float,
        imgsz: int,
        classes: list[int] | None,
        stream: bool,
        vid_stride: int,
        augment: bool,
        save: bool,
        project: str,
        name: str,
        save_dir: str | Path | None,
        exist_ok: bool,
        verbose: bool,
        backend: str = "opencv",
        gst_pipeline: str | None = None,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
        iou: float = 0.85,
        result_processor: Callable[[Results], Results] | None = None,
        run_mode: str = "predict",
        run_metadata: dict[str, object] | None = None,
        frame_sink: FrameSink | None = None,
        return_probs: bool = False,
    ) -> list | Generator:
        """Iterate over source and return results (list or generator if stream=True)."""
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if self.task == "semantic" and classes is not None:
            raise ValueError("Semantic prediction does not support classes filtering")
        loader = LoadSource(
            source,
            imgsz=imgsz,
            device=self.device,
            vid_stride=vid_stride,
            backend=backend,
            gst_pipeline=gst_pipeline,
            rtsp_latency=rtsp_latency,
            rtsp_transport=rtsp_transport,
            rtsp_username=rtsp_username,
            rtsp_password=rtsp_password,
        )
        if save:
            from dfine.utils.runs import resolve_run_dir, write_run_metadata

            save_dir = resolve_run_dir(
                project=project, name=name, save_dir=save_dir, exist_ok=exist_ok
            )
            metadata: dict[str, object] = {
                "mode": run_mode,
                "source": source,
                "conf": conf,
                "mask_threshold": mask_threshold,
                "imgsz": imgsz,
                "classes": classes,
                "stream": stream,
                "vid_stride": vid_stride,
                "augment": augment,
                "save": save,
                "project": project,
                "name": name,
                "save_dir": str(save_dir),
                "exist_ok": exist_ok,
                "verbose": verbose,
                "backend": backend,
                "gst_pipeline": gst_pipeline,
                "rtsp_latency": rtsp_latency,
                "rtsp_transport": rtsp_transport,
                "rtsp_username": rtsp_username,
                "rtsp_authenticated": rtsp_username is not None,
                "iou": iou,
                "return_probs": return_probs,
            }
            if run_metadata:
                metadata.update(run_metadata)
            write_run_metadata(save_dir, metadata)
        else:
            save_dir = None
        gen = self._infer(
            loader,
            conf,
            classes,
            mask_threshold=mask_threshold,
            augment=augment,
            iou=iou,
            save_dir=save_dir,
            result_processor=result_processor,
            frame_sink=frame_sink,
            return_probs=return_probs,
        )
        return gen if stream else list(gen)

    def _infer(
        self,
        loader: LoadSource,
        conf,
        classes,
        mask_threshold: float = 0.5,
        augment: bool = False,
        iou: float = 0.85,
        save_dir: Path | None = None,
        result_processor: Callable[[Results], Results] | None = None,
        frame_sink: FrameSink | None = None,
        return_probs: bool = False,
    ) -> Generator:
        """Yield one Results object per frame/image."""
        seen: dict[str, int] = {}
        semantic_seen: dict[str, int] = {}
        video_sink: OpenCVVideoSink | None = None
        video_output_path: Path | None = None
        source_iter = loader.iter_samples()
        index = 0
        try:
            while True:
                preprocess_start = time.perf_counter()
                try:
                    sample = next(source_iter)
                except StopIteration:
                    break
                preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
                index += 1

                tensor = sample.tensor
                orig_img = sample.frame.image
                path = sample.frame.metadata.source_id

                h, w = orig_img.shape[:2]
                orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=self.device)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    raw = self.model(tensor)
                    detections = self._postprocessor(raw, orig_size)
                    if not isinstance(detections, list):
                        raise RuntimeError("Prediction postprocessor returned an invalid result")
                    merged_det = detections[0]
                    if self.task != "semantic":
                        merged_det["num_orig"] = len(merged_det["boxes"])

                    if augment:
                        # Run prediction on horizontally flipped image
                        tensor_flipped = torch.flip(tensor, dims=[3])
                        raw_flipped = self.model(tensor_flipped)
                        detections_flipped = self._postprocessor(raw_flipped, orig_size)
                        if not isinstance(detections_flipped, list):
                            raise RuntimeError(
                                "Prediction postprocessor returned an invalid result"
                            )
                        det_flipped = detections_flipped[0]
                        if self.task == "semantic":
                            flipped_logits = torch.flip(det_flipped["semantic_logits"], dims=[2])
                            merged_det = {
                                "semantic_logits": (merged_det["semantic_logits"] + flipped_logits)
                                / 2
                            }
                            det_flipped = {}
                        else:
                            self._merge_augmented_detections(
                                merged_det,
                                det_flipped,
                                width=w,
                            )
                inference_ms = (time.perf_counter() - inference_start) * 1000

                postprocess_start = time.perf_counter()
                result = self._postprocess(
                    merged_det,
                    orig_img,
                    path,
                    conf,
                    classes,
                    mask_threshold=mask_threshold,
                    augment=augment,
                    iou=iou,
                    frame_metadata=sample.frame.metadata,
                    return_probs=return_probs,
                )
                result.speed = {
                    "preprocess": preprocess_ms,
                    "inference": inference_ms,
                    "postprocess": (time.perf_counter() - postprocess_start) * 1000,
                }
                if result_processor is not None:
                    result = result_processor(result)
                if frame_sink is not None:
                    frame_sink.write(Frame(image=result.plot(), metadata=sample.frame.metadata))
                if save_dir is not None:
                    if loader.mode == "video":
                        if video_sink is None:
                            video_output_path = self._resolve_output_path(
                                save_dir / f"{Path(path).stem}.mp4",
                                seen,
                            )
                            video_sink = self._create_video_sink(
                                video_output_path,
                                frame_size=(w, h),
                                source_fps=sample.frame.metadata.fps,
                                vid_stride=loader.vid_stride,
                            )
                        assert video_output_path is not None
                        video_sink.write(Frame(image=result.plot(), metadata=sample.frame.metadata))
                        result.save_path = str(video_output_path)
                        self._save_semantic_mask(
                            result,
                            save_dir,
                            self._output_filename(path, index),
                            semantic_seen,
                        )
                    else:
                        save_path = self._save_result(result, save_dir, index, seen)
                        result.save_path = str(save_path)
                        self._save_semantic_mask(result, save_dir, save_path.name, semantic_seen)
                yield result
        finally:
            if video_sink is not None:
                video_sink.close()
            if frame_sink is not None:
                frame_sink.close()
            loader.close()

    def _save_result(
        self,
        result: Results,
        save_dir: Path,
        index: int,
        seen: dict[str, int],
    ) -> Path:
        """Save an annotated prediction image and return the output path."""
        out_path = self._resolve_output_path(
            save_dir / self._output_filename(result.path, index),
            seen,
        )
        result.save(str(out_path))
        return out_path

    def _save_semantic_mask(
        self,
        result: Results,
        save_dir: Path,
        output_name: str,
        seen: dict[str, int],
    ) -> None:
        """Save a lossless class-ID PNG alongside semantic visualizations."""
        if result.semantic_mask is None:
            return
        mask_path = self._resolve_output_path(
            save_dir / "masks" / f"{Path(output_name).stem}.png",
            seen,
        )
        result.save_semantic(mask_path)
        result.semantic_save_path = str(mask_path)

    @staticmethod
    def _merge_augmented_detections(
        merged_det: dict,
        det_flipped: dict,
        *,
        width: int,
    ) -> None:
        """Merge a flipped detection view into the original-view dictionary in place."""
        masks_flipped_back = det_flipped.get("masks")
        if masks_flipped_back is not None:
            masks_flipped_back = torch.flip(masks_flipped_back, dims=[2])
        boxes_flipped_back = det_flipped["boxes"].clone()
        if len(boxes_flipped_back) > 0:
            x1 = width - boxes_flipped_back[:, 2]
            x2 = width - boxes_flipped_back[:, 0]
            boxes_flipped_back[:, 0] = x1
            boxes_flipped_back[:, 2] = x2
        merged_det["labels"] = torch.cat([merged_det["labels"], det_flipped["labels"]], dim=0)
        merged_det["boxes"] = torch.cat([merged_det["boxes"], boxes_flipped_back], dim=0)
        merged_det["scores"] = torch.cat([merged_det["scores"], det_flipped["scores"]], dim=0)
        if "masks" in merged_det and masks_flipped_back is not None:
            merged_det["masks"] = torch.cat([merged_det["masks"], masks_flipped_back], dim=0)

    def _resolve_output_path(self, out_path: Path, seen: dict[str, int]) -> Path:
        count = seen.get(out_path.name, 0)
        seen[out_path.name] = count + 1
        if count:
            out_path = out_path.with_name(f"{out_path.stem}_{count + 1}{out_path.suffix}")

        while out_path.exists():
            count += 1
            out_path = out_path.with_name(f"{out_path.stem}_{count + 1}{out_path.suffix}")

        return out_path

    def _create_video_sink(
        self,
        output_path: Path,
        frame_size: tuple[int, int],
        source_fps: float | None,
        vid_stride: int,
    ) -> OpenCVVideoSink:
        fps = (source_fps or 30.0) / float(vid_stride)
        fps = max(fps, 1.0)
        return OpenCVVideoSink(output_path, frame_size=frame_size, fps=fps)

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
        self,
        det: dict,
        orig_img,
        path: str,
        conf_thr,
        classes,
        mask_threshold: float = 0.5,
        augment: bool = False,
        iou: float = 0.85,
        frame_metadata: FrameMetadata | None = None,
        return_probs: bool = False,
    ) -> Results:
        """
        det is one element from DFINEPostProcessor output:
            {labels: [N], boxes: [N, 4] xyxy in pixel coords, scores: [N]}
        """
        if self.task == "semantic":
            logits = det.get("semantic_logits")
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Semantic postprocessor did not return logits")
            probabilities = torch.softmax(logits, dim=0)
            if probabilities.shape[0] != len(self.names):
                raise RuntimeError(
                    "Semantic output channel count does not match checkpoint taxonomy: "
                    f"channels={probabilities.shape[0]}, names={len(self.names)}"
                )
            class_ids = probabilities.argmax(dim=0).to(torch.int64)
            h, w = orig_img.shape[:2]
            return Results(
                orig_img=orig_img,
                path=path,
                names=self.names,
                semantic_mask=SemanticMask(
                    class_ids,
                    orig_shape=(h, w),
                    probs=probabilities if return_probs else None,
                ),
                frame_metadata=frame_metadata,
            )

        labels = det["labels"]
        boxes = det["boxes"]
        scores = det["scores"]
        masks = det.get("masks")
        num_orig = det.get("num_orig", len(boxes))

        # Assign view tracking labels (0 = original view, 1 = flipped TTA view)
        views = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
        views[num_orig:] = 1

        mask = scores > conf_thr
        labels, boxes, scores, views = labels[mask], boxes[mask], scores[mask], views[mask]
        if masks is not None:
            masks = masks[mask]

        if classes is not None:
            cls_tensor = torch.tensor(classes, device=labels.device)
            class_mask = torch.isin(labels, cls_tensor)
            labels, boxes, scores, views = (
                labels[class_mask],
                boxes[class_mask],
                scores[class_mask],
                views[class_mask],
            )
            if masks is not None:
                masks = masks[class_mask]

        h, w = orig_img.shape[:2]
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
            if masks is not None:
                masks = masks[keep_indices]

        boxes = clip_boxes(boxes, (h, w))
        result_masks = None
        if masks is not None:
            masks = (
                torch.nn.functional.interpolate(
                    masks[:, None].float(), size=(h, w), mode="bilinear", align_corners=False
                )[:, 0]
                if len(masks)
                else torch.zeros((0, h, w), device=masks.device)
            )
            masks = masks >= mask_threshold
            masks = crop_masks_to_boxes(masks, boxes)
            result_masks = Masks(masks.to(torch.uint8), orig_shape=(h, w))

        if len(boxes):
            data = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
        else:
            data = torch.zeros((0, 6), device=boxes.device)

        return Results(
            orig_img=orig_img,
            path=path,
            names=self.names,
            boxes=Boxes(data, orig_shape=(h, w)),
            masks=result_masks,
            frame_metadata=frame_metadata,
        )
