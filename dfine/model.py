"""
DFINE — public entry point. Mirrors the ultralytics.YOLO interface.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, TypeVar, Union, cast

import numpy as np
import torch.nn as nn

from dfine.media import FrameSink, FrameSource
from dfine.utils.device import resolve_device

if TYPE_CHECKING:
    from dfine.tracking import ResultTracker

Source = Union[str, Path, int, np.ndarray, list, FrameSource]
ModelCallback = Callable[..., object]
_T = TypeVar("_T")


class _DefaultTrainOption:
    def __repr__(self) -> str:
        return "default"


_TRAIN_DEFAULT = _DefaultTrainOption()


def _resolve_train_option(value: _T | _DefaultTrainOption, default: _T) -> _T:
    if value is _TRAIN_DEFAULT:
        return default
    return cast(_T, value)


class DFINE:
    """
    D-FINE object detection and segmentation wrapper.

    Args:
        model:   D-FINE architecture name or path to a wrapped .pth checkpoint.
        task:    ``"detect"``, ``"segment"``, ``"semantic"`` (alias ``"sem_seg"``), or ``"pose"``.
        weights: Official weight variant: ``default``, ``obj2coco``, or ``coco``.
        backend: ``"torch"`` (default) or ``"openvino"``. ``"openvino"`` routes
                 ``predict()``/``track()`` through OpenVINO Runtime, enabling
                 Intel NPU and integrated GPU inference; every other method
                 (``train``, ``val``, ``export``, ``info``) keeps running on
                 PyTorch/CPU regardless of `backend`.
        device:  Meaning depends on `backend`. For ``backend="torch"``: "cuda",
                 "cpu", "cuda:N", or None for auto-select. For
                 ``backend="openvino"``: an OpenVINO device string ("CPU",
                 "GPU" for Intel integrated GPU, "NPU"), or "auto"/None to
                 prefer NPU, then GPU, then CPU.
        verbose: Print model info on load.

    Example:
        model = DFINE("dfine_l", task="detect", weights="obj2coco")
        results = model("image.jpg", conf=0.5)
        model.train(data="coco.yaml", epochs=50)
        model.export(format="tensorrt")

        # Intel NPU / integrated GPU inference via OpenVINO Runtime:
        model = DFINE("dfine_l", backend="openvino", device="NPU")
        results = model("image.jpg", conf=0.5)
    """

    def __init__(
        self,
        model: str | Path = "dfine_l",
        *,
        task: str = "detect",
        weights: str = "default",
        backend: str = "torch",
        device: str | int | None = None,
        verbose: bool = True,
    ) -> None:
        if backend not in ("torch", "openvino"):
            raise ValueError(f"backend must be 'torch' or 'openvino', got {backend!r}")
        self._backend = backend
        if backend == "openvino":
            from dfine.nn.openvino_runtime import resolve_openvino_device

            # Fail fast at construction, not on the first predict() call.
            self._openvino_device: str | None = resolve_openvino_device(
                device if isinstance(device, str) else None
            )
            self._device_str: str = "cpu"
        else:
            self._openvino_device = None
            self._device_str = resolve_device(device)
        self.verbose = verbose
        self._model: nn.Module
        self._cfg: dict[str, Any]
        self._names: dict[int, str]
        self._path: str
        self._weights: str | None = None
        self._deployed_model: nn.Module | None = None
        self._deployed_model_device: str | None = None
        self._openvino_cache: dict[int, Any] = {}
        # RLock: _get_openvino_model() calls _get_deployed_model() while
        # already holding this lock, from the same thread.
        self._model_lock = threading.RLock()
        from dfine.tasks import normalize_task

        self._task = normalize_task(task)
        self._callbacks: dict[str, list[ModelCallback]] = {}
        self._load(str(model), task=self._task, weights=weights)

    # ── Inference ──────────────────────────────────────────────────────────

    def __call__(self, source: Source, **kwargs) -> list[Any] | Generator[Any, None, None]:
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Source,
        conf: float = 0.5,
        mask_threshold: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        stream: bool = False,
        vid_stride: int = 1,
        augment: bool = False,
        save: bool = False,
        project: str | None = None,
        name: str = "exp",
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        verbose: bool = True,
        backend: str = "opencv",
        gst_pipeline: str | None = None,
        reconnect: bool = False,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_attempts: int | None = None,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
        hardware_profile: str | None = None,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
        iou: float = 0.85,
        sink: FrameSink | None = None,
        return_probs: bool = False,
    ) -> list | Generator:
        """
        Run detection, instance segmentation, or semantic segmentation on a source.

        Returns list[Results] when stream=False,
        Generator[Results] when stream=True.
        """
        return self._predictor_for(imgsz).run(
            source,
            conf=conf,
            mask_threshold=mask_threshold,
            imgsz=imgsz,
            classes=classes,
            stream=stream,
            vid_stride=vid_stride,
            augment=augment,
            save=save,
            project=project or f"runs/{self.task}",
            name=name,
            save_dir=save_dir,
            exist_ok=exist_ok,
            verbose=verbose,
            backend=backend,
            gst_pipeline=gst_pipeline,
            reconnect=reconnect,
            reconnect_initial_delay=reconnect_initial_delay,
            reconnect_max_delay=reconnect_max_delay,
            reconnect_attempts=reconnect_attempts,
            rtsp_latency=rtsp_latency,
            rtsp_transport=rtsp_transport,
            hardware_profile=hardware_profile,
            rtsp_username=rtsp_username,
            rtsp_password=rtsp_password,
            iou=iou,
            frame_sink=sink,
            return_probs=return_probs,
        )

    def track(
        self,
        source: Source,
        conf: float = 0.1,
        imgsz: int = 640,
        classes: list[int] | None = None,
        stream: bool = False,
        vid_stride: int = 1,
        augment: bool = False,
        save: bool = False,
        project: str = "runs/track",
        name: str = "exp",
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        verbose: bool = True,
        backend: str = "opencv",
        gst_pipeline: str | None = None,
        reconnect: bool = False,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        reconnect_attempts: int | None = None,
        rtsp_latency: int = 200,
        rtsp_transport: str = "tcp",
        hardware_profile: str | None = None,
        rtsp_username: str | None = None,
        rtsp_password: str | None = None,
        iou: float = 0.85,
        tracker: str | ResultTracker = "bytetrack",
        tracker_kwargs: dict[str, Any] | None = None,
        sink: FrameSink | None = None,
    ) -> list | Generator:
        """Run detection and assign persistent object IDs across source frames."""
        from dfine.tracking import DFINETracker

        if self.task != "detect":
            raise ValueError("track() currently supports task='detect' only")

        return DFINETracker(self._predictor_for(imgsz)).run(
            source,
            tracker=tracker,
            tracker_kwargs=tracker_kwargs,
            conf=conf,
            mask_threshold=0.5,
            imgsz=imgsz,
            classes=classes,
            stream=stream,
            vid_stride=vid_stride,
            augment=augment,
            save=save,
            project=project,
            name=name,
            save_dir=save_dir,
            exist_ok=exist_ok,
            verbose=verbose,
            backend=backend,
            gst_pipeline=gst_pipeline,
            reconnect=reconnect,
            reconnect_initial_delay=reconnect_initial_delay,
            reconnect_max_delay=reconnect_max_delay,
            reconnect_attempts=reconnect_attempts,
            rtsp_latency=rtsp_latency,
            rtsp_transport=rtsp_transport,
            hardware_profile=hardware_profile,
            rtsp_username=rtsp_username,
            rtsp_password=rtsp_password,
            iou=iou,
            frame_sink=sink,
        )

    # ── Training ────────────────────────────────────────────────────────────

    def train(
        self,
        data: str,
        epochs: int | _DefaultTrainOption = _TRAIN_DEFAULT,
        imgsz: int = 640,
        batch: int | _DefaultTrainOption = _TRAIN_DEFAULT,
        lr0: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        backbone_lr: float | None | _DefaultTrainOption = _TRAIN_DEFAULT,
        lrf: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        cos_lr: bool | _DefaultTrainOption = _TRAIN_DEFAULT,
        warmup_epochs: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        warmup_momentum: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        warmup_bias_lr: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        optimizer: str | _DefaultTrainOption = _TRAIN_DEFAULT,
        momentum: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        weight_decay: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        clip_grad: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        resume: bool = False,
        amp: bool | _DefaultTrainOption = _TRAIN_DEFAULT,
        ema: bool | _DefaultTrainOption = _TRAIN_DEFAULT,
        ema_decay: float | _DefaultTrainOption = _TRAIN_DEFAULT,
        device: str | None = None,
        project: str = "runs/train",
        name: str = "exp",
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        patience: int = 100,
        save: bool = True,
        save_period: int = 1,
        val: bool = True,
        plots: bool = True,
        val_period: int = 1,
        workers: int = 0,
        cache: bool | str = False,
        seed: int = 0,
        deterministic: bool = True,
        freeze: int | list[int | str] | str | None = None,
        classes: list[int] | None = None,
        single_cls: bool = False,
        fraction: float = 1.0,
        accumulate: int = 1,
        multi_scale: bool = False,
        augment: bool = True,
        fliplr: float = 0.5,
        scale: float = 0.5,
        translate: float = 0.1,
        crop: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        mosaic: float = 0.0,
        mixup: float = 0.0,
        close_mosaic: int = 10,
        time: float | None = None,
        recipe: str = "default",
        verbose: bool = True,
        callbacks: object | None = None,
        wandb: bool | dict[str, Any] = False,
        mlflow: bool | dict[str, Any] = False,
    ) -> dict:
        """Fine-tune on a custom dataset. Returns final metrics plus per-epoch history."""
        if self._backend == "openvino":
            raise ValueError(
                "train() requires backend='torch' — the openvino backend is inference-only"
            )
        from dfine.nn.transfer import adapt_model_to_classes
        from dfine.trainer import DFINETrainer
        from dfine.training_recipes import (
            apply_training_recipe_to_config,
            default_train_options,
            resolve_training_recipe,
            training_recipe_policy,
        )
        from dfine.utils.data import load_data_yaml, normalize_names

        resolved_recipe = resolve_training_recipe(
            recipe, task=str(getattr(self, "_task", "detect"))
        )
        train_defaults = default_train_options(recipe=resolved_recipe, config=self._cfg)
        recipe_policy = training_recipe_policy(recipe=resolved_recipe, config=self._cfg)
        resolved_epochs = _resolve_train_option(epochs, train_defaults.epochs)
        resolved_batch = _resolve_train_option(batch, train_defaults.batch)
        resolved_lr0 = _resolve_train_option(lr0, train_defaults.lr0)
        resolved_backbone_lr = _resolve_train_option(backbone_lr, train_defaults.backbone_lr)
        resolved_lrf = _resolve_train_option(lrf, train_defaults.lrf)
        resolved_cos_lr = _resolve_train_option(cos_lr, train_defaults.cos_lr)
        resolved_warmup_epochs = _resolve_train_option(warmup_epochs, train_defaults.warmup_epochs)
        resolved_warmup_momentum = _resolve_train_option(
            warmup_momentum, train_defaults.warmup_momentum
        )
        resolved_warmup_bias_lr = _resolve_train_option(
            warmup_bias_lr, train_defaults.warmup_bias_lr
        )
        resolved_optimizer = _resolve_train_option(optimizer, train_defaults.optimizer)
        resolved_momentum = _resolve_train_option(momentum, train_defaults.momentum)
        resolved_weight_decay = _resolve_train_option(weight_decay, train_defaults.weight_decay)
        resolved_clip_grad = _resolve_train_option(clip_grad, train_defaults.clip_grad)
        resolved_amp = _resolve_train_option(amp, train_defaults.amp)
        resolved_ema = _resolve_train_option(ema, train_defaults.ema)
        resolved_ema_decay = _resolve_train_option(ema_decay, train_defaults.ema_decay)

        data_config = load_data_yaml(data)
        dataset_names = {0: "object"} if single_cls else normalize_names(data_config)
        declared_nc = data_config.get("nc")
        if isinstance(declared_nc, int) and declared_nc != len(dataset_names):
            raise ValueError(
                f"Dataset YAML declares nc={declared_nc} but defines {len(dataset_names)} names"
            )
        if getattr(self, "_task", "detect") == "semantic":
            if single_cls or classes is not None:
                raise ValueError(
                    "Semantic training does not support single_cls or classes filtering; "
                    "define the desired contiguous taxonomy in the dataset masks and YAML"
                )
            ignore_index = data_config.get("ignore_index", 255)
            if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
                raise ValueError("Data YAML ignore_index must be an integer")
            if ignore_index < 0:
                raise ValueError("Data YAML ignore_index must be non-negative")
            if 0 <= ignore_index < len(dataset_names):
                raise ValueError(f"ignore_index={ignore_index} overlaps valid semantic class IDs")
            self._cfg["SemSegCriterion"]["ignore_index"] = ignore_index
            self._cfg["SemanticSegmentation"]["ignore_index"] = ignore_index
            class_weights = data_config.get("class_weights")
            if class_weights is not None:
                if not isinstance(class_weights, list) or len(class_weights) != len(dataset_names):
                    raise ValueError(
                        "Data YAML class_weights must contain one numeric value per semantic class"
                    )
                self._cfg["SemSegCriterion"]["class_weights"] = [
                    float(value) for value in class_weights
                ]
        transfer = adapt_model_to_classes(
            self._model,
            self._cfg,
            self._names,
            dataset_names,
        )
        self._model = transfer.model
        self._cfg = apply_training_recipe_to_config(transfer.config, recipe=resolved_recipe)
        self._names = dataset_names
        self._invalidate_deployed_cache()
        if transfer.changed and self.verbose:
            mapped = list(transfer.mapped_proposal_scorer) or "unavailable"
            print(
                f"[D-FINE] Configured {len(dataset_names)} dataset classes: "
                f"transferred {transfer.transferred} pretrained tensors; "
                f"mean-mapped proposal scorer {mapped}; "
                f"initialized {len(transfer.initialized)} class-specific tensors"
            )

        trainer = DFINETrainer(
            model=self._model,
            cfg=self._cfg,
            device=device or self._device_str,
            names=self._names,
            callbacks=self._callbacks,
        )
        try:
            metrics = trainer.train(
                data=data,
                epochs=resolved_epochs,
                imgsz=imgsz,
                batch=resolved_batch,
                lr0=resolved_lr0,
                backbone_lr=resolved_backbone_lr,
                lrf=resolved_lrf,
                cos_lr=resolved_cos_lr,
                warmup_epochs=resolved_warmup_epochs,
                warmup_momentum=resolved_warmup_momentum,
                warmup_bias_lr=resolved_warmup_bias_lr,
                optimizer=resolved_optimizer,
                momentum=resolved_momentum,
                weight_decay=resolved_weight_decay,
                clip_grad=resolved_clip_grad,
                resume=resume,
                amp=resolved_amp,
                ema=resolved_ema,
                ema_decay=resolved_ema_decay,
                project=project,
                name=name,
                save_dir=save_dir,
                exist_ok=exist_ok,
                patience=patience,
                save=save,
                save_period=save_period,
                val=val,
                plots=plots,
                val_period=val_period,
                workers=workers,
                cache=cache,
                seed=seed,
                deterministic=deterministic,
                freeze=freeze,
                classes=classes,
                single_cls=single_cls,
                fraction=fraction,
                accumulate=accumulate,
                multi_scale=multi_scale,
                augment=augment,
                fliplr=fliplr,
                scale=scale,
                translate=translate,
                crop=crop,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                mosaic=mosaic,
                mixup=mixup,
                close_mosaic=close_mosaic,
                time_limit=time,
                recipe=resolved_recipe.name,
                scheduler=recipe_policy.scheduler,
                warmup_iter=recipe_policy.warmup_iter,
                flat_epochs=recipe_policy.flat_epochs,
                no_aug_epochs=recipe_policy.no_aug_epochs,
                lr_gamma=recipe_policy.lr_gamma,
                collate_mixup_prob=recipe_policy.collate_mixup_prob,
                collate_mixup_epochs=recipe_policy.collate_mixup_epochs,
                verbose=verbose,
                callbacks=callbacks,
                wandb=wandb,
                mlflow=mlflow,
            )
            self._invalidate_deployed_cache()
            return metrics
        except BaseException as error:
            self._invalidate_deployed_cache()
            trainer._handle_train_error(error)
            raise

    def add_callback(self, event: str, callback: ModelCallback) -> None:
        """Register a persistent callback on the model, similar to Ultralytics."""
        from dfine.trainer import DFINETrainer

        if event not in DFINETrainer.CALLBACK_EVENTS:
            supported = ", ".join(DFINETrainer.CALLBACK_EVENTS)
            raise ValueError(f"Unknown callback event '{event}'. Supported events: {supported}")
        if not callable(callback):
            raise TypeError(
                f"Callback for '{event}' must be callable, got {type(callback).__name__}"
            )
        self._callbacks.setdefault(event, [])
        callback_identity = self._callback_identity(callback)
        if not any(
            self._callback_identity(existing) == callback_identity
            for existing in self._callbacks[event]
        ):
            self._callbacks[event].append(callback)

    def clear_callbacks(self, event: str | None = None) -> None:
        """Remove persistent callbacks registered via add_callback()."""
        from dfine.trainer import DFINETrainer

        if event is None:
            self._callbacks.clear()
            return
        if event not in DFINETrainer.CALLBACK_EVENTS:
            supported = ", ".join(DFINETrainer.CALLBACK_EVENTS)
            raise ValueError(f"Unknown callback event '{event}'. Supported events: {supported}")
        self._callbacks.pop(event, None)

    def _callback_identity(self, callback: ModelCallback) -> object:
        bound_self = getattr(callback, "__self__", None)
        bound_func = getattr(callback, "__func__", None)
        if bound_self is not None and bound_func is not None:
            return (id(bound_self), id(bound_func))
        return id(callback)

    # ── Validation ──────────────────────────────────────────────────────────

    def val(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.001,
        split: str = "val",
        project: str = "runs/val",
        name: str = "exp",
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        plots: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Evaluate on val/test split. Returns mAP50, mAP50-95, etc."""
        if self._backend == "openvino":
            raise ValueError(
                "val() requires backend='torch' — the openvino backend is inference-only"
            )
        from dfine.utils.runs import resolve_run_dir, write_run_metadata
        from dfine.validator import DFINEValidator

        resolved_dir = resolve_run_dir(
            project=project, name=name, save_dir=save_dir, exist_ok=exist_ok
        )
        args = {
            "mode": "val",
            "data": data,
            "imgsz": imgsz,
            "batch": batch,
            "conf": conf,
            "split": split,
            "project": project,
            "name": name,
            "save_dir": str(resolved_dir),
            "exist_ok": exist_ok,
            "plots": plots,
            "verbose": verbose,
        }
        write_run_metadata(resolved_dir, args)
        validator = DFINEValidator(self._model, self._cfg, self._device_str, self._names)
        return validator.run(
            data=data,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            split=split,
            verbose=verbose,
            save_dir=resolved_dir,
            plots=plots,
        )

    # ── Export ──────────────────────────────────────────────────────────────

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        batch: int = 1,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 17,
        half: bool = False,
        device: str | None = None,
        verbose: bool = True,
        project: str = "runs/export",
        name: str = "exp",
        save_dir: str | Path | None = None,
        output: str | Path | None = None,
        exist_ok: bool = False,
    ) -> Path:
        """Export to ONNX, OpenVINO, TensorRT, or TorchScript. Returns output path."""
        if self.task == "semantic" and format.lower() not in {"onnx", "openvino"}:
            raise ValueError(
                "Semantic segmentation currently supports format='onnx' or 'openvino' only"
            )
        from dfine.exporter import DFINEExporter

        exporter = DFINEExporter(
            self._get_deployed_model(device or self._device_str),
            self._cfg,
            device=device or self._device_str,
        )
        return exporter.export(
            format=format,
            imgsz=imgsz,
            batch=batch,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            half=half,
            verbose=verbose,
            project=project,
            name=name,
            save_dir=save_dir,
            output=output,
            exist_ok=exist_ok,
        )

    # ── Utilities ───────────────────────────────────────────────────────────

    def info(self, detailed: bool = False, verbose: bool = True) -> dict:
        """
        Return model info: parameter count, GFLOPs, and on-disk size.

        Args:
            detailed: If True, also break down parameters per layer.
            verbose:  Print a one-line summary.

        Returns:
            dict with keys ``params``, ``params_trainable``, ``gflops``, ``size_mb``.
            ``gflops`` is ``None`` when profiling fails.
            ``size_mb`` is ``None`` when the checkpoint path no longer exists.
        """
        from pathlib import Path

        import torch

        n_params = sum(p.numel() for p in self._model.parameters())
        n_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        # On-disk size
        p = Path(self._path)
        size_mb = p.stat().st_size / 1e6 if p.exists() else None

        # GFLOPs via torch.profiler (counts conv/linear/matmul without extra deps)
        gflops = None
        try:
            h, w = self._cfg.get("eval_spatial_size", [640, 640])
            dummy = torch.zeros(1, 3, h, w, device=self._device_str)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_flops=True,
            ) as prof:
                with torch.no_grad():
                    self._model(dummy)
            total_flops = sum(e.flops for e in prof.key_averages())
            gflops = total_flops / 1e9
        except Exception:
            pass

        result = {
            "params": n_params,
            "params_trainable": n_trainable,
            "gflops": gflops,
            "size_mb": size_mb,
        }

        if verbose:
            gflop_str = f"{gflops:.1f} GFLOPs" if gflops is not None else "GFLOPs n/a"
            size_str = f"{size_mb:.1f} MB" if size_mb is not None else "size n/a"
            print(
                f"[D-FINE] {n_params / 1e6:.1f}M params "
                f"({n_trainable / 1e6:.1f}M trainable)  "
                f"{gflop_str}  {size_str}"
            )

        if detailed:
            result["layers"] = {name: p.numel() for name, p in self._model.named_parameters()}

        return result

    @property
    def names(self) -> dict[int, str]:
        """Class id → class name mapping from checkpoint."""
        return self._names

    @property
    def device(self) -> str:
        """Effective compute device: an OpenVINO device string for
        `backend="openvino"`, otherwise the torch device string."""
        if self._backend == "openvino":
            assert self._openvino_device is not None
            return self._openvino_device
        return self._device_str

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def task(self) -> str:
        return self._task

    @property
    def weights(self) -> str | None:
        """Resolved official weight variant, or None for a user checkpoint path."""
        return self._weights

    @property
    def predictor(self):
        """Predictor built for the default 640 imgsz. `predict()`/`track()`
        build one sized to the requested `imgsz` instead — for
        `backend="openvino"` with a non-default `imgsz`, prefer those over
        this property."""
        return self._predictor_for(640)

    def _predictor_for(self, imgsz: int):
        from dfine.predictor import DFINEPredictor

        if self._backend == "openvino":
            return DFINEPredictor(
                self._get_openvino_model(imgsz),
                self._cfg,
                "cpu",
                self._names,
            )
        return DFINEPredictor(
            self._get_deployed_model(),
            self._cfg,
            self._device_str,
            self._names,
        )

    def _get_openvino_model(self, imgsz: int):
        """Return the OpenVINO-compiled raw model for `imgsz`, compiling and
        caching it on first use (OpenVINO compiles per fixed input shape).
        Locked so concurrent callers (e.g. parallel requests against a
        shared DFINE instance) don't each compile the same shape."""
        cached = self._openvino_cache.get(imgsz)
        if cached is not None:
            return cached

        with self._model_lock:
            cached = self._openvino_cache.get(imgsz)
            if cached is not None:
                return cached

            from dfine.exporter import DFINEExporter
            from dfine.nn.openvino_runtime import compile_raw_openvino

            exporter = DFINEExporter(self._get_deployed_model(), self._cfg, device=self._device_str)
            assert self._openvino_device is not None
            compiled = compile_raw_openvino(exporter, imgsz=imgsz, ov_device=self._openvino_device)
            self._openvino_cache[imgsz] = compiled
            return compiled

    # ── Internal ────────────────────────────────────────────────────────────

    def _invalidate_deployed_cache(self) -> None:
        """Discard the cached deployed inference copy after model changes."""
        self._deployed_model = None
        self._deployed_model_device = None
        self._openvino_cache.clear()

    def _get_deployed_model(self, device: str | None = None) -> nn.Module:
        """Return a cached deployed copy without mutating the trainable model.
        Locked so concurrent callers don't each deep-copy/deploy the model."""
        target_device = device or self._device_str
        if self._deployed_model is not None and self._deployed_model_device == target_device:
            return self._deployed_model

        with self._model_lock:
            if self._deployed_model is None or self._deployed_model_device != target_device:
                deployed = copy.deepcopy(self._model)
                deployed.to(target_device)
                deployed.eval()
                deploy_fn = getattr(deployed, "deploy", None)
                if callable(deploy_fn) and not bool(getattr(deployed, "_deployed", False)):
                    deploy_fn()
                    setattr(deployed, "_deployed", True)
                self._deployed_model = deployed
                self._deployed_model_device = target_device
            return self._deployed_model

    def _load(self, path: str, *, task: str, weights: str = "default") -> None:
        """Load checkpoint, deserialise config, build model."""
        from pathlib import Path

        from dfine.tasks import normalize_task
        from dfine.utils.checkpoint import load_checkpoint
        from dfine.utils.device import resolve_device
        from dfine.utils.downloads import download_model, get_model_asset

        self._device_str = resolve_device(self._device_str)
        resolved_task = normalize_task(task)
        if not isinstance(weights, str):
            raise TypeError("weights must be a string")
        default_weights_requested = weights.lower().replace("-", "_") == "default"

        path_obj = Path(path)
        if path_obj.exists():
            if not default_weights_requested:
                raise ValueError(
                    "weights= selects official registry weights and cannot be combined with "
                    f"an existing checkpoint path: {path_obj}"
                )
            self._weights = None
        else:
            # Check if it is a known model name or alias (e.g. "dfine_l", "dfine_l.pth", etc.)
            name_to_check = path_obj.name
            if name_to_check.endswith(".pth"):
                name_to_check = name_to_check[:-4]
            if name_to_check.endswith("_wrapped"):
                name_to_check = name_to_check[:-8]

            try:
                asset = get_model_asset(name_to_check, weights=weights, task=resolved_task)
                if path_obj.suffix == ".pth":
                    resolved_path = download_model(
                        asset.model, task=resolved_task, weights=asset.weights, output=path_obj
                    )
                else:
                    parent = path_obj.parent
                    if str(parent) in (".", ""):
                        resolved_path = download_model(
                            asset.model, task=resolved_task, weights=asset.weights, output=None
                        )
                    else:
                        resolved_path = download_model(
                            asset.model, task=resolved_task, weights=asset.weights, output=parent
                        )
                path = str(resolved_path)
                self._weights = asset.weights
            except ValueError:
                if not default_weights_requested:
                    raise
                # Not a known model/alias, let load_checkpoint raise FileNotFoundError
                pass

        self._path = str(path)
        self._model, self._cfg, self._names = load_checkpoint(path, device=self._device_str)
        checkpoint_task = normalize_task(str(self._cfg.get("task", "detect")))
        if checkpoint_task != resolved_task:
            raise ValueError(
                f"Checkpoint task is {checkpoint_task!r}, but task={resolved_task!r} was requested"
            )
        self._task = resolved_task
        self._model.eval()
        self._invalidate_deployed_cache()
        if self.verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"[D-FINE] Loaded '{path}' — {n_params / 1e6:.1f}M params on {self._device_str}")
