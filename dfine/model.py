"""
DFINE — public entry point. Mirrors the ultralytics.YOLO interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, Union

import numpy as np
import torch.nn as nn

from dfine.utils.device import resolve_device

Source = Union[str, Path, int, np.ndarray, list]
ModelCallback = Callable[..., object]


class DFINE:
    """
    D-FINE object detection wrapper.

    Args:
        model:   Path to .pth checkpoint (config serialised inside).
        device:  "cuda", "cpu", "cuda:N", or None for auto-select.
        verbose: Print model info on load.

    Example:
        model = DFINE("dfine_l.pth")
        results = model("image.jpg", conf=0.5)
        model.train(data="coco.yaml", epochs=50)
        model.export(format="tensorrt")
    """

    def __init__(
        self,
        model: str = "dfine_l.pth",
        device: str | int | None = None,
        verbose: bool = True,
    ) -> None:
        self._device_str: str = resolve_device(device)
        self.verbose = verbose
        self._model: nn.Module
        self._cfg: dict[str, Any]
        self._names: dict[int, str]
        self._path: str
        self._callbacks: dict[str, list[ModelCallback]] = {}
        self._load(model)

    # ── Inference ──────────────────────────────────────────────────────────

    def __call__(self, source: Source, **kwargs) -> list[Any] | Generator[Any, None, None]:
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Source,
        conf: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        stream: bool = False,
        vid_stride: int = 1,
        augment: bool = False,
        save: bool = False,
        project: str = "runs/detect",
        name: str = "exp",
        save_dir: str | Path | None = None,
        exist_ok: bool = False,
        verbose: bool = True,
        iou: float = 0.85,
    ) -> list | Generator:
        """
        Run detection on source.

        Returns list[Results] when stream=False,
        Generator[Results] when stream=True.
        """
        return self.predictor.run(
            source,
            conf=conf,
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
            iou=iou,
        )

    # ── Training ────────────────────────────────────────────────────────────

    def train(
        self,
        data: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 1e-4,
        lrf: float = 0.01,
        cos_lr: bool = False,
        warmup_epochs: float = 0.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        optimizer: str = "AdamW",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        clip_grad: float = 0.1,
        resume: bool = False,
        amp: bool = False,
        ema: bool = False,
        ema_decay: float = 0.9999,
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
        verbose: bool = True,
        callbacks: object | None = None,
        wandb: bool | dict[str, Any] = False,
        mlflow: bool | dict[str, Any] = False,
    ) -> dict:
        """Fine-tune on a custom dataset. Returns final metrics plus per-epoch history."""
        from dfine.trainer import DFINETrainer

        trainer = DFINETrainer(
            model=self._model,
            cfg=self._cfg,
            device=device or self._device_str,
            names=self._names,
            callbacks=self._callbacks,
        )
        try:
            return trainer.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                lrf=lrf,
                cos_lr=cos_lr,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                optimizer=optimizer,
                momentum=momentum,
                weight_decay=weight_decay,
                clip_grad=clip_grad,
                resume=resume,
                amp=amp,
                ema=ema,
                ema_decay=ema_decay,
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
                verbose=verbose,
                callbacks=callbacks,
                wandb=wandb,
                mlflow=mlflow,
            )
        except BaseException as error:
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
        from dfine.exporter import DFINEExporter

        exporter = DFINEExporter(
            self._model,
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
        return self._device_str

    @property
    def task(self) -> str:
        return "detect"

    @property
    def predictor(self):
        from dfine.predictor import DFINEPredictor

        return DFINEPredictor(self._model, self._cfg, self._device_str, self._names)

    # ── Internal ────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Load checkpoint, deserialise config, build model."""
        from pathlib import Path

        from dfine.utils.checkpoint import load_checkpoint
        from dfine.utils.device import resolve_device
        from dfine.utils.downloads import download_model, get_model_asset

        self._device_str = resolve_device(self._device_str)

        path_obj = Path(path)
        if not path_obj.exists():
            # Check if it is a known model name or alias (e.g. "dfine_l", "dfine_l.pth", etc.)
            name_to_check = path_obj.name
            if name_to_check.endswith(".pth"):
                name_to_check = name_to_check[:-4]
            if name_to_check.endswith("_wrapped"):
                name_to_check = name_to_check[:-8]

            try:
                asset = get_model_asset(name_to_check)
                if path_obj.suffix == ".pth":
                    resolved_path = download_model(asset.name, output=path_obj)
                else:
                    parent = path_obj.parent
                    if str(parent) in (".", ""):
                        resolved_path = download_model(asset.name, output=None)
                    else:
                        resolved_path = download_model(asset.name, output=parent)
                path = str(resolved_path)
            except ValueError:
                # Not a known model/alias, let load_checkpoint raise FileNotFoundError
                pass

        self._path = str(path)
        self._model, self._cfg, self._names = load_checkpoint(path, device=self._device_str)
        self._model.eval()
        if self.verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"[D-FINE] Loaded '{path}' — {n_params / 1e6:.1f}M params on {self._device_str}")
