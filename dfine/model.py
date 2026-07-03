"""
DFINE — public entry point. Mirrors the ultralytics.YOLO interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Union

import numpy as np

Source = Union[str, Path, int, np.ndarray, list]


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
        self._device_str = device
        self.verbose = verbose
        self._model = None  # torch.nn.Module, loaded lazily
        self._cfg = None  # dict, deserialised from checkpoint
        self._names: dict[int, str] = {}
        self._load(model)

    # ── Inference ──────────────────────────────────────────────────────────

    def __call__(self, source: Source, **kwargs) -> list:
        return self.predict(source, **kwargs)

    def predict(
        self,
        source: Source,
        conf: float = 0.5,
        imgsz: int = 640,
        classes: list[int] | None = None,
        stream: bool = False,
        augment: bool = False,
        verbose: bool = True,
    ) -> list | Generator:
        """
        Run detection on source.

        Returns list[Results] when stream=False,
        Generator[Results] when stream=True.
        """
        from dfine.predictor import DFINEPredictor

        predictor = DFINEPredictor(self._model, self._cfg, self._device_str, self._names)
        return predictor.run(
            source,
            conf=conf,
            imgsz=imgsz,
            classes=classes,
            stream=stream,
            augment=augment,
            verbose=verbose,
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
        optimizer: str = "AdamW",
        resume: bool = False,
        amp: bool = False,
        ema: bool = False,
        ema_decay: float = 0.9999,
        device: str | None = None,
        project: str = "runs/train",
        name: str = "exp",
        verbose: bool = True,
    ) -> dict:
        """Fine-tune on a custom dataset. Returns final metrics dict."""
        from dfine.trainer import DFINETrainer

        trainer = DFINETrainer(
            model=self._model,
            cfg=self._cfg,
            device=device or self._device_str,
            names=self._names,
        )
        return trainer.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            optimizer=optimizer,
            resume=resume,
            amp=amp,
            ema=ema,
            ema_decay=ema_decay,
            project=project,
            name=name,
            verbose=verbose,
        )

    # ── Validation ──────────────────────────────────────────────────────────

    def val(
        self,
        data: str,
        imgsz: int = 640,
        batch: int = 16,
        conf: float = 0.001,
        split: str = "val",
        verbose: bool = True,
    ) -> dict:
        """Evaluate on val/test split. Returns mAP50, mAP50-95, etc."""
        from dfine.validator import DFINEValidator

        validator = DFINEValidator(self._model, self._cfg, self._device_str, self._names)
        return validator.run(
            data=data,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            split=split,
            verbose=verbose,
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
    ) -> Path:
        """Export to ONNX, TensorRT, or TorchScript. Returns output path."""
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

    # ── Internal ────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        """Load checkpoint, deserialise config, build model."""
        from dfine.utils.checkpoint import load_checkpoint
        from dfine.utils.device import resolve_device

        self._device_str = resolve_device(self._device_str)
        self._path = str(path)
        self._model, self._cfg, self._names = load_checkpoint(path, device=self._device_str)
        self._model.eval()
        if self.verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"[D-FINE] Loaded '{path}' — {n_params / 1e6:.1f}M params on {self._device_str}")
