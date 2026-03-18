"""
DFINEExporter — model export to ONNX, TensorRT, TorchScript.
Called internally by DFINE.export(). Not part of the public API.
"""
from __future__ import annotations
from pathlib import Path
import torch
from dfine.utils.logging import LOGGER


class DFINEExporter:
    def __init__(self, model, cfg: dict, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

    def export(
        self,
        format: str,
        imgsz: int,
        batch: int,
        dynamic: bool,
        simplify: bool,
        opset: int,
        verbose: bool,
    ) -> Path:
        format = format.lower()
        if format == "onnx":
            return self._to_onnx(imgsz, batch, dynamic, simplify, opset, verbose)
        if format == "tensorrt":
            return self._to_tensorrt(imgsz, batch, dynamic, verbose)
        if format == "torchscript":
            return self._to_torchscript(imgsz, batch, verbose)
        raise ValueError(f"Unsupported export format: {format!r}. Choose: onnx, tensorrt, torchscript")

    def _to_onnx(self, imgsz, batch, dynamic, simplify, opset, verbose) -> Path:
        import onnx
        out = Path(f"dfine_{imgsz}.onnx")
        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
        torch.onnx.export(
            self.model,
            dummy,
            str(out),
            opset_version=opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        if simplify:
            import onnxsim
            model_onnx = onnx.load(str(out))
            model_onnx, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_onnx, str(out))
        LOGGER.info(f"ONNX export saved to {out}")
        return out

    def _to_tensorrt(self, imgsz, batch, dynamic, verbose) -> Path:
        # Requires: pip install tensorrt
        # Workflow: ONNX → trtexec or TRT Python API
        raise NotImplementedError("TensorRT export — coming in Phase 2")

    def _to_torchscript(self, imgsz, batch, verbose) -> Path:
        out = Path(f"dfine_{imgsz}.torchscript")
        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        scripted = torch.jit.trace(self.model, dummy)
        scripted.save(str(out))
        LOGGER.info(f"TorchScript export saved to {out}")
        return out
