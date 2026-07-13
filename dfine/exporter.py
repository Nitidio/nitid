"""
DFINEExporter — model export to ONNX, TensorRT, TorchScript.
Called internally by DFINE.export(). Not part of the public API.
"""

from __future__ import annotations

from pathlib import Path

import torch

from dfine.utils.logging import LOGGER


class DeployModel(torch.nn.Module):
    def __init__(self, model, postprocessor) -> None:
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.eval()

    def forward(self, images):
        outputs = self.model(images)
        B = images.shape[0]
        H = images.shape[2]
        W = images.shape[3]
        h_t = torch.as_tensor(H, dtype=torch.float32, device=images.device)
        w_t = torch.as_tensor(W, dtype=torch.float32, device=images.device)
        orig_target_sizes = torch.stack([w_t, h_t]).unsqueeze(0).repeat(B, 1)
        return self.postprocessor(outputs, orig_target_sizes)


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
        half: bool,
        verbose: bool,
    ) -> Path:
        format = format.lower()
        if format == "onnx":
            return self._to_onnx(imgsz, batch, dynamic, simplify, opset, verbose)
        if format == "tensorrt":
            return self._to_tensorrt(imgsz, batch, dynamic, half, verbose)
        if format == "torchscript":
            return self._to_torchscript(imgsz, batch, verbose)
        raise ValueError(
            f"Unsupported export format: {format!r}. Choose: onnx, tensorrt, torchscript"
        )

    # ── ONNX ────────────────────────────────────────────────────────────────

    def _to_onnx(self, imgsz, batch, dynamic, simplify, opset, verbose) -> Path:
        import onnx

        out = Path(f"dfine_{imgsz}.onnx")
        self._export_onnx_to_path(out, imgsz, batch, dynamic, opset)
        if simplify:
            import onnxsim

            model_onnx = onnx.load(str(out))
            model_onnx, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_onnx, str(out))
        LOGGER.info(f"ONNX export saved to {out}")
        return out

    def _export_onnx_to_path(
        self, path: Path, imgsz: int, batch: int, dynamic: bool, opset: int
    ) -> None:
        """Trace the model to ONNX at an explicit output path."""
        from typing import Any

        from dfine.nn.build import build_postprocessor

        # Deploy raw model if available
        if hasattr(self.model, "deploy"):
            self.model.deploy()

        postprocessor: Any = build_postprocessor(self.cfg)
        if hasattr(postprocessor, "deploy"):
            postprocessor.deploy()
        postprocessor.to(self.device)

        wrapped_model = DeployModel(self.model, postprocessor)
        wrapped_model.eval()

        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "labels": {0: "batch"},
                "boxes": {0: "batch"},
                "scores": {0: "batch"},
            }
            if dynamic
            else None
        )

        torch.onnx.export(
            wrapped_model,
            (dummy,),
            str(path),
            dynamo=False,
            opset_version=opset,
            input_names=["images"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
        )

    # ── TensorRT ─────────────────────────────────────────────────────────────

    def _to_tensorrt(
        self, imgsz: int, batch: int, dynamic: bool, half: bool, verbose: bool
    ) -> Path:
        """
        Export to a TensorRT serialised engine via the TRT Python API.

        Workflow: trace model → temp ONNX → parse with TRT → write .engine file.
        Requires ``tensorrt`` (``uv sync --extra tensorrt``).
        """
        import tempfile

        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is not installed. Install it with: uv sync --extra tensorrt"
            ) from None

        out = Path(f"dfine_{imgsz}.engine")

        # ── Step 1: trace to a temporary ONNX ────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_onnx = Path(f.name)

        try:
            self._export_onnx_to_path(tmp_onnx, imgsz, batch, dynamic, opset=17)

            # ── Step 2: build TRT engine from ONNX ───────────────────────────
            trt_logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, trt_logger)

            with open(str(tmp_onnx), "rb") as f:
                if not parser.parse(f.read()):
                    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"TensorRT failed to parse ONNX:\n{errors}")

            config = builder.create_builder_config()

            # Workspace: 1 GB — handle API change between TRT 8.x and 8.5+
            workspace = 1 << 30
            if hasattr(trt, "MemoryPoolType"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
            else:
                config.max_workspace_size = workspace  # type: ignore[attr-defined]

            if half:
                if not builder.platform_has_fast_fp16:
                    LOGGER.warning("half=True but GPU reports no native FP16 support")
                config.set_flag(trt.BuilderFlag.FP16)

            if dynamic:
                profile = builder.create_optimization_profile()
                # min=1, opt=batch, max=batch*4
                profile.set_shape(
                    "images",
                    (1, 3, imgsz, imgsz),
                    (batch, 3, imgsz, imgsz),
                    (batch * 4, 3, imgsz, imgsz),
                )
                config.add_optimization_profile(profile)

            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("TensorRT engine build failed — check GPU and TRT logs")

            out.write_bytes(engine_bytes)

        finally:
            tmp_onnx.unlink(missing_ok=True)

        LOGGER.info(f"TensorRT engine saved to {out}")
        return out

    # ── TorchScript ──────────────────────────────────────────────────────────

    def _to_torchscript(self, imgsz, batch, verbose) -> Path:
        out = Path(f"dfine_{imgsz}.torchscript")
        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        # D-FINE forward returns a dict; strict=False allows tracing dict outputs
        scripted = torch.jit.trace(self.model, dummy, strict=False)
        scripted.save(str(out))
        LOGGER.info(f"TorchScript export saved to {out}")
        return out
