"""
DFINEExporter — model export to ONNX, OpenVINO, TensorRT, TorchScript.
Called internally by DFINE.export(). Not part of the public API.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from dfine.tasks import normalize_task
from dfine.utils.logging import LOGGER
from dfine.utils.runs import atomic_output_path, resolve_run_dir, write_run_metadata

# Raw (undecoded) model output names per task — the dict keys the deployed
# model itself returns, as opposed to the postprocessed (labels, boxes,
# scores) contract used by the public export() formats.
_RAW_OUTPUT_NAMES: dict[str, list[str]] = {
    "detect": ["pred_logits", "pred_boxes"],
    "segment": ["pred_logits", "pred_boxes", "pred_masks"],
    "semantic": ["sem_seg_logits"],
}


# Postprocessed output names per task — the decoded contract produced by
# DeployModel (model + postprocessor in deploy mode).
_POSTPROCESSED_OUTPUT_NAMES: dict[str, list[str]] = {
    "detect": ["labels", "boxes", "scores"],
    "segment": ["labels", "boxes", "scores", "masks"],
    "semantic": ["semantic_logits"],
}


class DeployModel(torch.nn.Module):
    def __init__(self, model, postprocessor, *, semantic: bool = False) -> None:
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
        self.semantic = semantic
        self.eval()

    def forward(self, images):
        outputs = self.model(images)
        if self.semantic:
            return outputs["sem_seg_logits"]
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
        postprocess: bool,
        verbose: bool,
        project: str,
        name: str,
        save_dir: str | Path | None,
        output: str | Path | None,
        exist_ok: bool,
    ) -> Path:
        format = format.lower()
        suffixes = {
            "onnx": ".onnx",
            "openvino": ".xml",
            "tensorrt": ".engine",
            "torchscript": ".torchscript",
        }
        if format not in suffixes:
            raise ValueError(
                "Unsupported export format: "
                f"{format!r}. Choose: onnx, openvino, tensorrt, torchscript"
            )
        if output is not None:
            out = Path(output)
            if out.exists() and not exist_ok:
                raise FileExistsError(f"Export output already exists: '{out}'")
            run_dir = out.parent
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = resolve_run_dir(
                project=project, name=name, save_dir=save_dir, exist_ok=exist_ok
            )
            out = run_dir / f"dfine_{imgsz}{suffixes[format]}"
        if format == "openvino":
            if out.suffix.lower() != ".xml":
                raise ValueError("OpenVINO output must use the .xml suffix")
            weights_out = out.with_suffix(".bin")
            if weights_out.exists() and not exist_ok:
                raise FileExistsError(f"Export output already exists: '{weights_out}'")
        write_run_metadata(
            run_dir,
            {
                "mode": "export",
                "task": str(self.cfg.get("task", "detect")),
                "format": format,
                "imgsz": imgsz,
                "batch": batch,
                "dynamic": dynamic,
                "simplify": simplify,
                "opset": opset,
                "half": half,
                "postprocess": postprocess,
                "project": project,
                "name": name,
                "save_dir": str(run_dir),
                "output": str(out),
                "exist_ok": exist_ok,
                "verbose": verbose,
            },
        )
        if format == "onnx":
            return self._to_onnx(
                out, imgsz, batch, dynamic, simplify, opset, verbose, postprocess=postprocess
            )
        if format == "openvino":
            return self._to_openvino(
                out,
                imgsz,
                batch,
                dynamic,
                simplify,
                opset,
                half,
                verbose,
                postprocess=postprocess,
            )
        if format == "tensorrt":
            return self._to_tensorrt(
                out, imgsz, batch, dynamic, half, verbose, postprocess=postprocess
            )
        if format == "torchscript":
            return self._to_torchscript(out, imgsz, batch, verbose)
        raise AssertionError("unreachable")

    # ── ONNX ────────────────────────────────────────────────────────────────

    def _to_onnx(
        self, out, imgsz, batch, dynamic, simplify, opset, verbose, *, postprocess: bool = True
    ) -> Path:
        import onnx

        with atomic_output_path(out) as temporary:
            self._export_onnx_to_path(
                temporary, imgsz, batch, dynamic, opset, postprocess=postprocess
            )
            if simplify:
                import onnxsim

                model_onnx = onnx.load(str(temporary))
                model_onnx, ok = onnxsim.simplify(model_onnx)
                if ok:
                    onnx.save(model_onnx, str(temporary))
        if verbose:
            LOGGER.info("ONNX export saved to %s", out)
        return out

    # ── OpenVINO ────────────────────────────────────────────────────────────

    def _to_openvino(
        self,
        out: Path,
        imgsz: int,
        batch: int,
        dynamic: bool,
        simplify: bool,
        opset: int,
        half: bool,
        verbose: bool,
        *,
        postprocess: bool = True,
    ) -> Path:
        """Export the corrected ONNX graph, then convert it to OpenVINO IR."""
        try:
            import openvino as ov
        except ImportError:
            raise ImportError(
                "OpenVINO is not installed. Install it with: uv sync --extra openvino"
            ) from None

        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{out.stem}.", dir=out.parent) as directory:
            staging_dir = Path(directory)
            onnx_path = staging_dir / "model.onnx"
            ir_path = staging_dir / "model.xml"
            self._to_onnx(
                onnx_path,
                imgsz,
                batch,
                dynamic,
                simplify,
                opset,
                verbose=False,
                postprocess=postprocess,
            )

            ov_model = ov.convert_model(onnx_path)
            ov.save_model(ov_model, ir_path, compress_to_fp16=half)

            # Ensure the serialized IR can be loaded and compiled before publishing it.
            core = ov.Core()
            core.compile_model(core.read_model(ir_path), "CPU")

            staged_weights = ir_path.with_suffix(".bin")
            if not staged_weights.is_file():
                raise RuntimeError("OpenVINO conversion did not produce the expected .bin file")
            for staged_file in (staged_weights, ir_path):
                with staged_file.open("rb") as stream:
                    os.fsync(stream.fileno())

            # Publish weights first and XML last, so a visible XML always has complete weights.
            os.replace(staged_weights, out.with_suffix(".bin"))
            os.replace(ir_path, out)

        LOGGER.info("OpenVINO IR export saved to %s", out)
        return out

    def _export_onnx_to_path(
        self,
        path: Path,
        imgsz: int,
        batch: int,
        dynamic: bool,
        opset: int,
        *,
        postprocess: bool = True,
    ) -> None:
        """
        Trace the model to ONNX at an explicit output path.

        With ``postprocess=False`` the deployed model is traced on its own, so the
        graph exposes the decoder's raw outputs (``pred_logits``/``pred_boxes``/…)
        instead of the decoded ``(labels, boxes, scores, …)`` contract.
        """
        from typing import Any

        from dfine.nn.build import build_postprocessor

        task = normalize_task(str(self.cfg.get("task", "detect")))

        if postprocess:
            postprocessor: Any = build_postprocessor(self.cfg)
            if hasattr(postprocessor, "deploy"):
                postprocessor.deploy()
            postprocessor.to(self.device)
            traced_model: torch.nn.Module = DeployModel(
                self.model, postprocessor, semantic=task == "semantic"
            )
            traced_model.eval()
            output_names = _POSTPROCESSED_OUTPUT_NAMES[task]
        else:
            traced_model, output_names = self.prepare_raw_trace(imgsz)

        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        dynamic_axes = (
            {name: {0: "batch"} for name in ("images", *output_names)} if dynamic else None
        )

        torch.onnx.export(
            traced_model,
            (dummy,),
            str(path),
            dynamo=False,
            opset_version=opset,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # ── Raw (non-postprocessed) trace — used by the OpenVINO runtime ────────

    def prepare_raw_trace(self, imgsz: int) -> tuple[torch.nn.Module, list[str]]:
        """
        Return ``(model, output_names)`` for a raw trace of the deployed model
        at ``imgsz`` — its own ``pred_logits``/``pred_boxes``/etc. dict, not
        the postprocessed ``(labels, boxes, scores)`` graph ``export()``
        produces by default. Used by ``export(..., postprocess=False)`` and by
        ``dfine.nn.openvino_runtime.compile_raw_openvino``.
        """
        task = normalize_task(str(self.cfg.get("task", "detect")))
        return self.model, _RAW_OUTPUT_NAMES[task]

    # ── TensorRT ─────────────────────────────────────────────────────────────

    def _to_tensorrt(
        self,
        out: Path,
        imgsz: int,
        batch: int,
        dynamic: bool,
        half: bool,
        verbose: bool,
        *,
        postprocess: bool = True,
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

        # ── Step 1: trace to a temporary ONNX ────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_onnx = Path(f.name)

        try:
            self._export_onnx_to_path(
                tmp_onnx, imgsz, batch, dynamic, opset=17, postprocess=postprocess
            )

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

            with atomic_output_path(out) as temporary:
                temporary.write_bytes(engine_bytes)

        finally:
            tmp_onnx.unlink(missing_ok=True)

        LOGGER.info(f"TensorRT engine saved to {out}")
        return out

    # ── TorchScript ──────────────────────────────────────────────────────────

    def _to_torchscript(self, out, imgsz, batch, verbose) -> Path:
        dummy = torch.zeros(batch, 3, imgsz, imgsz, device=self.device)
        # D-FINE forward returns a dict; strict=False allows tracing dict outputs
        scripted = torch.jit.trace(self.model, dummy, strict=False)
        with atomic_output_path(out) as temporary:
            scripted.save(str(temporary))
        LOGGER.info(f"TorchScript export saved to {out}")
        return out
