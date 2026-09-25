"""Download official D-FINE checkpoints and wrap them for nitid."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch

from dfine.nn.configs import make_model_config
from dfine.tasks import normalize_task
from nitid.convert_checkpoint import convert as convert_checkpoint

_ROOT = Path(__file__).parents[2]
_RELEASE_ROOT = "https://github.com/Peterande/storage/releases/download/dfinev1.0"
_SEGMENT_RELEASE_ROOT = "https://huggingface.co/ArgoSA/D-FINE-seg/resolve/main"


@dataclass(frozen=True)
class ModelAsset:
    """One pretrained-weight variant for a D-FINE architecture."""

    model: str
    task: str
    weights: str
    url: str
    filename: str
    sha256: str | None = None

    @property
    def name(self) -> str:
        """Compatibility label used in messages and checkpoint metadata."""
        return self.model


def _asset(
    model: str,
    weights: str,
    checkpoint: str,
    *,
    task: str = "detect",
    release_root: str = _RELEASE_ROOT,
) -> ModelAsset:
    return ModelAsset(
        model=model,
        task=task,
        weights=weights,
        url=f"{release_root}/{checkpoint}",
        filename=(
            f"dfine_seg_{model.removeprefix('dfine_')}_{weights}_wrapped.pth"
            if task == "segment"
            else (
                f"dfine_semantic_{model.removeprefix('dfine_')}_{weights}_init_wrapped.pth"
                if task == "semantic"
                else f"{model}_{weights}_wrapped.pth"
            )
        ),
    )


MODEL_REGISTRY: dict[str, dict[str, ModelAsset]] = {
    "dfine_s": {
        "obj2coco": _asset(
            "dfine_s",
            "obj2coco",
            "dfine_s_obj2coco.pth",
        ),
        "coco": _asset("dfine_s", "coco", "dfine_s_coco.pth"),
    },
    "dfine_m": {
        "obj2coco": _asset(
            "dfine_m",
            "obj2coco",
            "dfine_m_obj2coco.pth",
        ),
        "coco": _asset("dfine_m", "coco", "dfine_m_coco.pth"),
    },
    "dfine_l": {
        "obj2coco": _asset(
            "dfine_l",
            "obj2coco",
            "dfine_l_obj2coco_e25.pth",
        ),
        "coco": _asset("dfine_l", "coco", "dfine_l_coco.pth"),
    },
    "dfine_x": {
        "obj2coco": _asset(
            "dfine_x",
            "obj2coco",
            "dfine_x_obj2coco.pth",
        ),
        "coco": _asset("dfine_x", "coco", "dfine_x_coco.pth"),
    },
}

SEGMENT_MODEL_REGISTRY: dict[str, dict[str, ModelAsset]] = {
    model: {
        "coco": _asset(
            model,
            "coco",
            f"dfine_seg_{model.removeprefix('dfine_')}_coco.pt",
            task="segment",
            release_root=_SEGMENT_RELEASE_ROOT,
        )
    }
    for model in ("dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x")
}
SEMANTIC_MODEL_REGISTRY: dict[str, dict[str, ModelAsset]] = {
    model: {
        "coco": _asset(
            model,
            "coco",
            f"dfine_seg_{model.removeprefix('dfine_')}_coco.pt",
            task="semantic",
            release_root=_SEGMENT_RELEASE_ROOT,
        )
    }
    for model in ("dfine_n", "dfine_s", "dfine_m", "dfine_l", "dfine_x")
}

DEFAULT_WEIGHTS = "obj2coco"
_MODEL_ALIASES = {
    "n": "dfine_n",
    "s": "dfine_s",
    "m": "dfine_m",
    "l": "dfine_l",
    "x": "dfine_x",
    "d_fine_n": "dfine_n",
    "d_fine_s": "dfine_s",
    "d_fine_m": "dfine_m",
    "d_fine_l": "dfine_l",
    "d_fine_x": "dfine_x",
}
_WEIGHT_ALIASES = {
    "default": DEFAULT_WEIGHTS,
    "objects365_coco": "obj2coco",
    "objects365_to_coco": "obj2coco",
    "coco_only": "coco",
}


def _normalize_model(model: str) -> str:
    key = model.lower().replace("-", "_")
    return _MODEL_ALIASES.get(key, key)


def _normalize_weights(weights: str) -> str:
    if not isinstance(weights, str):
        raise TypeError("weights must be a string")
    key = weights.lower().replace("-", "_")
    return _WEIGHT_ALIASES.get(key, key)


def _registry(task: str) -> dict[str, dict[str, ModelAsset]]:
    resolved_task = normalize_task(task)
    if resolved_task == "detect":
        return MODEL_REGISTRY
    if resolved_task == "segment":
        return SEGMENT_MODEL_REGISTRY
    return SEMANTIC_MODEL_REGISTRY


def list_models(task: str = "detect") -> list[str]:
    """Return supported D-FINE architecture names."""
    return sorted(_registry(task))


def list_weights(model: str, *, task: str = "detect") -> list[str]:
    """Return canonical pretrained-weight variants for an architecture."""
    model_key = _normalize_model(model)
    registry = _registry(task)
    if model_key not in registry:
        choices = ", ".join(list_models(task))
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")
    return sorted(registry[model_key])


def get_model_asset(
    model: str,
    weights: str = "default",
    *,
    task: str = "detect",
) -> ModelAsset:
    """Resolve a model architecture and pretrained-weight variant."""
    model_key = _normalize_model(model)
    resolved_task = normalize_task(task)
    registry = _registry(resolved_task)
    if model_key not in registry:
        choices = ", ".join(list_models(resolved_task))
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")

    weights_key = _normalize_weights(weights)
    if resolved_task in {"segment", "semantic"} and weights_key == DEFAULT_WEIGHTS:
        weights_key = "coco"
    variants = registry[model_key]
    if weights_key not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown weights {weights!r} for {model_key}. Choose one of: {choices}, default"
        )
    return variants[weights_key]


def _verify_sha256(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"Checksum mismatch for downloaded checkpoint {path.name}: "
            f"expected {expected}, got {digest}"
        )


def resolve_output_path(asset: ModelAsset, output: str | Path | None = None) -> Path:
    """Resolve output file path from a file path, directory path, or None."""
    if output is None:
        return Path(asset.filename)

    path = Path(output)
    if path.suffix == ".pth":
        return path
    return path / asset.filename


def download_model(
    model: str,
    *,
    task: str = "detect",
    weights: str = "default",
    output: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and wrap an official checkpoint for ``model`` and ``weights``."""
    asset = get_model_asset(model, weights, task=task)
    out_path = resolve_output_path(asset, output)

    if out_path.exists() and not force:
        print(f"{out_path} already exists. Use force=true to overwrite.")
        return out_path

    names = _ROOT / "configs" / "datasets" / "coco.yml"
    if not names.exists():
        raise FileNotFoundError(f"Class names file not found: {names}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_filename = Path(urlparse(asset.url).path).name
    with tempfile.TemporaryDirectory(prefix="nitid-download-") as tmp_dir:
        raw_path = Path(tmp_dir) / raw_filename
        print(f"Downloading {asset.model} weights={asset.weights} from {asset.url}")
        urlretrieve(asset.url, raw_path)
        _verify_sha256(raw_path, asset.sha256)
        print(f"Converting to nitid checkpoint: {out_path}")
        if asset.task == "semantic":
            _wrap_semantic_initialization(raw_path, asset, names, out_path)
        else:
            convert_checkpoint(
                weights=str(raw_path),
                config=make_model_config(asset.model, task=asset.task),
                names_file=str(names),
                output=str(out_path),
            )

    return out_path


def _wrap_semantic_initialization(
    raw_path: Path,
    asset: ModelAsset,
    names_path: Path,
    output_path: Path,
) -> None:
    """Build a semantic checkpoint initialized from compatible instance weights."""
    from collections.abc import Mapping

    from dfine.nn.build import build_model
    from dfine.nn.transfer import compatible_pretrained_state
    from dfine.utils.checkpoint import save_checkpoint

    raw_checkpoint = torch.load(raw_path, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, Mapping):
        raise TypeError("Downloaded checkpoint must contain a state mapping")
    ema = raw_checkpoint.get("ema")
    if isinstance(ema, Mapping) and isinstance(ema.get("module"), Mapping):
        state = dict(ema["module"])
    else:
        model_state = raw_checkpoint.get("model", raw_checkpoint)
        if not isinstance(model_state, Mapping):
            raise TypeError("Downloaded checkpoint has no model state mapping")
        state = dict(model_state)

    with names_path.open() as names_file:
        import yaml

        names_config = yaml.safe_load(names_file) or {}
    raw_names = names_config.get("names", {})
    names = (
        {index: str(name) for index, name in enumerate(raw_names)}
        if isinstance(raw_names, list)
        else {int(index): str(name) for index, name in raw_names.items()}
    )
    config = make_model_config(asset.model, task="semantic", num_classes=len(names))
    model = build_model(config)
    compatible = compatible_pretrained_state(state, model)
    load_result = model.load_state_dict(compatible, strict=False)
    mask_fuser_keys = {key for key in model.state_dict() if key.startswith("decoder.mask_decoder.")}
    if not mask_fuser_keys.issubset(compatible):
        raise RuntimeError(
            "Instance checkpoint is incompatible with the semantic feature-fusion head"
        )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected semantic initialization keys: {load_result.unexpected_keys}"
        )
    save_checkpoint(output_path, model, config, names)
