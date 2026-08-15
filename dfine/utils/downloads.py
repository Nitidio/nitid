"""Download official D-FINE checkpoints and wrap them for nitid."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from dfine.nn.configs import make_detection_config
from tools.convert_checkpoint import convert as convert_checkpoint

_ROOT = Path(__file__).parents[2]
_RELEASE_ROOT = "https://github.com/Peterande/storage/releases/download/dfinev1.0"


@dataclass(frozen=True)
class ModelAsset:
    """One pretrained-weight variant for a D-FINE architecture."""

    model: str
    weights: str
    url: str
    filename: str

    @property
    def name(self) -> str:
        """Compatibility label used in messages and checkpoint metadata."""
        return self.model


def _asset(
    model: str,
    weights: str,
    checkpoint: str,
) -> ModelAsset:
    return ModelAsset(
        model=model,
        weights=weights,
        url=f"{_RELEASE_ROOT}/{checkpoint}",
        filename=f"{model}_{weights}_wrapped.pth",
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

DEFAULT_WEIGHTS = "obj2coco"
_MODEL_ALIASES = {
    "s": "dfine_s",
    "m": "dfine_m",
    "l": "dfine_l",
    "x": "dfine_x",
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


def list_models() -> list[str]:
    """Return supported D-FINE architecture names."""
    return sorted(MODEL_REGISTRY)


def list_weights(model: str) -> list[str]:
    """Return canonical pretrained-weight variants for an architecture."""
    model_key = _normalize_model(model)
    if model_key not in MODEL_REGISTRY:
        choices = ", ".join(list_models())
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")
    return sorted(MODEL_REGISTRY[model_key])


def get_model_asset(model: str, weights: str = "default") -> ModelAsset:
    """Resolve a model architecture and pretrained-weight variant."""
    model_key = _normalize_model(model)
    if model_key not in MODEL_REGISTRY:
        choices = ", ".join(list_models())
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")

    weights_key = _normalize_weights(weights)
    variants = MODEL_REGISTRY[model_key]
    if weights_key not in variants:
        choices = ", ".join(sorted(variants))
        raise ValueError(
            f"Unknown weights {weights!r} for {model_key}. Choose one of: {choices}, default"
        )
    return variants[weights_key]


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
    weights: str = "default",
    output: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Download and wrap an official checkpoint for ``model`` and ``weights``."""
    asset = get_model_asset(model, weights)
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
        print(f"Converting to nitid checkpoint: {out_path}")
        convert_checkpoint(
            weights=str(raw_path),
            config=make_detection_config(asset.model),
            names_file=str(names),
            output=str(out_path),
        )

    return out_path
