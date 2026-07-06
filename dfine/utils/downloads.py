"""Download official D-FINE checkpoints and wrap them for nitid."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from tools.convert_checkpoint import convert as convert_checkpoint

_ROOT = Path(__file__).parents[2]


@dataclass(frozen=True)
class ModelAsset:
    name: str
    url: str
    config: Path
    filename: str


MODEL_REGISTRY = {
    "dfine_s": ModelAsset(
        name="dfine_s",
        url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth",
        config=_ROOT / "extern" / "dfine" / "configs" / "dfine" / "dfine_hgnetv2_s_coco.yml",
        filename="dfine_s_wrapped.pth",
    ),
    "dfine_m": ModelAsset(
        name="dfine_m",
        url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
        config=_ROOT / "extern" / "dfine" / "configs" / "dfine" / "dfine_hgnetv2_m_coco.yml",
        filename="dfine_m_wrapped.pth",
    ),
    "dfine_l": ModelAsset(
        name="dfine_l",
        url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth",
        config=_ROOT / "extern" / "dfine" / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml",
        filename="dfine_l_wrapped.pth",
    ),
    "dfine_x": ModelAsset(
        name="dfine_x",
        url="https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth",
        config=_ROOT / "extern" / "dfine" / "configs" / "dfine" / "dfine_hgnetv2_x_coco.yml",
        filename="dfine_x_wrapped.pth",
    ),
}

_ALIASES = {
    "s": "dfine_s",
    "m": "dfine_m",
    "l": "dfine_l",
    "x": "dfine_x",
    "d_fine_s": "dfine_s",
    "d_fine_m": "dfine_m",
    "d_fine_l": "dfine_l",
    "d_fine_x": "dfine_x",
}


def list_models() -> list[str]:
    """Return supported model names."""
    return sorted(MODEL_REGISTRY)


def get_model_asset(model: str) -> ModelAsset:
    """Return metadata for a supported D-FINE model name."""
    key = model.lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in MODEL_REGISTRY:
        choices = ", ".join(list_models())
        raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")
    return MODEL_REGISTRY[key]


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
    output: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download a raw D-FINE checkpoint and convert it to nitid's wrapped format.

    Args:
        model: Supported model name: dfine_s, dfine_m, dfine_l, or dfine_x.
        output: Optional output file path or directory. Defaults to current directory.
        force: Overwrite an existing wrapped checkpoint when True.

    Returns:
        Path to the wrapped checkpoint.
    """
    asset = get_model_asset(model)
    out_path = resolve_output_path(asset, output)

    if out_path.exists() and not force:
        print(f"{out_path} already exists. Use force=true to overwrite.")
        return out_path

    if not asset.config.exists():
        raise FileNotFoundError(
            f"Config not found: {asset.config}. Run git submodule update --init."
        )

    names = _ROOT / "configs" / "datasets" / "coco.yml"
    if not names.exists():
        raise FileNotFoundError(f"Class names file not found: {names}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nitid-download-") as tmp_dir:
        raw_path = Path(tmp_dir) / f"{asset.name}_coco.pth"
        print(f"Downloading {asset.name} from {asset.url}")
        urlretrieve(asset.url, raw_path)
        print(f"Converting to nitid checkpoint: {out_path}")
        convert_checkpoint(
            weights=str(raw_path),
            config=str(asset.config),
            names_file=str(names),
            output=str(out_path),
        )

    return out_path
