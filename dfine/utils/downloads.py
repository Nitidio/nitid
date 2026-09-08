"""Download official D-FINE checkpoints and wrap them for nitid."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch

from dfine.nn.configs import make_model_config, make_pose_config
from dfine.nn.rio import DOTA_OBB_NAMES, make_rio_obb_config
from dfine.pose_contract import get_pose_checkpoint
from dfine.tasks import normalize_task
from tools.convert_checkpoint import convert as convert_checkpoint

_ROOT = Path(__file__).parents[2]
_RELEASE_ROOT = "https://github.com/Peterande/storage/releases/download/dfinev1.0"
_SEGMENT_RELEASE_ROOT = "https://huggingface.co/ArgoSA/D-FINE-seg/resolve/main"
_RIO_OBB_RELEASE_ROOT = "https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main"


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


def _pose_asset(model: str, weights: str) -> ModelAsset:
    checkpoint = get_pose_checkpoint(model, "coco")
    return ModelAsset(
        model=model,
        task="pose",
        weights=weights,
        url=checkpoint.url,
        filename=f"{model}_coco_wrapped.pth",
    )


def _rio_obb_asset(model: str, weights: str, sha256: str) -> ModelAsset:
    size = model[-1]
    return ModelAsset(
        model=model,
        task="obb",
        weights=weights,
        url=f"{_RIO_OBB_RELEASE_ROOT}/{weights}/rtdetrv2_obb_hgnetv2_{size}_{weights}.pth",
        filename=f"{model}_{weights}_wrapped.pth",
        sha256=sha256,
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

POSE_MODEL_REGISTRY: dict[str, dict[str, ModelAsset]] = {
    model: {
        "coco": _pose_asset(model, "coco"),
        "default": _pose_asset(model, "default"),
    }
    for model in ("detrpose_n", "detrpose_s", "detrpose_m", "detrpose_l", "detrpose_x")
}

RIO_OBB_MODEL_REGISTRY: dict[str, dict[str, ModelAsset]] = {
    "nitid1n": {
        "diorr": _rio_obb_asset(
            "nitid1n", "diorr", "6d00665413e48d60bc942defb7c12204b3d080cfd08b9a7b407dc4d80e80f0fc"
        ),
        "dota_1_ss": _rio_obb_asset(
            "nitid1n",
            "dota_1_ss",
            "b87de5d18f7abfa0ea83aaa27e5f7a4d6545553c6ee559afbff9ac95a7d41006",
        ),
    },
    "nitid1s": {
        "diorr": _rio_obb_asset(
            "nitid1s", "diorr", "b37c73b8030c1ef784e8f35ca9fab3285e3e60ba6c6035057ea38257ffee01f1"
        ),
        "dota_1_ss": _rio_obb_asset(
            "nitid1s",
            "dota_1_ss",
            "ef0c728aaeb4d85950134431617fb576b3ad6a9ac85878088ce97c0075df2026",
        ),
    },
    "nitid1m": {
        "diorr": _rio_obb_asset(
            "nitid1m", "diorr", "2a6384fe713679e4c3269d5176e8549c773fdfa87bb818da782e79b2b372d362"
        ),
        "dota_1_ss": _rio_obb_asset(
            "nitid1m",
            "dota_1_ss",
            "2cbfc60909a7d1209192f35881aed407f8790317fe9ce051cefa3e17ed88e0c6",
        ),
        "dota_1_ms": _rio_obb_asset(
            "nitid1m",
            "dota_1_ms",
            "a01e166b85ad6b7a19390c26c6a1773350ae483c1d7ec7379f9658752b6d5b21",
        ),
    },
    "nitid1l": {
        "diorr": _rio_obb_asset(
            "nitid1l", "diorr", "3a403d589585403091a6b154a32499d7ccf6e3196c7ac8b91e9bf1c182457125"
        ),
        "dota_1_ss": _rio_obb_asset(
            "nitid1l",
            "dota_1_ss",
            "503b790952f1d8b3ad9773b5a61a175246b8ba81aae50dbf02f1fc3c5b792cdd",
        ),
    },
    "nitid1x": {
        "diorr": _rio_obb_asset(
            "nitid1x", "diorr", "67ef7436857489460b6abd9cce984c723dfe4f183f62ad0e43ba49136b2711e8"
        ),
        "dota_1_ss": _rio_obb_asset(
            "nitid1x",
            "dota_1_ss",
            "9121e1da4c79e37fa204d8caaad3107e6edc4b8780d9d936f0747fd35330476d",
        ),
        "dota_1_ms": _rio_obb_asset(
            "nitid1x",
            "dota_1_ms",
            "034c52993da3cfb8d3643a44e69ea56d1907cdd6c8e7907590def041a5c57008",
        ),
    },
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
    "detrpose_n": "detrpose_n",
    "detrpose_s": "detrpose_s",
    "detrpose_m": "detrpose_m",
    "detrpose_l": "detrpose_l",
    "detrpose_x": "detrpose_x",
    "pose_n": "detrpose_n",
    "pose_s": "detrpose_s",
    "pose_m": "detrpose_m",
    "pose_l": "detrpose_l",
    "pose_x": "detrpose_x",
    "rio_n": "nitid1n",
    "rio_s": "nitid1s",
    "rio_m": "nitid1m",
    "rio_l": "nitid1l",
    "rio_x": "nitid1x",
    "rtdetrv2_obb_n": "nitid1n",
    "rtdetrv2_obb_s": "nitid1s",
    "rtdetrv2_obb_m": "nitid1m",
    "rtdetrv2_obb_l": "nitid1l",
    "rtdetrv2_obb_x": "nitid1x",
}
_WEIGHT_ALIASES = {
    "default": DEFAULT_WEIGHTS,
    "objects365_coco": "obj2coco",
    "objects365_to_coco": "obj2coco",
    "coco_only": "coco",
}
_OBB_WEIGHT_ALIASES = {
    "default": "dota_1_ss",
    "dota": "dota_1_ss",
    "dota_ss": "dota_1_ss",
    "dota_1": "dota_1_ss",
    "dota_1_0": "dota_1_ss",
    "dota_ms": "dota_1_ms",
    "dota_1_ms": "dota_1_ms",
    "dior": "diorr",
    "dior_r": "diorr",
    "diorr": "diorr",
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
    if resolved_task == "pose":
        return POSE_MODEL_REGISTRY
    if resolved_task == "semantic":
        return SEMANTIC_MODEL_REGISTRY
    return RIO_OBB_MODEL_REGISTRY


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

    if resolved_task == "obb":
        raw_weights_key = weights.lower().replace("-", "_")
        weights_key = _OBB_WEIGHT_ALIASES.get(raw_weights_key, raw_weights_key)
    else:
        weights_key = _normalize_weights(weights)
    if resolved_task in {"segment", "semantic"} and weights_key == DEFAULT_WEIGHTS:
        weights_key = "coco"
    if resolved_task == "pose" and weights_key == DEFAULT_WEIGHTS:
        weights_key = "default"
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
        elif asset.task == "obb":
            _wrap_rio_obb_checkpoint(raw_path, asset, out_path)
        elif asset.task == "pose":
            convert_checkpoint(
                weights=str(raw_path),
                config=make_pose_config(asset.model),
                names_file=str(names),
                output=str(out_path),
            )
        else:
            convert_checkpoint(
                weights=str(raw_path),
                config=make_model_config(asset.model, task=asset.task),
                names_file=str(names),
                output=str(out_path),
            )

    return out_path


def _wrap_rio_obb_checkpoint(raw_path: Path, asset: ModelAsset, output_path: Path) -> None:
    """Wrap a RiO-DETR RT-DETRv2-OBB checkpoint in nitid's checkpoint format."""
    names_path = output_path.parent / ".dota_obb_names.yaml"
    import yaml

    try:
        with names_path.open("w", encoding="utf-8") as names_file:
            yaml.safe_dump({"names": list(DOTA_OBB_NAMES)}, names_file)
        convert_checkpoint(
            weights=str(raw_path),
            config=make_rio_obb_config(asset.model, num_classes=len(DOTA_OBB_NAMES)),
            names_file=str(names_path),
            output=str(output_path),
        )
    finally:
        names_path.unlink(missing_ok=True)


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
