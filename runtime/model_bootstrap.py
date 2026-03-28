"""Minimal runtime bootstrap helpers for local weight configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in scaffold stage
    yaml = None


class ModelBootstrapError(RuntimeError):
    """Raised when runtime model bootstrap configuration is invalid."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file for the demo runtime."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise ModelBootstrapError(
            f"Configuration file does not exist: {config_file}"
        )

    if yaml is None:
        raise ModelBootstrapError(
            "PyYAML is required to read configuration files. "
            "Install it before running the demo bootstrap."
        )

    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ModelBootstrapError(
            f"Configuration root must be a mapping: {config_file}"
        )
    return data


def resolve_model_weight_path(
    config: Mapping[str, Any], project_root: str | Path | None = None
) -> str:
    """Resolve the MoME model weight path from deploy or research-style config."""
    weight_path = _extract_weight_path(config)
    if not weight_path:
        raise ModelBootstrapError(
            "Missing MoME weight path in configuration. "
            "Expected 'weights.mome_model' or 'paths.weights.mome_model'."
        )

    base_dir = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    resolved_path = (base_dir / weight_path).resolve() if not Path(weight_path).is_absolute() else Path(weight_path)
    return str(resolved_path)


def validate_weight_exists(weight_path: str) -> None:
    """Validate that the local MoME weight file exists for offline runtime."""
    path_obj = Path(weight_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            "MoME weight file not found: "
            f"{path_obj}. Offline demo runtime requires a local checkpoint and "
            "will not download weights automatically."
        )
    if not path_obj.is_file():
        raise FileNotFoundError(
            f"MoME weight path is not a file: {path_obj}"
        )


def bootstrap_model_config(config_path: str | Path) -> str:
    """Load config and return a validated local MoME weight path."""
    config = load_config(config_path)
    weight_path = resolve_model_weight_path(config)
    validate_weight_exists(weight_path)
    # TODO: Audit models/backbones loading flow for any implicit online download path.
    return weight_path


def _extract_weight_path(config: Mapping[str, Any]) -> str:
    """Extract the MoME weight path from supported configuration layouts."""
    weights_config = config.get("weights")
    if isinstance(weights_config, Mapping):
        mome_weight = weights_config.get("mome_model")
        if isinstance(mome_weight, str):
            return mome_weight

    paths_config = config.get("paths")
    if isinstance(paths_config, Mapping):
        nested_weights = paths_config.get("weights")
        if isinstance(nested_weights, Mapping):
            mome_weight = nested_weights.get("mome_model")
            if isinstance(mome_weight, str):
                return mome_weight

    return ""
