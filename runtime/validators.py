"""Runtime validation helpers for offline demo startup and pre-run checks.

This module only validates local inputs and filesystem state. It does not load
models, run inference, perform visualization, or export outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from core.exceptions import RoadMomeDemoError
from data_access.index_loader import IndexLoadError, load_index
from runtime.logger import (
    get_app_logger,
    log_sample_validation_failure,
    log_startup_failure,
    log_startup_success,
)
from runtime.model_bootstrap import ModelBootstrapError, resolve_model_weight_path

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in scaffold stage
    yaml = None

_REQUIRED_CONFIG_FILES = ("config.yaml", "deploy.yaml", "ui.yaml")
_REQUIRED_INDEX_FIELDS = ("jpg", "npz")
_REQUIRED_SAMPLE_FIELDS = (
    "image",
    "phys_8d",
    "deep_512d",
    "patch_corners_uv",
    "meta",
)


class ValidationError(RoadMomeDemoError):
    """Base exception for demo runtime validation failures."""


class ConfigValidationError(ValidationError):
    """Raised when local configuration files are missing or malformed."""


class WeightValidationError(ValidationError):
    """Raised when the configured offline model weight is invalid."""


class DemoDataValidationError(ValidationError):
    """Raised when demo_data layout or sample assets are invalid."""


class IndexValidationError(DemoDataValidationError):
    """Raised when demo_data/index.json is missing, empty, or inconsistent."""


class OutputValidationError(ValidationError):
    """Raised when output directories cannot be created or written."""


class SampleInputValidationError(ValidationError):
    """Raised when a sample payload does not satisfy runtime input rules."""


def validate_config_files(config_dir: str | Path) -> None:
    """Validate that required YAML config files exist and can be parsed."""

    config_root = Path(config_dir)
    if not config_root.exists():
        raise ConfigValidationError(f"Configuration directory does not exist: {config_root}")
    if not config_root.is_dir():
        raise ConfigValidationError(f"Configuration path is not a directory: {config_root}")

    for filename in _REQUIRED_CONFIG_FILES:
        config_path = config_root / filename
        _load_yaml_mapping(config_path)


def validate_weight_file(weight_path: str | Path) -> None:
    """Validate that the offline MoME weight path exists and is a file."""

    weight_file = Path(weight_path)
    if not weight_file.exists():
        message = f"Weight file does not exist: {weight_file}"
        _log_error("权重校验失败 | path=%s | error=%s", weight_file, message)
        raise WeightValidationError(message)
    if not weight_file.is_file():
        message = f"Weight path is not a file: {weight_file}"
        _log_error("权重校验失败 | path=%s | error=%s", weight_file, message)
        raise WeightValidationError(message)


def validate_demo_data_layout(demo_root: str | Path) -> None:
    """Validate the expected demo_data directory layout."""

    demo_dir = Path(demo_root)
    if not demo_dir.exists():
        raise DemoDataValidationError(f"Demo data root does not exist: {demo_dir}")
    if not demo_dir.is_dir():
        raise DemoDataValidationError(f"Demo data root is not a directory: {demo_dir}")

    jpg_dir = demo_dir / "jpg"
    npz_dir = demo_dir / "npz"
    index_path = demo_dir / "index.json"

    if not jpg_dir.exists():
        raise DemoDataValidationError(f"Missing required demo image directory: {jpg_dir}")
    if not jpg_dir.is_dir():
        raise DemoDataValidationError(f"Demo image path is not a directory: {jpg_dir}")

    if not npz_dir.exists():
        raise DemoDataValidationError(f"Missing required demo NPZ directory: {npz_dir}")
    if not npz_dir.is_dir():
        raise DemoDataValidationError(f"Demo NPZ path is not a directory: {npz_dir}")

    if not index_path.exists():
        raise DemoDataValidationError(f"Missing required demo index file: {index_path}")
    if not index_path.is_file():
        raise DemoDataValidationError(f"Demo index path is not a file: {index_path}")


def validate_index_file(
    index_path: str | Path,
    demo_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Validate index.json structure and referenced local asset files."""

    index_file = Path(index_path)
    demo_dir = Path(demo_root)

    try:
        normalized_index = load_index(index_file)
    except IndexLoadError as exc:
        _log_error("索引校验失败 | index_path=%s | error=%s", index_file, exc)
        raise IndexValidationError(str(exc)) from exc

    if not normalized_index:
        message = f"Index file is empty: {index_file}. The demo requires at least one sample."
        _log_error("索引校验失败 | index_path=%s | error=%s", index_file, message)
        raise IndexValidationError(message)

    resolved_demo_dir = demo_dir.resolve()
    total_samples = len(normalized_index)
    missing_jpg_count = 0
    missing_npz_count = 0
    problems: list[str] = []

    for frame_id, entry in normalized_index.items():
        missing_fields = [field for field in _REQUIRED_INDEX_FIELDS if field not in entry]
        if missing_fields:
            message = (
                f"Index entry for frame '{frame_id}' is missing required fields: {missing_fields}"
            )
            _log_error("索引校验失败 | frame_id=%s | error=%s", frame_id, message)
            raise IndexValidationError(message)

        try:
            jpg_path = _resolve_demo_relative_path(
                demo_root=resolved_demo_dir,
                frame_id=frame_id,
                field_name="jpg",
                relative_path=entry["jpg"],
            )
            npz_path = _resolve_demo_relative_path(
                demo_root=resolved_demo_dir,
                frame_id=frame_id,
                field_name="npz",
                relative_path=entry["npz"],
            )
        except IndexValidationError as exc:
            _log_error("索引校验失败 | frame_id=%s | error=%s", frame_id, exc)
            raise

        if not jpg_path.exists() or not jpg_path.is_file():
            missing_jpg_count += 1
            if len(problems) < 8:
                problems.append(f"frame '{frame_id}' jpg missing: {jpg_path}")

        if not npz_path.exists() or not npz_path.is_file():
            missing_npz_count += 1
            if len(problems) < 8:
                problems.append(f"frame '{frame_id}' npz missing: {npz_path}")

    if missing_jpg_count or missing_npz_count:
        summary = (
            f"Index asset validation failed for {index_file}. "
            f"total_samples={total_samples}, missing_jpg={missing_jpg_count}, "
            f"missing_npz={missing_npz_count}."
        )
        if problems:
            summary = f"{summary} Examples: {'; '.join(problems)}"
        _log_error(
            "索引校验失败 | index_path=%s | missing_jpg=%s | missing_npz=%s | error=%s",
            index_file,
            missing_jpg_count,
            missing_npz_count,
            summary,
        )
        raise IndexValidationError(summary)

    return normalized_index


def validate_output_dirs(
    visualization_dir: str | Path,
    log_dir: str | Path,
) -> None:
    """Ensure output directories exist and are writable."""

    for label, raw_path in (
        ("visualization output directory", visualization_dir),
        ("log output directory", log_dir),
    ):
        directory = Path(raw_path)
        if directory.exists() and not directory.is_dir():
            message = f"{label.capitalize()} is not a directory: {directory}"
            _log_error("输出目录校验失败 | path=%s | error=%s", directory, message)
            raise OutputValidationError(message)

        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = f"Failed to create {label}: {directory}: {exc}"
            _log_error("输出目录校验失败 | path=%s | error=%s", directory, message)
            raise OutputValidationError(message) from exc

        _ensure_directory_writable(directory, label=label)


def validate_runtime_sample(
    sample: Mapping[str, Any] | Any,
    *,
    expected_patch_count: int = 63,
    expected_phys_dim: int = 8,
    expected_deep_dim: int = 384,
) -> None:
    """Validate the critical runtime sample fields before inference."""

    try:
        for field_name in _REQUIRED_SAMPLE_FIELDS:
            value = _get_field(sample, field_name, default=None)
            if value is None:
                raise SampleInputValidationError(
                    f"Sample is missing required field '{field_name}'."
                )

        image_array = np.asarray(_get_field(sample, "image", default=None))
        if image_array.size == 0:
            raise SampleInputValidationError("Sample image is empty.")
        if image_array.ndim == 2:
            pass
        elif image_array.ndim == 3 and image_array.shape[2] in {1, 3, 4}:
            pass
        else:
            raise SampleInputValidationError(
                f"Sample image must have shape [H, W], [H, W, 1], [H, W, 3], or [H, W, 4], got {image_array.shape}."
            )

        phys_array = _as_numpy_array(
            _get_field(sample, "phys_8d", default=None),
            field_name="phys_8d",
            expected_ndim=2,
            dtype=np.float32,
        )
        deep_array = _as_numpy_array(
            _get_field(sample, "deep_512d", default=None),
            field_name="deep_512d",
            expected_ndim=2,
            dtype=np.float32,
        )
        corners_array = _as_numpy_array(
            _get_field(sample, "patch_corners_uv", default=None),
            field_name="patch_corners_uv",
            expected_ndim=3,
            dtype=np.float32,
        )
        meta_array = _as_numpy_array(
            _get_field(sample, "meta", default=None),
            field_name="meta",
            expected_ndim=2,
        )

        if phys_array.shape != (expected_patch_count, expected_phys_dim):
            raise SampleInputValidationError(
                f"phys_8d must have shape [{expected_patch_count}, {expected_phys_dim}], got {phys_array.shape}."
            )
        if deep_array.shape != (expected_patch_count, expected_deep_dim):
            raise SampleInputValidationError(
                f"deep_512d must have shape [{expected_patch_count}, {expected_deep_dim}], got {deep_array.shape}."
            )
        if corners_array.shape != (expected_patch_count, 4, 2):
            raise SampleInputValidationError(
                f"patch_corners_uv must have shape [{expected_patch_count}, 4, 2], got {corners_array.shape}."
            )
        if meta_array.shape != (expected_patch_count, 3):
            raise SampleInputValidationError(
                f"meta must have shape [{expected_patch_count}, 3], got {meta_array.shape}."
            )

        valid_mask = np.asarray(meta_array[:, 2], dtype=np.float32)
        if not np.any(valid_mask > 0.5):
            raise SampleInputValidationError(
                "meta[:, 2] contains no active patches, so runtime inference cannot proceed."
            )

        quality_2d = _get_field(sample, "quality_2d", default=None)
        if quality_2d is not None:
            q2_array = np.asarray(quality_2d, dtype=np.float32)
            if q2_array.ndim == 1:
                if q2_array.shape[0] != expected_patch_count:
                    raise SampleInputValidationError(
                        f"quality_2d must have {expected_patch_count} entries when provided as a 1D array."
                    )
            elif q2_array.shape != (expected_patch_count, 1):
                raise SampleInputValidationError(
                    f"quality_2d must have shape [{expected_patch_count}, 1], got {q2_array.shape}."
                )
    except SampleInputValidationError as exc:
        logger = _get_logger()
        if logger is not None:
            log_sample_validation_failure(logger, _extract_frame_id(sample), exc)
        raise


def run_startup_validation(
    *,
    project_root: str | Path | None = None,
    config_dir: str | Path | None = None,
    demo_root: str | Path | None = None,
    weight_path: str | Path | None = None,
    visualization_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run end-to-end startup validation and return a compact status summary."""

    logger = _get_logger()

    try:
        root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        resolved_config_dir = Path(config_dir) if config_dir is not None else root / "config"

        validate_config_files(resolved_config_dir)
        deploy_config = _load_yaml_mapping(resolved_config_dir / "deploy.yaml")

        resolved_weight_path = (
            Path(weight_path)
            if weight_path is not None
            else _resolve_weight_path_from_config(deploy_config, project_root=root)
        )
        resolved_demo_root = (
            Path(demo_root)
            if demo_root is not None
            else _resolve_demo_root_from_config(deploy_config, project_root=root)
        )
        resolved_index_path = _resolve_index_path_from_config(
            deploy_config,
            project_root=root,
            fallback_demo_root=resolved_demo_root,
        )
        resolved_visualization_dir = (
            Path(visualization_dir)
            if visualization_dir is not None
            else _resolve_output_dir_from_config(
                deploy_config,
                key="visualization_dir",
                project_root=root,
                default_path=root / "outputs" / "visualizations",
            )
        )
        resolved_log_dir = (
            Path(log_dir)
            if log_dir is not None
            else _resolve_output_dir_from_config(
                deploy_config,
                key="log_dir",
                project_root=root,
                default_path=root / "outputs" / "logs",
            )
        )

        validate_weight_file(resolved_weight_path)
        validate_demo_data_layout(resolved_demo_root)
        index_data = validate_index_file(
            index_path=resolved_index_path,
            demo_root=resolved_demo_root,
        )
        validate_output_dirs(
            visualization_dir=resolved_visualization_dir,
            log_dir=resolved_log_dir,
        )

        summary = {
            "config_ok": True,
            "weight_ok": True,
            "demo_data_ok": True,
            "output_dirs_ok": True,
            "sample_count": len(index_data),
            "weight_path": str(resolved_weight_path.resolve()),
            "demo_root": str(resolved_demo_root.resolve()),
            "index_path": str(resolved_index_path.resolve()),
            "visualization_dir": str(resolved_visualization_dir.resolve()),
            "log_dir": str(resolved_log_dir.resolve()),
            "index_data": index_data,
        }
    except ValidationError as exc:
        if logger is not None:
            log_startup_failure(logger, exc)
        raise

    if logger is not None:
        log_startup_success(logger, summary)
    return summary


def _load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    """Load one YAML file and require a mapping root."""

    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise ConfigValidationError(f"Configuration path is not a file: {path}")
    if yaml is None:
        raise ConfigValidationError(
            f"PyYAML is required to read configuration file: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Failed to parse YAML file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigValidationError(f"Failed to read configuration file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigValidationError(
            f"Configuration root must be a mapping in file: {path}"
        )
    return payload


def _resolve_weight_path_from_config(
    deploy_config: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    try:
        return Path(resolve_model_weight_path(deploy_config, project_root=project_root))
    except (ModelBootstrapError, RuntimeError, ValueError) as exc:
        raise WeightValidationError(
            f"Failed to resolve weight path from deploy config: {exc}"
        ) from exc


def _resolve_demo_root_from_config(
    deploy_config: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    data_config = deploy_config.get("data")
    if not isinstance(data_config, Mapping):
        return project_root / "demo_data"

    index_file = data_config.get("index_file")
    if isinstance(index_file, str) and index_file.strip():
        return _resolve_project_path(project_root, index_file).parent

    parent_dirs: list[Path] = []
    for key in ("demo_jpg_dir", "demo_npz_dir"):
        value = data_config.get(key)
        if isinstance(value, str) and value.strip():
            parent_dirs.append(_resolve_project_path(project_root, value).parent)

    if parent_dirs and any(path != parent_dirs[0] for path in parent_dirs[1:]):
        raise DemoDataValidationError(
            "Configured demo_jpg_dir and demo_npz_dir do not share the same demo root."
        )
    if parent_dirs:
        return parent_dirs[0]
    return project_root / "demo_data"


def _resolve_index_path_from_config(
    deploy_config: Mapping[str, Any],
    *,
    project_root: Path,
    fallback_demo_root: Path,
) -> Path:
    data_config = deploy_config.get("data")
    if isinstance(data_config, Mapping):
        value = data_config.get("index_file")
        if isinstance(value, str) and value.strip():
            return _resolve_project_path(project_root, value)
    return fallback_demo_root / "index.json"


def _resolve_output_dir_from_config(
    deploy_config: Mapping[str, Any],
    *,
    key: str,
    project_root: Path,
    default_path: Path,
) -> Path:
    outputs_config = deploy_config.get("outputs")
    if isinstance(outputs_config, Mapping):
        value = outputs_config.get(key)
        if isinstance(value, str) and value.strip():
            return _resolve_project_path(project_root, value)
    return default_path


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def _resolve_demo_relative_path(
    *,
    demo_root: Path,
    frame_id: str,
    field_name: str,
    relative_path: Any,
) -> Path:
    if not isinstance(relative_path, Path):
        raise IndexValidationError(
            f"Index entry for frame '{frame_id}' field '{field_name}' was not normalized to Path."
        )

    resolved_path = (demo_root / relative_path).resolve()
    try:
        resolved_path.relative_to(demo_root)
    except ValueError as exc:
        raise IndexValidationError(
            f"Index entry for frame '{frame_id}' field '{field_name}' escapes demo_root: {relative_path}"
        ) from exc
    return resolved_path


def _ensure_directory_writable(directory: Path, *, label: str) -> None:
    probe_path = directory / f".write_test_{uuid4().hex}.tmp"
    try:
        with probe_path.open("wb") as handle:
            handle.write(b"ok")
    except OSError as exc:
        message = f"{label.capitalize()} is not writable: {directory}: {exc}"
        _log_error("输出目录不可写 | path=%s | error=%s", directory, message)
        raise OutputValidationError(message) from exc
    finally:
        try:
            if probe_path.exists():
                probe_path.unlink()
        except OSError:
            pass


def _get_field(source: Mapping[str, Any] | Any, field_name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _as_numpy_array(
    value: Any,
    *,
    field_name: str,
    expected_ndim: int,
    dtype: Any | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except Exception as exc:  # noqa: BLE001 - normalization should keep original cause
        raise SampleInputValidationError(
            f"Field '{field_name}' could not be converted to a NumPy array: {exc}"
        ) from exc

    if array.ndim != expected_ndim:
        raise SampleInputValidationError(
            f"Field '{field_name}' must have ndim={expected_ndim}, got shape {array.shape}."
        )
    return array


def _extract_frame_id(sample: Mapping[str, Any] | Any) -> str | None:
    value = _get_field(sample, "frame_id", default=None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _get_logger() -> logging.Logger | None:
    """Best-effort logger acquisition that does not change validation behavior."""

    try:
        return get_app_logger()
    except Exception:
        return None


def _log_error(message: str, *args: object) -> None:
    logger = _get_logger()
    if logger is not None:
        logger.error(message, *args)
