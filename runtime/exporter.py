"""Output export utilities for saving single-frame demo inference artifacts."""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.exceptions import RoadMomeDemoError
from runtime.logger import get_app_logger, log_export_failure, log_export_success

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional at import time
    Image = None


class ExportError(RoadMomeDemoError):
    """Raised when exporter inputs are invalid or output files cannot be written."""


class ResultExporter:
    """Persist visualization images and structured logs for one demo frame."""

    DEFAULT_IMAGE_SUFFIX = ".jpg"
    DEFAULT_JSON_PREFIX = "result_"
    DEFAULT_VIS_PREFIX = "vis_"
    DEFAULT_CSV_NAME = "result_summary.csv"

    def __init__(
        self,
        export_root: str | Path | None = None,
        *,
        visualization_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        image_suffix: str = DEFAULT_IMAGE_SUFFIX,
        enable_csv_summary: bool = True,
        csv_filename: str = DEFAULT_CSV_NAME,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_output_root = project_root / "outputs"
        base_root = Path(export_root) if export_root is not None else default_output_root

        self.visualization_dir = (
            Path(visualization_dir)
            if visualization_dir is not None
            else base_root / "visualizations"
        )
        self.log_dir = Path(log_dir) if log_dir is not None else base_root / "logs"
        self.image_suffix = _normalize_image_suffix(image_suffix)
        self.enable_csv_summary = bool(enable_csv_summary)
        self.csv_log_path = self.log_dir / csv_filename

        _ensure_directory(self.visualization_dir)
        _ensure_directory(self.log_dir)

    def save_visualization(self, frame_id: str, image_rgb: np.ndarray) -> Path:
        """Save one RGB visualization image to the configured visualization directory."""
        if Image is None:
            raise ExportError(
                "Pillow is required for exporting visualization images but is not installed."
            )

        safe_frame_id = _normalize_frame_id(frame_id)
        image_array = _normalize_image_rgb(image_rgb)
        output_path = self.visualization_dir / (
            f"{self.DEFAULT_VIS_PREFIX}{safe_frame_id}{self.image_suffix}"
        )

        try:
            image = Image.fromarray(image_array, mode="RGB")
            save_kwargs: dict[str, Any] = {}
            if self.image_suffix in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
            image.save(output_path, **save_kwargs)
        except Exception as exc:
            message = f"Failed to save visualization image to {output_path}: {exc}"
            _log_error("结果图保存失败 | frame_id=%s | error=%s", frame_id, message)
            raise ExportError(message) from exc

        _log_info("结果图保存成功 | frame_id=%s | path=%s", frame_id, output_path)
        return output_path

    def save_result_json(
        self,
        sample: Mapping[str, Any] | Any,
        result: Mapping[str, Any] | Any,
        result_image_path: str | Path | None = None,
    ) -> Path:
        """Write a structured per-frame JSON log and return its path."""
        sample_context = _extract_sample_context(sample, require_paths=True)
        result_payload = _extract_result_payload(
            result=result,
            valid_mask=sample_context["valid_mask"],
        )
        timestamp = _build_timestamp()

        json_payload: dict[str, Any] = {
            "frame_id": sample_context["frame_id"],
            "image_path": str(sample_context["image_path"]),
            "npz_path": str(sample_context["npz_path"]),
            "valid_patch_count": result_payload["valid_patch_count"],
            "abnormal_patch_count": result_payload["abnormal_patch_count"],
            "avg_pred_prob": result_payload["avg_pred_prob"],
            "mean_phys_weight": result_payload["mean_phys_weight"],
            "mean_geom_weight": result_payload["mean_geom_weight"],
            "mean_tex_weight": result_payload["mean_tex_weight"],
            "threshold": result_payload["threshold"],
            "result_image_path": _stringify_optional_path(result_image_path),
            "timestamp": timestamp,
            "patch_probabilities": result_payload["patch_probs"].tolist(),
            "phys_weights": result_payload["phys_weights"].tolist(),
            "geom_weights": result_payload["geom_weights"].tolist(),
            "tex_weights": result_payload["tex_weights"].tolist(),
        }

        json_path = self.log_dir / (
            f"{self.DEFAULT_JSON_PREFIX}{_normalize_frame_id(sample_context['frame_id'])}.json"
        )

        try:
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(json_payload, handle, ensure_ascii=False, indent=2)
        except TypeError as exc:
            message = (
                f"Failed to serialize JSON export payload for frame {sample_context['frame_id']}: {exc}"
            )
            _log_error("JSON 保存失败 | frame_id=%s | error=%s", sample_context["frame_id"], message)
            raise ExportError(message) from exc
        except OSError as exc:
            message = f"Failed to write JSON export file {json_path}: {exc}"
            _log_error("JSON 保存失败 | frame_id=%s | error=%s", sample_context["frame_id"], message)
            raise ExportError(message) from exc

        _log_info("JSON 保存成功 | frame_id=%s | path=%s", sample_context["frame_id"], json_path)
        return json_path

    def append_result_csv(
        self,
        sample: Mapping[str, Any] | Any,
        result: Mapping[str, Any] | Any,
        result_image_path: str | Path | None = None,
    ) -> Path:
        """Append one result row to the configured CSV summary file."""
        sample_context = _extract_sample_context(sample, require_paths=True)
        result_payload = _extract_result_payload(
            result=result,
            valid_mask=sample_context["valid_mask"],
        )
        timestamp = _build_timestamp()

        header = [
            "timestamp",
            "frame_id",
            "image_path",
            "npz_path",
            "result_image_path",
            "threshold",
            "valid_patch_count",
            "abnormal_patch_count",
            "avg_pred_prob",
            "mean_phys_weight",
            "mean_geom_weight",
            "mean_tex_weight",
        ]
        row = [
            timestamp,
            sample_context["frame_id"],
            str(sample_context["image_path"]),
            str(sample_context["npz_path"]),
            _stringify_optional_path(result_image_path) or "",
            result_payload["threshold"],
            result_payload["valid_patch_count"],
            result_payload["abnormal_patch_count"],
            result_payload["avg_pred_prob"],
            result_payload["mean_phys_weight"],
            result_payload["mean_geom_weight"],
            result_payload["mean_tex_weight"],
        ]

        csv_path = self.csv_log_path
        try:
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
        except OSError as exc:
            message = f"Failed to append CSV export file {csv_path}: {exc}"
            _log_error("CSV 追加失败 | frame_id=%s | error=%s", sample_context["frame_id"], message)
            raise ExportError(message) from exc
        except csv.Error as exc:
            message = f"Failed to write CSV export row for frame {sample_context['frame_id']}: {exc}"
            _log_error("CSV 追加失败 | frame_id=%s | error=%s", sample_context["frame_id"], message)
            raise ExportError(message) from exc

        _log_info("CSV 追加成功 | frame_id=%s | path=%s", sample_context["frame_id"], csv_path)
        return csv_path

    def export_all(
        self,
        sample: Mapping[str, Any] | Any,
        result: Mapping[str, Any] | Any,
        image_rgb: np.ndarray,
    ) -> dict[str, Path | None]:
        """Save visualization image, JSON log, and optional CSV summary in one call."""
        sample_context = _extract_sample_context(sample, require_paths=True)

        try:
            visualization_path = self.save_visualization(
                frame_id=sample_context["frame_id"],
                image_rgb=image_rgb,
            )
            json_log_path = self.save_result_json(
                sample=sample,
                result=result,
                result_image_path=visualization_path,
            )
            csv_log_path: Path | None = None
            if self.enable_csv_summary:
                csv_log_path = self.append_result_csv(
                    sample=sample,
                    result=result,
                    result_image_path=visualization_path,
                )
        except ExportError as exc:
            logger = _get_logger()
            if logger is not None:
                log_export_failure(logger, sample_context["frame_id"], exc)
            raise

        export_payload = {
            "visualization_path": visualization_path,
            "json_log_path": json_log_path,
            "csv_log_path": csv_log_path,
        }
        logger = _get_logger()
        if logger is not None:
            log_export_success(logger, sample_context["frame_id"], export_payload)
        return export_payload


def _extract_sample_context(
    sample: Mapping[str, Any] | Any,
    *,
    require_paths: bool,
) -> dict[str, Any]:
    """Read and validate frame-level metadata from a sample object."""
    frame_id = _require_frame_id(_get_field(sample, "frame_id", default=None))
    meta = _get_field(sample, "meta", default=None)
    valid_mask = _extract_valid_mask(meta)

    image_path = _get_field(sample, "image_path", default=None)
    npz_path = _get_field(sample, "npz_path", default=None)
    if require_paths:
        if image_path is None:
            raise ExportError("Missing required sample.image_path for export logging.")
        if npz_path is None:
            raise ExportError("Missing required sample.npz_path for export logging.")

    return {
        "frame_id": frame_id,
        "image_path": Path(image_path) if image_path is not None else None,
        "npz_path": Path(npz_path) if npz_path is not None else None,
        "valid_mask": valid_mask,
    }


def _extract_result_payload(
    *,
    result: Mapping[str, Any] | Any,
    valid_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Normalize exporter-facing result values with compatibility for field aliases."""
    patch_probs = _extract_vector(
        result=result,
        field_names=("patch_probs", "final_probabilities", "probabilities", "probs"),
        target_name="result probabilities",
    )
    phys_weights, geom_weights, tex_weights = _extract_expert_weight_vectors(
        result=result,
        patch_count=patch_probs.size,
    )

    if valid_mask is not None and valid_mask.size != patch_probs.size:
        raise ExportError(
            "sample.meta-derived valid_mask does not match the number of result patches."
        )

    usable_mask = (
        valid_mask.astype(bool, copy=False)
        if valid_mask is not None
        else np.ones(patch_probs.shape[0], dtype=bool)
    )

    threshold = _extract_numeric_field(
        result=result,
        primary_name="threshold",
        fallback=float(0.5),
    )
    valid_patch_count = _extract_numeric_field(
        result=result,
        primary_name="valid_patch_count",
        fallback=int(np.count_nonzero(usable_mask)),
        cast=int,
    )
    abnormal_patch_count = _extract_numeric_field(
        result=result,
        primary_name="abnormal_patch_count",
        fallback=int(np.count_nonzero(patch_probs[usable_mask] >= threshold)),
        cast=int,
    )
    avg_pred_prob = _extract_numeric_field(
        result=result,
        primary_name="avg_pred_prob",
        fallback=float(np.mean(patch_probs[usable_mask])) if np.any(usable_mask) else 0.0,
    )
    mean_phys_weight = _extract_numeric_field(
        result=result,
        primary_name="mean_phys_weight",
        fallback=float(np.mean(phys_weights[usable_mask])) if np.any(usable_mask) else 0.0,
    )
    mean_geom_weight = _extract_numeric_field(
        result=result,
        primary_name="mean_geom_weight",
        fallback=float(np.mean(geom_weights[usable_mask])) if np.any(usable_mask) else 0.0,
    )
    mean_tex_weight = _extract_numeric_field(
        result=result,
        primary_name="mean_tex_weight",
        fallback=float(np.mean(tex_weights[usable_mask])) if np.any(usable_mask) else 0.0,
    )

    return {
        "threshold": float(threshold),
        "patch_probs": patch_probs,
        "phys_weights": phys_weights,
        "geom_weights": geom_weights,
        "tex_weights": tex_weights,
        "valid_patch_count": int(valid_patch_count),
        "abnormal_patch_count": int(abnormal_patch_count),
        "avg_pred_prob": float(avg_pred_prob),
        "mean_phys_weight": float(mean_phys_weight),
        "mean_geom_weight": float(mean_geom_weight),
        "mean_tex_weight": float(mean_tex_weight),
    }


def _extract_expert_weight_vectors(
    *,
    result: Mapping[str, Any] | Any,
    patch_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Phys/Geom/Tex vectors from combined or split result fields."""
    expert_weights = _get_field(result, "expert_weights", default=None)
    if expert_weights is not None:
        weight_matrix = np.asarray(expert_weights, dtype=np.float32)
        if weight_matrix.ndim == 3 and weight_matrix.shape[0] == 1:
            weight_matrix = weight_matrix[0]
        if weight_matrix.shape != (patch_count, 3):
            raise ExportError(
                f"result.expert_weights must have shape ({patch_count}, 3), got {weight_matrix.shape}."
            )
        return weight_matrix[:, 0], weight_matrix[:, 1], weight_matrix[:, 2]

    phys = _extract_vector(
        result=result,
        field_names=("phys_weights",),
        target_name="result.phys_weights",
    )
    geom = _extract_vector(
        result=result,
        field_names=("geom_weights",),
        target_name="result.geom_weights",
    )
    tex = _extract_vector(
        result=result,
        field_names=("tex_weights",),
        target_name="result.tex_weights",
    )

    if phys.size != patch_count or geom.size != patch_count or tex.size != patch_count:
        raise ExportError(
            "Split expert weight vectors must match the number of patch probabilities."
        )
    return phys, geom, tex


def _extract_vector(
    *,
    result: Mapping[str, Any] | Any,
    field_names: tuple[str, ...],
    target_name: str,
) -> np.ndarray:
    """Read one vector field from a mapping-like or attribute-based result object."""
    for field_name in field_names:
        value = _get_field(result, field_name, default=None)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ExportError(f"{target_name} is empty.")
        return array

    raise ExportError(
        f"Missing {target_name}. Checked fields: {', '.join(field_names)}."
    )


def _extract_numeric_field(
    *,
    result: Mapping[str, Any] | Any,
    primary_name: str,
    fallback: int | float,
    cast: type[int] | type[float] = float,
) -> int | float:
    """Read one numeric scalar from a result object with an explicit fallback."""
    value = _get_field(result, primary_name, default=None)
    if value is None:
        return cast(fallback)
    if not isinstance(value, (int, float)):
        raise ExportError(
            f"result.{primary_name} must be numeric, got {type(value).__name__}."
        )
    return cast(value)


def _extract_valid_mask(meta: Any) -> np.ndarray | None:
    """Extract a boolean valid-mask from sample.meta when available."""
    if meta is None:
        return None

    meta_array = np.asarray(meta)
    if meta_array.ndim != 2 or meta_array.shape[1] < 3:
        raise ExportError(
            "sample.meta must have shape [N, 3] or wider so valid_mask can be read from meta[:, 2]."
        )
    return np.asarray(meta_array[:, 2], dtype=np.float32) > 0.5


def _normalize_image_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Normalize exported visualization image data to RGB uint8 HWC format."""
    image_array = np.asarray(image_rgb)
    if image_array.size == 0:
        raise ExportError("image_rgb is empty and cannot be exported.")
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ExportError(
            f"image_rgb must have shape [H, W, 3], got {image_array.shape}."
        )

    if np.issubdtype(image_array.dtype, np.floating):
        max_value = float(np.nanmax(image_array))
        if max_value <= 1.0:
            image_array = image_array * 255.0

    image_array = np.clip(image_array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(image_array)


def _ensure_directory(path: Path) -> None:
    """Create one directory when needed and fail with a clear export error."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"Failed to create export directory {path}: {exc}") from exc


def _normalize_frame_id(frame_id: str) -> str:
    """Make frame_id safe for filenames without changing the logical frame id elsewhere."""
    cleaned = re.sub(r"[<>:\"/\\\\|?*]+", "_", frame_id.strip())
    cleaned = cleaned.strip(". ")
    if not cleaned:
        raise ExportError("frame_id is empty or invalid for export.")
    return cleaned


def _require_frame_id(value: Any) -> str:
    """Validate that frame_id exists and is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ExportError("Missing required sample.frame_id for export.")
    return value.strip()


def _normalize_image_suffix(value: str) -> str:
    """Validate the configured visualization image suffix."""
    suffix = value.lower().strip()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ExportError(
            f"Unsupported exporter image suffix '{value}'. Use .jpg, .jpeg, or .png."
        )
    return suffix


def _build_timestamp() -> str:
    """Return an ISO 8601 timestamp suitable for JSON and CSV logging."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stringify_optional_path(path_value: str | Path | None) -> str | None:
    """Convert an optional path-like value to string for structured logs."""
    if path_value is None:
        return None
    return str(Path(path_value))


def _get_field(
    obj: Mapping[str, Any] | Any,
    field_name: str,
    *,
    default: Any,
) -> Any:
    """Read a field from a mapping or attribute-based object."""
    if isinstance(obj, Mapping):
        return obj.get(field_name, default)
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    return default


def _get_logger() -> logging.Logger | None:
    """Best-effort logger acquisition that does not affect export behavior."""

    try:
        return get_app_logger()
    except Exception:
        return None


def _log_info(message: str, *args: object) -> None:
    logger = _get_logger()
    if logger is not None:
        logger.info(message, *args)


def _log_error(message: str, *args: object) -> None:
    logger = _get_logger()
    if logger is not None:
        logger.error(message, *args)
