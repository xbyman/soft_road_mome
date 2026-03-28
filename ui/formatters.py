"""Lightweight text and value formatting helpers for the Dash UI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def read_field(source: Mapping[str, Any] | Any, field_name: str) -> Any:
    """Read a field from either a mapping or an object."""

    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def extract_stats(result: Mapping[str, Any] | Any, threshold_value: float) -> dict[str, Any]:
    """Extract the statistics currently displayed in the result panel."""

    return {
        "valid_patch_count": read_field(result, "valid_patch_count"),
        "abnormal_patch_count": read_field(result, "abnormal_patch_count"),
        "avg_pred_prob": read_field(result, "avg_pred_prob"),
        "mean_phys_weight": read_field(result, "mean_phys_weight"),
        "mean_geom_weight": read_field(result, "mean_geom_weight"),
        "mean_tex_weight": read_field(result, "mean_tex_weight"),
        "threshold": read_field(result, "threshold") or threshold_value,
    }


def format_float(value: Any) -> str:
    """Format a numeric value for display or fall back to N/A."""

    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(numeric_value):
        return "N/A"
    return f"{numeric_value:.4f}"


def format_int(value: Any) -> str:
    """Format an integer value for display or fall back to N/A."""

    if value is None:
        return "N/A"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def format_threshold_text(threshold_value: float) -> str:
    """Format the threshold slider label."""

    return f"当前阈值：{threshold_value:.2f}"


def build_export_path_text(export_result: Mapping[str, Any] | None) -> str | None:
    """Combine export paths into one compact display string."""

    if not export_result:
        return None

    parts: list[str] = []
    visualization_path = export_result.get("visualization_path")
    json_log_path = export_result.get("json_log_path")
    csv_log_path = export_result.get("csv_log_path")

    if visualization_path is not None:
        parts.append(f"vis: {visualization_path}")
    if json_log_path is not None:
        parts.append(f"json: {json_log_path}")
    if csv_log_path is not None:
        parts.append(f"csv: {csv_log_path}")

    return " | ".join(parts) if parts else None
