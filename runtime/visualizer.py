"""Single-frame 4-panel visualization utilities for the offline Road-MoME demo."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from core.exceptions import RoadMomeDemoError

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None

_INFERNO_STOPS_RGB = np.array(
    [
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [252, 193, 33],
        [252, 255, 164],
    ],
    dtype=np.float32,
)


class VisualizationError(RoadMomeDemoError):
    """Raised when a sample or inference result cannot be visualized."""


def draw_4panel_result(
    sample: Mapping[str, Any] | Any,
    result: Mapping[str, Any] | Any,
    save_path: str | Path | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Render a 4-panel visualization from ``sample + result`` and return an RGB image.

    Assumptions:
    - ``sample`` carries the original image, frame_id, patch geometry, and meta.
    - ``result`` carries only inference outputs and summary values.
    - Current ``InferenceResult`` stores ``patch_probs`` and ``expert_weights``.
      This module also accepts a legacy or future shape using
      ``final_probabilities`` / ``phys_weights`` / ``geom_weights`` / ``tex_weights``.
    """
    _require_cv2()
    alpha = _validate_alpha(alpha)

    sample_payload = _extract_sample_payload(sample)
    result_payload = _extract_result_payload(
        result=result,
        patch_count=sample_payload["patch_count"],
        valid_mask=sample_payload["valid_mask"],
    )

    image_bgr = _rgb_to_bgr(sample_payload["image"])
    patch_corners_uv = sample_payload["patch_corners_uv"]
    valid_mask = sample_payload["valid_mask"]

    pred_title = (
        "Prediction  "
        f"avg={result_payload['avg_pred_prob']:.3f}  "
        f"abn={result_payload['abnormal_patch_count']}"
    )
    phys_title = f"Phys Weight  mean={result_payload['mean_phys_weight']:.3f}"
    geom_title = f"Geom Weight  mean={result_payload['mean_geom_weight']:.3f}"
    tex_title = f"Tex Weight   mean={result_payload['mean_tex_weight']:.3f}"

    panel_pred = _render_panel(
        image_bgr=image_bgr,
        patch_corners_uv=patch_corners_uv,
        valid_mask=valid_mask,
        values=result_payload["patch_probs"],
        title=pred_title,
        alpha=alpha,
    )
    panel_phys = _render_panel(
        image_bgr=image_bgr,
        patch_corners_uv=patch_corners_uv,
        valid_mask=valid_mask,
        values=result_payload["phys_weights"],
        title=phys_title,
        alpha=alpha,
    )
    panel_geom = _render_panel(
        image_bgr=image_bgr,
        patch_corners_uv=patch_corners_uv,
        valid_mask=valid_mask,
        values=result_payload["geom_weights"],
        title=geom_title,
        alpha=alpha,
    )
    panel_tex = _render_panel(
        image_bgr=image_bgr,
        patch_corners_uv=patch_corners_uv,
        valid_mask=valid_mask,
        values=result_payload["tex_weights"],
        title=tex_title,
        alpha=alpha,
    )

    row_top = np.hstack([panel_pred, panel_phys])
    row_bottom = np.hstack([panel_geom, panel_tex])
    grid_bgr = np.vstack([row_top, row_bottom])
    banner_bgr = _build_summary_banner(
        width=grid_bgr.shape[1],
        frame_id=sample_payload["frame_id"],
        stats=result_payload,
    )
    colorbar_bgr = _build_colorbar(width=grid_bgr.shape[1])
    final_bgr = np.vstack([banner_bgr, colorbar_bgr, grid_bgr])
    final_rgb = _bgr_to_rgb(final_bgr)

    if save_path is not None:
        _save_rgb_image(final_rgb, save_path)

    return final_rgb


def build_visualization(
    sample: Mapping[str, Any] | Any,
    result: Mapping[str, Any] | Any,
    save_path: str | Path | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Compatibility wrapper around :func:`draw_4panel_result`."""
    return draw_4panel_result(
        sample=sample,
        result=result,
        save_path=save_path,
        alpha=alpha,
    )


def _extract_sample_payload(sample: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate and normalize sample-side visualization inputs."""
    image = _require_field(sample, "image", target_name="sample.image")
    patch_corners_uv = _require_field(
        sample,
        "patch_corners_uv",
        target_name="sample.patch_corners_uv",
    )
    meta = _require_field(sample, "meta", target_name="sample.meta")
    frame_id = _get_field(sample, "frame_id", default="<unknown frame>")

    image_array = _normalize_rgb_image(image)
    patch_corners_array = np.asarray(patch_corners_uv, dtype=np.float32)
    meta_array = np.asarray(meta)

    if patch_corners_array.ndim != 3 or patch_corners_array.shape[1:] != (4, 2):
        raise VisualizationError(
            "sample.patch_corners_uv must have shape [N, 4, 2] for panel rendering."
        )
    if meta_array.ndim != 2 or meta_array.shape[1] < 3:
        raise VisualizationError(
            "sample.meta must have shape [N, 3] or wider so valid_mask can be read from meta[:, 2]."
        )
    if patch_corners_array.shape[0] != meta_array.shape[0]:
        raise VisualizationError(
            "sample.patch_corners_uv and sample.meta must describe the same number of patches."
        )

    valid_mask = np.asarray(meta_array[:, 2], dtype=np.float32) > 0.5

    return {
        "image": image_array,
        "frame_id": str(frame_id),
        "patch_corners_uv": patch_corners_array,
        "valid_mask": valid_mask,
        "patch_count": int(patch_corners_array.shape[0]),
    }


def _extract_result_payload(
    *,
    result: Mapping[str, Any] | Any,
    patch_count: int,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Normalize inference outputs and statistics into visualization-ready arrays."""
    patch_probs = _extract_vector(
        result=result,
        field_names=("final_probabilities", "patch_probs", "probabilities", "probs"),
        patch_count=patch_count,
        target_name="result probabilities",
    )

    phys_weights, geom_weights, tex_weights = _extract_expert_weight_vectors(
        result=result,
        patch_count=patch_count,
    )

    threshold = _coerce_float(_get_field(result, "threshold", default=0.5), 0.5)
    valid_patch_count = _coerce_int(
        _get_field(result, "valid_patch_count", default=int(valid_mask.sum())),
        int(valid_mask.sum()),
    )
    abnormal_patch_count = _coerce_int(
        _get_field(
            result,
            "abnormal_patch_count",
            default=int(np.count_nonzero(patch_probs[valid_mask] >= threshold)),
        ),
        int(np.count_nonzero(patch_probs[valid_mask] >= threshold)),
    )
    avg_pred_prob = _coerce_float(
        _get_field(
            result,
            "avg_pred_prob",
            default=float(np.mean(patch_probs[valid_mask])) if np.any(valid_mask) else 0.0,
        ),
        0.0,
    )
    mean_phys_weight = _coerce_float(
        _get_field(
            result,
            "mean_phys_weight",
            default=float(np.mean(phys_weights[valid_mask])) if np.any(valid_mask) else 0.0,
        ),
        0.0,
    )
    mean_geom_weight = _coerce_float(
        _get_field(
            result,
            "mean_geom_weight",
            default=float(np.mean(geom_weights[valid_mask])) if np.any(valid_mask) else 0.0,
        ),
        0.0,
    )
    mean_tex_weight = _coerce_float(
        _get_field(
            result,
            "mean_tex_weight",
            default=float(np.mean(tex_weights[valid_mask])) if np.any(valid_mask) else 0.0,
        ),
        0.0,
    )

    return {
        "patch_probs": patch_probs,
        "phys_weights": phys_weights,
        "geom_weights": geom_weights,
        "tex_weights": tex_weights,
        "valid_patch_count": valid_patch_count,
        "abnormal_patch_count": abnormal_patch_count,
        "avg_pred_prob": avg_pred_prob,
        "mean_phys_weight": mean_phys_weight,
        "mean_geom_weight": mean_geom_weight,
        "mean_tex_weight": mean_tex_weight,
    }


def _extract_expert_weight_vectors(
    *,
    result: Mapping[str, Any] | Any,
    patch_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Phys/Geom/Tex patch weights from either split or combined fields."""
    expert_weights = _get_field(result, "expert_weights", default=None)
    if expert_weights is not None:
        weight_matrix = np.asarray(expert_weights, dtype=np.float32)
        if weight_matrix.ndim == 3 and weight_matrix.shape[0] == 1:
            weight_matrix = weight_matrix[0]
        if weight_matrix.shape != (patch_count, 3):
            raise VisualizationError(
                f"result.expert_weights must have shape ({patch_count}, 3), got {weight_matrix.shape}."
            )
        return weight_matrix[:, 0], weight_matrix[:, 1], weight_matrix[:, 2]

    phys = _extract_vector(
        result=result,
        field_names=("phys_weights",),
        patch_count=patch_count,
        target_name="result.phys_weights",
    )
    geom = _extract_vector(
        result=result,
        field_names=("geom_weights",),
        patch_count=patch_count,
        target_name="result.geom_weights",
    )
    tex = _extract_vector(
        result=result,
        field_names=("tex_weights",),
        patch_count=patch_count,
        target_name="result.tex_weights",
    )
    return phys, geom, tex


def _extract_vector(
    *,
    result: Mapping[str, Any] | Any,
    field_names: tuple[str, ...],
    patch_count: int,
    target_name: str,
) -> np.ndarray:
    """Read one patch-level vector from a result object using compatible field names."""
    for field_name in field_names:
        value = _get_field(result, field_name, default=None)
        if value is None:
            continue

        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != patch_count:
            raise VisualizationError(
                f"{target_name} must contain {patch_count} values, got {array.size}."
            )
        return array

    raise VisualizationError(
        f"Missing {target_name}. Checked fields: {', '.join(field_names)}."
    )


def _render_panel(
    *,
    image_bgr: np.ndarray,
    patch_corners_uv: np.ndarray,
    valid_mask: np.ndarray,
    values: np.ndarray,
    title: str,
    alpha: float,
) -> np.ndarray:
    """Render one overlay panel with a title header."""
    panel_body = image_bgr.copy()
    overlay = image_bgr.copy()

    for patch_index in range(patch_corners_uv.shape[0]):
        if not valid_mask[patch_index]:
            continue

        corners = _polygon_to_int32(
            patch_corners_uv[patch_index],
            width=image_bgr.shape[1],
            height=image_bgr.shape[0],
        )
        color = _value_to_inferno_bgr(values[patch_index])
        cv2.fillPoly(overlay, [corners], color)
        cv2.polylines(
            panel_body,
            [corners],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, panel_body, 1.0 - alpha, 0.0, panel_body)
    header = _build_panel_header(width=panel_body.shape[1], title=title)
    return np.vstack([header, panel_body])


def _build_panel_header(*, width: int, title: str) -> np.ndarray:
    """Build a compact header strip for one panel."""
    header = np.zeros((56, width, 3), dtype=np.uint8)
    header[:] = (44, 44, 44)
    cv2.putText(
        header,
        title,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return header


def _build_summary_banner(*, width: int, frame_id: str, stats: Mapping[str, Any]) -> np.ndarray:
    """Build the top summary banner for the 4-panel canvas."""
    banner = np.zeros((82, width, 3), dtype=np.uint8)
    banner[:] = (28, 28, 28)

    title = f"Road MoME Demo   Frame: {frame_id}"
    stats_line = (
        f"Valid patches: {stats['valid_patch_count']}    "
        f"Abnormal patches: {stats['abnormal_patch_count']}    "
        f"Avg pred prob: {stats['avg_pred_prob']:.3f}    "
        f"Phys: {stats['mean_phys_weight']:.3f}    "
        f"Geom: {stats['mean_geom_weight']:.3f}    "
        f"Tex: {stats['mean_tex_weight']:.3f}"
    )

    cv2.putText(
        banner,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (248, 248, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        stats_line,
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return banner


def _build_colorbar(*, width: int) -> np.ndarray:
    """Build a compact horizontal inferno colorbar similar to the reference script."""
    bar_height = 28
    label_height = 22
    bar_strip = np.zeros((bar_height, width, 3), dtype=np.uint8)
    for x in range(width):
        bar_strip[:, x] = _value_to_inferno_bgr(x / max(width - 1, 1))

    label_strip = np.zeros((label_height, width, 3), dtype=np.uint8)
    label_strip[:] = (30, 30, 30)
    _draw_colorbar_label(label_strip, position=0.0, text="0.0")
    _draw_colorbar_label(label_strip, position=0.5, text="0.5")
    _draw_colorbar_label(label_strip, position=1.0, text="1.0")
    return np.vstack([bar_strip, label_strip])


def _draw_colorbar_label(strip: np.ndarray, *, position: float, text: str) -> None:
    """Draw one label on the compact colorbar strip."""
    x = int(position * max(strip.shape[1] - 1, 1))
    text_width = 12 * len(text)
    if position <= 0.0:
        text_x = 4
    elif position >= 1.0:
        text_x = max(4, strip.shape[1] - text_width - 4)
    else:
        text_x = max(4, x - text_width // 2)

    cv2.putText(
        strip,
        text,
        (text_x, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def _save_rgb_image(image_rgb: np.ndarray, save_path: str | Path) -> None:
    """Save an RGB image to disk with Unicode-safe path handling."""
    save_file = Path(save_path)
    if save_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        if save_file.suffix:
            raise VisualizationError(
                f"Unsupported save_path suffix '{save_file.suffix}'. Use .jpg, .jpeg, .png, or .bmp."
            )
        save_file = save_file.with_suffix(".jpg")

    try:
        save_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VisualizationError(
            f"Failed to prepare visualization output directory for {save_file}: {exc}"
        ) from exc

    image_bgr = _rgb_to_bgr(image_rgb)
    encode_ext = save_file.suffix.lower()
    encode_params: list[int] = []
    if encode_ext in {".jpg", ".jpeg"}:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    success, buffer = cv2.imencode(encode_ext, image_bgr, encode_params)
    if not success:
        raise VisualizationError(
            f"Failed to encode visualization image for saving: {save_file}"
        )

    try:
        save_file.write_bytes(buffer.tobytes())
    except OSError as exc:
        raise VisualizationError(
            f"Failed to write visualization image to {save_file}: {exc}"
        ) from exc


def _normalize_rgb_image(image: Any) -> np.ndarray:
    """Normalize input image data to RGB uint8 HWC format."""
    if image is None:
        raise VisualizationError("sample.image is missing.")

    image_array = np.asarray(image)
    if image_array.size == 0:
        raise VisualizationError("sample.image is empty.")
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[..., None], 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = np.repeat(image_array, 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[..., :3]
    elif image_array.ndim != 3 or image_array.shape[2] != 3:
        raise VisualizationError(
            f"sample.image must have shape [H, W, 3], got {image_array.shape}."
        )

    if np.issubdtype(image_array.dtype, np.floating):
        max_value = float(np.nanmax(image_array))
        if max_value <= 1.0:
            image_array = image_array * 255.0

    image_array = np.clip(image_array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(image_array)


def _polygon_to_int32(corners: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Clip one patch polygon to image bounds and convert it to int32 for OpenCV."""
    polygon = np.asarray(corners, dtype=np.float32)
    if polygon.shape != (4, 2):
        raise VisualizationError(
            f"Each patch polygon must have shape [4, 2], got {polygon.shape}."
        )

    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(width - 1, 0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(height - 1, 0))
    return np.round(polygon).astype(np.int32)


def _value_to_inferno_bgr(value: float) -> tuple[int, int, int]:
    """Map a [0, 1] scalar to an inferno-like BGR color."""
    clipped = float(np.clip(value, 0.0, 1.0))
    scaled = clipped * (_INFERNO_STOPS_RGB.shape[0] - 1)
    low_index = int(scaled)
    high_index = min(low_index + 1, _INFERNO_STOPS_RGB.shape[0] - 1)
    blend = scaled - low_index
    rgb = _INFERNO_STOPS_RGB[low_index] * (1.0 - blend) + _INFERNO_STOPS_RGB[
        high_index
    ] * blend
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR for OpenCV drawing."""
    return np.ascontiguousarray(image_rgb[..., ::-1])


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for downstream UI use."""
    return np.ascontiguousarray(image_bgr[..., ::-1])


def _require_field(
    obj: Mapping[str, Any] | Any,
    field_name: str,
    *,
    target_name: str,
) -> Any:
    """Read a required field from a mapping or attribute-based object."""
    value = _get_field(obj, field_name, default=None)
    if value is None:
        raise VisualizationError(f"Missing required field {target_name}.")
    return value


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


def _coerce_float(value: Any, fallback: float) -> float:
    """Convert one scalar to float or fall back safely."""
    if isinstance(value, (int, float)):
        return float(value)
    return fallback


def _coerce_int(value: Any, fallback: int) -> int:
    """Convert one scalar to int or fall back safely."""
    if isinstance(value, (int, float)):
        return int(value)
    return fallback


def _validate_alpha(alpha: float) -> float:
    """Validate the patch overlay alpha parameter."""
    if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
        raise VisualizationError(
            f"alpha must be a numeric value within [0.0, 1.0], got {alpha!r}."
        )
    return float(alpha)


def _require_cv2() -> None:
    """Ensure OpenCV is available before any drawing or saving work begins."""
    if cv2 is None:
        raise VisualizationError(
            "OpenCV is required for runtime visualization but is not installed."
        )
