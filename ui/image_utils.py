"""Image helpers for in-memory Dash display."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np

try:
    from dash import html
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Dash is required to use ui.image_utils. Please install Dash in the active Python environment."
    ) from exc

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional for image encoding
    Image = None


def encode_image_to_data_url(image_rgb: np.ndarray) -> str:
    """Convert an RGB NumPy image to an in-memory PNG data URL."""

    image_array = np.asarray(image_rgb)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("可视化结果必须是 shape=[H, W, 3] 的 RGB 图像。")

    image_uint8 = np.clip(image_array, 0, 255).astype(np.uint8, copy=False)

    if Image is None:
        raise ValueError("缺少 Pillow，无法将可视化图像编码为页面预览。")

    buffer = BytesIO()
    Image.fromarray(image_uint8, mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_result_image(image_rgb: np.ndarray) -> html.Img:
    """Build the Dash image component used in the result panel."""

    return html.Img(
        src=encode_image_to_data_url(image_rgb),
        style={
            "width": "100%",
            "height": "auto",
            "maxHeight": "760px",
            "objectFit": "contain",
            "borderRadius": "16px",
            "border": "1px solid #e2e8f0",
            "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.10)",
        },
    )
