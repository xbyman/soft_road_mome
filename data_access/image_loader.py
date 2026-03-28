"""Utilities for loading local JPG or PNG demo images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.exceptions import RoadMomeDemoError

_SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ImageLoadError(RoadMomeDemoError):
    """Raised when a demo image cannot be loaded or validated."""


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a demo image file and return it as a NumPy array."""
    image_file = Path(image_path)
    if not image_file.exists():
        raise ImageLoadError(f"Image file does not exist: {image_file}")
    if not image_file.is_file():
        raise ImageLoadError(f"Image path is not a file: {image_file}")

    suffix = image_file.suffix.lower()
    if suffix not in _SUPPORTED_IMAGE_SUFFIXES:
        raise ImageLoadError(
            f"Unsupported image file suffix '{suffix}' for {image_file}. "
            f"Expected one of: {sorted(_SUPPORTED_IMAGE_SUFFIXES)}"
        )

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ImageLoadError(
            "Pillow is required to load demo images. Install it before running the data access layer."
        ) from exc

    try:
        with Image.open(image_file) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA", "L"}:
                image = image.convert("RGB")
            image_array = np.asarray(image)
    except UnidentifiedImageError as exc:
        raise ImageLoadError(f"Unsupported or corrupted image file: {image_file}") from exc
    except OSError as exc:
        raise ImageLoadError(f"Failed to read image file {image_file}: {exc}") from exc

    _validate_image_array(image_array=image_array, image_path=image_file)
    return image_array


def _validate_image_array(*, image_array: np.ndarray, image_path: Path) -> None:
    """Perform minimal sanity checks on the loaded image array."""
    if image_array.size == 0:
        raise ImageLoadError(f"Loaded image is empty: {image_path}")
    if image_array.ndim not in (2, 3):
        raise ImageLoadError(
            f"Loaded image must be 2D or 3D, got shape {image_array.shape} for {image_path}"
        )

    height, width = image_array.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageLoadError(
            f"Loaded image has invalid spatial size {image_array.shape[:2]} for {image_path}"
        )

    if image_array.ndim == 3:
        channels = image_array.shape[2]
        if channels not in (1, 3, 4):
            raise ImageLoadError(
                f"Loaded image must have 1, 3, or 4 channels, got {channels} for {image_path}"
            )
