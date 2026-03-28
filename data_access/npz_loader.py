"""Utilities for loading and validating local NPZ feature packages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.exceptions import RoadMomeDemoError

_REQUIRED_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "phys_8d": ((63, 8), np.dtype(np.float32)),
    "deep_512d": ((63, 384), np.dtype(np.float32)),
    "patch_corners_uv": ((63, 4, 2), np.dtype(np.float32)),
    "meta": ((63, 3), np.dtype(np.float64)),
}
_OPTIONAL_FIELDS = ("sampled_pts", "patch_uv", "centers", "neighborhood")


class NpzLoadError(RoadMomeDemoError):
    """Raised when an NPZ sample is missing required fields or has invalid content."""


def load_npz_sample(npz_path: str | Path) -> dict[str, np.ndarray | None]:
    """Load and validate a single NPZ feature package."""
    npz_file = Path(npz_path)
    if not npz_file.exists():
        raise NpzLoadError(f"NPZ file does not exist: {npz_file}")
    if not npz_file.is_file():
        raise NpzLoadError(f"NPZ path is not a file: {npz_file}")
    if npz_file.suffix.lower() != ".npz":
        raise NpzLoadError(f"Unsupported NPZ file suffix for {npz_file}. Expected '.npz'.")

    try:
        with np.load(npz_file, allow_pickle=False) as archive:
            normalized_sample: dict[str, np.ndarray | None] = {}

            for field_name, (expected_shape, expected_dtype) in _REQUIRED_FIELDS.items():
                if field_name not in archive.files:
                    raise NpzLoadError(
                        f"NPZ file {npz_file} is missing required field '{field_name}'."
                    )

                array = np.asarray(archive[field_name])
                _validate_required_array(
                    field_name=field_name,
                    array=array,
                    expected_shape=expected_shape,
                    expected_dtype=expected_dtype,
                    npz_path=npz_file,
                )
                normalized_sample[field_name] = array

            for field_name in _OPTIONAL_FIELDS:
                if field_name not in archive.files:
                    normalized_sample[field_name] = None
                    continue

                array = np.asarray(archive[field_name])
                _validate_optional_array(
                    field_name=field_name,
                    array=array,
                    npz_path=npz_file,
                )
                normalized_sample[field_name] = array
    except ValueError as exc:
        raise NpzLoadError(f"Failed to parse NPZ file {npz_file}: {exc}") from exc
    except OSError as exc:
        raise NpzLoadError(f"Failed to read NPZ file {npz_file}: {exc}") from exc

    return normalized_sample


def _validate_required_array(
    *,
    field_name: str,
    array: np.ndarray,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
    npz_path: Path,
) -> None:
    """Validate a required NPZ field using strict shape and dtype checks."""
    if array.shape != expected_shape:
        raise NpzLoadError(
            f"NPZ field '{field_name}' in {npz_path} has invalid shape {array.shape}, "
            f"expected {expected_shape}."
        )
    if array.dtype != expected_dtype:
        raise NpzLoadError(
            f"NPZ field '{field_name}' in {npz_path} has invalid dtype {array.dtype}, "
            f"expected {expected_dtype}."
        )


def _validate_optional_array(
    *,
    field_name: str,
    array: np.ndarray,
    npz_path: Path,
) -> None:
    """Validate optional NPZ content with minimal engineering checks."""
    if array.dtype == np.dtype(object):
        raise NpzLoadError(
            f"Optional NPZ field '{field_name}' in {npz_path} must not use object dtype."
        )
