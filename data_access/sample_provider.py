"""Local sample provider for assembling image and NPZ data into one payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.exceptions import RoadMomeDemoError
from data_access.image_loader import ImageLoadError, load_image
from data_access.index_loader import IndexLoadError, load_index
from data_access.npz_loader import NpzLoadError, load_npz_sample


class SampleProviderError(RoadMomeDemoError):
    """Raised when a requested frame sample cannot be assembled."""


class LocalSampleProvider:
    """Read frame samples from local demo_data assets using index.json."""

    def __init__(self, index_path: str | Path, data_root: str | Path) -> None:
        self._index_path = Path(index_path)
        self._data_root = Path(data_root)

        if not self._data_root.exists():
            raise SampleProviderError(f"Data root does not exist: {self._data_root}")
        if not self._data_root.is_dir():
            raise SampleProviderError(f"Data root is not a directory: {self._data_root}")

        try:
            self._index = load_index(self._index_path)
        except IndexLoadError as exc:
            raise SampleProviderError(
                f"Failed to initialize LocalSampleProvider from index {self._index_path}: {exc}"
            ) from exc

    def list_frame_ids(self) -> list[str]:
        """Return all known frame IDs from the loaded index."""
        return list(self._index.keys())

    def get_sample(self, frame_id: str) -> dict[str, Any]:
        """Build a unified sample payload for downstream runtime inference."""
        if frame_id not in self._index:
            raise SampleProviderError(f"Frame id '{frame_id}' was not found in index.")

        entry = self._index[frame_id]
        image_path = self._resolve_data_path(entry["jpg"])
        npz_path = self._resolve_data_path(entry["npz"])

        try:
            image = load_image(image_path)
            npz_payload = load_npz_sample(npz_path)
        except (ImageLoadError, NpzLoadError) as exc:
            raise SampleProviderError(
                f"Failed to build sample for frame '{frame_id}': {exc}"
            ) from exc

        sample: dict[str, Any] = {
            "frame_id": frame_id,
            "image": image,
            "image_path": image_path,
            "npz_path": npz_path,
            **npz_payload,
        }

        extra_index_meta = {
            key: value
            for key, value in entry.items()
            if key not in {"jpg", "npz"}
        }
        if extra_index_meta:
            sample["index_meta"] = extra_index_meta

        # TODO: Replace the dict payload with core.entities.FrameSample once the entity is finalized.
        return sample

    def _resolve_data_path(self, relative_path: Path) -> Path:
        """Resolve an index entry path against the configured data root."""
        if not isinstance(relative_path, Path):
            raise SampleProviderError(
                "Index loader returned a non-Path entry. Expected normalized Path objects."
            )

        data_root = self._data_root.resolve()
        resolved_path = (data_root / relative_path).resolve()
        try:
            resolved_path.relative_to(data_root)
        except ValueError as exc:
            raise SampleProviderError(
                f"Index entry path escapes the configured data root: {relative_path}"
            ) from exc
        return resolved_path
