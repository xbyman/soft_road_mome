"""Utilities for loading and validating the local demo index.json file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.exceptions import RoadMomeDemoError

_REQUIRED_ENTRY_FIELDS = ("jpg", "npz")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class IndexLoadError(RoadMomeDemoError):
    """Raised when the demo index file is missing or malformed."""


def load_index(index_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a demo index.json mapping."""
    index_file = Path(index_path)
    if not index_file.exists():
        raise IndexLoadError(f"Index file does not exist: {index_file}")
    if not index_file.is_file():
        raise IndexLoadError(f"Index path is not a file: {index_file}")
    if index_file.suffix.lower() != ".json":
        raise IndexLoadError(
            f"Unsupported index file format for {index_file}. Expected a .json file."
        )

    try:
        with index_file.open("r", encoding="utf-8") as handle:
            raw_index = json.load(handle)
    except json.JSONDecodeError as exc:
        raise IndexLoadError(
            f"Failed to parse index JSON {index_file}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise IndexLoadError(f"Failed to read index file {index_file}: {exc}") from exc

    if not isinstance(raw_index, dict):
        raise IndexLoadError(
            f"Index file must contain a top-level JSON object: {index_file}"
        )

    normalized_index: dict[str, dict[str, Any]] = {}
    for frame_id, entry in raw_index.items():
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise IndexLoadError(
                f"Each index key must be a non-empty frame_id string. Got: {frame_id!r}"
            )
        if not isinstance(entry, dict):
            raise IndexLoadError(
                f"Index entry for frame '{frame_id}' must be an object."
            )

        missing_fields = [field for field in _REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing_fields:
            raise IndexLoadError(
                f"Index entry for frame '{frame_id}' is missing required fields: {missing_fields}"
            )

        normalized_entry: dict[str, Any] = {}
        normalized_entry["jpg"] = _normalize_relative_path(
            frame_id=frame_id,
            field_name="jpg",
            raw_value=entry["jpg"],
            expected_suffixes=_IMAGE_SUFFIXES,
        )
        normalized_entry["npz"] = _normalize_relative_path(
            frame_id=frame_id,
            field_name="npz",
            raw_value=entry["npz"],
            expected_suffixes={".npz"},
        )

        for extra_key, extra_value in entry.items():
            if extra_key not in _REQUIRED_ENTRY_FIELDS:
                normalized_entry[extra_key] = extra_value

        normalized_index[frame_id] = normalized_entry

    return normalized_index


def _normalize_relative_path(
    *,
    frame_id: str,
    field_name: str,
    raw_value: Any,
    expected_suffixes: set[str],
) -> Path:
    """Normalize a relative path string from index.json into Path."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise IndexLoadError(
            f"Index entry for frame '{frame_id}' field '{field_name}' must be a non-empty string."
        )

    normalized_path = Path(raw_value)
    if normalized_path.is_absolute():
        raise IndexLoadError(
            f"Index entry for frame '{frame_id}' field '{field_name}' must use a relative path, got: {raw_value}"
        )

    suffix = normalized_path.suffix.lower()
    if suffix not in expected_suffixes:
        raise IndexLoadError(
            f"Index entry for frame '{frame_id}' field '{field_name}' has unsupported suffix '{suffix}'. "
            f"Expected one of: {sorted(expected_suffixes)}"
        )

    return normalized_path
