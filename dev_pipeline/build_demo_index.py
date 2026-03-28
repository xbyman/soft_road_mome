"""Build a lightweight demo_data/index.json for the offline demo package.

This script scans:
- demo_data/jpg/
- demo_data/npz/

Then it matches files by frame_id and writes:
- demo_data/index.json

The generated JSON only includes samples that have both JPG and NPZ files.
All stored paths are relative to demo_data/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

FRAME_ID_PATTERN = re.compile(r"(\d{14}\.\d{3})")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class ScanReport:
    """Summary of scan and match results."""

    jpg_count: int = 0
    npz_count: int = 0
    matched_count: int = 0
    invalid_jpg_files: list[str] = field(default_factory=list)
    invalid_npz_files: list[str] = field(default_factory=list)
    duplicate_jpg_files: list[str] = field(default_factory=list)
    duplicate_npz_files: list[str] = field(default_factory=list)
    unmatched_jpg_files: list[str] = field(default_factory=list)
    unmatched_npz_files: list[str] = field(default_factory=list)


def extract_frame_id_from_npz_name(file_path: Path) -> str | None:
    """Extract frame_id from an NPZ filename.

    Expected primary pattern:
    - pkg_20230317074848.400.npz -> 20230317074848.400

    A regex fallback is kept so future naming variants are easier to support.
    """
    stem = file_path.stem

    if stem.startswith("pkg_"):
        candidate = stem.removeprefix("pkg_")
        if FRAME_ID_PATTERN.fullmatch(candidate):
            return candidate

    match = FRAME_ID_PATTERN.search(stem)
    if match:
        return match.group(1)

    return None


def extract_frame_id_from_jpg_name(file_path: Path) -> str | None:
    """Extract frame_id from an image filename.

    Current preferred pattern:
    - 20230317074848.400.jpg -> 20230317074848.400

    If the filename is not exactly equal to frame_id, this function falls back
    to regex matching. Adjust this function only if JPG naming rules change.
    """
    stem = file_path.stem

    if FRAME_ID_PATTERN.fullmatch(stem):
        return stem

    match = FRAME_ID_PATTERN.search(stem)
    if match:
        return match.group(1)

    return None


def collect_files(root_dir: Path, allowed_suffixes: set[str]) -> list[Path]:
    """Collect files recursively under root_dir with allowed suffixes."""
    return sorted(
        file_path
        for file_path in root_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in allowed_suffixes
    )


def scan_frame_map(
    *,
    file_paths: list[Path],
    extractor: Callable[[Path], str | None],
    demo_root: Path,
    invalid_bucket: list[str],
    duplicate_bucket: list[str],
) -> dict[str, Path]:
    """Scan files into a frame_id -> file_path map.

    Duplicate frame_ids are ignored after the first occurrence, with warnings
    recorded into duplicate_bucket.
    """
    frame_map: dict[str, Path] = {}

    for file_path in file_paths:
        frame_id = extractor(file_path)
        relative_path = to_demo_relative_path(file_path, demo_root)

        if frame_id is None:
            invalid_bucket.append(relative_path)
            continue

        if frame_id in frame_map:
            duplicate_bucket.append(relative_path)
            continue

        frame_map[frame_id] = file_path

    return frame_map


def build_demo_index(demo_root: Path) -> tuple[dict[str, dict[str, str]], ScanReport]:
    """Scan demo_data and build the final sorted index mapping."""
    jpg_dir = demo_root / "jpg"
    npz_dir = demo_root / "npz"

    ensure_directory_exists(jpg_dir, "JPG")
    ensure_directory_exists(npz_dir, "NPZ")

    jpg_files = collect_files(jpg_dir, IMAGE_SUFFIXES)
    npz_files = collect_files(npz_dir, {".npz"})

    report = ScanReport(jpg_count=len(jpg_files), npz_count=len(npz_files))

    jpg_map = scan_frame_map(
        file_paths=jpg_files,
        extractor=extract_frame_id_from_jpg_name,
        demo_root=demo_root,
        invalid_bucket=report.invalid_jpg_files,
        duplicate_bucket=report.duplicate_jpg_files,
    )
    npz_map = scan_frame_map(
        file_paths=npz_files,
        extractor=extract_frame_id_from_npz_name,
        demo_root=demo_root,
        invalid_bucket=report.invalid_npz_files,
        duplicate_bucket=report.duplicate_npz_files,
    )

    matched_frame_ids = sorted(set(jpg_map) & set(npz_map))
    report.matched_count = len(matched_frame_ids)

    for frame_id in sorted(set(jpg_map) - set(npz_map)):
        report.unmatched_jpg_files.append(
            to_demo_relative_path(jpg_map[frame_id], demo_root)
        )

    for frame_id in sorted(set(npz_map) - set(jpg_map)):
        report.unmatched_npz_files.append(
            to_demo_relative_path(npz_map[frame_id], demo_root)
        )

    index_data: dict[str, dict[str, str]] = {}
    for frame_id in matched_frame_ids:
        index_data[frame_id] = {
            "jpg": to_demo_relative_path(jpg_map[frame_id], demo_root),
            "npz": to_demo_relative_path(npz_map[frame_id], demo_root),
        }

    return index_data, report


def write_index_json(index_data: dict[str, dict[str, str]], output_path: Path) -> None:
    """Write the matched index mapping to JSON using UTF-8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(index_data, file, ensure_ascii=False, indent=2)


def ensure_directory_exists(directory: Path, label: str) -> None:
    """Ensure a required directory exists and is a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {directory}")


def to_demo_relative_path(file_path: Path, demo_root: Path) -> str:
    """Convert an absolute path to a demo_data-relative POSIX-style path."""
    return file_path.relative_to(demo_root).as_posix()


def print_report(report: ScanReport) -> None:
    """Print scan and match summary to the console."""
    print("[INFO] Demo index build summary")
    print(f"[INFO] Scanned JPG files: {report.jpg_count}")
    print(f"[INFO] Scanned NPZ files: {report.npz_count}")
    print(f"[INFO] Matched pairs: {report.matched_count}")

    print_warning_group("Invalid JPG filenames", report.invalid_jpg_files)
    print_warning_group("Invalid NPZ filenames", report.invalid_npz_files)
    print_warning_group("Duplicate JPG frame_ids", report.duplicate_jpg_files)
    print_warning_group("Duplicate NPZ frame_ids", report.duplicate_npz_files)
    print_warning_group("JPG files without matching NPZ", report.unmatched_jpg_files)
    print_warning_group("NPZ files without matching JPG", report.unmatched_npz_files)


def print_warning_group(title: str, items: list[str]) -> None:
    """Print one warning group if there are any items."""
    if not items:
        return

    print(f"[WARN] {title}: {len(items)}")
    for item in items:
        print(f"  - {item}")


def main() -> int:
    """Build demo_data/index.json under the current demo project."""
    project_root = Path(__file__).resolve().parents[1]
    demo_root = project_root / "demo_data"
    index_path = demo_root / "index.json"

    index_data, report = build_demo_index(demo_root)
    write_index_json(index_data, index_path)
    print_report(report)
    print(f"[INFO] Wrote index file: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
