"""Exception-path tests for runtime validators."""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.validators import (
    IndexValidationError,
    SampleInputValidationError,
    run_startup_validation,
    validate_index_file,
    validate_runtime_sample,
)


class ValidatorsExceptionPathTests(unittest.TestCase):
    """Cover startup and pre-run validation failures that block the demo."""

    def test_run_startup_validation_rejects_non_object_index(self) -> None:
        project_root = self._build_minimal_project(index_content="1111")

        with self.assertRaises(IndexValidationError) as context:
            run_startup_validation(project_root=project_root)

        self.assertIn("top-level JSON object", str(context.exception))

    def test_run_startup_validation_rejects_empty_index(self) -> None:
        project_root = self._build_minimal_project(index_content="{}")

        with self.assertRaises(IndexValidationError) as context:
            run_startup_validation(project_root=project_root)

        self.assertIn("Index file is empty", str(context.exception))

    def test_validate_index_file_reports_missing_assets(self) -> None:
        project_root = self._build_minimal_project(
            index_content=json.dumps(
                {
                    "frame_001": {
                        "jpg": "jpg/frame_001.jpg",
                        "npz": "npz/frame_001.npz",
                    }
                }
            ),
            create_assets=False,
        )

        with self.assertRaises(IndexValidationError) as context:
            validate_index_file(
                index_path=project_root / "demo_data" / "index.json",
                demo_root=project_root / "demo_data",
            )

        error_text = str(context.exception)
        self.assertIn("missing_jpg=1", error_text)
        self.assertIn("missing_npz=1", error_text)

    def test_validate_runtime_sample_rejects_no_active_patches(self) -> None:
        sample = {
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "phys_8d": np.zeros((63, 8), dtype=np.float32),
            "deep_512d": np.zeros((63, 384), dtype=np.float32),
            "patch_corners_uv": np.zeros((63, 4, 2), dtype=np.float32),
            "meta": np.zeros((63, 3), dtype=np.float32),
        }

        with self.assertRaises(SampleInputValidationError) as context:
            validate_runtime_sample(sample)

        self.assertIn("no active patches", str(context.exception))

    def _build_minimal_project(
        self,
        *,
        index_content: str,
        create_assets: bool = True,
    ) -> Path:
        temp_root = Path.cwd() / "tests" / "_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / f"validator_case_{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(shutil.rmtree, temp_dir, True)

        root = temp_dir
        (root / "config").mkdir(parents=True, exist_ok=True)
        (root / "weights").mkdir(parents=True, exist_ok=True)
        (root / "demo_data" / "jpg").mkdir(parents=True, exist_ok=True)
        (root / "demo_data" / "npz").mkdir(parents=True, exist_ok=True)

        (root / "config" / "config.yaml").write_text("app:\n  name: test\n", encoding="utf-8")
        (root / "config" / "ui.yaml").write_text("defaults:\n  threshold: 0.8\n", encoding="utf-8")
        (root / "config" / "deploy.yaml").write_text(
            "\n".join(
                [
                    "weights:",
                    '  mome_model: "./weights/road_mome_v12_best.pth"',
                    "data:",
                    '  index_file: "./demo_data/index.json"',
                    '  demo_jpg_dir: "./demo_data/jpg"',
                    '  demo_npz_dir: "./demo_data/npz"',
                    "outputs:",
                    '  visualization_dir: "./outputs/visualizations"',
                    '  log_dir: "./outputs/logs"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "weights" / "road_mome_v12_best.pth").write_bytes(b"test-weight")
        (root / "demo_data" / "index.json").write_text(index_content, encoding="utf-8")

        if create_assets:
            (root / "demo_data" / "jpg" / "frame_001.jpg").write_bytes(b"jpg")
            (root / "demo_data" / "npz" / "frame_001.npz").write_bytes(b"npz")

        return root


if __name__ == "__main__":
    unittest.main()
