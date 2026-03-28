"""Single-frame runtime inference engine for the offline Road-MoME demo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.exceptions import RoadMomeDemoError
from runtime.model_bootstrap import (
    ModelBootstrapError,
    load_config as load_yaml_config,
    resolve_model_weight_path,
    validate_weight_exists,
)


class InferenceEngineError(RoadMomeDemoError):
    """Base error raised by the runtime inference engine."""


class SampleValidationError(InferenceEngineError):
    """Raised when a frame sample does not satisfy the runtime input contract."""


class ModelLoadError(InferenceEngineError):
    """Raised when the MoME model cannot be constructed or restored offline."""


@dataclass(slots=True)
class InferenceResult:
    """Normalized single-frame inference output for downstream runtime modules.

    This result object intentionally stores only model outputs and summary
    statistics. Input-side context such as the original image, patch_corners_uv,
    meta, and frame_id must stay in the sample object. The future visualizer
    should therefore consume ``sample + result`` together, not ``result`` alone.
    """

    threshold: float
    patch_probs: np.ndarray
    expert_weights: np.ndarray
    valid_patch_count: int
    abnormal_patch_count: int
    avg_pred_prob: float
    mean_phys_weight: float
    mean_geom_weight: float
    mean_tex_weight: float


def generate_rois_from_patch_corners(
    patch_corners_uv: np.ndarray | torch.Tensor,
    valid_mask: np.ndarray | torch.Tensor,
    batch_idx: int = 0,
) -> np.ndarray:
    """Build valid ROI Align-style boxes in the original image coordinate space."""
    corners_array = _as_numpy_array(
        patch_corners_uv,
        field_name="patch_corners_uv",
        expected_ndim=3,
    )
    valid_array = _as_numpy_array(
        valid_mask,
        field_name="valid_mask",
        expected_ndim=1,
    )

    if corners_array.shape[1:] != (4, 2):
        raise SampleValidationError(
            "patch_corners_uv must have shape [N, 4, 2] for ROI generation."
        )
    if corners_array.shape[0] != valid_array.shape[0]:
        raise SampleValidationError(
            "patch_corners_uv and valid_mask must describe the same number of patches."
        )

    rois: list[list[float]] = []
    for patch_index, is_valid in enumerate(valid_array > 0.5):
        if not is_valid:
            continue

        x1, y1, x2, y2 = _corners_to_box(corners_array[patch_index])
        rois.append([float(batch_idx), x1, y1, x2, y2])

    if not rois:
        raise SampleValidationError(
            "No valid patch ROI could be generated because valid_mask contains no active patches."
        )

    return np.asarray(rois, dtype=np.float32)


class MomeInferEngine:
    """Offline single-frame inference runtime compatible with the MoME reference flow."""

    TARGET_IMAGE_WIDTH = 1008
    TARGET_IMAGE_HEIGHT = 560
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    MAX_PATCH_COUNT = 63

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config = self._load_runtime_config(config_path=config_path, config=config)
        self.device = self._resolve_device(device_override=device)
        self._default_threshold = self._resolve_default_threshold(self.config)
        self.ablation_cfg = self._extract_ablation_config(self.config)
        self.dim_phys = self._extract_feature_dim(
            config=self.config,
            branch_name="phys",
            fallback=8,
        )
        self.dim_3d = self._extract_feature_dim(
            config=self.config,
            branch_name="3d",
            fallback=384,
        )
        self.weight_path = self._resolve_weight_path(self.config)
        self._model: torch.nn.Module | None = None

    def predict(
        self,
        sample: Mapping[str, Any] | Any,
        threshold: float | None = None,
    ) -> InferenceResult:
        """Run single-frame inference and return normalized patch-level results."""
        frame_id = self._get_optional_sample_field(sample, "frame_id")
        image = self._require_sample_field(sample, "image")
        phys_8d = self._require_sample_field(sample, "phys_8d")
        deep_512d = self._require_sample_field(sample, "deep_512d")
        patch_corners_uv = self._require_sample_field(sample, "patch_corners_uv")
        meta = self._require_sample_field(sample, "meta")
        quality_2d = self._get_optional_sample_field(sample, "quality_2d")

        normalized = self._normalize_sample_arrays(
            image=image,
            phys_8d=phys_8d,
            deep_512d=deep_512d,
            patch_corners_uv=patch_corners_uv,
            meta=meta,
            quality_2d=quality_2d,
        )
        resolved_threshold = self._resolve_threshold(threshold)

        # Run a strict valid-only ROI pass first so malformed patch corners fail
        # fast with a readable error. The real forward path still uses the dense
        # 63-slot ROI tensor expected by the current MoMEEngine implementation.
        generate_rois_from_patch_corners(
            normalized["patch_corners_uv"],
            normalized["valid_mask"],
            batch_idx=0,
        )

        model = self._ensure_model_loaded()
        image_tensor = self._prepare_image_tensor(normalized["image"]).to(self.device)
        rois_tensor = self._build_dense_model_rois(
            patch_corners_uv=normalized["patch_corners_uv"],
            valid_mask=normalized["valid_mask"],
            image_height=normalized["image"].shape[0],
            image_width=normalized["image"].shape[1],
        ).to(self.device)

        phys_tensor = (
            torch.from_numpy(normalized["phys_8d"]).unsqueeze(0).to(self.device)
        )
        deep_tensor = (
            torch.from_numpy(normalized["deep_512d"]).unsqueeze(0).to(self.device)
        )
        q2_tensor = (
            torch.from_numpy(normalized["quality_2d"]).unsqueeze(0).to(self.device)
        )
        valid_mask_tensor = (
            torch.from_numpy(normalized["valid_mask"]).unsqueeze(0).to(self.device)
        )

        try:
            with torch.no_grad():
                forward_output = model(
                    image_tensor,
                    rois_tensor,
                    phys_tensor,
                    deep_tensor,
                    q2_tensor,
                    valid_mask_tensor,
                    ablation_cfg=self.ablation_cfg,
                )
        except Exception as exc:
            sample_name = frame_id or "<unknown frame>"
            raise InferenceEngineError(
                f"MoMEEngine.forward failed for frame {sample_name}: {exc}"
            ) from exc

        final_logit, internals = self._unpack_forward_output(forward_output)
        patch_probs = self._extract_patch_probs(
            final_logit=final_logit,
            patch_count=self.MAX_PATCH_COUNT,
        )
        expert_weights = self._extract_expert_weights(
            internals=internals,
            patch_count=self.MAX_PATCH_COUNT,
        )

        valid_index = normalized["valid_mask"] > 0.5
        valid_patch_count = int(valid_index.sum())
        abnormal_patch_count = int(
            np.count_nonzero(patch_probs[valid_index] >= resolved_threshold)
        )
        avg_pred_prob = (
            float(np.mean(patch_probs[valid_index])) if valid_patch_count else 0.0
        )
        mean_phys_weight = (
            float(np.mean(expert_weights[valid_index, 0])) if valid_patch_count else 0.0
        )
        mean_geom_weight = (
            float(np.mean(expert_weights[valid_index, 1])) if valid_patch_count else 0.0
        )
        mean_tex_weight = (
            float(np.mean(expert_weights[valid_index, 2])) if valid_patch_count else 0.0
        )

        return InferenceResult(
            threshold=resolved_threshold,
            patch_probs=patch_probs,
            expert_weights=expert_weights,
            valid_patch_count=valid_patch_count,
            abnormal_patch_count=abnormal_patch_count,
            avg_pred_prob=avg_pred_prob,
            mean_phys_weight=mean_phys_weight,
            mean_geom_weight=mean_geom_weight,
            mean_tex_weight=mean_tex_weight,
        )

    def _load_runtime_config(
        self,
        *,
        config_path: str | Path | None,
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Load runtime configuration from an explicit path or an already parsed dict."""
        if config is not None:
            return dict(config)

        resolved_config_path = (
            Path(config_path) if config_path is not None else self._discover_default_config_path()
        )
        try:
            loaded = load_yaml_config(resolved_config_path)
        except ModelBootstrapError as exc:
            raise InferenceEngineError(
                f"Failed to load runtime configuration from {resolved_config_path}: {exc}"
            ) from exc

        return dict(loaded)

    def _discover_default_config_path(self) -> Path:
        """Find a default runtime config file inside the packaged project tree."""
        candidates = (
            self.project_root / "config" / "deploy.yaml",
            self.project_root / "config" / "config.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise InferenceEngineError(
            "No runtime configuration file was found. Expected config/deploy.yaml or config/config.yaml."
        )

    def _resolve_device(self, device_override: str | None) -> torch.device:
        """Resolve the target torch device from arguments or runtime config."""
        configured_value = device_override
        if configured_value is None:
            runtime_config = self.config.get("runtime")
            if isinstance(runtime_config, Mapping):
                raw_device = runtime_config.get("device")
                if isinstance(raw_device, str):
                    configured_value = raw_device

        if configured_value is None or configured_value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if configured_value.startswith("cuda") and not torch.cuda.is_available():
            raise InferenceEngineError(
                f"Configured device '{configured_value}' is unavailable in the current runtime."
            )

        return torch.device(configured_value)

    def _resolve_weight_path(self, config: Mapping[str, Any]) -> Path:
        """Resolve and validate the packaged MoME weight file path."""
        try:
            resolved = Path(
                resolve_model_weight_path(config, project_root=self.project_root)
            )
            validate_weight_exists(str(resolved))
        except (FileNotFoundError, ModelBootstrapError) as exc:
            raise ModelLoadError(
                f"Offline MoME checkpoint is unavailable: {exc}"
            ) from exc

        return resolved

    def _resolve_threshold(self, explicit_threshold: float | None) -> float:
        """Resolve inference threshold with one consistent fallback path."""
        if explicit_threshold is not None:
            return _validate_threshold(explicit_threshold)

        return self._default_threshold

    def _resolve_default_threshold(self, config: Mapping[str, Any]) -> float:
        """Resolve the engine default threshold from config with safe fallback.

        Priority is intentionally fixed:
        1. deploy-style default threshold fields
        2. config inference threshold
        3. hard fallback to 0.5

        Missing, non-numeric, or out-of-range config values do not break engine
        initialization; they safely fall back to 0.5 instead.
        """
        deploy_paths = (
            ("defaults", "threshold"),
            ("ui", "defaults", "threshold"),
            ("runtime", "threshold"),
        )
        for path in deploy_paths:
            found, value = _nested_get_with_presence(config, path)
            if found:
                return _coerce_config_threshold(value)

        found, value = _nested_get_with_presence(config, ("inference", "threshold"))
        if found:
            return _coerce_config_threshold(value)

        return 0.5

    def _extract_ablation_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Extract training ablation switches used by the reference model forward path."""
        ablation = _nested_get(config, ("training", "ablation"))
        return dict(ablation) if isinstance(ablation, Mapping) else {}

    def _extract_feature_dim(
        self,
        *,
        config: Mapping[str, Any],
        branch_name: str,
        fallback: int,
    ) -> int:
        """Read feature dimensions from config.yaml style structure."""
        value = _nested_get(config, ("features", branch_name, "input_dim"))
        return int(value) if isinstance(value, (int, float)) else fallback

    def _ensure_model_loaded(self) -> torch.nn.Module:
        """Construct and restore the MoME model lazily for offline runtime inference."""
        if self._model is not None:
            return self._model

        try:
            from models.mome_model import MoMEEngine
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to import MoMEEngine from models.mome_model: {exc}"
            ) from exc

        try:
            # TODO: models/mome_model.py currently constructs ConvNeXt with
            # ConvNeXt_Base_Weights.DEFAULT. If those backbone weights are not cached
            # locally, torchvision may try to download them. Keep model creation
            # isolated here so offline packaging can replace that behavior later.
            model = MoMEEngine(
                dim_f3_stats=self.dim_phys,
                dim_f3_mae=self.dim_3d,
            ).to(self.device)
        except Exception as exc:
            raise ModelLoadError(
                "Failed to initialize MoMEEngine. Offline risk note: the current "
                "model constructor may still rely on cached torchvision backbone "
                "weights and can fail if it attempts an implicit download."
            ) from exc

        try:
            checkpoint = torch.load(
                self.weight_path,
                map_location=self.device,
                weights_only=False,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if not isinstance(state_dict, Mapping):
                raise ModelLoadError(
                    f"Unsupported checkpoint structure in {self.weight_path}: expected a state_dict mapping."
                )
            clean_state_dict = {
                str(key).replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }
            model.load_state_dict(clean_state_dict, strict=True)
            model.eval()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load MoME checkpoint from {self.weight_path}: {exc}"
            ) from exc

        self._model = model
        return model

    def _normalize_sample_arrays(
        self,
        *,
        image: Any,
        phys_8d: Any,
        deep_512d: Any,
        patch_corners_uv: Any,
        meta: Any,
        quality_2d: Any,
    ) -> dict[str, np.ndarray]:
        """Validate and normalize sample fields into runtime-ready NumPy arrays."""
        image_array = self._normalize_image_array(image)
        phys_array = _as_numpy_array(
            phys_8d,
            field_name="phys_8d",
            expected_ndim=2,
            dtype=np.float32,
        )
        deep_array = _as_numpy_array(
            deep_512d,
            field_name="deep_512d",
            expected_ndim=2,
            dtype=np.float32,
        )
        corners_array = _as_numpy_array(
            patch_corners_uv,
            field_name="patch_corners_uv",
            expected_ndim=3,
            dtype=np.float32,
        )
        meta_array = _as_numpy_array(
            meta,
            field_name="meta",
            expected_ndim=2,
        )

        if image_array.size == 0:
            raise SampleValidationError("Sample image is empty and cannot be used for inference.")
        if corners_array.shape != (self.MAX_PATCH_COUNT, 4, 2):
            raise SampleValidationError(
                "patch_corners_uv must have shape [63, 4, 2] for the current MoME runtime."
            )
        if meta_array.shape != (self.MAX_PATCH_COUNT, 3):
            raise SampleValidationError(
                "meta must have shape [63, 3] for the current MoME runtime."
            )
        if phys_array.shape != (self.MAX_PATCH_COUNT, self.dim_phys):
            raise SampleValidationError(
                f"phys_8d must have shape [63, {self.dim_phys}] for runtime inference."
            )
        if deep_array.shape != (self.MAX_PATCH_COUNT, self.dim_3d):
            raise SampleValidationError(
                f"deep_512d must have shape [63, {self.dim_3d}] for runtime inference."
            )

        valid_mask = np.asarray(meta_array[:, 2], dtype=np.float32)
        if valid_mask.shape[0] != self.MAX_PATCH_COUNT:
            raise SampleValidationError(
                "meta[:, 2] did not produce a valid_mask with 63 entries."
            )
        if not np.any(valid_mask > 0.5):
            raise SampleValidationError(
                "valid_mask contains no active patches, so inference cannot proceed."
            )

        q2_array = self._normalize_quality_2d(
            quality_2d=quality_2d,
            patch_count=self.MAX_PATCH_COUNT,
        )

        return {
            "image": image_array,
            "phys_8d": phys_array,
            "deep_512d": deep_array,
            "patch_corners_uv": corners_array,
            "meta": meta_array.astype(np.float64, copy=False),
            "quality_2d": q2_array,
            "valid_mask": valid_mask,
        }

    def _normalize_image_array(self, image: Any) -> np.ndarray:
        """Normalize a loaded image into an RGB HWC NumPy array."""
        if image is None:
            raise SampleValidationError("Sample image is missing.")

        image_array = np.asarray(image)
        if image_array.size == 0:
            raise SampleValidationError("Sample image is empty.")
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=2)
        elif image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        elif image_array.ndim == 3 and image_array.shape[2] == 4:
            image_array = image_array[..., :3]
        elif image_array.ndim != 3 or image_array.shape[2] != 3:
            raise SampleValidationError(
                f"Sample image must have shape [H, W, 3], got {image_array.shape}."
            )

        if np.issubdtype(image_array.dtype, np.floating):
            max_value = float(np.nanmax(image_array))
            if max_value <= 1.0:
                image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.float32, copy=False)

        return np.ascontiguousarray(image_array)

    def _normalize_quality_2d(
        self,
        *,
        quality_2d: Any,
        patch_count: int,
    ) -> np.ndarray:
        """Normalize optional q_2d input into shape [63, 1]."""
        if quality_2d is None:
            return np.ones((patch_count, 1), dtype=np.float32)

        q2_array = _as_numpy_array(
            quality_2d,
            field_name="quality_2d",
            dtype=np.float32,
        )
        if q2_array.ndim == 1:
            if q2_array.shape[0] != patch_count:
                raise SampleValidationError(
                    "quality_2d must have 63 entries when provided as a 1D array."
                )
            q2_array = q2_array[:, None]
        if q2_array.shape != (patch_count, 1):
            raise SampleValidationError(
                "quality_2d must have shape [63, 1] for the current MoME runtime."
            )
        return q2_array

    def _prepare_image_tensor(self, image_array: np.ndarray) -> torch.Tensor:
        """Resize and normalize RGB image data to the reference model input format."""
        chw_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float() / 255.0
        chw_tensor = chw_tensor.unsqueeze(0)
        chw_tensor = F.interpolate(
            chw_tensor,
            size=(self.TARGET_IMAGE_HEIGHT, self.TARGET_IMAGE_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        mean_tensor = torch.tensor(self.IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(self.IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
        return (chw_tensor - mean_tensor) / std_tensor

    def _build_dense_model_rois(
        self,
        *,
        patch_corners_uv: np.ndarray,
        valid_mask: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> torch.Tensor:
        """Generate the fixed 63 ROI tensor expected by the current MoME forward path."""
        scale_x = self.TARGET_IMAGE_WIDTH / float(image_width)
        scale_y = self.TARGET_IMAGE_HEIGHT / float(image_height)

        dense_rois: list[list[float]] = []
        for patch_index in range(self.MAX_PATCH_COUNT):
            if valid_mask[patch_index] <= 0.5:
                dense_rois.append([0.0, 0.0, 1.0, 1.0])
                continue

            x1, y1, x2, y2 = _corners_to_box(
                patch_corners_uv[patch_index],
                scale_x=scale_x,
                scale_y=scale_y,
                clip_width=float(self.TARGET_IMAGE_WIDTH),
                clip_height=float(self.TARGET_IMAGE_HEIGHT),
            )
            dense_rois.append([x1, y1, x2, y2])

        rois_tensor = torch.tensor(dense_rois, dtype=torch.float32)
        batch_index_column = torch.zeros((self.MAX_PATCH_COUNT, 1), dtype=torch.float32)
        return torch.cat([batch_index_column, rois_tensor], dim=1)

    def _unpack_forward_output(
        self,
        forward_output: Any,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        """Unpack the model forward result into logit tensor and internals mapping.

        The trusted reference path currently returns ``(final_logit, internals)``.
        This parser keeps that contract as the primary path but also accepts a
        close mapping-style variant if a future wrapper exposes named keys.
        """
        if isinstance(forward_output, tuple):
            if len(forward_output) < 2:
                raise InferenceEngineError(
                    "MoMEEngine.forward returned a tuple without internals. "
                    "Expected (final_logit, internals_dict)."
                )
            final_logit = forward_output[0]
            internals = forward_output[1]
        elif isinstance(forward_output, Mapping):
            final_logit = forward_output.get("final_logit")
            if "internals" in forward_output and isinstance(
                forward_output["internals"], Mapping
            ):
                internals = forward_output["internals"]
            else:
                internals = forward_output
        else:
            raise InferenceEngineError(
                "MoMEEngine.forward returned an unexpected structure. "
                "Expected (final_logit, internals_dict)."
            )

        if not isinstance(final_logit, torch.Tensor):
            raise InferenceEngineError(
                "MoMEEngine.forward did not return final_logit as a torch.Tensor."
            )
        if not isinstance(internals, Mapping):
            raise InferenceEngineError(
                "MoMEEngine.forward did not return internals as a mapping."
            )
        if "weights" not in internals:
            raise InferenceEngineError(
                "MoMEEngine.forward internals do not contain the required 'weights' output."
            )

        return final_logit, internals

    def _extract_patch_probs(
        self,
        *,
        final_logit: torch.Tensor,
        patch_count: int,
    ) -> np.ndarray:
        """Convert final logits to a flat patch probability array."""
        probs_tensor = torch.sigmoid(final_logit).detach().cpu()
        probs_array = np.asarray(probs_tensor, dtype=np.float32).reshape(-1)

        if probs_array.size != patch_count:
            raise InferenceEngineError(
                f"Expected {patch_count} patch probabilities, got {probs_array.size}."
            )

        return probs_array

    def _extract_expert_weights(
        self,
        *,
        internals: Mapping[str, Any],
        patch_count: int,
    ) -> np.ndarray:
        """Extract the [63, 3] expert weight matrix from model internals."""
        raw_weights = internals.get("weights")
        if raw_weights is None:
            raise InferenceEngineError(
                "MoMEEngine.forward internals do not contain expert weights."
            )
        if not isinstance(raw_weights, torch.Tensor):
            raise InferenceEngineError(
                "MoMEEngine.forward internals['weights'] is not a torch.Tensor."
            )

        weights_array = np.asarray(raw_weights.detach().cpu(), dtype=np.float32)
        if weights_array.ndim == 3 and weights_array.shape[0] == 1:
            weights_array = weights_array[0]
        if weights_array.shape != (patch_count, 3):
            raise InferenceEngineError(
                f"Expected expert weights with shape ({patch_count}, 3), got {weights_array.shape}."
            )

        return weights_array

    def _require_sample_field(
        self,
        sample: Mapping[str, Any] | Any,
        field_name: str,
    ) -> Any:
        """Read a required field from a mapping-like or attribute-based sample object."""
        value = self._get_optional_sample_field(sample, field_name)
        if value is None:
            raise SampleValidationError(
                f"Sample is missing required field '{field_name}'."
            )
        return value

    def _get_optional_sample_field(
        self,
        sample: Mapping[str, Any] | Any,
        field_name: str,
    ) -> Any | None:
        """Safely read a possibly missing field from a sample object."""
        if isinstance(sample, Mapping):
            return sample.get(field_name)
        if hasattr(sample, field_name):
            return getattr(sample, field_name)
        return None


def _as_numpy_array(
    value: Any,
    *,
    field_name: str,
    expected_ndim: int | None = None,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray:
    """Convert runtime inputs to NumPy arrays with optional ndim and dtype checks."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if expected_ndim is not None and array.ndim != expected_ndim:
        raise SampleValidationError(
            f"{field_name} must have {expected_ndim} dimensions, got shape {array.shape}."
        )

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    return array


def _corners_to_box(
    corners: np.ndarray,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    clip_width: float | None = None,
    clip_height: float | None = None,
) -> tuple[float, float, float, float]:
    """Convert one patch's four UV corners into a stable axis-aligned ROI box."""
    x_coords = np.asarray(corners[:, 0], dtype=np.float32) * scale_x
    y_coords = np.asarray(corners[:, 1], dtype=np.float32) * scale_y

    x1 = float(np.min(x_coords))
    y1 = float(np.min(y_coords))
    x2 = float(np.max(x_coords))
    y2 = float(np.max(y_coords))

    if clip_width is not None:
        x1 = float(np.clip(x1, 0.0, clip_width))
        x2 = float(np.clip(x2, 0.0, clip_width))
    if clip_height is not None:
        y1 = float(np.clip(y1, 0.0, clip_height))
        y2 = float(np.clip(y2, 0.0, clip_height))

    # Keep ROI Align inputs strictly positive even when a patch projects to a
    # degenerate box after clipping. This preserves the reference script's
    # placeholder behavior without changing the upstream valid_mask semantics.
    if x2 <= x1:
        x2 = x1 + 1.0
    if y2 <= y1:
        y2 = y1 + 1.0

    return x1, y1, x2, y2


def _nested_get(
    mapping: Mapping[str, Any],
    path: tuple[str, ...],
) -> Any | None:
    """Read a nested mapping value without assuming intermediate keys exist."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _nested_get_with_presence(
    mapping: Mapping[str, Any],
    path: tuple[str, ...],
) -> tuple[bool, Any | None]:
    """Read a nested mapping value while preserving key presence information."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _coerce_config_threshold(value: Any) -> float:
    """Convert a config threshold value to float, falling back to 0.5 on error."""
    if not isinstance(value, (int, float)):
        return 0.5
    try:
        return _validate_threshold(float(value))
    except InferenceEngineError:
        return 0.5


def _validate_threshold(value: float) -> float:
    """Validate threshold range for anomaly counting."""
    if not 0.0 <= float(value) <= 1.0:
        raise InferenceEngineError(
            f"Inference threshold must be within [0.0, 1.0], got {value}."
        )
    return float(value)
