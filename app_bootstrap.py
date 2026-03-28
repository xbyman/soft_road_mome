"""Application bootstrap and runtime context assembly for the Dash demo."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_access.sample_provider import LocalSampleProvider
from runtime.exporter import ResultExporter
from runtime.infer_engine import MomeInferEngine
from runtime.logger import get_app_logger, log_startup_failure, log_startup_success
from runtime.validators import (
    ConfigValidationError,
    DemoDataValidationError,
    IndexValidationError,
    OutputValidationError,
    SampleInputValidationError,
    ValidationError,
    WeightValidationError,
    run_startup_validation,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "deploy.yaml"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "demo_data" / "index.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "demo_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "weights" / "road_mome_v12_best.pth"
DEFAULT_THRESHOLD = 0.8
OFFLINE_MODE = True


@dataclass(slots=True)
class RuntimeContext:
    """Container for startup-time runtime objects and UI-facing init state."""

    sample_provider: LocalSampleProvider | None
    infer_engine: MomeInferEngine | None
    exporter: ResultExporter | None
    frame_ids: list[str]
    init_messages: list[str]
    init_errors: list[str]
    weight_path: str
    default_threshold: float
    offline_mode: bool
    sample_hint: str | None
    startup_summary: dict[str, Any] | None
    startup_error_title: str | None


def initialize_runtime() -> RuntimeContext:
    """Initialize runtime components once at startup without crashing the page."""

    logger = _get_logger()
    _log_info(logger, "应用初始化开始")

    init_messages: list[str] = []
    init_errors: list[str] = []
    frame_ids: list[str] = []
    sample_hint: str | None = None

    sample_provider: LocalSampleProvider | None = None
    infer_engine: MomeInferEngine | None = None
    exporter: ResultExporter | None = None
    weight_path = str(DEFAULT_WEIGHT_PATH)
    default_threshold = DEFAULT_THRESHOLD
    startup_summary: dict[str, Any] | None = None
    startup_error_title: str | None = None

    try:
        startup_summary = run_startup_validation(project_root=PROJECT_ROOT)
        weight_path = str(startup_summary.get("weight_path", DEFAULT_WEIGHT_PATH))
        init_messages.append("启动校验通过。")
        init_messages.append(f"样本数：{startup_summary.get('sample_count', 0)}")
        init_messages.append(
            f"输出图目录：{startup_summary.get('visualization_dir', str(DEFAULT_OUTPUT_ROOT / 'visualizations'))}"
        )
        init_messages.append(
            f"日志目录：{startup_summary.get('log_dir', str(DEFAULT_OUTPUT_ROOT / 'logs'))}"
        )
        if logger is not None:
            log_startup_success(logger, startup_summary)
    except ValidationError as exc:
        if logger is not None:
            log_startup_failure(logger, exc)
        startup_error_title, sample_hint = describe_validation_error(exc)
        init_errors.append(sample_hint)
        runtime = RuntimeContext(
            sample_provider=None,
            infer_engine=None,
            exporter=None,
            frame_ids=[],
            init_messages=init_messages,
            init_errors=init_errors,
            weight_path=weight_path,
            default_threshold=default_threshold,
            offline_mode=OFFLINE_MODE,
            sample_hint=sample_hint,
            startup_summary=startup_summary,
            startup_error_title=startup_error_title,
        )
        _log_error(logger, "RuntimeContext 组装失败 | reason=%s", sample_hint)
        return runtime

    try:
        sample_provider = LocalSampleProvider(
            index_path=DEFAULT_INDEX_PATH,
            data_root=DEFAULT_DATA_ROOT,
        )
        frame_ids = sample_provider.list_frame_ids()
        init_messages.append(f"样本提供器初始化成功，已加载 {len(frame_ids)} 个样本。")
        _log_info(logger, "LocalSampleProvider 初始化成功 | sample_count=%s", len(frame_ids))
        if not frame_ids:
            sample_hint = "当前没有可用演示样本，请先检查 demo_data/index.json。"
    except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
        summary = summarize_exception("样本提供器初始化失败", exc)
        init_errors.append(summary)
        _log_error(logger, "LocalSampleProvider 初始化失败 | error=%s", summary)

    try:
        infer_engine = MomeInferEngine(config_path=DEFAULT_CONFIG_PATH)
        weight_path = str(getattr(infer_engine, "weight_path", DEFAULT_WEIGHT_PATH))
        default_threshold = clamp_threshold(
            getattr(infer_engine, "_default_threshold", DEFAULT_THRESHOLD)
        )
        init_messages.append("推理引擎初始化成功。")
        _log_info(logger, "MomeInferEngine 初始化成功 | weight_path=%s", weight_path)
    except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
        summary = summarize_exception("推理引擎初始化失败", exc)
        init_errors.append(summary)
        _log_error(logger, "MomeInferEngine 初始化失败 | error=%s", summary)

    try:
        exporter = ResultExporter(export_root=DEFAULT_OUTPUT_ROOT)
        init_messages.append("结果导出器初始化成功。")
        _log_info(logger, "ResultExporter 初始化成功 | export_root=%s", DEFAULT_OUTPUT_ROOT)
    except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
        summary = summarize_exception("结果导出器初始化失败", exc)
        init_errors.append(summary)
        _log_error(logger, "ResultExporter 初始化失败 | error=%s", summary)

    runtime = RuntimeContext(
        sample_provider=sample_provider,
        infer_engine=infer_engine,
        exporter=exporter,
        frame_ids=frame_ids,
        init_messages=init_messages,
        init_errors=init_errors,
        weight_path=weight_path,
        default_threshold=default_threshold,
        offline_mode=OFFLINE_MODE,
        sample_hint=sample_hint,
        startup_summary=startup_summary,
        startup_error_title=startup_error_title,
    )

    if init_errors:
        _log_error(logger, "RuntimeContext 组装失败 | error_count=%s", len(init_errors))
    else:
        _log_info(logger, "RuntimeContext 组装成功 | sample_count=%s", len(frame_ids))
    return runtime


def has_startup_validation_failure(runtime: RuntimeContext) -> bool:
    """Return whether startup validation failed before runtime initialization."""

    return runtime.startup_summary is None and bool(runtime.init_errors)


def describe_validation_error(exc: ValidationError) -> tuple[str, str]:
    """Map one validation exception to a user-facing title and summary."""

    if isinstance(exc, ConfigValidationError):
        title = "配置校验失败"
    elif isinstance(exc, WeightValidationError):
        title = "权重校验失败"
    elif isinstance(exc, IndexValidationError):
        title = "索引校验失败"
    elif isinstance(exc, DemoDataValidationError):
        title = "演示数据校验失败"
    elif isinstance(exc, OutputValidationError):
        title = "输出目录校验失败"
    elif isinstance(exc, SampleInputValidationError):
        title = "样本校验失败"
    else:
        title = "启动校验失败"

    return title, summarize_exception(title, exc)


def clamp_threshold(value: Any) -> float:
    """Clamp UI threshold input into the supported [0, 1] range."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(1.0, max(0.0, numeric_value))


def summarize_exception(prefix: str, exc: Exception, limit: int = 220) -> str:
    """Convert an exception into one short UI-friendly summary string."""

    detail = str(exc).strip() or exc.__class__.__name__
    summary = f"{prefix}：{detail}"
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3]}..."


def _get_logger() -> logging.Logger | None:
    """Best-effort logger acquisition that does not block app startup."""

    try:
        return get_app_logger()
    except Exception:
        return None


def _log_info(logger: logging.Logger | None, message: str, *args: object) -> None:
    if logger is not None:
        logger.info(message, *args)


def _log_error(logger: logging.Logger | None, message: str, *args: object) -> None:
    if logger is not None:
        logger.error(message, *args)
