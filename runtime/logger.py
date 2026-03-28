"""Minimal unified logging helpers for the Road-MoME demo runtime.

This module provides one reusable application logger with console + file output
and a few lightweight event helpers for startup, validation, inference, and
export flows. The implementation stays intentionally small and human-readable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.exceptions import RoadMomeDemoError

DEFAULT_LOGGER_NAME = "road_mome_demo"
DEFAULT_LOG_PATH = Path("outputs") / "logs" / "app.log"
DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LoggerSetupError(RoadMomeDemoError):
    """Raised when the runtime logger cannot be initialized safely."""


def get_logger_name() -> str:
    """Return the default application logger name."""

    return DEFAULT_LOGGER_NAME


def get_app_logger(
    *,
    name: str = DEFAULT_LOGGER_NAME,
    log_path: str | Path = DEFAULT_LOG_PATH,
    level: str | int = DEFAULT_LEVEL,
) -> logging.Logger:
    """Return a configured application logger without duplicating handlers."""

    logger_name = _normalize_logger_name(name)
    resolved_level = _normalize_log_level(level)
    resolved_log_path = _resolve_log_path(log_path)

    _ensure_log_destination_ready(resolved_log_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    if not _has_console_handler(logger):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        console_handler._road_mome_handler_kind = "console"  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if not _has_file_handler(logger, resolved_log_path):
        file_handler = logging.FileHandler(
            resolved_log_path,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        file_handler._road_mome_handler_kind = "file"  # type: ignore[attr-defined]
        file_handler._road_mome_log_path = str(resolved_log_path.resolve())  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    _sync_handler_levels(logger, resolved_level, formatter)
    return logger


def setup_logger(
    *,
    name: str = DEFAULT_LOGGER_NAME,
    log_path: str | Path = DEFAULT_LOG_PATH,
    level: str | int = DEFAULT_LEVEL,
) -> logging.Logger:
    """Compatibility alias for :func:`get_app_logger`."""

    return get_app_logger(name=name, log_path=log_path, level=level)


def log_startup_success(logger: logging.Logger, summary: dict[str, Any]) -> None:
    """Log one startup validation success summary."""

    logger.info(
        "启动校验通过%s",
        _format_context(
            sample_count=summary.get("sample_count"),
            weight_path=summary.get("weight_path"),
            visualization_dir=summary.get("visualization_dir"),
            log_dir=summary.get("log_dir"),
        ),
    )


def log_startup_failure(logger: logging.Logger, error: Exception) -> None:
    """Log one startup validation or initialization failure."""

    logger.error("启动校验失败%s", _format_context(error=str(error)))


def log_sample_validation_success(
    logger: logging.Logger,
    frame_id: str,
) -> None:
    """Log one successful runtime sample validation."""

    logger.info("样本校验通过%s", _format_context(frame_id=frame_id))


def log_sample_validation_failure(
    logger: logging.Logger,
    frame_id: str | None,
    error: Exception,
) -> None:
    """Log one runtime sample validation failure."""

    logger.error(
        "样本校验失败%s",
        _format_context(frame_id=frame_id, error=str(error)),
    )


def log_inference_start(
    logger: logging.Logger,
    frame_id: str,
    *,
    threshold: float | None = None,
) -> None:
    """Log one inference start event."""

    logger.info(
        "开始推理%s",
        _format_context(frame_id=frame_id, threshold=threshold),
    )


def log_inference_success(
    logger: logging.Logger,
    frame_id: str,
    *,
    abnormal_patch_count: int | None = None,
    avg_pred_prob: float | None = None,
) -> None:
    """Log one inference success event."""

    logger.info(
        "推理完成%s",
        _format_context(
            frame_id=frame_id,
            abnormal_patch_count=abnormal_patch_count,
            avg_pred_prob=avg_pred_prob,
        ),
    )


def log_inference_failure(
    logger: logging.Logger,
    frame_id: str | None,
    error: Exception,
) -> None:
    """Log one inference failure event."""

    logger.error(
        "推理失败%s",
        _format_context(frame_id=frame_id, error=str(error)),
    )


def log_export_success(
    logger: logging.Logger,
    frame_id: str,
    path_info: dict[str, Any],
) -> None:
    """Log one export success event."""

    logger.info(
        "导出完成%s",
        _format_context(
            frame_id=frame_id,
            visualization_path=path_info.get("visualization_path"),
            json_log_path=path_info.get("json_log_path"),
            csv_log_path=path_info.get("csv_log_path"),
        ),
    )


def log_export_failure(
    logger: logging.Logger,
    frame_id: str | None,
    error: Exception,
) -> None:
    """Log one export failure event."""

    logger.error(
        "导出失败%s",
        _format_context(frame_id=frame_id, error=str(error)),
    )


def log_page_error(
    logger: logging.Logger,
    message: str,
    *,
    frame_id: str | None = None,
    error: Exception | None = None,
) -> None:
    """Log one page-level or callback-level error summary."""

    logger.error(
        "%s%s",
        message,
        _format_context(frame_id=frame_id, error=str(error) if error is not None else None),
    )


def _normalize_logger_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LoggerSetupError("Logger name must be a non-empty string.")
    return name.strip()


def _normalize_log_level(level: str | int) -> int:
    if isinstance(level, int):
        if level < 0:
            raise LoggerSetupError(f"Logger level must be non-negative, got: {level}")
        return level

    if not isinstance(level, str) or not level.strip():
        raise LoggerSetupError("Logger level must be a non-empty string or int.")

    normalized = level.strip().upper()
    if normalized not in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise LoggerSetupError(
            f"Unsupported logger level '{level}'. Expected INFO, WARNING, ERROR, or CRITICAL."
        )
    return getattr(logging, normalized)


def _resolve_log_path(log_path: str | Path) -> Path:
    path = Path(log_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.exists() and path.is_dir():
        raise LoggerSetupError(f"Log path points to a directory, expected a file: {path}")
    if path.name.strip() == "":
        raise LoggerSetupError(f"Log path must include a file name: {path}")
    return path


def _ensure_log_destination_ready(log_path: Path) -> None:
    log_dir = log_path.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoggerSetupError(
            f"Failed to create log directory '{log_dir}': {exc}"
        ) from exc

    try:
        with log_path.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise LoggerSetupError(
            f"Failed to open log file '{log_path}' for writing: {exc}"
        ) from exc


def _has_console_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if getattr(handler, "_road_mome_handler_kind", None) == "console":
            return True
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            logging.FileHandler,
        ):
            return True
    return False


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    normalized_path = str(log_path.resolve())
    for handler in logger.handlers:
        existing_path = getattr(handler, "_road_mome_log_path", None)
        if existing_path == normalized_path:
            return True
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == log_path.resolve():
                    return True
            except OSError:
                continue
    return False


def _sync_handler_levels(
    logger: logging.Logger,
    level: int,
    formatter: logging.Formatter,
) -> None:
    for handler in logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)


def _format_context(**context: Any) -> str:
    parts: list[str] = []
    for key, value in context.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return ""
    return " | " + " | ".join(parts)
