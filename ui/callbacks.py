"""Dash callback registration for the Road-MoME single-page demo."""

from __future__ import annotations

import logging

try:
    from dash import Dash, Input, Output, State, ctx
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Dash is required to use ui.callbacks. Please install Dash in the active Python environment."
    ) from exc

from app_bootstrap import (
    RuntimeContext,
    clamp_threshold,
    has_startup_validation_failure,
    summarize_exception,
)
from runtime.logger import (
    get_app_logger,
    log_export_failure,
    log_export_success,
    log_inference_failure,
    log_inference_start,
    log_inference_success,
    log_page_error,
    log_sample_validation_failure,
    log_sample_validation_success,
)
from runtime.validators import ValidationError, validate_runtime_sample
from runtime.visualizer import draw_4panel_result
from ui.formatters import build_export_path_text, extract_stats, format_threshold_text
from ui.image_utils import build_result_image
from ui.layout import build_result_placeholder, build_stats_panel, build_status_panel


def register_callbacks(app: Dash, runtime: RuntimeContext) -> None:
    """Register all callbacks required by the single-page demo UI."""

    @app.callback(Output("threshold-text", "children"), Input("threshold-slider", "value"))
    def update_threshold_text(threshold_value: float | None) -> str:
        threshold = clamp_threshold(threshold_value)
        return format_threshold_text(threshold)

    @app.callback(
        Output("result-panel", "children"),
        Output("stats-panel", "children"),
        Output("status-panel", "children"),
        Input("run-button", "n_clicks"),
        Input("export-button", "n_clicks"),
        State("frame-dropdown", "value"),
        State("threshold-slider", "value"),
        prevent_initial_call=True,
    )
    def run_demo(
        run_clicks: int,
        export_clicks: int,
        frame_id: str | None,
        threshold_value: float | None,
    ):
        del run_clicks, export_clicks

        logger = _get_logger()
        triggered_id = ctx.triggered_id or ""
        should_export = triggered_id == "export-button"
        threshold = clamp_threshold(threshold_value)

        if should_export:
            _log_info(logger, "用户点击运行并导出 | frame_id=%s | threshold=%.2f", frame_id, threshold)
        else:
            _log_info(logger, "用户点击运行检测 | frame_id=%s | threshold=%.2f", frame_id, threshold)

        if runtime.sample_provider is None or runtime.infer_engine is None:
            if has_startup_validation_failure(runtime):
                error_text = runtime.sample_hint or "启动校验失败，请先查看状态区中的详细信息。"
            else:
                error_text = "运行时尚未正确初始化，请先查看状态区中的初始化失败信息。"
            if logger is not None:
                log_page_error(logger, "运行前检查失败", frame_id=frame_id)
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=frame_id, stats={"threshold": threshold}, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        if not frame_id:
            error_text = "当前没有可运行的样本，请先检查 demo_data/index.json。"
            if logger is not None:
                log_page_error(logger, "未提供 frame_id")
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=None, stats={"threshold": threshold}, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        try:
            sample = runtime.sample_provider.get_sample(frame_id)
            _log_info(logger, "样本读取成功 | frame_id=%s", frame_id)
        except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
            error_text = summarize_exception("样本读取失败", exc)
            _log_error(logger, "样本读取失败 | frame_id=%s | error=%s", frame_id, error_text)
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=frame_id, stats={"threshold": threshold}, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        try:
            validate_runtime_sample(sample)
            if logger is not None:
                log_sample_validation_success(logger, frame_id)
        except ValidationError as exc:
            if logger is not None:
                log_sample_validation_failure(logger, frame_id, exc)
            error_text = summarize_exception("样本校验失败", exc)
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=frame_id, stats={"threshold": threshold}, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        try:
            if logger is not None:
                log_inference_start(logger, frame_id, threshold=threshold)
            result = runtime.infer_engine.predict(sample, threshold=threshold)
            stats = extract_stats(result, threshold)
            if logger is not None:
                log_inference_success(
                    logger,
                    frame_id,
                    abnormal_patch_count=stats.get("abnormal_patch_count"),
                    avg_pred_prob=stats.get("avg_pred_prob"),
                )
        except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
            if logger is not None:
                log_inference_failure(logger, frame_id, exc)
            error_text = summarize_exception("推理失败", exc)
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=frame_id, stats={"threshold": threshold}, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        try:
            image_rgb = draw_4panel_result(sample, result, save_path=None)
            result_panel = build_result_image(image_rgb)
            _log_info(logger, "可视化成功 | frame_id=%s", frame_id)
        except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
            error_text = summarize_exception("可视化失败", exc)
            _log_error(logger, "可视化失败 | frame_id=%s | error=%s", frame_id, error_text)
            return (
                build_result_placeholder(runtime, error_text),
                build_stats_panel(frame_id=frame_id, stats=stats, export_path=None),
                build_status_panel(runtime=runtime, run_state="失败", message=error_text),
            )

        export_path_text: str | None = None
        run_state = "成功"
        message = "运行完成。"

        if should_export:
            if runtime.exporter is None:
                run_state = "失败"
                message = "导出器未初始化，无法执行“运行并导出”。"
                if logger is not None:
                    log_export_failure(logger, frame_id, RuntimeError("exporter not initialized"))
            else:
                try:
                    export_result = runtime.exporter.export_all(sample, result, image_rgb)
                    export_path_text = build_export_path_text(export_result)
                    message = "运行并导出完成。"
                    if logger is not None:
                        log_export_success(logger, frame_id, export_result)
                except Exception as exc:  # noqa: BLE001 - keep room for future centralized logger
                    if logger is not None:
                        log_export_failure(logger, frame_id, exc)
                    run_state = "失败"
                    message = summarize_exception("导出失败", exc)

        return (
            result_panel,
            build_stats_panel(frame_id=frame_id, stats=stats, export_path=export_path_text),
            build_status_panel(runtime=runtime, run_state=run_state, message=message),
        )


def _get_logger() -> logging.Logger | None:
    """Best-effort logger acquisition that does not affect callback flow."""

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
