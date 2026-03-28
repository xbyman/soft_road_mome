"""Static layout builders for the Road-MoME single-page Dash demo."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from dash import dcc, html
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Dash is required to use ui.layout. Please install Dash in the active Python environment."
    ) from exc

from app_bootstrap import RuntimeContext, has_startup_validation_failure
from ui.formatters import format_float, format_int, format_threshold_text

PAGE_BG = "#eef3f9"
SURFACE_BG = "#ffffff"
SURFACE_ALT = "#f7f9fc"
BORDER = "#d9e2ef"
BORDER_STRONG = "#c4d2e4"
TEXT_MAIN = "#122033"
TEXT_SUB = "#5f6f86"
TEXT_MUTED = "#7f8ba0"
ACCENT = "#1e3a5f"
ACCENT_SOFT = "#edf4ff"
ACCENT_LINE = "#bfcee6"
SUCCESS_BG = "#edf8f2"
SUCCESS_TEXT = "#14784a"
INFO_BG = "#eef5ff"
INFO_TEXT = "#244f84"
WARN_BG = "#fff4e8"
WARN_TEXT = "#a15a1b"
ERROR_BG = "#fff0f1"
ERROR_TEXT = "#b33a47"
CARD_SHADOW = "0 20px 48px rgba(15, 23, 42, 0.08)"
CARD_RADIUS = "22px"


def build_layout(runtime: RuntimeContext) -> html.Div:
    """Build the single-page Dash layout."""

    selected_frame = runtime.frame_ids[0] if runtime.frame_ids else None
    run_disabled = (
        runtime.sample_provider is None
        or runtime.infer_engine is None
        or not runtime.frame_ids
    )
    export_disabled = run_disabled or runtime.exporter is None

    return html.Div(
        style=_page_style(),
        children=[
            html.Div(
                style=_shell_style(),
                children=[
                    _build_header(runtime),
                    html.Div(
                        style=_workspace_style(),
                        children=[
                            _build_control_panel(
                                runtime=runtime,
                                selected_frame=selected_frame,
                                run_disabled=run_disabled,
                                export_disabled=export_disabled,
                            ),
                            _build_result_section(runtime),
                        ],
                    ),
                    html.Div(
                        style=_bottom_grid_style(),
                        children=[
                            _build_stats_section(selected_frame),
                            _build_status_section(runtime),
                        ],
                    ),
                ],
            )
        ],
    )


def build_result_placeholder(runtime: RuntimeContext, error_text: str | None = None) -> html.Div:
    """Build a placeholder or error box for the result panel."""

    if error_text:
        title = "结果显示失败"
        message = error_text
        tone = "error"
        kicker = "RESULT"
    elif has_startup_validation_failure(runtime):
        title = runtime.startup_error_title or "启动校验失败"
        message = runtime.sample_hint or "启动校验未通过，请先修复配置、权重、demo_data 或输出目录。"
        tone = "warn"
        kicker = "SETUP"
    elif runtime.sample_hint:
        title = "暂无样本"
        message = runtime.sample_hint
        tone = "warn"
        kicker = "DATA"
    elif runtime.init_errors:
        title = "初始化未完成"
        message = "请先查看状态区中的初始化失败信息。"
        tone = "warn"
        kicker = "INIT"
    else:
        title = "等待运行"
        message = "请选择一个样本，然后点击“运行检测”或“运行并导出”。"
        tone = "info"
        kicker = "READY"

    palette = _tone_palette(tone)
    return html.Div(
        style={
            "width": "100%",
            "minHeight": "560px",
            "borderRadius": "20px",
            "border": f"1px dashed {palette['border']}",
            "background": palette["panel"],
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "28px",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={"maxWidth": "640px", "textAlign": "center"},
                children=[
                    html.Div(
                        kicker,
                        style={
                            "display": "inline-flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "height": "34px",
                            "padding": "0 14px",
                            "borderRadius": "999px",
                            "background": palette["badge_bg"],
                            "border": f"1px solid {palette['badge_border']}",
                            "color": palette["accent"],
                            "fontSize": "11px",
                            "fontWeight": "800",
                            "letterSpacing": "0.18em",
                            "marginBottom": "18px",
                        },
                    ),
                    html.Div(
                        title,
                        style={
                            "fontSize": "30px",
                            "fontWeight": "800",
                            "color": TEXT_MAIN,
                            "letterSpacing": "-0.02em",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        message,
                        style={
                            "fontSize": "15px",
                            "lineHeight": "1.8",
                            "color": palette["accent"],
                            "wordBreak": "break-word",
                        },
                    ),
                ],
            )
        ],
    )


def build_stats_panel(
    *,
    frame_id: str | None,
    stats: Mapping[str, Any] | None,
    export_path: str | None,
) -> html.Div:
    """Build the lower statistics section."""

    stat_source = stats or {}
    rows = [
        ("frame_id", frame_id or "N/A"),
        ("valid_patch_count", format_int(stat_source.get("valid_patch_count"))),
        ("abnormal_patch_count", format_int(stat_source.get("abnormal_patch_count"))),
        ("avg_pred_prob", format_float(stat_source.get("avg_pred_prob"))),
        ("mean_phys_weight", format_float(stat_source.get("mean_phys_weight"))),
        ("mean_geom_weight", format_float(stat_source.get("mean_geom_weight"))),
        ("mean_tex_weight", format_float(stat_source.get("mean_tex_weight"))),
        ("threshold", format_float(stat_source.get("threshold"))),
        ("最近一次导出路径", export_path or "N/A"),
    ]

    children: list[html.Div] = []
    for label, value in rows:
        is_na = value == "N/A"
        is_wide = label in {"frame_id", "最近一次导出路径"}
        children.append(
            html.Div(
                style=_metric_card_style(wide=is_wide),
                children=[
                    html.Div(
                        label,
                        style={
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "color": TEXT_MUTED,
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(
                        value,
                        style={
                            "fontSize": "15px" if is_wide else "28px",
                            "fontWeight": "600" if is_na else ("700" if is_wide else "800"),
                            "color": TEXT_MUTED if is_na else TEXT_MAIN,
                            "lineHeight": "1.35",
                            "wordBreak": "break-word",
                            "letterSpacing": "-0.02em",
                        },
                    ),
                ],
            )
        )

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
            "gap": "14px",
        },
        children=children,
    )


def build_status_panel(
    *,
    runtime: RuntimeContext,
    run_state: str,
    message: str | None,
) -> html.Div:
    """Build the status area with init status, run state, and simplified errors."""

    if has_startup_validation_failure(runtime):
        init_status = runtime.startup_error_title or "启动校验失败"
    else:
        init_status = "已就绪" if not runtime.init_errors else "初始化失败"
    message_text = message or "无"

    detail_lines = runtime.init_messages + runtime.init_errors
    if runtime.startup_summary is not None:
        detail_lines.extend(
            [
                f"启动样本数：{runtime.startup_summary.get('sample_count', 'N/A')}",
                f"权重路径：{runtime.startup_summary.get('weight_path', runtime.weight_path)}",
                f"输出图目录：{runtime.startup_summary.get('visualization_dir', 'N/A')}",
                f"日志目录：{runtime.startup_summary.get('log_dir', 'N/A')}",
            ]
        )
    if not detail_lines:
        detail_lines = ["初始化信息为空。"]

    return html.Div(
        children=[
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "12px",
                    "marginBottom": "18px",
                },
                children=[
                    _build_status_line("初始化状态", init_status),
                    _build_status_line("当前运行状态", run_state),
                    _build_status_line("简洁错误信息", message_text, emphasize=run_state == "失败"),
                ],
            ),
            html.Div(
                style={"borderTop": f"1px solid {BORDER}", "paddingTop": "16px"},
                children=[
                    html.Div(
                        "初始化明细",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "800",
                            "color": TEXT_MAIN,
                            "marginBottom": "12px",
                            "letterSpacing": "0.02em",
                        },
                    ),
                    html.Div(
                        style={"display": "grid", "gap": "10px"},
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "flex-start",
                                    "gap": "10px",
                                    "padding": "12px 14px",
                                    "borderRadius": "14px",
                                    "background": SURFACE_ALT,
                                    "border": f"1px solid {BORDER}",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "width": "8px",
                                            "height": "8px",
                                            "borderRadius": "50%",
                                            "background": ACCENT,
                                            "marginTop": "7px",
                                            "flex": "0 0 auto",
                                        }
                                    ),
                                    html.Div(
                                        line,
                                        style={
                                            "fontSize": "13px",
                                            "lineHeight": "1.7",
                                            "color": TEXT_SUB,
                                            "wordBreak": "break-word",
                                        },
                                    ),
                                ],
                            )
                            for line in detail_lines
                        ],
                    ),
                ],
            ),
        ]
    )


def _build_header(runtime: RuntimeContext) -> html.Div:
    sample_count = len(runtime.frame_ids)
    overall_status = (
        (runtime.startup_error_title or "启动校验失败")
        if has_startup_validation_failure(runtime)
        else ("已就绪" if not runtime.init_errors else "初始化失败")
    )

    return html.Div(
        style=_header_style(),
        children=[
            html.Div(
                style={"flex": "1 1 520px"},
                children=[
                    html.Div(
                        "ROAD-MOME DASHBOARD",
                        style={
                            "fontSize": "12px",
                            "fontWeight": "800",
                            "letterSpacing": "0.18em",
                            "color": "#385475",
                            "marginBottom": "12px",
                        },
                    ),
                    html.H1(
                        "Road-MoME 单机浏览器演示平台",
                        style={
                            "margin": "0 0 10px",
                            "fontSize": "36px",
                            "fontWeight": "800",
                            "color": TEXT_MAIN,
                            "letterSpacing": "-0.03em",
                        },
                    ),
                    html.P(
                        "离线单页仪表盘，面向答辩展示场景，统一承载样本选择、推理可视化、统计概览与导出状态。",
                        style={
                            "margin": 0,
                            "fontSize": "15px",
                            "lineHeight": "1.8",
                            "color": TEXT_SUB,
                            "maxWidth": "760px",
                        },
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                    "gap": "12px",
                    "flex": "1 1 360px",
                    "minWidth": "280px",
                },
                children=[
                    _build_header_metric("运行模式", "离线" if runtime.offline_mode else "在线"),
                    _build_header_metric("样本数", str(sample_count)),
                    _build_header_metric("系统状态", overall_status),
                ],
            ),
        ],
    )


def _build_control_panel(
    *,
    runtime: RuntimeContext,
    selected_frame: str | None,
    run_disabled: bool,
    export_disabled: bool,
) -> html.Div:
    return html.Div(
        style=_card_style(min_height="0", flex="1 1 360px"),
        children=[
            _build_section_heading(
                eyebrow="CONTROL PANEL",
                title="运行控制台",
                subtitle="配置当前样本和阈值后，执行单帧检测或运行后导出。",
            ),
            html.Div(
                style={"display": "grid", "gap": "16px", "marginTop": "22px"},
                children=[
                    _build_info_block("当前权重路径", runtime.weight_path),
                    _build_info_block("离线模式状态", "已启用" if runtime.offline_mode else "未启用"),
                    html.Div(
                        style=_field_group_style(),
                        children=[
                            html.Label("样本选择（frame_id）", style=_label_style()),
                            html.Div(
                                style=_field_shell_style(),
                                children=[
                                    dcc.Dropdown(
                                        id="frame-dropdown",
                                        options=[
                                            {"label": frame_id, "value": frame_id}
                                            for frame_id in runtime.frame_ids
                                        ],
                                        value=selected_frame,
                                        clearable=False,
                                        placeholder="当前没有可用样本",
                                        disabled=not runtime.frame_ids,
                                        style={"fontSize": "14px", "color": TEXT_MAIN},
                                    )
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style=_field_group_style(accent=True),
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "gap": "12px",
                                    "marginBottom": "10px",
                                },
                                children=[
                                    html.Label("检测阈值", style=_label_style(margin_bottom="0")),
                                    html.Div(
                                        id="threshold-text",
                                        style=_threshold_badge_style(),
                                        children=format_threshold_text(runtime.default_threshold),
                                    ),
                                ],
                            ),
                            dcc.Slider(
                                id="threshold-slider",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=runtime.default_threshold,
                                marks={
                                    0.0: "0.0",
                                    0.25: "0.25",
                                    0.5: "0.5",
                                    0.75: "0.75",
                                    1.0: "1.0",
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gap": "12px"},
                        children=[
                            html.Button(
                                "运行检测",
                                id="run-button",
                                n_clicks=0,
                                disabled=run_disabled,
                                style=_button_style(primary=True),
                            ),
                            html.Button(
                                "运行并导出",
                                id="export-button",
                                n_clicks=0,
                                disabled=export_disabled,
                                style=_button_style(primary=False),
                            ),
                        ],
                    ),
                    html.Div(
                        style=_hint_panel_style(),
                        children=[
                            html.Div(
                                "提示",
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "800",
                                    "letterSpacing": "0.08em",
                                    "textTransform": "uppercase",
                                    "color": "#385475",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                runtime.sample_hint
                                or "选择一个样本后即可单帧运行。导出会同时写入图像、JSON 和 CSV。",
                                style={
                                    "fontSize": "13px",
                                    "lineHeight": "1.8",
                                    "color": TEXT_SUB,
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_result_section(runtime: RuntimeContext) -> html.Div:
    return html.Div(
        style=_card_style(min_height="0", flex="1.8 1 760px"),
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": "18px",
                    "marginBottom": "18px",
                    "flexWrap": "wrap",
                },
                children=[
                    _build_section_heading(
                        eyebrow="PRIMARY VIEW",
                        title="四宫格结果图",
                        subtitle="主结果区聚焦当前样本的检测可视化，用于答辩展示和快速判断。",
                    ),
                    html.Div(
                        style={
                            "padding": "10px 14px",
                            "borderRadius": "999px",
                            "background": ACCENT_SOFT,
                            "border": f"1px solid {ACCENT_LINE}",
                            "fontSize": "12px",
                            "fontWeight": "700",
                            "color": "#385475",
                            "whiteSpace": "nowrap",
                        },
                        children="Main Result",
                    ),
                ],
            ),
            html.Div(
                style={
                    "borderRadius": "24px",
                    "padding": "18px",
                    "background": "linear-gradient(180deg, #fbfdff 0%, #f3f7fc 100%)",
                    "border": f"1px solid {BORDER}",
                    "boxShadow": "inset 0 1px 0 rgba(255, 255, 255, 0.8)",
                },
                children=[
                    dcc.Loading(
                        type="default",
                        children=html.Div(
                            id="result-panel",
                            style={
                                "height": "100%",
                                "minHeight": "560px",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "borderRadius": "20px",
                                "overflow": "hidden",
                                "background": "#f9fbfe",
                                "border": f"1px solid {BORDER}",
                                "padding": "14px",
                                "boxSizing": "border-box",
                            },
                            children=build_result_placeholder(runtime),
                        ),
                    )
                ],
            ),
        ],
    )


def _build_stats_section(selected_frame: str | None) -> html.Div:
    return html.Div(
        style=_card_style(min_height="0", flex="1.4 1 700px"),
        children=[
            _build_section_heading(
                eyebrow="RESULT OVERVIEW",
                title="结果信息",
                subtitle="以统一指标卡形式展示当前帧的关键统计，便于快速浏览和答辩讲解。",
            ),
            html.Div(
                id="stats-panel",
                style={"marginTop": "20px"},
                children=build_stats_panel(frame_id=selected_frame, stats=None, export_path=None),
            ),
        ],
    )


def _build_status_section(runtime: RuntimeContext) -> html.Div:
    return html.Div(
        style=_card_style(min_height="0", flex="0.95 1 360px"),
        children=[
            _build_section_heading(
                eyebrow="SYSTEM STATUS",
                title="状态区",
                subtitle="分层展示初始化状态、当前运行状态与错误摘要，弱化但不隐藏系统信息。",
            ),
            html.Div(
                id="status-panel",
                style={"marginTop": "20px"},
                children=build_status_panel(
                    runtime=runtime,
                    run_state="未运行",
                    message=runtime.sample_hint,
                ),
            ),
        ],
    )


def _build_status_line(label: str, value: str, *, emphasize: bool = False) -> html.Div:
    tone = _status_tone(value=value, emphasize=emphasize)
    palette = _tone_palette(tone)
    return html.Div(
        style={
            "padding": "16px",
            "borderRadius": "18px",
            "background": palette["panel"],
            "border": f"1px solid {palette['border']}",
            "minHeight": "118px",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                label,
                style={
                    "fontSize": "12px",
                    "fontWeight": "800",
                    "letterSpacing": "0.08em",
                    "textTransform": "uppercase",
                    "color": TEXT_MUTED,
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "19px" if label != "简洁错误信息" else "15px",
                    "fontWeight": "800" if label != "简洁错误信息" else "600",
                    "lineHeight": "1.5",
                    "color": palette["accent"],
                    "wordBreak": "break-word",
                },
            ),
        ],
    )


def _build_header_metric(title: str, value: str) -> html.Div:
    return html.Div(
        style={
            "padding": "16px 18px",
            "borderRadius": "18px",
            "background": "rgba(255, 255, 255, 0.72)",
            "border": f"1px solid {BORDER}",
            "boxShadow": "0 10px 24px rgba(15, 23, 42, 0.04)",
        },
        children=[
            html.Div(
                title,
                style={
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "letterSpacing": "0.08em",
                    "textTransform": "uppercase",
                    "color": TEXT_MUTED,
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "22px",
                    "fontWeight": "800",
                    "color": TEXT_MAIN,
                    "lineHeight": "1.3",
                },
            ),
        ],
    )


def _build_info_block(title: str, value: str) -> html.Div:
    return html.Div(
        style=_field_group_style(),
        children=[
            html.Div(title, style=_label_style()),
            html.Div(
                value,
                style={
                    "padding": "13px 14px",
                    "borderRadius": "16px",
                    "background": SURFACE_ALT,
                    "border": f"1px solid {BORDER}",
                    "fontSize": "13px",
                    "color": TEXT_MAIN,
                    "wordBreak": "break-word",
                    "lineHeight": "1.7",
                },
            ),
        ],
    )


def _build_section_heading(*, eyebrow: str, title: str, subtitle: str) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                eyebrow,
                style={
                    "fontSize": "11px",
                    "fontWeight": "800",
                    "letterSpacing": "0.14em",
                    "textTransform": "uppercase",
                    "color": "#466180",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                title,
                style={
                    "fontSize": "28px",
                    "fontWeight": "800",
                    "letterSpacing": "-0.02em",
                    "color": TEXT_MAIN,
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                subtitle,
                style={
                    "fontSize": "14px",
                    "lineHeight": "1.8",
                    "color": TEXT_SUB,
                    "maxWidth": "760px",
                },
            ),
        ]
    )


def _label_style(*, margin_bottom: str = "8px") -> dict[str, str]:
    return {
        "display": "block",
        "marginBottom": margin_bottom,
        "fontSize": "12px",
        "fontWeight": "800",
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "color": TEXT_MUTED,
    }


def _button_style(*, primary: bool) -> dict[str, str]:
    if primary:
        background = "linear-gradient(180deg, #203a5d 0%, #162b46 100%)"
        color = "#ffffff"
        border = "#203a5d"
        shadow = "0 14px 28px rgba(17, 34, 58, 0.18)"
    else:
        background = "#ffffff"
        color = TEXT_MAIN
        border = BORDER_STRONG
        shadow = "0 10px 20px rgba(15, 23, 42, 0.05)"

    return {
        "width": "100%",
        "padding": "14px 18px",
        "borderRadius": "16px",
        "border": f"1px solid {border}",
        "background": background,
        "color": color,
        "fontSize": "14px",
        "fontWeight": "800",
        "letterSpacing": "0.01em",
        "cursor": "pointer",
        "boxShadow": shadow,
    }


def _page_style() -> dict[str, str]:
    return {
        "minHeight": "100vh",
        "padding": "28px",
        "background": (
            "radial-gradient(circle at top left, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0) 36%), "
            "linear-gradient(180deg, #f4f7fb 0%, #eaf0f7 100%)"
        ),
        "fontFamily": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
        "boxSizing": "border-box",
    }


def _shell_style() -> dict[str, str]:
    return {
        "maxWidth": "1480px",
        "margin": "0 auto",
        "display": "grid",
        "gap": "22px",
    }


def _header_style() -> dict[str, str]:
    return {
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "20px",
        "alignItems": "stretch",
        "padding": "28px 30px",
        "borderRadius": CARD_RADIUS,
        "background": "linear-gradient(180deg, rgba(255, 255, 255, 0.98) 0%, rgba(249, 251, 255, 0.92) 100%)",
        "border": f"1px solid {BORDER}",
        "boxShadow": CARD_SHADOW,
    }


def _workspace_style() -> dict[str, str]:
    return {
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "22px",
        "alignItems": "stretch",
    }


def _bottom_grid_style() -> dict[str, str]:
    return {
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "22px",
        "alignItems": "stretch",
    }


def _card_style(*, min_height: str, flex: str | None = None) -> dict[str, str]:
    style = {
        "background": SURFACE_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": CARD_RADIUS,
        "padding": "26px",
        "boxShadow": CARD_SHADOW,
        "minHeight": min_height,
    }
    if flex is not None:
        style["flex"] = flex
    return style


def _field_group_style(*, accent: bool = False) -> dict[str, str]:
    return {
        "padding": "16px",
        "borderRadius": "18px",
        "background": ACCENT_SOFT if accent else SURFACE_BG,
        "border": f"1px solid {ACCENT_LINE if accent else BORDER}",
    }


def _field_shell_style() -> dict[str, str]:
    return {
        "padding": "8px",
        "borderRadius": "14px",
        "background": SURFACE_BG,
        "border": f"1px solid {BORDER}",
    }


def _threshold_badge_style() -> dict[str, str]:
    return {
        "padding": "8px 12px",
        "borderRadius": "999px",
        "background": "#ffffff",
        "border": f"1px solid {ACCENT_LINE}",
        "fontSize": "12px",
        "fontWeight": "700",
        "color": "#385475",
        "whiteSpace": "nowrap",
    }


def _hint_panel_style() -> dict[str, str]:
    return {
        "padding": "16px 18px",
        "borderRadius": "18px",
        "background": "linear-gradient(180deg, #f7fbff 0%, #eef5fc 100%)",
        "border": f"1px solid {ACCENT_LINE}",
    }


def _metric_card_style(*, wide: bool) -> dict[str, str]:
    style = {
        "padding": "16px 18px",
        "borderRadius": "18px",
        "background": SURFACE_ALT,
        "border": f"1px solid {BORDER}",
        "minHeight": "112px",
        "boxSizing": "border-box",
    }
    if wide:
        style["gridColumn"] = "1 / -1"
    return style


def _tone_palette(tone: str) -> dict[str, str]:
    if tone == "success":
        return {
            "panel": SUCCESS_BG,
            "border": "#cce7d7",
            "accent": SUCCESS_TEXT,
            "badge_bg": "#ffffff",
            "badge_border": "#cce7d7",
        }
    if tone == "warn":
        return {
            "panel": WARN_BG,
            "border": "#f2d3b1",
            "accent": WARN_TEXT,
            "badge_bg": "#fffaf4",
            "badge_border": "#f2d3b1",
        }
    if tone == "error":
        return {
            "panel": ERROR_BG,
            "border": "#efc8cf",
            "accent": ERROR_TEXT,
            "badge_bg": "#fff7f8",
            "badge_border": "#efc8cf",
        }
    return {
        "panel": INFO_BG,
        "border": "#cadefb",
        "accent": INFO_TEXT,
        "badge_bg": "#f8fbff",
        "badge_border": "#cadefb",
    }


def _status_tone(*, value: str, emphasize: bool) -> str:
    if emphasize or "失败" in value:
        return "error"
    if any(token in value for token in ("成功", "通过", "已就绪")):
        return "success"
    if any(token in value for token in ("未运行", "等待", "未初始化")):
        return "info"
    if value == "无":
        return "info"
    return "warn"
