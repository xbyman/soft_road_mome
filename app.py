"""Thin Dash entrypoint for the Road-MoME single-page browser demo."""

from __future__ import annotations

try:
    from dash import Dash
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(
        "Dash is required to run app.py. Please install Dash in the active Python environment."
    ) from exc

from app_bootstrap import initialize_runtime
from ui.callbacks import register_callbacks
from ui.layout import build_layout

APP_TITLE = "Road-MoME 单机浏览器演示平台"
DEFAULT_PORT = 8050

runtime = initialize_runtime()
app = Dash(
    __name__,
    title=APP_TITLE,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.layout = build_layout(runtime)
register_callbacks(app, runtime)


def main() -> None:
    """Launch the local Dash development server."""

    app.run(host="127.0.0.1", port=DEFAULT_PORT, debug=False)


if __name__ == "__main__":
    main()
