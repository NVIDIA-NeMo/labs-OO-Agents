# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme behavior for the long-lived TUI input composer."""

from __future__ import annotations

from types import SimpleNamespace

from nooa_cli.tui import theme
from nooa_cli.tui.commands import ThemeCommand
from nooa_cli.tui.tui_application import TUIApplication


def test_input_uses_terminal_background_across_themes() -> None:
    original_theme = theme.get_theme()
    try:
        theme.set_theme("mocha")
        app = TUIApplication()
        dark = app._app.style.get_attrs_for_style_str("class:input-area")
        assert dark.bgcolor == ""
        assert dark.color == theme.COLORS["text"].lstrip("#")

        theme.set_theme("latte")
        app.refresh_style()
        light = app._app.style.get_attrs_for_style_str("class:input-area")
        assert light.bgcolor == ""
        assert light.color == theme.COLORS["text"].lstrip("#")
        assert light.color != dark.color
    finally:
        theme.set_theme(original_theme)


async def test_theme_command_refreshes_live_composer_style() -> None:
    class LiveApp:
        refreshed = False

        def refresh_style(self) -> None:
            self.refreshed = True

    original_theme = theme.get_theme()
    app = LiveApp()
    frontend = SimpleNamespace(_app=app, _input_handler=None)
    command = ThemeCommand(frontend, config=object(), agent=object())
    try:
        await command.execute(["latte"])
        assert app.refreshed is True
    finally:
        theme.set_theme(original_theme)
