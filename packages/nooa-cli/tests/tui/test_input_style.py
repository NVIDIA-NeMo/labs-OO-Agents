# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme behavior for the long-lived TUI input composer."""

from __future__ import annotations

from types import SimpleNamespace

from nooa_cli.tui import theme
from nooa_cli.tui.commands import ThemeCommand
from nooa_cli.tui.config import Config, TUIConfig
from nooa_cli.tui.frontend import TerminalFrontend
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


async def test_theme_command_refreshes_and_persists_selection(
    tmp_path, monkeypatch
) -> None:
    import yaml

    class LiveApp:
        refreshed = False

        def refresh_style(self) -> None:
            self.refreshed = True

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    original_theme = theme.get_theme()
    app = LiveApp()
    frontend = SimpleNamespace(_app=app, _input_handler=None)
    config = TUIConfig()
    command = ThemeCommand(frontend, config=config, agent=object())
    try:
        result = await command.execute(["latte"])

        assert result.success is True
        assert app.refreshed is True
        assert config.theme == "latte"
        saved = yaml.safe_load((project_dir / "settings.yaml").read_text())
        assert saved["tui"]["theme"] == "latte"
    finally:
        theme.set_theme(original_theme)


def test_terminal_frontend_restores_persisted_theme(tmp_path, monkeypatch) -> None:
    project_dir = tmp_path / ".nooa"
    project_dir.mkdir()
    (project_dir / "settings.yaml").write_text("tui:\n  theme: vslight\n")
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    original_theme = theme.get_theme()
    try:
        config = Config.load()
        frontend = TerminalFrontend(config)

        assert config.tui.theme == "vslight"
        assert theme.get_theme() == "vslight"
        assert frontend._console.console.get_style("agent").color is not None
        assert frontend._console.console.get_style("agent").color.triplet == (111, 66, 193)
    finally:
        theme.set_theme(original_theme)
