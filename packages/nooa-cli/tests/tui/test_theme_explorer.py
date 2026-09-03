# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the full-screen theme browser."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui import theme
from nooa_cli.tui.commands import ThemeCommand
from nooa_cli.tui.fullscreen_browser import ExplorerBrowser
from nooa_cli.tui.output import TextOutput
from nooa_cli.tui.terminal_safety import strip_safe_ansi
from nooa_cli.tui.theme_explorer import ThemeExplorerView, build_theme_rows
from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Size
from prompt_toolkit.output import DummyOutput


def test_theme_rows_expose_metadata_and_semantic_preview() -> None:
    rows = build_theme_rows()

    assert [row.id for row in rows[:4]] == ["mocha", "latte", "vsdark", "vslight"]
    assert "Catppuccin" in rows[0].title
    assert "builtin" in rows[0].search_text
    rendered = "\n".join(ThemeExplorerView(rows).detail_lines(rows[0], 100))
    plain = strip_safe_ansi(rendered)
    assert "Semantic highlights" in rendered
    assert "inline code" in rendered
    assert "success" in rendered and "error" in rendered
    assert "Syntax-highlighted code" in rendered
    assert "def greet" in plain and "return" in plain
    assert "Unified diff" in rendered
    assert "-old_value = 1" in plain and "+new_value = 2" in plain
    assert "\x1b[38;2;" in rendered


@pytest.mark.parametrize("name", ["latte", "vslight"])
def test_light_theme_diff_preview_uses_distinct_semantic_colors(name: str) -> None:
    row = next(row for row in build_theme_rows() if row.id == name)
    rendered = "\n".join(ThemeExplorerView([row]).detail_lines(row, 100))
    palette = row.record.palette

    for role in ("diff_added", "diff_removed"):
        color = palette[role].lstrip("#")
        rgb = ";".join(str(int(color[index : index + 2], 16)) for index in (0, 2, 4))
        assert f"38;2;{rgb}" in rendered
    assert palette["diff_added"] != palette["diff_removed"]


def test_theme_browser_live_previews_and_rolls_back_on_close() -> None:
    original = theme.get_theme()
    refreshed: list[str] = []
    try:
        theme.set_theme("mocha")
        view = ThemeExplorerView(refresh=lambda: refreshed.append(theme.get_theme()))
        output = DummyOutput()
        output.get_size = lambda: Size(rows=24, columns=100)  # type: ignore[method-assign]
        app = Application(output=output, full_screen=True)
        browser = ExplorerBrowser(view, app)
        app.layout = app.layout.__class__(browser.container)
        browser.handle_key("down")
        assert theme.get_theme() == "latte"
        browser.on_close()
        assert theme.get_theme() == "mocha"
        assert refreshed == ["latte", "mocha"]
    finally:
        theme.set_theme(original)


def test_theme_browser_q_closes_from_initial_list_focus() -> None:
    view = ThemeExplorerView(build_theme_rows())
    output = DummyOutput()
    output.get_size = lambda: Size(rows=24, columns=100)  # type: ignore[method-assign]
    app = Application(output=output, full_screen=True)
    browser = ExplorerBrowser(view, app)
    app.layout = app.layout.__class__(browser.container)

    assert browser.active_control == "list"
    main = browser.container._get_container()
    assert browser.explorer_list_height.preferred == 4
    assert browser.explorer_list_height.max == 4
    list_area = main.content.children[3]
    assert list_area.height.min == 7
    assert list_area.height.preferred == 7
    assert list_area.height.max == 7
    assert browser.handle_key("quit") == "close"
    assert browser.buffer.text == ""


def test_theme_rows_do_not_shift_when_active_theme_changes() -> None:
    original = theme.get_theme()
    rows = build_theme_rows()
    view = ThemeExplorerView(rows)
    try:
        theme.set_theme("mocha")
        mocha = view.format_row(rows[0], 100)
        latte_before = view.format_row(rows[1], 100)
        theme.set_theme("latte")
        mocha_after = view.format_row(rows[0], 100)
        latte = view.format_row(rows[1], 100)

        assert mocha.index("mocha") == mocha_after.index("mocha")
        assert latte.index("latte") == latte_before.index("latte")
        assert "●" not in mocha + latte
    finally:
        theme.set_theme(original)


def test_theme_browser_surfaces_skipped_theme_count(monkeypatch) -> None:
    monkeypatch.setattr(theme, "reload_themes", lambda: theme.theme_names())
    monkeypatch.setattr(theme, "THEME_DIAGNOSTICS", ["bad one", "bad two"])

    view = ThemeExplorerView()

    assert "2 invalid skipped" in view.title


def test_theme_browser_enter_commits_current_theme() -> None:
    original = theme.get_theme()
    persisted: list[str] = []
    try:
        theme.set_theme("mocha")
        view = ThemeExplorerView(refresh=lambda: None, persist=persisted.append)
        view.model.move(1)
        view.on_selection_changed()
        assert view.handle_action("enter", view.model.current) == "close"
        view.on_close()
        assert theme.get_theme() == "latte"
        assert persisted == ["latte"]
    finally:
        theme.set_theme(original)


@pytest.mark.asyncio
async def test_theme_command_without_args_opens_browser() -> None:
    frontend = MagicMock()
    frontend.open_theme_explorer = AsyncMock()
    command = ThemeCommand(frontend, MagicMock(), MagicMock())

    result = await command.execute([])

    assert result.success is True
    assert isinstance(result.outputs[0], TextOutput)
    frontend.open_theme_explorer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tui_application_builds_theme_browser_with_persistence(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import yaml
    from nooa_cli.tui.config import TUIConfig
    from nooa_cli.tui.tui_application import TUIApplication

    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / ".nooa"))
    app = object.__new__(TUIApplication)
    app._config = SimpleNamespace(tui=TUIConfig())
    app.open_subview = AsyncMock()
    refresh = MagicMock()

    await app.open_theme_explorer(refresh=refresh)
    view = app.open_subview.await_args.args[0]
    target = next(row for row in view.model.rows if row.id == "latte")
    assert view.handle_action("enter", target) == "close"

    assert app._config.tui.theme == "latte"
    saved = yaml.safe_load((tmp_path / ".nooa" / "settings.yaml").read_text())
    assert saved["tui"]["theme"] == "latte"


@pytest.mark.asyncio
async def test_theme_browser_does_not_mutate_runtime_config_when_save_fails(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from nooa_cli.tui.config import TUIConfig
    from nooa_cli.tui.tui_application import TUIApplication

    app = object.__new__(TUIApplication)
    app._config = SimpleNamespace(tui=TUIConfig())
    app.open_subview = AsyncMock()

    def fail_write(_updates) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("nooa_cli.tui.settings.write_settings_updates", fail_write)
    await app.open_theme_explorer(refresh=MagicMock())
    view = app.open_subview.await_args.args[0]
    target = next(row for row in view.model.rows if row.id == "latte")

    with pytest.raises(OSError, match="disk full"):
        view.handle_action("enter", target)

    assert app._config.tui.theme == "mocha"
    view.on_close()


@pytest.mark.asyncio
async def test_theme_command_accepts_discovered_theme(monkeypatch) -> None:
    original = theme.get_theme()
    custom = theme.ThemeRecord(
        "custom-night",
        "Custom Night",
        dict(theme.THEMES["mocha"]),
        "monokai",
        "dark",
        "test",
    )
    monkeypatch.setitem(theme.THEME_RECORDS, custom.id, custom)
    monkeypatch.setitem(theme.THEMES, custom.id, custom.palette)
    monkeypatch.setitem(theme.SYNTAX_THEMES, custom.id, custom.syntax_theme)
    monkeypatch.setattr(theme, "reload_themes", lambda: theme.theme_names())
    frontend = MagicMock()
    config = MagicMock()
    command = ThemeCommand(frontend, config, MagicMock())
    monkeypatch.setattr(command, "_persist_tui_setting", lambda *_args: "settings.yaml")
    try:
        assert command.validate_args([custom.id]) == (True, None)
        result = await command.execute([custom.id])
        assert result.success is True
        assert theme.get_theme() == custom.id
        frontend.refresh_theme.assert_called_once_with()
    finally:
        theme.set_theme(original)


@pytest.mark.asyncio
async def test_terminal_frontend_delegates_theme_browser_with_full_refresh() -> None:
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import TerminalFrontend

    frontend = TerminalFrontend(Config())
    frontend._app = MagicMock()
    frontend._app.open_theme_explorer = AsyncMock()

    await frontend.open_theme_explorer()

    frontend._app.open_theme_explorer.assert_awaited_once_with(refresh=frontend.refresh_theme)


@pytest.mark.asyncio
async def test_full_application_theme_browser_previews_and_rolls_back() -> None:
    from .tui_app_harness import MutableRecordingOutput, TUIHarness

    original = theme.get_theme()
    theme.set_theme("mocha")
    try:
        async with TUIHarness(output=MutableRecordingOutput(100, 24), full_screen=True) as harness:
            opened = asyncio.create_task(harness.app.open_theme_explorer())
            await harness.wait_for(lambda: isinstance(harness.app.active_subview, ExplorerBrowser))
            await harness.press("down")
            await harness.wait_for(lambda: theme.get_theme() == "latte")
            await harness.press("escape")
            await asyncio.wait_for(opened, 2)
            assert theme.get_theme() == "mocha"
    finally:
        theme.set_theme(original)
