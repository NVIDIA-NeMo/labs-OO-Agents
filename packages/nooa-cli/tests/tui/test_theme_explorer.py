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
    inline_rgb = ";".join(
        str(int(rows[0].record.palette["inline_code_fg"].lstrip("#")[index : index + 2], 16))
        for index in (0, 2, 4)
    )
    assert f"\x1b[1;38;2;{inline_rgb}m inline code " in rendered
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
async def test_theme_picker_action_opens_installed_browser() -> None:
    frontend = MagicMock()
    frontend.open_theme_explorer = AsyncMock()
    command = ThemeCommand(frontend, MagicMock(), MagicMock())

    result = await command.execute(["picker"])

    assert result.success is True
    frontend.open_theme_explorer.assert_awaited_once_with()


def test_theme_action_validation_is_local_and_does_not_load_gallery(monkeypatch) -> None:
    from nooa_cli.tui import theme_gallery

    monkeypatch.setattr(
        theme_gallery,
        "ensure_gallery_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("gallery must remain lazy")),
    )
    command = ThemeCommand(MagicMock(), MagicMock(), MagicMock())

    assert command.validate_args(["picker"]) == (True, None)
    assert command.validate_args(["gallery"]) == (True, None)
    assert command.validate_args(["update"]) == (True, None)


@pytest.mark.asyncio
async def test_theme_update_downloads_catalog_without_opening_browser(monkeypatch) -> None:
    from nooa_cli.tui import theme_gallery

    catalog = theme_gallery.GalleryCatalog({"base16-ocean": MagicMock()}, ("bad",))
    update = MagicMock(return_value=catalog)
    monkeypatch.setattr(theme_gallery, "update_gallery_catalog", update)
    frontend = MagicMock()
    command = ThemeCommand(frontend, MagicMock(), MagicMock())

    result = await command.execute(["update"])

    assert result.success is True
    assert "1 schemes (1 skipped)" in result.outputs[0].content
    update.assert_called_once_with()
    frontend.open_theme_explorer.assert_not_called()


@pytest.mark.asyncio
async def test_theme_gallery_loads_on_demand_and_opens_browser(monkeypatch) -> None:
    from nooa_cli.tui import theme_gallery

    catalog = theme_gallery.GalleryCatalog({"base16-ocean": MagicMock()})
    ensure = MagicMock(return_value=catalog)
    monkeypatch.setattr(theme_gallery, "ensure_gallery_catalog", ensure)
    frontend = MagicMock()
    frontend.open_theme_gallery = AsyncMock()
    command = ThemeCommand(frontend, MagicMock(), MagicMock())

    result = await command.execute(["gallery"])

    assert result.success is True
    ensure.assert_called_once_with()
    frontend.open_theme_gallery.assert_awaited_once_with(catalog)


def test_gallery_view_previews_without_changing_active_theme() -> None:
    from nooa_cli.tui.theme_explorer import ThemeGalleryView
    from nooa_cli.tui.theme_gallery import GalleryTheme

    original = theme.get_theme()
    remote = theme.ThemeRecord(
        "base16-remote",
        "Remote",
        dict(theme.THEMES["latte"]),
        "vs",
        "light",
        "gallery:base16",
    )
    entry = GalleryTheme(remote.id, "base16", remote, {"name": "Remote"})
    try:
        theme.set_theme("mocha")
        view = ThemeGalleryView({entry.id: entry}, install=MagicMock())
        view.on_open()
        assert theme.get_theme() == "mocha"
        assert "Remote" in "\n".join(view.detail_lines(view.model.current, 80))
    finally:
        theme.set_theme(original)


def test_gallery_view_keeps_open_and_reports_install_failure() -> None:
    from nooa_cli.tui.theme_explorer import ThemeGalleryView
    from nooa_cli.tui.theme_gallery import GalleryTheme

    remote = theme.ThemeRecord(
        "base16-remote",
        "Remote",
        dict(theme.THEMES["latte"]),
        "vs",
        "light",
        "gallery:base16",
    )
    entry = GalleryTheme(remote.id, "base16", remote, {"name": "Remote"})
    view = ThemeGalleryView(
        {entry.id: entry},
        install=MagicMock(side_effect=OSError("disk full")),
    )

    assert view.handle_action("enter", view.model.current) == "handled"
    assert "Install failed: disk full" in "\n".join(view.detail_lines(view.model.current, 80))


@pytest.mark.asyncio
async def test_tui_application_installs_and_applies_gallery_theme(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import yaml
    from nooa_cli.tui.config import TUIConfig
    from nooa_cli.tui.theme_gallery import GalleryCatalog, GalleryTheme
    from nooa_cli.tui.tui_application import TUIApplication

    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    remote = theme.ThemeRecord(
        "base16-remote",
        "Remote",
        dict(theme.THEMES["latte"]),
        "vs",
        "light",
        "gallery:base16",
    )
    document = {
        "system": "base16",
        "name": "Remote",
        "variant": "light",
        "palette": {
            f"base{index:02X}": value
            for index, value in enumerate(
                [
                    "eff1f5",
                    "e6e9ef",
                    "dce0e8",
                    "acb0be",
                    "6c6f85",
                    "4c4f69",
                    "3c3f59",
                    "24273a",
                    "d20f39",
                    "fe640b",
                    "df8e1d",
                    "40a02b",
                    "179299",
                    "1e66f5",
                    "8839ef",
                    "dc8a78",
                ]
            )
        },
    }
    entry = GalleryTheme(remote.id, "base16", remote, document)
    app = object.__new__(TUIApplication)
    app._config = SimpleNamespace(tui=TUIConfig())
    app.open_subview = AsyncMock()
    original = theme.get_theme()
    try:
        await app.open_theme_gallery(GalleryCatalog({entry.id: entry}), refresh=MagicMock())
        view = app.open_subview.await_args.args[0]
        target = view.model.current
        assert view.handle_action("enter", target) == "close"

        installed = tmp_path / "user" / "themes" / "base16-remote.yaml"
        assert installed.exists()
        assert yaml.safe_load(installed.read_text(encoding="utf-8"))["name"] == "Remote"
        assert app._config.tui.theme == "base16-remote"
        assert theme.get_theme() == "base16-remote"
    finally:
        theme.set_theme(original)
        theme.reload_themes()


@pytest.mark.asyncio
async def test_gallery_install_rolls_back_when_settings_write_fails(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import yaml
    from nooa_cli.tui.config import TUIConfig
    from nooa_cli.tui.theme_gallery import GalleryCatalog, GalleryTheme
    from nooa_cli.tui.tui_application import TUIApplication

    user_dir = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))
    target = user_dir / "themes" / "base16-remote.yaml"
    target.parent.mkdir(parents=True)
    original_document = {
        "system": "base16",
        "name": "Original",
        "variant": "light",
        "palette": {
            f"base{index:02X}": value
            for index, value in enumerate(
                [
                    "eff1f5",
                    "e6e9ef",
                    "dce0e8",
                    "acb0be",
                    "6c6f85",
                    "4c4f69",
                    "3c3f59",
                    "24273a",
                    "d20f39",
                    "fe640b",
                    "df8e1d",
                    "40a02b",
                    "179299",
                    "1e66f5",
                    "8839ef",
                    "dc8a78",
                ]
            )
        },
    }
    target.write_text(yaml.safe_dump(original_document), encoding="utf-8")
    theme.reload_themes()
    remote = theme.ThemeRecord(
        "base16-remote",
        "Replacement",
        dict(theme.THEMES["latte"]),
        "vs",
        "light",
        "gallery:base16",
    )
    replacement_document = {**original_document, "name": "Replacement"}
    entry = GalleryTheme(remote.id, "base16", remote, replacement_document)
    app = object.__new__(TUIApplication)
    app._config = SimpleNamespace(tui=TUIConfig())
    app.open_subview = AsyncMock()
    monkeypatch.setattr(
        "nooa_cli.tui.settings.write_settings_updates",
        MagicMock(side_effect=OSError("settings unavailable")),
    )

    await app.open_theme_gallery(GalleryCatalog({entry.id: entry}), refresh=MagicMock())
    view = app.open_subview.await_args.args[0]

    assert view.handle_action("enter", view.model.current) == "handled"
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["name"] == "Original"
    assert "settings unavailable" in "\n".join(view.detail_lines(view.model.current, 80))
    theme.reload_themes()


@pytest.mark.asyncio
async def test_gallery_refresh_failure_does_not_misreport_committed_install(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import yaml
    from nooa_cli.tui.config import TUIConfig
    from nooa_cli.tui.theme_gallery import GalleryCatalog, GalleryTheme
    from nooa_cli.tui.tui_application import TUIApplication

    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    remote = theme.ThemeRecord(
        "base16-remote", "Remote", dict(theme.THEMES["latte"]), "vs", "light", "gallery:base16"
    )
    document = {
        "system": "base16",
        "name": "Remote",
        "variant": "light",
        "palette": {
            f"base{index:02X}": value
            for index, value in enumerate(
                [
                    "eff1f5",
                    "e6e9ef",
                    "dce0e8",
                    "acb0be",
                    "6c6f85",
                    "4c4f69",
                    "3c3f59",
                    "24273a",
                    "d20f39",
                    "fe640b",
                    "df8e1d",
                    "40a02b",
                    "179299",
                    "1e66f5",
                    "8839ef",
                    "dc8a78",
                ]
            )
        },
    }
    entry = GalleryTheme(remote.id, "base16", remote, document)
    app = object.__new__(TUIApplication)
    app._config = SimpleNamespace(tui=TUIConfig())
    app.open_subview = AsyncMock()
    original = theme.get_theme()
    try:
        await app.open_theme_gallery(
            GalleryCatalog({entry.id: entry}),
            refresh=MagicMock(side_effect=RuntimeError("paint failed")),
        )
        view = app.open_subview.await_args.args[0]

        assert view.handle_action("enter", view.model.current) == "close"
        assert (user_dir / "themes" / "base16-remote.yaml").exists()
        assert (
            yaml.safe_load((project_dir / "settings.yaml").read_text())["tui"]["theme"] == entry.id
        )
        assert app._config.tui.theme == entry.id
        assert theme.get_theme() == entry.id
    finally:
        theme.set_theme(original)
        theme.reload_themes()


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
