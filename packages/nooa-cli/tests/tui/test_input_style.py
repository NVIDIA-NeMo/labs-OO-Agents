# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Theme behavior for the long-lived TUI input composer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize(
    "style_name",
    [
        "fullscreen-browser.row",
        "fullscreen-browser.meta",
        "fullscreen-browser.detail",
        "fullscreen-browser.search-label",
        "fullscreen-browser.control",
        "fullscreen-browser.heading",
        "fullscreen-browser.preview",
        "fullscreen-browser.preview-user",
        "fullscreen-browser.preview-agent",
        "fullscreen-browser.footer",
        "fullscreen-browser.control-focused",
        "fullscreen-browser.active-rail-active",
    ],
)
def test_fullscreen_browser_standard_text_uses_terminal_default_foreground(style_name: str) -> None:
    """Standard browser text matches scrollback: the terminal default foreground.

    The theme does not paint ordinary text; only focus/selection/match roles
    carry color. Focus and preview-user roles keep a background.
    """
    app = TUIApplication()

    attrs = app._app.style.get_attrs_for_style_str(f"class:{style_name}")

    assert attrs.color == ""
    if style_name not in {"fullscreen-browser.preview-user", "fullscreen-browser.control-focused"}:
        assert attrs.bgcolor == ""


def test_agent_message_bodies_use_terminal_default_foreground() -> None:
    """Scrollback agent text must stay "no color" like live agent messages.

    Both live rendering and resume replay render plain bodies without a
    foreground SGR so the terminal default foreground (not the theme
    palette) paints agent text, matching the event viewer chrome.
    """
    import re

    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import (
        TerminalFrontend,
        render_history_replay_to_ansi,
    )
    from nooa_cli.tui.output import AgentMessage, HistoryReplay, HistoryTurn

    frontend = TerminalFrontend(Config())
    live = frontend._render_output_to_ansi(AgentMessage("body words"), 40)
    replay = render_history_replay_to_ansi(
        HistoryReplay(
            turns=[HistoryTurn("agent", "body words")],
            session_id="t",
            show_header=False,
            show_footer=False,
        ),
        40,
    )
    for rendered in (live, replay):
        body = next(line for line in rendered.splitlines() if "body words" in line)
        assert not re.search(r"\x1b\[[0-9;]*m", body), rendered


def test_inline_code_spans_follow_the_active_theme() -> None:
    """Inline ``code`` spans must not keep Rich's fixed cyan-on-black style.

    The default markdown.code style ignores the palette, so light themes
    rendered unreadable blue-on-black chips.
    """
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import TerminalFrontend
    from nooa_cli.tui.output import AgentMessage

    frontend = TerminalFrontend(Config())
    original = theme.get_theme()
    bodies = {}
    try:
        for name in ("mocha", "latte"):
            theme.set_theme(name)
            frontend._console.refresh_theme()
            rendered = frontend._render_output_to_ansi(AgentMessage("see `chip` here"), 40)
            bodies[name] = next(line for line in rendered.splitlines() if "chip" in line)
        for name, body in bodies.items():
            assert "\x1b[40m" not in body, (name, body)
            assert "chip" in body
        assert bodies["mocha"] != bodies["latte"]
    finally:
        theme.set_theme(original)
        frontend._console.refresh_theme()


async def test_theme_command_refreshes_and_persists_selection(tmp_path, monkeypatch) -> None:
    import yaml

    class LiveApp:
        refreshed = False

        def refresh_style(self) -> None:
            self.refreshed = True

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    original_theme = theme.get_theme()
    app = LiveApp()
    frontend = SimpleNamespace(refresh_theme=app.refresh_style)
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


@pytest.mark.parametrize(
    ("name", "syntax_theme"),
    [
        ("mocha", "monokai"),
        ("latte", "gruvbox-light"),
        ("vsdark", "github-dark"),
        ("vslight", "vs"),
    ],
)
def test_each_ui_theme_has_a_matching_syntax_theme(name: str, syntax_theme: str) -> None:
    from pygments.styles import get_style_by_name

    original_theme = theme.get_theme()
    try:
        theme.set_theme(name)
        assert theme.get_syntax_theme() == syntax_theme
        assert get_style_by_name(syntax_theme) is not None
    finally:
        theme.set_theme(original_theme)


def test_unknown_syntax_theme_name_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown theme 'ultraviolet'"):
        theme.get_syntax_theme("ultraviolet")


def test_retained_markdown_tracks_live_theme_but_explicit_override_does_not() -> None:
    import io

    from nooa_cli.tui.copyable_markdown import TerminalMarkdown
    from rich.console import Console

    original_theme = theme.get_theme()
    active = TerminalMarkdown("```python\nprint('hello')\n```")
    explicit = TerminalMarkdown("```python\nprint('hello')\n```", code_theme="ansi_dark")
    try:
        for name in theme.THEMES:
            theme.set_theme(name)
            for renderable in (active, explicit):
                Console(file=io.StringIO(), width=80).print(renderable)
            assert active.code_theme == theme.get_syntax_theme(name)
            assert explicit.code_theme == "ansi_dark"
    finally:
        theme.set_theme(original_theme)


def test_console_theme_refresh_replaces_console_without_disturbing_temporary_themes() -> None:
    import io

    from nooa_cli.tui.console import TUIConsole
    from rich.console import Console
    from rich.theme import Theme

    output = io.StringIO()
    tui_console = TUIConsole()
    original = Console(
        file=output,
        force_terminal=True,
        color_system="256",
        width=73,
        theme=theme.create_theme(),
    )
    tui_console.replace_console(original)
    original.push_theme(Theme({"temporary": "bold red"}))
    original_width = original.width

    tui_console.refresh_theme()

    assert tui_console.console is not original
    assert tui_console.console.file is output
    assert tui_console.console.width == original_width
    assert tui_console.console.color_system == "256"
    assert tui_console.console.get_style("agent").color is not None
    # The old console's temporary scope remains balanced and independently usable.
    assert original.get_style("temporary").bold is True
    original.pop_theme()


def test_console_theme_refresh_rebinds_active_spinner(monkeypatch) -> None:
    from nooa_cli.tui import console as console_module
    from nooa_cli.tui.console import TUIConsole

    instances = []

    class FakeLive:
        def __init__(self, _renderable, *, console, **_kwargs) -> None:
            self.console = console
            self.started = False
            self.stopped = False
            instances.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    monkeypatch.setattr(console_module, "Live", FakeLive)
    tui_console = TUIConsole()
    original_console = tui_console.console
    tui_console.start_spinner("still working")
    old_live = tui_console._live_spinner

    tui_console.refresh_theme()

    assert old_live is not None
    assert old_live.stopped is True
    assert tui_console.console is not original_console
    assert tui_console._live_spinner is instances[-1]
    assert tui_console._live_spinner.console is tui_console.console
    assert tui_console._live_spinner.started is True
    assert tui_console._spinner_message == "still working"

    tui_console.stop_spinner()
    assert instances[-1].stopped is True
    assert tui_console._live_spinner is None
