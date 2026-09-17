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
    """Inline ``code`` uses bold theme accent text without a filled chip."""
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import TerminalFrontend
    from nooa_cli.tui.output import AgentMessage

    frontend = TerminalFrontend(Config())
    original = theme.get_theme()
    bodies = {}
    try:
        for name in theme.THEMES:
            theme.set_theme(name)
            frontend._console.refresh_theme()
            rendered = frontend._render_output_to_ansi(AgentMessage("see `chip` here"), 40)
            body = next(line for line in rendered.splitlines() if "chip" in line)
            bodies[name] = body
            style = frontend._console.console.get_style("markdown.code")
            assert style.color is not None
            assert style.bold is True
            assert style.bgcolor is None
            assert style.color.triplet == tuple(
                int(theme.COLORS["inline_code_fg"][index : index + 2], 16) for index in (1, 3, 5)
            )
            assert "\x1b[40m" not in body, (name, body)
            assert "chip" in body
        assert len(set(bodies.values())) == len(theme.THEMES)
    finally:
        theme.set_theme(original)
        frontend._console.refresh_theme()


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("name", theme.THEMES)
def test_each_theme_has_accessible_inline_code_colors(name: str) -> None:
    palette = theme.THEMES[name]
    assert _contrast_ratio(palette["inline_code_fg"], palette["base"]) >= 4.5
    assert palette["inline_code_bg"] == palette["base"]


@pytest.mark.parametrize("name", theme.THEMES)
def test_semantic_highlight_roles_reach_prompt_toolkit(name: str) -> None:
    original = theme.get_theme()
    try:
        theme.set_theme(name)
        app = TUIApplication()
        style = app._app.style
        palette = theme.THEMES[name]

        selected = style.get_attrs_for_style_str("class:fullscreen-browser.selected")
        match = style.get_attrs_for_style_str("class:transcript-search-match")
        current = style.get_attrs_for_style_str("class:transcript-search-current")
        rail = style.get_attrs_for_style_str("class:fullscreen-browser.active-rail-active")

        assert selected.color == palette["selection_fg"].lstrip("#")
        assert selected.bgcolor == palette["selection_bg"].lstrip("#")
        assert match.color == palette["search_match_fg"].lstrip("#")
        assert match.bgcolor == palette["search_match_bg"].lstrip("#")
        assert current.color == palette["search_current_fg"].lstrip("#")
        assert current.bgcolor == palette["search_current_bg"].lstrip("#")
        assert rail.color == palette["focus_accent"].lstrip("#")
    finally:
        theme.set_theme(original)


@pytest.mark.parametrize("name", theme.THEMES)
def test_feedback_roles_reach_rich_theme(name: str) -> None:
    original = theme.get_theme()
    try:
        theme.set_theme(name)
        rich_theme = theme.create_theme()
        palette = theme.THEMES[name]
        for style_name, role in (
            ("success", "feedback_success"),
            ("error", "feedback_error"),
            ("warning", "feedback_warning"),
            ("info", "feedback_info"),
        ):
            color = rich_theme.styles[style_name].color
            assert color is not None
            assert color.triplet == tuple(
                int(palette[role][index : index + 2], 16) for index in (1, 3, 5)
            )
    finally:
        theme.set_theme(original)


@pytest.mark.parametrize("name", theme.THEMES)
def test_semantic_diff_colors_override_pygments_theme(name: str) -> None:
    from pygments.token import Generic

    palette = theme.THEMES[name]
    syntax_theme = theme.create_syntax_theme(name)

    assert syntax_theme.get_style_for_token(Generic.Inserted).color.triplet == tuple(
        int(palette["diff_added"][index : index + 2], 16) for index in (1, 3, 5)
    )
    assert syntax_theme.get_style_for_token(Generic.Deleted).color.triplet == tuple(
        int(palette["diff_removed"][index : index + 2], 16) for index in (1, 3, 5)
    )


def test_active_pane_rail_uses_theme_highlight_color() -> None:
    """The active-pane rail is the theme's highlight color and follows the palette."""
    app = TUIApplication()

    def rail_color() -> str:
        attrs = app._app.style.get_attrs_for_style_str(
            "class:fullscreen-browser.active-rail-active"
        )
        return attrs.color or ""

    original = theme.get_theme()
    try:
        assert rail_color() == theme.COLORS["focus_accent"].lstrip("#")
        theme.set_theme("latte")
        app.refresh_style()
        assert rail_color() == theme.COLORS["focus_accent"].lstrip("#")
    finally:
        theme.set_theme(original)
        app.refresh_style()


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
