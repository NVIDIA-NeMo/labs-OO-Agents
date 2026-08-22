# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responsive NOOA splash rendering."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nooa_cli.tui.splash import NOOA_ASCII, show_splash, splash_variant
from rich.console import Console


def _render(width: int) -> str:
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=width,
        color_system=None,
        force_terminal=False,
    )
    show_splash(console)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("width", "expected"),
    [(120, "wide"), (80, "wide"), (79, "standard"), (48, "standard"), (47, "compact")],
)
def test_splash_selects_responsive_variant(width: int, expected: str) -> None:
    assert splash_variant(width) == expected


@pytest.mark.parametrize("width", [32, 47, 48, 64, 72, 100, 160])
def test_splash_never_exceeds_terminal_width(width: int) -> None:
    rendered = _render(width)
    assert rendered
    assert max(len(line) for line in rendered.splitlines()) <= width


def test_compact_splash_keeps_full_identity_readable() -> None:
    rendered = _render(40)
    assert "NVIDIA LABS · NOOA" in rendered
    assert "Object Oriented Agents" in rendered


def test_graphical_splash_uses_new_brand_not_legacy_copy() -> None:
    rendered = _render(100)
    assert "NVIDIA LABS" in rendered
    assert "OBJECT ORIENTED AGENTS" in rendered
    assert "NEMOTRON" not in rendered
    assert "licensed to vibe" not in rendered
    assert NOOA_ASCII == "NVIDIA LABS OBJECT ORIENTED AGENTS (NOOA)"


def test_graphical_splash_centers_nooa_without_the_nvidia_eye() -> None:
    rendered = _render(100)
    assert "████▀███▄ ▄███▀███▄" in rendered
    assert "████████████████████████" not in rendered


def test_splash_does_not_delay_startup_by_default() -> None:
    with patch("nooa_cli.tui.splash.time.sleep") as sleep:
        _render(80)
    sleep.assert_not_called()


def test_optional_delay_remains_available_for_embedders() -> None:
    stream = io.StringIO()
    console = Console(file=stream, width=80, color_system=None)
    with patch("nooa_cli.tui.splash.time.sleep") as sleep:
        show_splash(console, delay=0.25)
    sleep.assert_called_once_with(0.25)


def test_fullscreen_routes_splash_into_initial_app_output() -> None:
    from nooa_cli.tui.config import Config, DisplayMode
    from nooa_cli.tui.main import _prepare_splash
    from nooa_cli.tui.output import SplashScreen

    frontend = SimpleNamespace(raw_console=MagicMock())
    config = Config()
    config.tui.display_mode = DisplayMode.FULLSCREEN

    with patch("nooa_cli.tui.splash.show_splash") as show:
        outputs = _prepare_splash(config, frontend)

    assert outputs == [SplashScreen()]
    show.assert_not_called()


def test_native_mode_keeps_pre_application_splash() -> None:
    from nooa_cli.tui.config import Config, DisplayMode
    from nooa_cli.tui.main import _prepare_splash

    frontend = SimpleNamespace(raw_console=MagicMock())
    config = Config()
    config.tui.display_mode = DisplayMode.NATIVE

    with patch("nooa_cli.tui.splash.show_splash") as show:
        outputs = _prepare_splash(config, frontend)

    assert outputs == []
    show.assert_called_once_with(frontend.raw_console)


def test_fullscreen_splash_is_semantically_reflowable() -> None:
    from nooa_cli.tui.frontend import TerminalFrontend
    from nooa_cli.tui.output import SplashScreen
    from nooa_cli.tui.session import _EmitStream

    emitted = []
    current_width = 100

    def emit(text: str, replay=None) -> None:
        emitted.append((text, replay))

    stream = _EmitStream(
        emit,
        replay_width=lambda: current_width - 1,
        layout_width=lambda: current_width,
    )
    mock_console = MagicMock()
    mock_console.console.width = 120
    mock_console.console.file = stream
    frontend = TerminalFrontend.__new__(TerminalFrontend)
    frontend._console = mock_console

    frontend._render_splash(SplashScreen())

    assert len(emitted) == 1
    rendered, replay = emitted[0]
    assert "NVIDIA LABS" in rendered
    assert callable(replay)
    # The 80-column boundary must keep the wide variant rather than using
    # the native-scrollback width (79), which deliberately reserves a column.
    current_width = 80
    wide = replay()
    assert wide.count("\n") == 11

    current_width = 40
    compact = replay()
    assert "NVIDIA LABS" in compact
    assert compact != rendered
