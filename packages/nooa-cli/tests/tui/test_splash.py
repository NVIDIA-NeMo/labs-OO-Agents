# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responsive NOOA splash rendering."""

from __future__ import annotations

import io
from unittest.mock import patch

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
    [(120, "wide"), (80, "wide"), (79, "standard"), (56, "standard"), (55, "compact")],
)
def test_splash_selects_responsive_variant(width: int, expected: str) -> None:
    assert splash_variant(width) == expected


@pytest.mark.parametrize("width", [32, 48, 56, 64, 72, 100, 160])
def test_splash_never_exceeds_terminal_width(width: int) -> None:
    rendered = _render(width)
    assert rendered
    assert max(len(line) for line in rendered.splitlines()) <= width


def test_compact_splash_keeps_full_identity_readable() -> None:
    rendered = _render(48)
    assert "NVIDIA LABS · NOOA" in rendered
    assert "Object Oriented Agents" in rendered


def test_graphical_splash_uses_new_brand_not_legacy_copy() -> None:
    rendered = _render(100)
    assert "NVIDIA LABS" in rendered
    assert "OBJECT ORIENTED AGENTS" in rendered
    assert "NEMOTRON" not in rendered
    assert "licensed to vibe" not in rendered
    assert NOOA_ASCII == "NVIDIA LABS OBJECT ORIENTED AGENTS (NOOA)"


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
