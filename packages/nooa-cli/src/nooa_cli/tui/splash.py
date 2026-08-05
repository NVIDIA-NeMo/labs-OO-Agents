# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responsive splash for NVIDIA Labs Object Oriented Agents (NOOA)."""

from __future__ import annotations

import time
from typing import Literal

from rich.align import Align
from rich.console import Console, Group
from rich.text import Text

from .theme import COLORS

NVIDIA_GREEN = "#76b900"
NVIDIA_GREEN_BRIGHT = "#b7f36b"

NOOA_ASCII = "NVIDIA LABS OBJECT ORIENTED AGENTS (NOOA)"

_WIDE_MIN_WIDTH = 80
_STANDARD_MIN_WIDTH = 48

# fmt: off
_NOOA_WORDMARK: tuple[str, ...] = (
    "████▀███▄ ▄███▀███▄ ▄███▀███▄ ████▀████",
    "████ ████ ████ ████ ████ ████ ████ ████",
    "████ ████ ████ ████ ████ ████ ████ ████",
    "████ ████ ████ ████ ████ ████ ████ ████",
    "████ ████ ▀███▄███▀ ▀███▄███▀ ████▀████",
)
# fmt: on

_WORDMARK_ACCENTS = frozenset({1, 11, 21, 31})

SplashVariant = Literal["wide", "standard", "compact"]


def splash_variant(width: int) -> SplashVariant:
    """Choose the richest lockup that fits without wrapping."""
    if width >= _WIDE_MIN_WIDTH:
        return "wide"
    if width >= _STANDARD_MIN_WIDTH:
        return "standard"
    return "compact"


def _wordmark_text() -> Text:
    """Build the two-tone mark without embedding terminal escape codes."""
    result = Text(no_wrap=True)
    for row, line in enumerate(_NOOA_WORDMARK):
        for column, character in enumerate(line):
            style = (
                NVIDIA_GREEN_BRIGHT
                if row in {1, 2, 3} and column in _WORDMARK_ACCENTS
                else NVIDIA_GREEN
            )
            result.append(character, style=style)
        if row < len(_NOOA_WORDMARK) - 1:
            result.append("\n")
    return result


def build_splash(width: int) -> Group:
    """Build the responsive splash renderable for a terminal width."""
    variant = splash_variant(width)
    blank = Text("")
    if variant == "compact":
        title = Text("NVIDIA LABS", style=f"bold {NVIDIA_GREEN}")
        title.append(" · ", style=COLORS["overlay1"])
        title.append("NOOA", style=f"bold {COLORS['text']}")
        subtitle = Text("Object Oriented Agents", style=COLORS["subtext1"])
        return Group(blank, Align.center(title), Align.center(subtitle), blank)

    lab = Text("NVIDIA LABS", style=f"bold {COLORS['text']}")
    wordmark = _wordmark_text()
    subtitle = Text("OBJECT ORIENTED AGENTS", style=COLORS["subtext1"])
    if variant == "wide":
        return Group(
            blank,
            Align.center(lab),
            blank,
            Align.center(wordmark),
            blank,
            Align.center(subtitle),
            blank,
        )
    return Group(blank, Align.center(lab), Align.center(wordmark), Align.center(subtitle), blank)


def show_splash(console: Console, delay: float = 0.0) -> None:
    """Print the responsive NOOA splash without delaying normal startup."""
    console.print(build_splash(console.width))
    if delay > 0:
        time.sleep(delay)


__all__ = ["NOOA_ASCII", "build_splash", "show_splash", "splash_variant"]
