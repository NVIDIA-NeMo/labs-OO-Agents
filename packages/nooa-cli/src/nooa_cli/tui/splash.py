# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responsive splash for NVIDIA Labs Object Oriented Agents (NOOA).

The NVIDIA eye artwork is checked in as terminal-native half blocks. It was
generated from the ``Eye_Mark`` path in the official NVIDIA SVG served from
``nvidia.com/content/dam/en-zz/Solutions/about-nvidia/nvidia-brochure/images/``
``nvidia-logo-black.svg``, then hand-tuned at each terminal resolution.

Nothing is downloaded or rasterized at runtime. Wide and standard terminals
get an eye + NOOA lockup; narrow terminals get a readable text identity.
"""

from __future__ import annotations

import time
from typing import Literal

from rich.align import Align
from rich.console import Console, Group
from rich.text import Text

from .theme import COLORS

NVIDIA_GREEN = "#76b900"

NOOA_ASCII = "NVIDIA LABS OBJECT ORIENTED AGENTS (NOOA)"

_WIDE_MIN_WIDTH = 80
_STANDARD_MIN_WIDTH = 56
_LOCKUP_GAP = 3

# fmt: off
# Generated from the official NVIDIA eye at 38 columns × 28 half-cell pixels.
_WIDE_EYE: tuple[str, ...] = (
    "              ████████████████████████",
    "              ████████████████████████",
    "        ▄▄████     ▀▀▀████████████████",
    "     ▄███▀▀   █████▄▄   ▀█████████████",
    "  ▄███▀   ▄▄▄█▀ ▀▀▀███▄   ▀███████████",
    "▄███▀  ▄▄██▀▀▀█▄▄   ▀███▄   ▀█████████",
    "▀███▄  ███    ███▄  ▄███▀  ▄██████████",
    " ▀███  ▀███   ████████▀  ▄████▀▀██████",
    "  ▀███▄  ███▄ ██████▀  ▄████▀    ▀▀███",
    "    ███▄  ▀▀██▀▀▀  ▄▄████▀▀      ▄████",
    "     ▀████▄   ████████▀▀     ▄▄███████",
    "        ▀▀████▀▀▀▀     ▄▄▄████████████",
    "            ▀▀▄▄▄▄████████████████████",
    "              ████████████████████████",
)

# Generated independently at 30 columns × 20 half-cell pixels rather than
# scaling terminal glyphs, which keeps the inner eye open at standard widths.
_STANDARD_EYE: tuple[str, ...] = (
    "           ███████████████████",
    "       ▄▄▄▄▀▀▀▀▀██████████████",
    "   ▄▄██▀▀▀ ████▄▄ ▀▀██████████",
    " ▄██▀  ▄▄██▄  ▀▀██▄  ▀████████",
    "███▄ ███▀  ██▄  ▄██▀  ▄███████",
    " ▀██▄ ▀██  ██████▀ ▄▄██▀▀▀████",
    "  ▀██▄ ▀▀█▄▀▀▀▀ ▄▄███▀    ▄███",
    "    ▀██▄▄  ██████▀▀   ▄▄▄█████",
    "       ▀▀▀█   ▄▄▄▄▄███████████",
    "           ███████████████████",
)

_NOOA_WORDMARK: tuple[str, ...] = (
    "▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄     ▄▄▄▄",
    "████▄  ███ ▄███████▄ ▄███████▄ ▄██▀▀██▄",
    "███▀██▄███ ███   ███ ███   ███ ███  ███",
    "███  ▀████ ███▄▄▄███ ███▄▄▄███ ███▀▀███",
    "███    ███  ▀█████▀   ▀█████▀  ███  ███",
)
# fmt: on

SplashVariant = Literal["wide", "standard", "compact"]


def splash_variant(width: int) -> SplashVariant:
    """Choose the richest lockup that fits without wrapping."""
    if width >= _WIDE_MIN_WIDTH:
        return "wide"
    if width >= _STANDARD_MIN_WIDTH:
        return "standard"
    return "compact"


def _right_lockup(wordmark: tuple[str, ...]) -> list[tuple[str, str]]:
    return [
        ("NVIDIA LABS", f"bold {COLORS['text']}"),
        ("", ""),
        *((line, NVIDIA_GREEN) for line in wordmark),
        ("", ""),
        ("OBJECT ORIENTED AGENTS", COLORS["subtext1"]),
    ]


def _render_lockup(eye: tuple[str, ...], wordmark: tuple[str, ...]) -> Text:
    """Compose one vertically centered eye + wordmark Rich text block."""
    right = _right_lockup(wordmark)
    height = max(len(eye), len(right))
    eye_pad = (height - len(eye)) // 2
    right_pad = (height - len(right)) // 2
    eye_lines = ["" for _ in range(eye_pad)] + list(eye)
    right_lines = [("", "") for _ in range(right_pad)] + right
    eye_lines.extend("" for _ in range(height - len(eye_lines)))
    right_lines.extend(("", "") for _ in range(height - len(right_lines)))

    eye_width = max(len(line) for line in eye)
    result = Text(no_wrap=True, overflow="crop")
    for index, (eye_line, (right_line, right_style)) in enumerate(
        zip(eye_lines, right_lines, strict=True)
    ):
        result.append(eye_line.ljust(eye_width), style=NVIDIA_GREEN)
        result.append(" " * _LOCKUP_GAP)
        result.append(right_line, style=right_style)
        if index < height - 1:
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

    if variant == "wide":
        lockup = _render_lockup(_WIDE_EYE, _NOOA_WORDMARK)
        return Group(blank, Align.center(lockup), blank)

    eye = Text("\n".join(_STANDARD_EYE), style=NVIDIA_GREEN, no_wrap=True)
    lab = Text("NVIDIA LABS", style=f"bold {COLORS['text']}")
    wordmark = Text("\n".join(_NOOA_WORDMARK), style=NVIDIA_GREEN, no_wrap=True)
    subtitle = Text("OBJECT ORIENTED AGENTS", style=COLORS["subtext1"])
    return Group(
        blank,
        Align.center(eye),
        Align.center(lab),
        Text(""),
        Align.center(wordmark),
        Align.center(subtitle),
        blank,
    )


def show_splash(console: Console, delay: float = 0.0) -> None:
    """Print the responsive NOOA splash without delaying normal startup."""
    console.print(build_splash(console.width))
    if delay > 0:
        time.sleep(delay)


__all__ = ["NOOA_ASCII", "build_splash", "show_splash", "splash_variant"]
