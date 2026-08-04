# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Splash screen for the Nemotron Agents TUI.

A side-by-side lockup: the NVIDIA "eye" mark (green on its black square) next to
the ``NEMOTRON / OO / AGENTS`` wordmark, sized to fit an 80-column terminal.

The eye is half-block art (``█``/``▀``/``▄`` pack two pixel rows per line),
downsampled from the official logo. The wordmark uses a hand-built full-block
pixel font (each pixel is one solid ``█``) so the letters stay clean and chunky.
The logo and the text block are the same height and share top/bottom edges.
"""

import time

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

NVIDIA_GREEN = "#76b900"
BLACK = "#0a0a0a"
STEEL_LO = "#6e7d8c"

_EYE_W = 38  # columns
_WORD_W = 40  # columns

# fmt: off
# NVIDIA eye mark (green on black square). Sized to the wordmark's full height
# so the logo and the text block share the same top and bottom edges.
_EYE_LINES: list[str] = [
    "              ████████████████████████",
    "              ████████████████████████",
    "         ▄▄███      ▀▀████████████████",
    "     ▄▄██▀▀   ████▄▄    ▀█████████████",
    "  ▄▄██▀    ▄▄▄   ▀▀███▄   ▀███████████",
    "▄███▀   ▄██▀▀ ▄▄     ▀██▄   ▀█████████",
    "▀███   ███    ███▄  ▄███    ██████████",
    " ▀███   ██▄   ████████▀   ▄███▀▀██████",
    "  ▀███   ▀██▄ █████▀▀  ▄▄███▀    ▀▀███",
    "    ▀██▄   ▀██      ▄▄███▀       ▄████",
    "      ▀██▄▄   ███████▀▀      ▄▄███████",
    "        ▀▀████          ▄▄▄███████████",
    "              ▄▄▄▄▄███████████████████",
    "              ████████████████████████",
]

# NEMOTRON / OO / AGENTS wordmark. The OO echoes the object-oriented heritage
# and the eye mark.
_WORD_LINES: list[str] = [
    "█  █ ████ █   █ ████ ████ ███  ████ █  █",
    "██ █ ██   ██ ██ █  █  ██  █  █ █  █ ██ █",
    "█ ██ █    █ █ █ █  █  ██  ███  █  █ █ ██",
    "█  █ ████ █   █ ████  ██  █ ██ ████ █  █",
    "                                        ",
    "               ████ ████                ",
    "               █  █ █  █                ",
    "               █  █ █  █                ",
    "               ████ ████                ",
    "                                        ",
    "     ████ ████ ████ █  █ ████ ████      ",
    "     █  █ █    ██   ██ █  ██  ██        ",
    "     ████ █ ██ █    █ ██  ██    ██      ",
    "     █  █ ████ ████ █  █  ██  ████      ",
]
# fmt: on

# Plain-text marker kept for tests / log scraping.
NEMOTRON_ASCII = "NEMOTRON OO AGENTS"
NEMO_OO_ASCII = NEMOTRON_ASCII  # backwards-compatible alias


def _render_lockup() -> Text:
    """Render the eye + wordmark side by side as one styled block."""
    text = Text(justify="left", no_wrap=True)
    eye_style = f"{NVIDIA_GREEN} on {BLACK}"
    word_style = NVIDIA_GREEN
    for i, eye in enumerate(_EYE_LINES):
        text.append(eye.ljust(_EYE_W), style=eye_style)
        text.append("  ")
        text.append(_WORD_LINES[i].ljust(_WORD_W), style=word_style)
        if i < len(_EYE_LINES) - 1:
            text.append("\n")
    return text


def show_splash(console: Console, delay: float = 0.8) -> None:
    """Display the Nemotron Agents splash screen.

    The splash stays on screen and scrolls off naturally as the user chats.

    Args:
        console: Rich console instance
        delay: How long to pause after showing splash (seconds)
    """
    tagline = Align.center(
        Text("object-oriented agents · licensed to vibe", style=f"italic {STEEL_LO}")
    )

    panel = Panel(
        Group(Align.center(_render_lockup()), Text(""), tagline),
        border_style=NVIDIA_GREEN,
        padding=(1, 2),
        expand=False,
    )

    console.print(Align.center(panel))
    time.sleep(delay)
