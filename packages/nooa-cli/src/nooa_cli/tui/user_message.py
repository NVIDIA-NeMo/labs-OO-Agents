# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering for live and resumed user-message bars."""

from rich.cells import chop_cells, set_cell_size

from .terminal_safety import sanitize_live_text


def _hex_to_ansi256(hex_color: str) -> int:
    """Map an RGB hex color to the nearest xterm-256 color-cube entry."""
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))

    def quantize(channel: int) -> int:
        if channel < 48:
            return 0
        if channel < 115:
            return 1
        return (channel - 35) // 40

    return 16 + 36 * quantize(red) + 6 * quantize(green) + quantize(blue)


def render_user_bar(text: str, cols: int, colors: dict[str, str]) -> str:
    """Render one live or resumed user message as full-width highlighted rows."""
    width = max(int(cols), 1)
    foreground = _hex_to_ansi256(colors["text"])
    background = _hex_to_ansi256(colors["surface2"])
    style_on = f"\x1b[38;5;{foreground};48;5;{background}m"
    style_off = "\x1b[0m"

    rows: list[str] = []
    for index, line in enumerate(sanitize_live_text(text).split("\n")):
        shown = f" ❯ {line} " if index == 0 else f" {line} "
        # Rich's cell helpers preserve grapheme clusters and measure wide
        # glyphs correctly. Every row is exactly the safe content width.
        chunks = chop_cells(shown, width) or [""]
        rows.extend(f"{style_on}{set_cell_size(chunk, width)}{style_off}" for chunk in chunks)
    return "\n".join(rows) + "\n"


__all__ = ["render_user_bar"]
