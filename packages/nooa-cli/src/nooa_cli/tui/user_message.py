# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering for live and resumed user-message bars."""

from rich.cells import cell_len, split_graphemes

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
    """Render one live or resumed user message with highlighted breathing room."""
    width = max(int(cols), 1)
    foreground = _hex_to_ansi256(colors["text"])
    background = _hex_to_ansi256(colors["surface2"])
    style_on = f"\x1b[38;5;{foreground};48;5;{background}m"
    # Reverse the bar color over SGR's semantic default background. After
    # reversal, the glyph uses the terminal's real background while the rest
    # of each cell remains the user-bar color—without querying OSC 11.
    edge_style = f"\x1b[38;5;{background};49;7m"
    style_off = "\x1b[0m"

    inset = 1 if width >= 3 else 0
    inner_width = width - 2 * inset

    def styled_row(content: str = "") -> str:
        # Reserve one highlighted cell on both sides whenever the viewport can
        # still display content; pathological 1–2-cell terminals keep text.
        padding = " " * max(inner_width - cell_len(content), 0)
        edge = " " * inset
        return f"{style_on}{edge}{content}{padding}{edge}{style_off}"

    def edge_row(glyph: str) -> str:
        return f"{edge_style}{glyph * width}{style_off}"

    def take_chunk(value: str, available: int) -> tuple[str, str]:
        """Take one cell-bounded chunk, preferring a whitespace boundary."""
        if not value or available <= 0:
            return "", value
        spans, _ = split_graphemes(value)
        graphemes = [(value[start:stop], cells) for start, stop, cells in spans]
        used = 0
        stop = 0
        last_space = -1
        for index, (cluster, cells) in enumerate(graphemes):
            if stop and used + cells > available:
                break
            if not stop and cells > available:
                # Match fullscreen transcript behavior for a grapheme that is
                # physically wider than the entire available content row.
                remainder = "".join(item for item, _item_cells in graphemes[index + 1 :])
                return "…", remainder
            used += cells
            stop = index + 1
            if cluster.isspace():
                last_space = stop
        if stop < len(graphemes) and last_space > 0:
            stop = last_space
        chunk = "".join(cluster for cluster, _cells in graphemes[:stop])
        remainder = "".join(cluster for cluster, _cells in graphemes[stop:])
        return chunk, remainder

    # One-eighth block elements draw a subtle terminal-background edge while
    # preserving nearly a full row of highlighted breathing room. Consecutive
    # user messages therefore retain a narrow visual seam between their bars.
    rows: list[str] = [edge_row("▔")]
    first_visual_row = True
    for line in sanitize_live_text(text).split("\n"):
        remainder = line
        first_chunk = True
        while remainder or first_chunk:
            prefix = "❯ " if first_visual_row and inner_width >= 3 else ""
            available = max(inner_width - cell_len(prefix), 0)
            # The prompt makes row one narrower. If the first grapheme fits
            # an ordinary continuation row, move it intact before chunking so
            # it isn't replaced merely because the rest of its token is long.
            spans, _ = split_graphemes(remainder)
            first_cells = spans[0][2] if spans else 0
            first_token_cells = cell_len(remainder.partition(" ")[0])
            defer_first = (first_cells > available and first_cells <= inner_width) or (
                first_token_cells > available and first_token_cells <= inner_width
            )
            if prefix and defer_first:
                rows.append(styled_row(prefix))
                first_visual_row = False
                first_chunk = False
                continue
            chunk, remainder = take_chunk(remainder, available)
            rows.append(styled_row(f"{prefix}{chunk}"))
            first_visual_row = False
            first_chunk = False
    rows.append(edge_row("▁"))
    return "\n".join(rows) + "\n"


__all__ = ["render_user_bar"]
