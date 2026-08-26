# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering for live and resumed user-message bars."""

import unicodedata

from rich.cells import cell_len, split_graphemes

from .terminal_safety import sanitize_live_text


def _display_graphemes(value: str) -> list[tuple[str, int]]:
    """Return indivisible display clusters with nonzero terminal widths."""
    spans, _ = split_graphemes(value)
    raw: list[tuple[str, int]] = []
    for start, stop, cells in spans:
        cluster = value[start:stop]
        # A dangling joiner can combine with the row's layout padding after
        # wrapping, changing its measured width. Keep joiners inside valid
        # emoji clusters, but expose terminal-unstable trailing joiners.
        if cluster.endswith(("\u200c", "\u200d")):
            stable = cluster.rstrip("\u200c\u200d")
            replacements = "�" * (len(cluster) - len(stable))
            cluster = stable + replacements
            cells = cell_len(cluster)
        raw.append((cluster, cells))
    clusters: list[tuple[str, int]] = []
    index = 0
    while index < len(raw):
        cluster, cells = raw[index]
        # Rich intentionally separates regional indicators. Terminals render
        # a pair as one flag, so keep adjacent indicators in the same atom.
        if (
            len(cluster) == 1
            and 0x1F1E6 <= ord(cluster) <= 0x1F1FF
            and index + 1 < len(raw)
            and len(raw[index + 1][0]) == 1
            and 0x1F1E6 <= ord(raw[index + 1][0]) <= 0x1F1FF
        ):
            next_cluster, next_cells = raw[index + 1]
            clusters.append((cluster + next_cluster, cells + next_cells))
            index += 2
            continue
        # Standalone format controls may combine with prefixes or padding
        # after layout and change a completed row's measured width. Expose
        # those controls, while preserving whitespace bundled into Rich's
        # zero-cell span as its own breakable cluster.
        if cells <= 0 and any(unicodedata.category(char) == "Cf" for char in cluster):
            for char in cluster:
                visible = "�" if unicodedata.category(char) == "Cf" else char
                clusters.append((visible, max(cell_len(visible), 1)))
            index += 1
            continue
        # Attach a leading combining mark to the following base. A standalone
        # zero-cell cluster has no stable presentation, so expose it instead.
        if cells <= 0:
            if index + 1 < len(raw) and not raw[index + 1][0].isspace():
                next_cluster, next_cells = raw[index + 1]
                clusters.append((cluster + next_cluster, max(next_cells, 1)))
                index += 2
                continue
            clusters.append(("�", 1))
            index += 1
            continue
        clusters.append((cluster, cells))
        index += 1
    return clusters


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

    def take_chunk(graphemes: list[tuple[str, int]], start: int, available: int) -> tuple[str, int]:
        """Take one cell-bounded chunk, preferring a whitespace boundary."""
        if start >= len(graphemes) or available <= 0:
            return "", start
        used = 0
        stop = start
        last_space = -1
        for index in range(start, len(graphemes)):
            cluster, cells = graphemes[index]
            if stop > start and used + cells > available:
                break
            if stop == start and cells > available:
                # Match fullscreen transcript behavior for a grapheme that is
                # physically wider than the entire available content row.
                return "…", start + 1
            used += cells
            stop = index + 1
            if cluster.isspace():
                last_space = stop
        if stop < len(graphemes) and last_space > start:
            stop = last_space
        return "".join(cluster for cluster, _cells in graphemes[start:stop]), stop

    # One-eighth block elements draw a subtle terminal-background edge while
    # preserving nearly a full row of highlighted breathing room. Consecutive
    # user messages therefore retain a narrow visual seam between their bars.
    rows: list[str] = [edge_row("▔")]
    first_visual_row = True
    for line in sanitize_live_text(text).split("\n"):
        graphemes = _display_graphemes(line)
        # Cache each non-whitespace suffix's token width. This keeps prompt
        # deferral linear even for very large pasted, unbreakable tokens.
        token_cells = [0] * (len(graphemes) + 1)
        for index in range(len(graphemes) - 1, -1, -1):
            cluster, cells = graphemes[index]
            if not cluster.isspace():
                token_cells[index] = cells + token_cells[index + 1]
        start = 0
        first_chunk = True
        while start < len(graphemes) or first_chunk:
            prefix = "❯ " if first_visual_row and inner_width >= 3 else ""
            available = max(inner_width - cell_len(prefix), 0)
            # The prompt makes row one narrower. If the first grapheme or word
            # fits an ordinary continuation row, move it intact before
            # chunking so the prompt alone never forces a split/replacement.
            first_cells = graphemes[start][1] if start < len(graphemes) else 0
            first_token_cells = token_cells[start]
            defer_first = (first_cells > available and first_cells <= inner_width) or (
                first_token_cells > available and first_token_cells <= inner_width
            )
            if prefix and defer_first:
                rows.append(styled_row(prefix))
                first_visual_row = False
                first_chunk = False
                continue
            chunk, start = take_chunk(graphemes, start, available)
            rows.append(styled_row(f"{prefix}{chunk}"))
            first_visual_row = False
            first_chunk = False
    rows.append(edge_row("▁"))
    return "\n".join(rows) + "\n"


__all__ = ["render_user_bar"]
