# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic text snapshot of the resume picker for unit tests.

Migrated from ``nooa_cli.tui.resume_picker``: the renderer has no production
callers (the picker renders through the shared browser), so it lives here as a
test fixture. It snapshots the same model state the browser renders.
"""

from __future__ import annotations

from nooa_cli.tui.resume_picker import (
    ResumePickerModel,
    ResumePickerRow,
    _clip,
    _row_fragments,
    _single_line,
)
from nooa_cli.tui.terminal_safety import sanitize_live_text
from rich.cells import cell_len, split_graphemes


def _wrap(text: str, width: int) -> list[str]:
    """Wrap sanitized text by terminal cells while preserving grapheme clusters."""
    width = max(1, width)
    result: list[str] = []
    for source in sanitize_live_text(text).splitlines() or [""]:
        line, used = [], 0
        for start, stop, cells in split_graphemes(source)[0]:
            cluster = source[start:stop]
            if line and used + cells > width:
                result.append("".join(line))
                line, used = [], 0
            line.append(cluster)
            used += cells
        result.append("".join(line))
    return result


def _preview_lines(row: ResumePickerRow | None, width: int) -> list[list[tuple[str, str]]]:
    """Render transcript turns using the same visual language as live scrollback."""
    if row is None:
        return [[("class:fullscreen-browser.empty", "No session selected")]]
    if not row.turns:
        return [[("class:fullscreen-browser.empty", "No conversation preview")]]
    lines: list[list[tuple[str, str]]] = []
    width = max(1, width)
    for turn in row.turns:
        if turn.role == "user":
            edge = "▔" * width
            lines.append([("class:fullscreen-browser.preview-user-edge", edge)])
            for index, text in enumerate(_wrap(turn.content, max(1, width - 4))):
                prompt = "❯ " if index == 0 else "  "
                content = _clip(f" {prompt}{text}", width)
                lines.append(
                    [
                        (
                            "class:fullscreen-browser.preview-user",
                            content + " " * max(0, width - cell_len(content)),
                        )
                    ]
                )
            lines.append([("class:fullscreen-browser.preview-user-edge", "▁" * width)])
        else:
            lines.append([("class:fullscreen-browser.preview-agent", "OO:")])
            lines.extend(
                [("class:fullscreen-browser.preview", text)] for text in _wrap(turn.content, width)
            )
            lines.append([])
    if lines and not lines[-1]:
        lines.pop()
    return lines


def render_resume_picker(model: ResumePickerModel, width: int, height: int) -> str:
    """Render a deterministic text snapshot used by unit tests and narrow fallbacks."""
    if width < 48 or height < 13:
        return f"Terminal too small\nNeed 48 x 13; now {width} x {height}"
    separator = "─" * width
    filt = {
        "detached": "✓ Not attached",
        "attached": "✗ Attached",
        "all": "✓/✗ All",
    }[model.state_filter]
    sort = "Recent activity" if model.sort_updated else "Creation date"
    lines = [
        _clip(f"Resume a previous session · {len(model.matches)} sessions", width),
        _clip(f"[Search: {model.query}] [Filter: {filt}] [Sort: {sort}]", width),
        _clip(
            "Tab/Shift-Tab focus · arrows navigate · Space/↵ activate · Esc cancel",
            width,
        ),
        separator,
    ]
    list_height = min(5, max(1, height - 10))
    model.ensure_selection_visible(list_height)
    for index, match in model.visible(list_height):
        lines.extend(
            "".join(text for _, text in row)
            for row in _row_fragments(
                match, index == model.selected, width, sort_updated=model.sort_updated
            )
        )
    lines.extend(
        [
            separator,
            _clip(
                f"Preview · {_single_line(model.current.title) if model.current else 'No selection'}",
                width,
            ),
        ]
    )
    # The live browser keeps the preview viewport on the transcript; the
    # snapshot renders from the start (deterministic for tests).
    preview = _preview_lines(model.current, width)
    preview_height = max(1, height - len(lines) - 2)
    lines.extend(
        "".join(text for _, text in line) for line in preview[:preview_height]
    )
    lines.extend(
        [
            separator,
        ]
    )
    return "\n".join(lines[:height])
