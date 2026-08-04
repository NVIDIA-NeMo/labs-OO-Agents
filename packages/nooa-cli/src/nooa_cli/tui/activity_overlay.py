# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-screen in-app overlay for the /activity command."""

from __future__ import annotations

import io
import textwrap
from collections.abc import Iterable

from rich.console import Console as RichConsole
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .output import CodeExecution, Output, TableOutput, TextOutput
from .subapp import SubviewKeyResult
from .theme import COLORS

_BAR_STYLE = "\x1b[48;5;236;38;5;252m"


def _style_bar(text: str, *, ansi: bool) -> str:
    if not ansi:
        return text
    return f"{_BAR_STYLE}{text}\x1b[0m"


def _plain_wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        if raw == "":
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(
                    raw,
                    width=max(width, 1),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
    return lines or [""]


def _render_rich_lines(renderables: Iterable[object], width: int) -> list[str]:
    render_width = max(int(width), 20)
    buf = io.StringIO()
    console = RichConsole(
        file=buf,
        force_terminal=True,
        color_system="256",
        width=render_width,
        _environ={"COLUMNS": str(render_width), "LINES": "25"},
    )
    for renderable in renderables:
        console.print(renderable)
    return buf.getvalue().splitlines() or [""]


def _table_lines(output: TableOutput, width: int, *, ansi: bool) -> list[str]:
    if ansi:
        table = Table(
            title=output.title or None,
            show_header=output.show_header,
            box=None,
            expand=True,
        )
        for column in output.columns:
            table.add_column(column)
        for row in output.rows:
            table.add_row(*[str(cell) for cell in row])
        lines = _render_rich_lines([table], width)
        if output.footer:
            lines.append("")
            lines.extend(_render_rich_lines([Text(output.footer, style=COLORS["subtext1"])], width))
        return lines

    rows = output.rows
    if not rows:
        return []
    left_width = min(max(len(str(row[0])) for row in rows), max(width // 3, 12))
    lines: list[str] = []
    if output.title:
        lines.append(output.title)
        lines.append("")
    if output.show_header and output.columns:
        lines.append(
            f"{output.columns[0]:<{left_width}}  {output.columns[1] if len(output.columns) > 1 else ''}"
        )
    for row in rows:
        left = str(row[0]) if row else ""
        right = str(row[1]) if len(row) > 1 else ""
        prefix = f"{left:<{left_width}}  "
        wrapped = _plain_wrap(right, max(width - len(prefix), 1))
        lines.append(prefix + wrapped[0])
        for extra in wrapped[1:]:
            lines.append(" " * len(prefix) + extra)
    if output.footer:
        lines.append("")
        lines.extend(_plain_wrap(output.footer, width))
    return lines


def _code_lines(output: CodeExecution, width: int, *, ansi: bool) -> list[str]:
    lines: list[str] = []
    if output.code:
        code = output.code.rstrip() if output.start_line > 1 else output.code.strip()
        if ansi:
            highlight = {output.highlight_line} if output.highlight_line is not None else None
            lines.extend(
                _render_rich_lines(
                    [
                        Syntax(
                            code,
                            "python",
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True,
                            start_line=output.start_line,
                            highlight_lines=highlight,
                        )
                    ],
                    width,
                )
            )
        else:
            lines.extend(_plain_wrap(code, width))
    output_parts = [
        part for part in (output.stdout, output.stderr, output.error, output.value) if part
    ]
    if output_parts:
        if lines:
            lines.append("")
        for part in output_parts:
            lines.extend(_plain_wrap(str(part), width))
    return lines or ["(no source available)"]


def _output_lines(output: Output, width: int, *, ansi: bool) -> list[str]:
    if isinstance(output, TextOutput):
        return _plain_wrap(output.content, width)
    if isinstance(output, TableOutput):
        return _table_lines(output, width, ansi=ansi)
    if isinstance(output, CodeExecution):
        return _code_lines(output, width, ansi=ansi)
    return _plain_wrap(str(output), width)


class ActivityOverlayView:
    """Static activity snapshot shown as a full-screen in-app overlay."""

    title = "activity"

    def __init__(self, outputs: list[Output]) -> None:
        self.outputs = outputs
        self.offset = 0
        self._last_line_count = 0
        self._last_visible_lines = 0

    def render(self, width: int, height: int) -> str:
        return render_activity_overlay(self, width, height, ansi=True)

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action in {"quit", "escape", "enter"}:
            return "close"
        if action in {"down", "j", "scroll_down"}:
            self.scroll(+3 if action == "scroll_down" else +1)
            return "handled"
        if action in {"up", "k", "scroll_up"}:
            self.scroll(-3 if action == "scroll_up" else -1)
            return "handled"
        if action == "page_down":
            self.scroll(max(self._last_visible_lines, 1))
            return "handled"
        if action == "page_up":
            self.scroll(-max(self._last_visible_lines, 1))
            return "handled"
        if action == "home":
            self.offset = 0
            return "handled"
        if action == "end":
            self.offset = max(self._last_line_count - max(self._last_visible_lines, 1), 0)
            return "handled"
        return "handled" if action == "text" else "ignored"

    def scroll(self, delta: int) -> None:
        max_offset = max(self._last_line_count - max(self._last_visible_lines, 1), 0)
        self.offset = min(max(self.offset + delta, 0), max_offset)

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass


def render_activity_overlay(
    view: ActivityOverlayView, width: int, height: int, *, ansi: bool = False
) -> str:
    width = max(int(width), 40)
    height = max(int(height), 1)
    header = _style_bar(" Activity ".ljust(width, "─")[:width], ansi=ansi)
    footer = _style_bar(
        " ↑/↓ scroll  PgUp/PgDn page  Home/End  Enter/Esc/q close ".ljust(width, "─")[:width],
        ansi=ansi,
    )
    body_height = max(height - 2, 0)
    body: list[str] = []
    for i, output in enumerate(view.outputs):
        if i:
            body.append("")
        body.extend(_output_lines(output, width, ansi=ansi))
    if not body:
        body = ["No activity."]
    view._last_line_count = len(body)
    view._last_visible_lines = body_height
    view.scroll(0)
    visible = body[view.offset : view.offset + body_height]
    visible = [line if ansi and "\x1b[" in line else line.ljust(width)[:width] for line in visible]
    while len(visible) < body_height:
        visible.append("".ljust(width))
    return "\n".join([header, *visible, footer])
