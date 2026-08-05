# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared explorer framework for in-app subviews.

Provides the common layout, navigation, search, and rendering primitives
shared across session, event, job, and todo explorers. Each concrete explorer
supplies:

- A row dataclass (what shows in the list)
- A detail renderer (what shows in the detail pane)
- Optional custom actions (e.g. resume session, cancel job, add comment)

Layout (all explorers):
    ┌─ header bar ─────────────────────────────────┐
    │ list pane (scrollable, searchable rows)       │
    ├─ divider (FTS prompt) ───────────────────────┤
    │ detail pane (scrollable detail for selection) │
    └─ footer bar (mode, keybindings) ─────────────┘
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .subapp import SubviewKeyResult

# ─── ANSI styling primitives ─────────────────────────────────────────────────

BAR_STYLE = "\x1b[48;5;236;38;5;252m"
FTS_ACTIVE_STYLE = "\x1b[1;38;2;255;255;255;48;2;95;28;75m"


def style_bar(text: str, *, ansi: bool) -> str:
    """Style a header/footer bar line."""
    if not ansi:
        return text
    return f"{BAR_STYLE}{text}\x1b[0m"


def style_mode_label(text: str, *, active: bool, ansi: bool) -> str:
    """Style the mode indicator (FTS/BROWSE)."""
    if not ansi:
        return text
    if active:
        return f"{FTS_ACTIVE_STYLE}{text}\x1b[0m"
    return f"\x1b[1;30;46m{text}\x1b[0m"


def style_fts_prompt(text: str, *, active: bool, ansi: bool) -> str:
    """Style the search prompt in the divider."""
    if not ansi or not active:
        return text
    return f"{FTS_ACTIVE_STYLE}{text}\x1b[0m"


# ─── Text utilities ──────────────────────────────────────────────────────────


def wrap_plain_line(line: str, width: int) -> list[str]:
    """Wrap a single line to *width*, preserving leading indent."""
    if line == "":
        return [""]
    indent_len = len(line) - len(line.lstrip(" "))
    subsequent = " " * min(indent_len, max(width - 1, 0))
    return textwrap.wrap(
        line,
        width=max(width, 1),
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
        subsequent_indent=subsequent,
    ) or [""]


def search_terms(query: str) -> list[str]:
    """Split a search query into non-empty terms."""
    return [term for term in query.split() if term.strip()]


def highlight_terms(text: str, terms: list[str], *, current: bool = False) -> str:
    """Highlight search terms in text using ANSI colors."""
    if not terms:
        return text
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE
    )
    color = "30;106" if current else "30;43"
    return pattern.sub(lambda m: f"\x1b[{color}m{m.group(0)}\x1b[0m", text)


def display_line(
    text: str, width: int, terms: list[str], *, ansi: bool, current_match: bool = False
) -> str:
    """Fit a plain line to width, then optionally highlight search terms."""
    plain = text.ljust(width)[:width]
    if ansi and terms:
        return highlight_terms(plain, terms, current=current_match)
    return plain


def detail_match_lines(lines: list[str], terms: list[str]) -> list[int]:
    """Return line indices that contain any search term."""
    if not terms:
        return []
    lowered = [t.lower() for t in terms]
    return [i for i, line in enumerate(lines) if any(t in line.lower() for t in lowered)]


def render_markdown_lines(markdown: str, width: int) -> list[str]:
    """Render markdown to ANSI lines via Rich, falling back to plain wrap."""
    try:
        import io

        from rich.console import Console as RichConsole
        from rich.markdown import Markdown

        render_width = max(int(width), 20)
        buf = io.StringIO()
        console = RichConsole(
            file=buf,
            force_terminal=True,
            color_system="256",
            width=render_width,
            _environ={"COLUMNS": str(render_width), "LINES": "25"},
        )
        console.print(Markdown(markdown))
        return buf.getvalue().splitlines() or [""]
    except Exception:
        lines: list[str] = []
        for line in markdown.splitlines() or [""]:
            lines.extend(wrap_plain_line(line, width))
        return lines or [""]


# ─── Generic Explorer Model ──────────────────────────────────────────────────


class ExplorerModel:
    """Searchable, keyboard-navigable list with detail pane.

    Subclass or use directly — the model is generic over any row type.
    Rows must have a ``search_text: str`` attribute for FTS filtering.
    """

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.query = ""
        self.matches = list(range(len(rows)))
        self.cursor = 0
        self.detail_offset = 0
        self.focus = "list"
        self.search_active = False
        self.search_line_cursor = 0
        self._last_detail_line_count = 0
        self._last_detail_visible_lines = 0
        self._last_detail_match_lines: list[int] = []
        self._last_divider_y = 0

    @property
    def current_index(self) -> int | None:
        if not self.matches:
            return None
        self.cursor = min(max(self.cursor, 0), len(self.matches) - 1)
        return self.matches[self.cursor]

    @property
    def current(self) -> Any | None:
        idx = self.current_index
        return None if idx is None else self.rows[idx]

    def set_query(self, query: str) -> None:
        """Update search query and refilter matches."""
        self.query = query
        words = [w.lower() for w in query.split() if w.strip()]
        if not words:
            self.matches = list(range(len(self.rows)))
        else:
            self.matches = [
                i
                for i, row in enumerate(self.rows)
                if all(word in row.search_text.lower() for word in words)
            ]
        self.cursor = 0
        self.detail_offset = 0
        self.search_line_cursor = 0

    def edit_query(self, text: str) -> None:
        self.set_query(text)
        self.search_active = True

    def clear_query(self) -> None:
        self.set_query("")

    def move(self, delta: int) -> None:
        """Move cursor in the list."""
        if not self.matches:
            return
        self.cursor = min(max(self.cursor + delta, 0), len(self.matches) - 1)
        self.detail_offset = 0
        self.search_line_cursor = 0

    def move_or_scroll(self, delta: int) -> None:
        if self.search_active and self.focus != "list":
            self._move_detail_match(delta)
        elif self.focus == "list":
            self.move(delta)
        else:
            self.scroll_detail(delta)

    def _move_detail_match(self, delta: int) -> None:
        if not self._last_detail_match_lines:
            return
        count = len(self._last_detail_match_lines)
        self.search_line_cursor = min(max(self.search_line_cursor + delta, 0), count - 1)
        line = self._last_detail_match_lines[self.search_line_cursor]
        visible = max(self._last_detail_visible_lines, 1)
        self.detail_offset = max(line - visible // 2, 0)
        self.clamp_detail_offset(visible)

    def jump_home(self) -> None:
        self.cursor = 0
        self.detail_offset = 0
        self.search_line_cursor = 0

    def jump_end(self) -> None:
        if self.matches:
            self.cursor = len(self.matches) - 1
            self.detail_offset = 0
            self.search_line_cursor = 0

    def toggle_focus(self) -> None:
        self.focus = "detail" if self.focus == "list" else "list"

    def scroll_detail(self, delta: int) -> None:
        max_offset = max(self._last_detail_line_count - max(self._last_detail_visible_lines, 1), 0)
        self.detail_offset = min(max(self.detail_offset + delta, 0), max_offset)

    def page_detail(self, delta: int) -> None:
        self.scroll_detail(delta)

    def clamp_detail_offset(self, visible_lines: int) -> None:
        max_offset = max(self._last_detail_line_count - max(visible_lines, 1), 0)
        self.detail_offset = min(max(self.detail_offset, 0), max_offset)


# ─── Generic Explorer View ───────────────────────────────────────────────────


@dataclass
class ExplorerConfig:
    """Configuration for a concrete explorer instance.

    Attributes:
        title: Explorer title shown in header bar.
        detail_pane_name: Label for the detail focus (e.g. "dialog", "event text").
        empty_message: Shown when no rows exist.
        no_match_message: Template for no search results (gets .format(query=...)).
        list_ratio: Fraction of body height for the list pane (0.0-1.0).
        actions: Custom action names mapped to descriptions (for footer hints).
    """

    title: str = "Explorer"
    detail_pane_name: str = "detail"
    empty_message: str = "No items."
    no_match_message: str = "No matches for {query!r}."
    list_ratio: float = 0.33
    actions: dict[str, str] = field(default_factory=dict)


class ExplorerInteraction:
    """Shared mouse and native-selection behavior for explorer subviews."""

    detail_focus = "detail"
    native_selection = False

    @property
    def mouse_support(self) -> bool:
        """Disable terminal mouse reporting while native selection is active."""
        return not self.native_selection

    def handle_interaction_action(self, action: str) -> SubviewKeyResult:
        if action != "native_selection":
            return "ignored"
        self.native_selection = not self.native_selection
        return "handled"

    def handle_mouse(self, action: str, _x: int, y: int) -> SubviewKeyResult:
        """Route wheel input to the pane under the pointer."""
        if action not in {"scroll_up", "scroll_down"}:
            return "ignored"
        delta = -3 if action == "scroll_up" else 3
        if y <= getattr(self.model, "_last_divider_y", 0):
            self.model.focus = "list"
            self.model.move_or_scroll(delta)
        else:
            self.model.focus = self.detail_focus
            self.model.scroll_detail(delta)
        return "handled"

    def selection_hint(self) -> str:
        return "F2 mouse/wheel" if self.native_selection else "F2 select/copy"


class ExplorerView(ExplorerInteraction):
    """Generic in-app explorer subview.

    Concrete explorers subclass this and provide:
    - ``config``: an ExplorerConfig
    - ``format_row(row, width)``: format a row for the list pane
    - ``detail_lines(row, width)``: render detail content as plain-text lines
    - Optionally override ``handle_action(action, row)`` for custom actions
    """

    title: str = "explorer"

    def __init__(self, model: ExplorerModel, config: ExplorerConfig) -> None:
        self.model = model
        self.config = config
        self.title = config.title
        self.pending_input: str | None = None

    def format_row(self, row: Any, width: int) -> str:
        """Format a single row for the list. Override in subclasses."""
        return str(row)[:width]

    def detail_lines(self, row: Any, width: int) -> list[str]:
        """Return detail lines for the selected row. Override in subclasses."""
        return [str(row)]

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        """Handle a custom action on the current row. Override for custom behavior."""
        return "ignored"

    def render(self, width: int, height: int) -> str:
        return render_explorer(self, width, height, ansi=True)

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        model = self.model
        interaction = self.handle_interaction_action(action)
        if interaction != "ignored":
            return interaction
        if action == "quit":
            if model.search_active:
                model.edit_query(model.query + "q")
                return "handled"
            return "close"
        if action == "resume":
            if model.search_active:
                model.edit_query(model.query + "r")
                return "handled"
            return self.handle_action("resume", model.current)
        if action == "escape":
            if model.search_active:
                model.search_active = False
            elif model.query:
                model.clear_query()
            else:
                return "ignored"
        elif action == "enter":
            if model.search_active:
                model.search_active = False
            else:
                result = self.handle_action("enter", model.current)
                if result != "ignored":
                    return result
        elif action == "slash":
            if model.search_active:
                model.edit_query(model.query + "/")
            else:
                model.search_active = True
        elif action == "backspace":
            if model.search_active:
                model.edit_query(model.query[:-1])
        elif action == "tab":
            model.toggle_focus()
        elif action in ("down", "j"):
            if action == "j" and model.search_active:
                model.edit_query(model.query + "j")
            else:
                model.move_or_scroll(+1)
        elif action in ("up", "k"):
            if action == "k" and model.search_active:
                model.edit_query(model.query + "k")
            else:
                model.move_or_scroll(-1)
        elif action == "page_down":
            if not model.search_active:
                model.page_detail(+10)
        elif action == "page_up":
            if not model.search_active:
                model.page_detail(-10)
        elif action == "home":
            if not model.search_active:
                model.jump_home()
        elif action == "end":
            if not model.search_active:
                model.jump_end()
        elif action == "scroll_down":
            model.focus = "detail"
            model.scroll_detail(+3)
        elif action == "scroll_up":
            model.focus = "detail"
            model.scroll_detail(-3)
        elif action == "text":
            if model.search_active and value and value.isprintable():
                model.edit_query(model.query + value)
            else:
                return self.handle_action(f"text:{value}", model.current)
        else:
            return self.handle_action(action, model.current)
        return "handled"

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass


# ─── Generic Explorer Renderer ───────────────────────────────────────────────


def render_explorer(view: ExplorerView, width: int, height: int, *, ansi: bool = True) -> str:
    """Render an explorer view to a string frame.

    This is the shared layout engine. Concrete explorers customize via
    their ExplorerConfig, format_row(), and detail_lines() methods.
    """
    width = max(int(width), 40)
    height = max(int(height), 1)
    model = view.model
    config = view.config

    row = model.current
    match_count = len(model.matches)
    total = len(model.rows)

    # ── Header ──
    pos = f" {model.cursor + 1}/{match_count}" if match_count else " 0/0"
    match_label = f" match {model.cursor + 1}/{match_count}" if model.query and match_count else ""
    query_display = f" search={model.query!r}" if model.query else ""
    header = style_bar(
        f" {config.title}{pos} of {total}{match_label}{query_display} ".ljust(width, "\u2500")[
            :width
        ],
        ansi=ansi,
    )

    # ── Footer ──
    pane_label = "list" if model.focus == "list" else config.detail_pane_name
    if model.search_active:
        mode_text = "FTS MODE"
        search_prompt = f"FTS: {model.query}"
        enter_hint = "enter exit FTS"
    else:
        mode_text = "BROWSE MODE"
        search_prompt = "/ FTS"
        enter_hint = ""
    focus_label = f"{mode_text} \u00b7 pane={pane_label}"
    footer_parts = [
        focus_label,
        "\u2191/\u2193 nav/scroll",
        view.selection_hint(),
        "tab switch pane",
        search_prompt,
        enter_hint,
        "esc clear",
        "q close",
    ]
    # Add custom action hints
    for _action_name, hint in config.actions.items():
        footer_parts.append(hint)

    footer_plain = (" " + "  ".join(part for part in footer_parts if part) + " ").ljust(
        width, "\u2500"
    )[:width]
    if ansi:
        before_mode, after_mode = footer_plain.split(mode_text, 1)
        styled_mode = style_mode_label(mode_text, active=model.search_active, ansi=True)
        footer = f"{BAR_STYLE}{before_mode}{styled_mode}{BAR_STYLE}{after_mode}\x1b[0m"
    else:
        footer = footer_plain

    # ── Body ──
    body_height = max(height - 2, 0)
    model._last_divider_y = 0

    if not total:
        body = [config.empty_message]
    elif row is None:
        body = [config.no_match_message.format(query=model.query)]
    else:
        # List pane
        list_count = max(3, min(int(body_height * config.list_ratio), body_height - 4))
        half = list_count // 2
        start = max(0, model.cursor - half)
        end = min(match_count, start + list_count)
        start = max(0, end - list_count)
        terms = search_terms(model.query)

        body = []
        for visible_i in range(start, end):
            row_i = model.matches[visible_i]
            item = model.rows[row_i]
            marker = (
                "\u276f"
                if visible_i == model.cursor and model.focus == "list"
                else "\u2022"
                if visible_i == model.cursor
                else " "
            )
            row_text = view.format_row(item, width - 2)
            line = f"{marker} {row_text}"
            body.append(display_line(line, width, terms, ansi=ansi))

        # Divider
        divider_label = (
            f"[FTS: {model.query}] " if model.search_active or model.query else "[/: FTS] "
        )
        divider_plain = divider_label.ljust(width, "\u2500")[:width]
        if ansi and (model.search_active or model.query):
            divider = style_fts_prompt(divider_label, active=model.search_active, ansi=True)
            divider += "\u2500" * max(width - len(divider_label), 0)
            body.append(divider)
        else:
            body.append(divider_plain)
        # Screen row of the divider: one header row plus the zero-based body
        # row. Because ``body`` already includes the divider, ``len(body)`` is
        # exactly that absolute row number.
        model._last_divider_y = len(body)

        # Detail pane
        available = max(body_height - len(body), 0)
        # Reserve 1 line for scroll indicator when detail overflows
        detail_lines_list = view.detail_lines(row, width)
        model._last_detail_line_count = len(detail_lines_list)
        needs_indicator = len(detail_lines_list) > available
        detail_visible = max(available - (1 if needs_indicator else 0), 0)
        model._last_detail_visible_lines = detail_visible

        # Update match lines for search navigation
        terms_for_detail = search_terms(model.query)
        model._last_detail_match_lines = detail_match_lines(detail_lines_list, terms_for_detail)

        # Auto-center on match when searching
        if model.search_active and model._last_detail_match_lines:
            count = len(model._last_detail_match_lines)
            model.search_line_cursor = min(max(model.search_line_cursor, 0), count - 1)
            line = model._last_detail_match_lines[model.search_line_cursor]
            visible = max(detail_visible, 1)
            if not (model.detail_offset <= line < model.detail_offset + visible):
                model.detail_offset = max(line - visible // 2, 0)

        model.clamp_detail_offset(detail_visible)
        visible_slice = detail_lines_list[
            model.detail_offset : model.detail_offset + detail_visible
        ]

        # Highlight search terms in detail
        if ansi and terms_for_detail:
            current_match_line = (
                model._last_detail_match_lines[model.search_line_cursor]
                if model._last_detail_match_lines
                else None
            )
            highlighted = []
            for i, dl in enumerate(visible_slice):
                abs_line = model.detail_offset + i
                is_current = abs_line == current_match_line
                highlighted.append(
                    display_line(dl, width, terms_for_detail, ansi=True, current_match=is_current)
                )
            visible_slice = highlighted

        # Scroll indicator
        if model._last_detail_line_count > detail_visible and detail_visible > 0:
            remaining = model._last_detail_line_count - model.detail_offset - detail_visible
            if remaining > 0:
                indicator = f"  \u2193 {remaining} more lines"
                visible_slice.append(indicator[:width])

        body.extend(visible_slice)

    # Pad to fill height
    while len(body) < body_height:
        body.append("")

    lines = [header] + body[:body_height] + [footer]
    return "\n".join(lines)
