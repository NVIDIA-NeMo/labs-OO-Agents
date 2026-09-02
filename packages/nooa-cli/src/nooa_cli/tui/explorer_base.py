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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .subapp import SubviewKeyResult
from .theme import COLORS, create_theme, get_syntax_theme

# ─── ANSI styling primitives ─────────────────────────────────────────────────


def _rgb(hex_color: str) -> str:
    value = hex_color.lstrip("#")
    return ";".join(str(int(value[index : index + 2], 16)) for index in (0, 2, 4))


def bar_style_code() -> str:
    """Return explorer chrome colors for the active TUI theme."""
    return f"\x1b[48;2;{_rgb(COLORS['surface0'])};38;2;{_rgb(COLORS['text'])}m"


def highlight_style_code(*, current: bool = False) -> str:
    """Return search-match colors for the active TUI theme."""
    background = COLORS["sky"] if current else COLORS["yellow"]
    return f"\x1b[38;2;{_rgb(COLORS['crust'])};48;2;{_rgb(background)}m"


def style_bar(text: str, *, ansi: bool) -> str:
    """Style a header/footer bar line using the active TUI theme."""
    if not ansi:
        return text
    return f"{bar_style_code()}{text}\x1b[0m"


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


def matches_all_terms(terms: list[str], text: str) -> bool:
    """Shared word-AND search contract for every fullscreen browser.

    A query matches when *every* whitespace-separated term occurs
    (case-insensitively) somewhere in the searchable text. All explorers and
    the resume picker use this rule so search behaves identically everywhere.
    """
    if not terms:
        return True
    folded = text.casefold()
    return all(term.casefold() in folded for term in terms)


def highlight_terms(text: str, terms: list[str], *, current: bool = False) -> str:
    """Highlight search terms in text using ANSI colors."""
    if not terms:
        return text
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE
    )
    style = highlight_style_code(current=current)
    return pattern.sub(lambda match: f"{style}{match.group(0)}\x1b[0m", text)


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
            theme=create_theme(),
            width=render_width,
            _environ={"COLUMNS": str(render_width), "LINES": "25"},
        )
        console.print(Markdown(markdown, code_theme=get_syntax_theme()))
        return buf.getvalue().splitlines() or [""]
    except Exception:
        lines: list[str] = []
        for line in markdown.splitlines() or [""]:
            lines.extend(wrap_plain_line(line, width))
        return lines or [""]


@dataclass
class ExplorerOption:
    """One choice row in the shared transient explorer options mode."""

    key: str
    label: str
    choices: tuple[tuple[str, str], ...]
    value: str
    on_change: Callable[[str], None]

    @property
    def display_value(self) -> str:
        return dict(self.choices).get(self.value, self.value)

    def move(self, delta: int) -> None:
        if not self.choices or not delta:
            return
        values = [value for value, _label in self.choices]
        self.value = values[(values.index(self.value) + delta) % len(values)]
        self.on_change(self.value)

    def activate(self) -> None:
        self.move(1)


@dataclass
class ExplorerChecklistOption:
    """Multi-select dropdown option rendered with checkboxes."""

    key: str
    label: str
    choices: tuple[tuple[str, str], ...]
    checked: set[str]
    on_change: Callable[[set[str]], None]
    choice_cursor: int = 0
    dropdown: bool = True
    multi_select: bool = True
    all_value: str | None = None

    @property
    def selectable_values(self) -> set[str]:
        return {value for value, _label in self.choices if value != self.all_value}

    def is_checked(self, value: str) -> bool:
        if value == self.all_value:
            values = self.selectable_values
            return bool(values) and values <= self.checked
        return value in self.checked

    @property
    def display_value(self) -> str:
        values = self.selectable_values
        if not values:
            return "None"
        if values <= self.checked:
            return "All"
        return f"{len(self.checked & values)}/{len(values)}"

    def move(self, delta: int) -> None:
        if self.choices and delta:
            self.choice_cursor = (self.choice_cursor + delta) % len(self.choices)

    def activate(self) -> None:
        if not self.choices:
            return
        value = self.choices[self.choice_cursor][0]
        if value == self.all_value:
            values = self.selectable_values
            self.checked = set() if values <= self.checked else set(values)
        elif value in self.checked:
            self.checked.remove(value)
        else:
            self.checked.add(value)
        self.on_change(set(self.checked))


ExplorerOptionItem = ExplorerOption | ExplorerChecklistOption


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
        self._filter_predicate: Callable[[Any], bool] = lambda _row: True
        self._sort_key: Callable[[Any], Any] | None = None
        self._sort_reverse = False

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
        terms = [w for w in query.split() if w.strip()]
        self.matches = [
            i
            for i, row in enumerate(self.rows)
            if self._filter_predicate(row)
            and matches_all_terms(terms, row.search_text)
        ]
        if self._sort_key is not None:
            self.matches.sort(
                key=lambda index: self._sort_key(self.rows[index]),
                reverse=self._sort_reverse,
            )
        self.cursor = 0
        self.detail_offset = 0
        self.search_line_cursor = 0
        self._last_detail_match_lines = []

    def set_view(
        self,
        *,
        predicate: Callable[[Any], bool] | None = None,
        sort_key: Callable[[Any], Any] | None = None,
        reverse: bool = False,
    ) -> None:
        """Apply non-search filtering and ordering, then rebuild matches."""
        current = self.current
        self._filter_predicate = predicate or (lambda _row: True)
        self._sort_key = sort_key
        self._sort_reverse = reverse
        self.set_query(self.query)
        if current is not None:
            current_index = next(
                (i for i, row_index in enumerate(self.matches) if self.rows[row_index] is current),
                None,
            )
            if current_index is not None:
                self.cursor = current_index

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
        """Shared navigation contract: list focus moves rows; detail scrolls.

        Preview-match stepping is owned by the browser's transcript search
        (like /resume), not by the model.
        """
        if self.focus == "list":
            self.move(delta)
        else:
            self.scroll_detail(delta)

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
        actions: Custom action names mapped to descriptions (for footer hints).
    """

    title: str = "Explorer"
    actions: dict[str, str] = field(default_factory=dict)


class ExplorerInteraction:
    """Shared model/options configuration for full-screen browser views."""

    use_fullscreen_browser = True
    detail_focus = "detail"
    native_selection = False
    options: tuple[ExplorerOptionItem, ...] = ()
    option_cursor: int | None = None
    _options_y = 1
    _option_hit_boxes: list[tuple[int, int]] = []

    def configure_options(self, *options: ExplorerOptionItem) -> None:
        self.options = tuple(options)
        self.option_cursor = None

    @property
    def options_active(self) -> bool:
        return self.option_cursor is not None

    def toggle_options(self) -> None:
        if not self.options:
            return
        self.model.search_active = False
        self.option_cursor = 0 if self.option_cursor is None else None

    def close_options(self) -> bool:
        if self.option_cursor is None:
            return False
        self.option_cursor = None
        return True

    def handle_options_action(self, action: str) -> SubviewKeyResult:
        if action == "options":
            self.toggle_options()
            return "handled"
        if self.option_cursor is None:
            if action == "space" and self.model.search_active:
                self.model.edit_query(self.model.query + " ")
                return "handled"
            return "ignored"
        if action == "quit":
            return "ignored"
        if action in {"escape", "enter"}:
            self.option_cursor = None
        elif action == "left":
            self.option_cursor = (self.option_cursor - 1) % len(self.options)
        elif action == "right":
            self.option_cursor = (self.option_cursor + 1) % len(self.options)
        elif action == "up":
            self.options[self.option_cursor].move(-1)
        elif action == "down":
            self.options[self.option_cursor].move(1)
        elif action == "space":
            self.options[self.option_cursor].activate()
        elif action == "tab":
            self.option_cursor = None
            return "ignored"
        return "handled"

    @property
    def mouse_support(self) -> bool:
        """Disable terminal mouse reporting while native selection is active."""
        return not self.native_selection

    def handle_interaction_action(self, action: str) -> SubviewKeyResult:
        option_result = self.handle_options_action(action)
        if option_result != "ignored":
            return option_result
        if action != "native_selection":
            return "ignored"
        self.native_selection = not self.native_selection
        return "handled"

    def handle_mouse(self, action: str, _x: int, y: int) -> SubviewKeyResult:
        """Route wheel input to the pane under the pointer."""
        if action == "click":
            if y != self._options_y:
                return "ignored"
            for index, (start, stop) in enumerate(self._option_hit_boxes):
                if start <= _x < stop:
                    self.option_cursor = index
                    self.options[index].activate()
                    self.option_cursor = None
                    return "handled"
            return "ignored"
        if action not in {"scroll_up", "scroll_down"}:
            return "ignored"
        if self.options_active:
            return "handled"
        delta = -3 if action == "scroll_up" else 3
        if y <= getattr(self.model, "_last_divider_y", 0):
            self.model.focus = "list"
            self.model.move_or_scroll(delta)
        else:
            self.model.focus = self.detail_focus
            self.model.scroll_detail(delta)
        return "handled"

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

    def configure_row_options(
        self,
        *,
        filters: tuple[tuple[str, str, Callable[[Any], bool]], ...],
        sorts: tuple[tuple[str, str, Callable[[Any], Any], bool], ...],
    ) -> None:
        """Attach the standard grouped Filter/Sort controls to this explorer."""
        filter_map = {value: predicate for value, _label, predicate in filters}
        sort_map = {value: (key, reverse) for value, _label, key, reverse in sorts}

        def apply_filter(value: str) -> None:
            key, reverse = sort_map[sort_option.value]
            self.model.set_view(predicate=filter_map[value], sort_key=key, reverse=reverse)

        def apply_sort(value: str) -> None:
            key, reverse = sort_map[value]
            self.model.set_view(
                predicate=filter_map[filter_option.value], sort_key=key, reverse=reverse
            )

        filter_option = ExplorerOption(
            "filter",
            "Filter",
            tuple((value, label) for value, label, _predicate in filters),
            filters[0][0],
            apply_filter,
        )
        sort_option = ExplorerOption(
            "sort",
            "Sort",
            tuple((value, label) for value, label, _key, _reverse in sorts),
            sorts[0][0],
            apply_sort,
        )
        self.configure_options(filter_option, sort_option)
        apply_filter(filter_option.value)

    def format_row(self, row: Any, width: int) -> str:
        """Format a single row for the list. Override in subclasses."""
        return str(row)[:width]

    # Views that embed their own search highlighting in detail_lines (with
    # occurrence navigation) opt out of the browser's generic highlighting.
    handles_search_highlighting: bool = False

    def detail_lines(self, row: Any, width: int) -> list[str]:
        """Return detail lines for the selected row. Override in subclasses."""
        return [str(row)]

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        """Handle a custom action on the current row. Override for custom behavior."""
        return "ignored"

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
