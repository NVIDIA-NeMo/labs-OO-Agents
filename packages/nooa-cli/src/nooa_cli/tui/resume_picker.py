# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-application session chooser with separate list and conversation preview panes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import BufferControl, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
from rich.cells import cell_len, split_graphemes

from .fullscreen_browser import ExplorerBrowser, SelectablePreviewControl, build_fullscreen_browser
from .terminal_safety import sanitize_live_text


@dataclass(frozen=True, slots=True)
class ResumePickerTurn:
    role: Literal["user", "agent"]
    content: str


@dataclass(frozen=True, slots=True)
class ResumePickerRow:
    id: str
    title: str
    model: str
    agent: str
    working_directory: str
    last_active: float
    turn_count: int
    attached: bool = False
    current: bool = False
    created_at: float = 0
    turns: tuple[ResumePickerTurn, ...] = ()

    @property
    def preview(self) -> str:
        """Return the newest agent reply, falling back to the newest turn."""
        if not self.turns:
            return "No message preview"
        turn = next((item for item in reversed(self.turns) if item.role == "agent"), self.turns[-1])
        return _single_line(turn.content)

    @classmethod
    def from_meta(
        cls,
        meta: Any,
        *,
        attached: bool = False,
        current: bool = False,
        turns: tuple[ResumePickerTurn, ...] = (),
    ) -> ResumePickerRow:
        title = str(meta.name or "").strip()
        return cls(
            str(meta.id),
            title or "Untitled session",
            str(meta.model or "—"),
            str(meta.agent or "—"),
            str(meta.working_dir or "—"),
            float(meta.last_active),
            int(meta.turn_count),
            attached,
            current,
            float(getattr(meta, "started_at", 0) or 0),
            turns,
        )


@dataclass(frozen=True, slots=True)
class FieldMatch:
    row: ResumePickerRow
    field: str | None
    positions: tuple[int, ...]

    def __iter__(self):
        yield self.row
        yield self.positions


def _single_line(text: str) -> str:
    """Make untrusted session metadata safe and stable in a two-line row."""
    return " ".join(sanitize_live_text(text).split())


def literal_match(query: str, text: str) -> tuple[int, tuple[int, ...]] | None:
    """Find one case-insensitive contiguous match in original-text coordinates."""
    query = query.casefold().strip()
    parts: list[str] = []
    source: list[int] = []
    for index, char in enumerate(text):
        folded = char.casefold()
        parts.append(folded)
        source.extend([index] * len(folded))
    target = "".join(parts)
    if not query:
        return 0, ()
    found = target.find(query)
    if found < 0:
        return None
    stop = found + len(query)
    positions = tuple(dict.fromkeys(source[found:stop]))
    return -found, positions


class ResumePickerModel:
    def __init__(self, rows: list[ResumePickerRow]) -> None:
        self.rows = rows
        self._search_fields = [self._fields_for(row) for row in rows]
        self.query = ""
        self.state_filter: Literal["detached", "attached", "all"] = "detached"
        self.sort_updated = True
        self.selected = self.list_offset = 0
        self.preview_offset = 10**9
        self._query_matches: list[tuple[int, str, tuple[int, ...]] | None] = []
        self._matches: list[FieldMatch] = []
        self.set_query("")

    @staticmethod
    def _fields_for(row: ResumePickerRow) -> dict[str, str]:
        return {
            "title": _single_line(row.title),
            "preview": row.preview,
            "conversation": "\n".join(sanitize_live_text(turn.content) for turn in row.turns),
        }

    @property
    def matches(self) -> list[FieldMatch]:
        return self._matches

    @property
    def current_match(self) -> FieldMatch | None:
        return self._matches[self.selected] if self._matches else None

    @property
    def current(self) -> ResumePickerRow | None:
        return self.current_match.row if self.current_match else None

    @property
    def can_select(self) -> bool:
        return self.current is not None and not self.current.attached and not self.current.current

    def set_query(self, query: str) -> None:
        previous_id = self.current.id if self.current else None
        self.query = query
        normalized_query = query.strip()
        if normalized_query:
            query_matches: list[tuple[int, str, tuple[int, ...]] | None] = []
            for fields in self._search_fields:
                candidates = []
                for field, value in fields.items():
                    result = literal_match(normalized_query, value)
                    if result is not None:
                        candidates.append((result[0], field, result[1]))
                query_matches.append(
                    max(candidates, key=lambda item: item[0]) if candidates else None
                )
            self._query_matches = query_matches
        else:
            self._query_matches = [(0, "", ()) for _row in self.rows]
        self._rebuild_matches(previous_id)

    def _rebuild_matches(self, previous_id: str | None = None) -> None:
        ranked = []
        for index, row in enumerate(self.rows):
            attached = row.attached or row.current
            if self.state_filter == "detached" and attached:
                continue
            if self.state_filter == "attached" and not attached:
                continue
            query_match = self._query_matches[index]
            if query_match is None:
                continue
            score, field, positions = query_match
            stamp = row.last_active if self.sort_updated else row.created_at
            ranked.append(
                (
                    -stamp,
                    -score if self.query.strip() else 0,
                    index,
                    FieldMatch(row, field or None, positions),
                )
            )
        ranked.sort(key=lambda item: item[:3])
        self._matches = [item[3] for item in ranked]
        ids = [match.row.id for match in self._matches]
        self.selected = ids.index(previous_id) if previous_id in ids else 0
        self.list_offset = min(self.list_offset, self.selected)
        self.preview_offset = 10**9

    def cycle_filter(self, delta: int = 1) -> None:
        filters: tuple[Literal["detached", "attached", "all"], ...] = (
            "detached",
            "attached",
            "all",
        )
        self.state_filter = filters[(filters.index(self.state_filter) + delta) % len(filters)]
        self.selected = self.list_offset = 0
        self._rebuild_matches()

    def toggle_filter(self) -> None:
        self.cycle_filter()

    def toggle_sort(self) -> None:
        self.sort_updated = not self.sort_updated
        self.selected = self.list_offset = 0
        self._rebuild_matches()

    def move(self, delta: int) -> None:
        if not self._matches or not delta:
            return
        self.selected = (self.selected + delta) % len(self._matches)
        self.preview_offset = 10**9

    def select(self, index: int) -> None:
        if 0 <= index < len(self._matches) and index != self.selected:
            self.selected = index
            self.preview_offset = 10**9

    def ensure_selection_visible(self, rows: int) -> None:
        rows = max(1, rows)
        if self.selected < self.list_offset:
            self.list_offset = self.selected
        elif self.selected >= self.list_offset + rows:
            self.list_offset = self.selected - rows + 1

    def visible(self, rows: int) -> list[tuple[int, FieldMatch]]:
        """Return visible rows without mutating state during layout measurement."""
        rows = max(1, rows)
        return [
            (index, self._matches[index])
            for index in range(self.list_offset, min(len(self._matches), self.list_offset + rows))
        ]

    def scroll_list(self, delta: int, rows: int) -> None:
        """Scroll the list viewport without changing the selected session."""
        maximum = max(0, len(self._matches) - max(1, rows))
        self.list_offset = min(maximum, max(0, self.list_offset + delta))

    def scroll_preview(self, delta: int, line_count: int, height: int) -> None:
        maximum = max(0, line_count - max(1, height))
        current = min(self.preview_offset, maximum)
        self.preview_offset = min(maximum, max(0, current + delta))


def _clip(text: str, width: int) -> str:
    width = max(0, width)
    if cell_len(text) <= width:
        return text
    if not width:
        return ""
    kept = []
    used = 0
    for start, stop, cells in split_graphemes(text)[0]:
        if used + cells > width - 1:
            break
        kept.append(text[start:stop])
        used += cells
    return "".join(kept) + "…"


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


def _field_fragments(
    text: str, base: str, positions: tuple[int, ...] = ()
) -> list[tuple[str, str]]:
    matched = set(positions)
    return [
        (base + (" class:fullscreen-browser.match" if i in matched else ""), char)
        for i, char in enumerate(text)
    ]


def _clip_fragments(fragments: list[tuple[str, str]], width: int) -> list[tuple[str, str]]:
    plain = "".join(text for _, text in fragments)
    if cell_len(plain) <= width:
        return fragments
    if width <= 0:
        return []
    flattened = [(style, char) for style, text in fragments for char in text]
    result: list[tuple[str, str]] = []
    used = 0
    for start, stop, cells in split_graphemes(plain)[0]:
        if used + cells > width - 1:
            break
        cluster = flattened[start:stop]
        style = " ".join(dict.fromkeys(part for value, _ in cluster for part in value.split()))
        result.append((style, plain[start:stop]))
        used += cells
    result.append((fragments[0][0] if fragments else "", "…"))
    return result


def _relative(ts: float, now: float | None = None) -> str:
    seconds = max(0, int((now or time.time()) - ts))
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    if seconds < 86400 * 30:
        return f"{seconds // 86400}d ago"
    if seconds < 86400 * 365:
        return f"{seconds // (86400 * 30)}mo ago"
    return f"{seconds // (86400 * 365)}y ago"


def _row_title_width(width: int) -> int:
    # Marker (2), age (8), separators (4), and a two-cell emoji state.
    return min(30, max(10, max(0, width - 16) // 3))


def _row_fragments(
    match: FieldMatch, selected: bool, width: int, *, sort_updated: bool = True
) -> list[list[tuple[str, str]]]:
    """Render one compact row: age, resumability, title, and latest agent text."""
    row = match.row
    base = "class:fullscreen-browser.row" + (
        " class:fullscreen-browser.selected" if selected else ""
    )
    attached = row.current or row.attached
    timestamp = row.last_active if sort_updated else row.created_at
    state = "✗" if attached else "✓"
    prefix = ("❯ " if selected else "  ") + f"{_relative(timestamp):>8}  {state}  "
    title_width = _row_title_width(width)
    title_text = _single_line(row.title)
    title = _clip_fragments(
        _field_fragments(
            title_text,
            base + (" class:fullscreen-browser.unavailable" if attached else ""),
            match.positions if match.field == "title" else (),
        ),
        title_width,
    )
    title_cells = cell_len("".join(text for _style, text in title))
    title.append((base, " " * max(0, title_width - title_cells) + "  "))
    preview_positions = match.positions if match.field == "preview" else ()
    return [
        _clip_fragments(
            [
                (base, prefix),
                *title,
                *_field_fragments(
                    row.preview,
                    base + " class:fullscreen-browser.meta",
                    preview_positions,
                ),
            ],
            width,
        )
    ]


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


def _semantic_preview_selection(text: str) -> str:
    """Remove conversation-preview chrome from selected text."""
    output: list[str] = []
    in_user_bar = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and set(stripped) <= {"▔"}:
            in_user_bar = True
            continue
        if stripped and set(stripped) <= {"▁"}:
            in_user_bar = False
            continue
        if in_user_bar or line.startswith(" ❯ "):
            if line.startswith(" ❯ "):
                line = line[3:]
            elif in_user_bar and line.startswith("   "):
                line = line[3:]
            output.append(line.rstrip())
        elif stripped != "OO:":
            output.append(line)
    return "\n".join(output).strip("\n")


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
    preview = _preview_lines(model.current, width)
    preview_height = max(1, height - len(lines) - 2)
    maximum = max(0, len(preview) - preview_height)
    start = min(model.preview_offset, maximum)
    lines.extend(
        "".join(text for _, text in line) for line in preview[start : start + preview_height]
    )
    lines.extend(
        [
            separator,
        ]
    )
    return "\n".join(lines[:height])


class _PickerControl(FormattedTextControl):
    """Resume list control; preview gestures use SelectablePreviewControl."""

    def __init__(self, picker: ResumePicker):
        self.picker = picker
        self.viewport = (1, 1)
        super().__init__(self._text, focusable=True, show_cursor=False)

    def create_content(self, width: int, height: int):
        self.viewport = (max(1, width), max(1, height or 1))
        self._fragment_cache.clear()
        return super().create_content(width, height)

    def _text(self):
        width, height = self.viewport
        output = []
        for index, match in self.picker.model.visible(height):
            for row_line in _row_fragments(
                match,
                index == self.picker.model.selected,
                width,
                sort_updated=self.picker.model.sort_updated,
            ):
                if output:
                    output.append(("", "\n"))
                output.extend(row_line)
        return output or [("class:fullscreen-browser.empty", "No matching sessions")]

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
            or not self.picker.mouse_support
        ):
            return NotImplemented
        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self.picker.mouse_scroll("list", -3)
            return None
        if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self.picker.mouse_scroll("list", 3)
            return None
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.picker.activate_control("list")
            self.picker.select(self.picker.model.list_offset + mouse_event.position.y)
            return None
        return NotImplemented


class _PickerSearchControl(BufferControl):
    def __init__(self, picker: ResumePicker, buffer: Buffer):
        self.picker = picker
        super().__init__(buffer)

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type is MouseEventType.MOUSE_DOWN:
            self.picker.activate_control("list")
        return super().mouse_handler(mouse_event)


class _PickerButtonControl(FormattedTextControl):
    def __init__(self, picker: ResumePicker, kind: Literal["filter", "sort"]):
        self.picker = picker
        self.kind = kind
        super().__init__(self._text, focusable=False, show_cursor=False)

    def _text(self):
        focused = self.picker.option_cursor == self.kind
        style = "class:fullscreen-browser.control" + (
            " class:fullscreen-browser.control-focused" if focused else ""
        )
        if self.kind == "filter":
            value = {
                "detached": "✓ Not attached",
                "attached": "✗ Attached",
                "all": "✓/✗ All",
            }[self.picker.model.state_filter]
            return [(style, f"[Filter: {value}]")]
        value = "Recent activity" if self.picker.model.sort_updated else "Creation date"
        return [(style, f"[Sort: {value}]")]

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.picker.option_cursor = self.kind
            if self.kind == "filter":
                self.picker.cycle_filter()
            else:
                self.picker.toggle_sort()
            self.picker.option_cursor = None
            self.picker.invalidate()
            return None
        return NotImplemented


class ResumePicker(ExplorerBrowser):
    """Session-specific ExplorerBrowser with asynchronous replay previews.

    The host owns this picker's key dispatch (it never routes through
    ``ExplorerBrowser.handle_key``), but inherited helpers dereference
    ``self.view``. Expose a minimal facade so an accidental inherited call
    degrades gracefully instead of raising ``AttributeError``.
    """

    class _ViewFacade:
        config = SimpleNamespace(actions={})
        options: tuple[Any, ...] = ()
        item_name = "session"

        @staticmethod
        def handle_key(_action: str, _value: str = "") -> str:
            return "handled"

        @staticmethod
        def handle_action(_action: str, _row: Any) -> str:
            return "ignored"

    @property
    def view(self) -> ResumePicker._ViewFacade:
        return self._view_facade

    @property
    def model(self) -> ResumePickerModel:
        return self._resume_model

    @model.setter
    def model(self, value: ResumePickerModel) -> None:
        self._resume_model = value

    def __init__(
        self,
        rows: list[ResumePickerRow],
        app: Any,
        *,
        selection_copy_callback: Callable[[str], None] | None = None,
        selection_status: Callable[[], str] | None = None,
    ):
        self.app = app
        self._view_facade = ResumePicker._ViewFacade()
        self.model = ResumePickerModel(rows)
        self._selection_copy_callback = selection_copy_callback
        self._selection_status = selection_status
        self._preview_models: dict[tuple[str, int], Any] = {}
        self._preview_tasks: dict[tuple[str, int], Any] = {}
        self.native_selection = False
        self.active_control = "list"
        self.option_cursor: Literal["filter", "sort"] | None = None
        self.buffer = Buffer(multiline=False)
        self.buffer.on_text_changed += lambda _: self._query_changed()
        self.query_control = _PickerSearchControl(self, self.buffer)
        self.query_window = Window(
            self.query_control,
            width=Dimension(min=4, weight=1),
            height=1,
            style=lambda: (
                "class:fullscreen-browser.control-focused"
                if self.active_control == "list" and self.option_cursor is None
                else ""
            ),
        )
        self.filter_control = _PickerButtonControl(self, "filter")
        self.sort_control = _PickerButtonControl(self, "sort")
        self.search_label_control = FormattedTextControl(self._search_label)
        self.search_close_control = FormattedTextControl(self._search_close)
        search = VSplit(
            [
                Window(self.search_label_control, width=9, height=1),
                self.query_window,
                Window(self.search_close_control, width=1, height=1),
            ],
            padding=0,
        )
        selectors = VSplit(
            [
                Window(self.filter_control, width=Dimension(min=17, preferred=20), height=1),
                Window(FormattedTextControl(" "), width=1, height=1),
                Window(self.sort_control, width=Dimension(min=10, preferred=18), height=1),
            ],
            padding=0,
        )
        controls = VSplit(
            [search, Window(FormattedTextControl(" "), width=1, height=1), selectors],
            padding=0,
        )
        self.list_control = _PickerControl(self)
        self.preview_control = SelectablePreviewControl(self)

        self.title_control = FormattedTextControl(self._title)
        self.list_header_control = FormattedTextControl(self._list_header)
        self.preview_header_control = FormattedTextControl(self._preview_header)
        self.small_control = FormattedTextControl(
            [("class:fullscreen-browser.too-small", "Terminal too small")], focusable=True
        )
        self.container = build_fullscreen_browser(
            app=self.app,
            title_control=self.title_control,
            help_control=FormattedTextControl(
                self._help_text, style="class:fullscreen-browser.footer"
            ),
            controls=controls,
            list_header_control=self.list_header_control,
            list_control=self.list_control,
            preview_header_control=self.preview_header_control,
            preview_control=self.preview_control,
            active_rail=self._active_rail,
            small_control=self.small_control,
            small_text=self._small_text,
        )

    def _query_changed(self) -> None:
        self.model.set_query(self.buffer.text)
        self._prepare_current_preview()
        self.invalidate()

    @property
    def mouse_support(self) -> bool:
        """Disable mouse reporting while terminal-native selection is active."""
        return not self.native_selection

    def toggle_native_selection(self) -> None:
        """Switch between application selection and terminal-native selection."""
        self.preview_control.cancel_drag()
        self.native_selection = not self.native_selection
        self.invalidate()

    def _help_text(self):
        copy_status = self._selection_status() if self._selection_status is not None else ""
        if copy_status:
            return copy_status
        columns = self.app.output.get_size().columns
        if self.option_cursor is not None:
            if columns < 72:
                return "Options · ←→ select · ↑↓/Space change · Esc done"
            return "Options · ←→ select · ↑↓/Space change · Enter/Esc done"
        if columns < 72:
            return "Ctrl-O options · F2 native · ↵ resume · Esc"
        return "Ctrl-O options · Tab panes · F2 native · ↵ resume · Esc"

    def _title(self):
        count = len(self.model.matches)
        return [
            (
                "class:fullscreen-browser.title",
                f"Resume a previous session · {count} session{'s' if count != 1 else ''}",
            )
        ]

    def _list_header(self):
        age = "updated" if self.model.sort_updated else "created"
        width = max(1, self.app.output.get_size().columns - 1)
        title_width = _row_title_width(width)
        return [
            (
                "class:fullscreen-browser.heading",
                f"  {age:>8}  st {'title':<{title_width}}  last agent message",
            )
        ]

    def _preview_header(self):
        row = self.model.current
        title = _single_line(row.title) if row else "No selection"
        position = self.preview_search_position()
        suffix = f" · match {position[0]}/{position[1]}" if position[1] else ""
        return [
            (
                "class:fullscreen-browser.heading",
                _clip(
                    f"Conversation preview · {title}{suffix}",
                    self.app.output.get_size().columns,
                ),
            )
        ]

    def invalidate(self) -> None:
        self.list_control._fragment_cache.clear()
        self.preview_control._fragment_cache.clear()
        self.filter_control._fragment_cache.clear()
        self.sort_control._fragment_cache.clear()
        self.search_label_control._fragment_cache.clear()
        self.search_close_control._fragment_cache.clear()
        self.title_control._fragment_cache.clear()
        self.list_header_control._fragment_cache.clear()
        self.preview_header_control._fragment_cache.clear()
        self.app.invalidate()

    def activate_control(self, name: str) -> None:
        self.active_control = name
        controls = {
            "list": self.query_control,
            "preview": self.preview_control,
        }
        self.app.layout.focus(controls[name])
        self.invalidate()

    def toggle_options(self) -> None:
        """Enter or leave the inline filter/sort options mode."""
        self.active_control = "list"
        if self.option_cursor is None:
            self.option_cursor = "filter"
            self.app.layout.focus(self.list_control)
        else:
            self.option_cursor = None
            self.app.layout.focus(self.query_control)
        self.invalidate()

    def move_option(self, delta: int) -> None:
        if self.option_cursor is None or not delta:
            return
        self.option_cursor = "sort" if delta > 0 else "filter"
        self.invalidate()

    def change_option(self, delta: int = 1) -> None:
        if self.option_cursor == "filter":
            self.model.cycle_filter(delta)
            self._prepare_current_preview()
        elif self.option_cursor == "sort" and delta:
            self.model.toggle_sort()
            self._prepare_current_preview()
        self.invalidate()

    def close_options(self) -> bool:
        if self.option_cursor is None:
            return False
        self.option_cursor = None
        self.app.layout.focus(self.query_control)
        self.invalidate()
        return True

    def focus_previous(self) -> None:
        self.focus_next(-1)

    def move_horizontal(self, delta: int) -> None:
        if self.option_cursor is not None:
            self.move_option(delta)
            return
        if self.active_control != "list" or not delta:
            return
        if delta < 0:
            self.buffer.cursor_left(count=1)
        else:
            self.buffer.cursor_right(count=1)

    def navigate_vertical(self, delta: int) -> None:
        if self.option_cursor is not None:
            self.change_option(delta)
        elif self.active_control == "list":
            self.move(delta)
        elif self.active_control == "preview":
            width, height = self.preview_control.viewport
            transcript = self._preview_model(width)
            if transcript is None or not transcript.move_search_match(
                delta, width=max(1, width), height=max(1, height)
            ):
                self.scroll_preview(delta)
            else:
                self.invalidate()

    def cycle_filter(self) -> None:
        self.model.toggle_filter()
        self._prepare_current_preview()
        self.invalidate()

    def toggle_sort(self) -> None:
        self.model.toggle_sort()
        self._prepare_current_preview()
        self.invalidate()

    def page(self, delta: int) -> None:
        if self.active_control == "list":
            self.move(delta * max(1, self.list_control.viewport[1]))
        elif self.active_control == "preview":
            self.scroll_preview(delta * max(1, self.preview_control.viewport[1]))

    @staticmethod
    def _build_preview_model(row: ResumePickerRow, width: int, height: int):
        from .frontend import render_history_replay_to_ansi
        from .fullscreen_transcript import FullscreenTranscriptModel
        from .output import HistoryReplay, HistoryTurn

        replay = HistoryReplay(
            turns=[HistoryTurn(turn.role, turn.content) for turn in row.turns],
            session_id=row.id[:8],
            show_header=False,
            show_footer=False,
        )
        transcript = FullscreenTranscriptModel(show_trailing_blank=False)
        transcript.append(render_history_replay_to_ansi(replay, width))
        # Projection is the dominant first-render cost for long histories.
        transcript.formatted_text(width=width, height=max(1, height))
        return transcript

    def _preview_model(self, width: int):
        row = self.model.current
        if row is None:
            return None
        key = (row.id, max(1, width))
        transcript = self._preview_models.get(key)
        if transcript is not None:
            return transcript
        self._prepare_current_preview()
        # Synchronous callers (primarily deterministic unit tests) have no event
        # loop in which to schedule preparation, so retain a direct fallback.
        if key not in self._preview_tasks:
            transcript = self._build_preview_model(row, key[1], self.preview_control.viewport[1])
            self._preview_models[key] = transcript
            return transcript
        return None

    def _prepare_current_preview(self) -> None:
        import asyncio

        row = self.model.current
        if row is None or not row.turns:
            return
        width, height = self.preview_control.viewport
        key = (row.id, max(1, width))
        if key in self._preview_models or key in self._preview_tasks:
            return

        async def prepare() -> None:
            try:
                transcript = await asyncio.to_thread(
                    self._build_preview_model, row, key[1], max(1, height)
                )
            except Exception:
                return
            finally:
                self._preview_tasks.pop(key, None)
            self._preview_models.setdefault(key, transcript)
            self.invalidate()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._preview_tasks[key] = loop.create_task(prepare())

    def _preview_transcript(self, width: int, height: int):
        """Provide the asynchronously prepared replay to ExplorerBrowser."""
        return self._preview_model(width)

    def _selected_preview_text(self, transcript: Any) -> str:
        """Strip replay-only chrome while using the shared copy lifecycle."""
        return _semantic_preview_selection(transcript.selected_text())

    def preview_text(self, width: int, height: int):
        row = self.model.current
        if row is None:
            return [("class:fullscreen-browser.empty", "No session selected")]
        if not row.turns:
            return [("class:fullscreen-browser.empty", "No conversation preview")]
        transcript = self._preview_model(width)
        if transcript is None:
            return [("class:fullscreen-browser.empty", "Preparing conversation preview…")]
        transcript.set_search(self.model.query, width=max(1, width), height=max(1, height))
        return transcript.formatted_text(width=max(1, width), height=max(1, height))

    def preview_search_position(self) -> tuple[int, int]:
        width, height = self.preview_control.viewport
        row = self.model.current
        if row is None:
            return 0, 0
        transcript = self._preview_models.get((row.id, max(1, width)))
        if transcript is None:
            return 0, 0
        transcript.set_search(self.model.query, width=max(1, width), height=max(1, height))
        return transcript.search_position

    def _reset_preview(self) -> None:
        row = self.model.current
        if row is not None:
            for (session_id, _width), transcript in self._preview_models.items():
                if session_id == row.id:
                    if self.model.query.strip():
                        width = max(1, _width)
                        height = max(1, self.preview_control.viewport[1])
                        transcript.set_search("", width=width, height=height)
                        transcript.set_search(self.model.query, width=width, height=height)
                    else:
                        transcript.jump_to_tail()

    def move(self, delta: int) -> None:
        before = self.model.current.id if self.model.current else None
        self.model.move(delta)
        self.model.ensure_selection_visible(self.list_control.viewport[1])
        if self.model.current and self.model.current.id != before:
            self._reset_preview()
            self._prepare_current_preview()
        self.invalidate()

    def select(self, index: int) -> None:
        before = self.model.current.id if self.model.current else None
        self.model.select(index)
        self.model.ensure_selection_visible(self.list_control.viewport[1])
        if self.model.current and self.model.current.id != before:
            self._reset_preview()
            self._prepare_current_preview()
        self.invalidate()

    def scroll_preview(self, delta: int) -> None:
        width, height = self.preview_control.viewport
        transcript = self._preview_model(width)
        if transcript is not None:
            transcript.scroll_visual_lines(delta, width=max(1, width), height=max(1, height))
        self.invalidate()

    def mouse_scroll(self, pane: Literal["list", "preview"], delta: int) -> None:
        if pane == "list":
            self.model.scroll_list(delta, self.list_control.viewport[1])
            self.invalidate()
        else:
            self.scroll_preview(delta)

    def close(self) -> None:
        """Cancel preview preparation when the picker is dismissed."""
        self.preview_control.cancel_drag()
        for task in self._preview_tasks.values():
            task.cancel()
        self._preview_tasks.clear()

    def selected_id(self) -> str | None:
        return self.model.current.id if self.model.can_select and self.model.current else None
