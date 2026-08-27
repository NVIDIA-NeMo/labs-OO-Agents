# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-application session chooser with separate list and conversation preview panes."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import BufferControl, DynamicContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.cells import cell_len, split_graphemes

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


def fuzzy_match(query: str, text: str) -> tuple[int, tuple[int, ...]] | None:
    query = query.casefold().strip()
    parts, source = [], []
    for i, char in enumerate(text):
        folded = char.casefold()
        parts.append(folded)
        source.extend([i] * len(folded))
    target = "".join(parts)
    if not query:
        return 0, ()
    found, cursor = [], 0
    for char in query:
        pos = target.find(char, cursor)
        if pos < 0:
            return None
        found.append(pos)
        cursor = pos + 1
    span = found[-1] - found[0] + 1
    adjacency = sum(b == a + 1 for a, b in zip(found, found[1:], strict=False))
    boundary = sum(p == 0 or not target[p - 1].isalnum() for p in found)
    return adjacency * 20 + boundary * 8 - span - found[0], tuple(
        dict.fromkeys(source[p] for p in found)
    )


class ResumePickerModel:
    def __init__(self, rows: list[ResumePickerRow], cwd: str | None = None) -> None:
        self.rows = rows
        self.cwd = os.path.realpath(cwd or os.getcwd())
        self._resolved_directories = {
            row.id: os.path.realpath(row.working_directory) for row in rows
        }
        self._search_fields = [self._fields_for(row) for row in rows]
        self.query = ""
        self.filter_cwd = False
        self.sort_updated = True
        self.selected = self.list_offset = 0
        self.preview_offset = 10**9
        self._matches: list[FieldMatch] = []
        self.set_query("")

    @staticmethod
    def _fields_for(row: ResumePickerRow) -> dict[str, str]:
        return {
            "title": _single_line(row.title),
            "preview": row.preview,
            "conversation": "\n".join(sanitize_live_text(turn.content) for turn in row.turns),
            "id": row.id,
            "model": row.model,
            "agent": row.agent,
            "working_directory": row.working_directory,
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
        ranked = []
        for index, row in enumerate(self.rows):
            if self.filter_cwd and self._resolved_directories[row.id] != self.cwd:
                continue
            fields = self._search_fields[index]
            candidates = []
            for field, value in fields.items():
                result = fuzzy_match(query, value)
                if result is not None:
                    candidates.append((result[0], field, result[1]))
            if candidates:
                score, field, positions = max(candidates, key=lambda item: item[0])
                stamp = row.last_active if self.sort_updated else row.created_at
                ranked.append(
                    (
                        -score if query.strip() else 0,
                        -stamp,
                        index,
                        FieldMatch(row, field if query.strip() else None, positions),
                    )
                )
        ranked.sort(key=lambda item: item[:3])
        self._matches = [item[3] for item in ranked]
        ids = [match.row.id for match in self._matches]
        self.selected = ids.index(previous_id) if previous_id in ids else 0
        self.list_offset = min(self.list_offset, self.selected)
        self.preview_offset = 10**9

    def toggle_filter(self) -> None:
        self.filter_cwd = not self.filter_cwd
        self.selected = self.list_offset = 0
        self.set_query(self.query)

    def toggle_sort(self) -> None:
        self.sort_updated = not self.sort_updated
        self.selected = self.list_offset = 0
        self.set_query(self.query)

    def move(self, delta: int) -> None:
        if not self._matches or not delta:
            return
        self.selected = (self.selected + delta) % len(self._matches)
        self.preview_offset = 10**9

    def select(self, index: int) -> None:
        if 0 <= index < len(self._matches) and index != self.selected:
            self.selected = index
            self.preview_offset = 10**9

    def visible(self, rows: int) -> list[tuple[int, FieldMatch]]:
        rows = max(1, rows)
        if self.selected < self.list_offset:
            self.list_offset = self.selected
        elif self.selected >= self.list_offset + rows:
            self.list_offset = self.selected - rows + 1
        return [
            (index, self._matches[index])
            for index in range(self.list_offset, min(len(self._matches), self.list_offset + rows))
        ]

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
        (base + (" class:resume-picker.match" if i in matched else ""), char)
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


def _row_fragments(
    match: FieldMatch, selected: bool, width: int, *, sort_updated: bool = True
) -> list[list[tuple[str, str]]]:
    row = match.row
    base = "class:resume-picker.row" + (" class:resume-picker.selected" if selected else "")
    state = "attached" if row.current or row.attached else "detached"
    timestamp = row.last_active if sort_updated else row.created_at
    prefix = ("❯ " if selected else "  ") + f"{_relative(timestamp):>8}  {state:<8}  "
    title = _field_fragments(
        _single_line(row.title),
        base + (" class:resume-picker.unavailable" if state == "attached" else ""),
        match.positions if match.field == "title" else (),
    )
    first = _clip_fragments([(base, prefix), *title], width)
    preview_positions = match.positions if match.field == "preview" else ()
    preview_role = next(
        ("Agent" for turn in reversed(row.turns) if turn.role == "agent"),
        "You" if row.turns else "",
    )
    preview_prefix = f"    {preview_role}: " if preview_role else "    "
    second = _clip_fragments(
        [
            (base + " class:resume-picker.meta", preview_prefix),
            *_field_fragments(row.preview, base + " class:resume-picker.meta", preview_positions),
        ],
        width,
    )
    return [first, second]


def _preview_lines(row: ResumePickerRow | None, width: int) -> list[list[tuple[str, str]]]:
    if row is None:
        return [[("class:resume-picker.empty", "No session selected")]]
    if not row.turns:
        return [[("class:resume-picker.empty", "No conversation preview")]]
    lines: list[list[tuple[str, str]]] = []
    for turn in row.turns:
        label = "You" if turn.role == "user" else "Agent"
        style = (
            "class:resume-picker.preview-user"
            if turn.role == "user"
            else "class:resume-picker.preview-agent"
        )
        wrapped = _wrap(turn.content, max(1, width - 2))
        lines.append([(style, f"{label}:")])
        lines.extend([("class:resume-picker.preview", "  " + line)] for line in wrapped)
        lines.append([])
    return lines[:-1]


def render_resume_picker(model: ResumePickerModel, width: int, height: int) -> str:
    """Render a deterministic text snapshot used by unit tests and narrow fallbacks."""
    if width < 48 or height < 13:
        return f"Terminal too small\nNeed 48 x 13; now {width} x {height}"
    separator = "─" * width
    filt = "This directory" if model.filter_cwd else "All sessions"
    sort = "Recent activity" if model.sort_updated else "Creation date"
    lines = [
        _clip(f"Resume a previous session · {len(model.matches)} sessions", width),
        _clip(f"[Search: {model.query}] [Filter: {filt}] [Sort: {sort}]", width),
        separator,
    ]
    list_height = max(1, (height - 8) // 2)
    for index, match in model.visible(max(1, list_height // 2)):
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
            _clip(
                "Tab field · Space change · ↑↓ select · PgUp/PgDn preview · Enter resume · Esc cancel",
                width,
            ),
        ]
    )
    return "\n".join(lines[:height])


class _PickerControl(FormattedTextControl):
    def __init__(self, picker: ResumePicker, kind: Literal["list", "preview"]):
        self.picker = picker
        self.kind = kind
        self.viewport = (1, 1)
        super().__init__(self._text, focusable=False, show_cursor=False)

    def create_content(self, width: int, height: int):
        self.viewport = (max(1, width), max(1, height or 1))
        self._fragment_cache.clear()
        return super().create_content(width, height)

    def _text(self):
        width, height = self.viewport
        if self.kind == "list":
            output = []
            for index, match in self.picker.model.visible(max(1, height // 2)):
                for row_line in _row_fragments(
                    match,
                    index == self.picker.model.selected,
                    width,
                    sort_updated=self.picker.model.sort_updated,
                ):
                    if output:
                        output.append(("", "\n"))
                    output.extend(row_line)
            if not output:
                output = [("class:resume-picker.empty", "No matching sessions")]
            return output

        lines = _preview_lines(self.picker.model.current, width)
        maximum = max(0, len(lines) - height)
        start = min(self.picker.model.preview_offset, maximum)
        self.picker.model.preview_offset = start
        output = []
        for line in lines[start : start + height]:
            if output:
                output.append(("", "\n"))
            output.extend(line)
        return output

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self.picker.mouse_scroll(self.kind, -3)
            return None
        if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self.picker.mouse_scroll(self.kind, 3)
            return None
        if (
            self.kind == "list"
            and mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.picker.model.select(self.picker.model.list_offset + mouse_event.position.y // 2)
            self.picker.invalidate()
            return None
        return NotImplemented


class _PickerSearchControl(BufferControl):
    def __init__(self, picker: ResumePicker, buffer: Buffer):
        self.picker = picker
        super().__init__(buffer)

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type is MouseEventType.MOUSE_DOWN:
            self.picker.active_control = "search"
            self.picker.invalidate()
        return super().mouse_handler(mouse_event)


class _PickerButtonControl(FormattedTextControl):
    def __init__(self, picker: ResumePicker, kind: Literal["filter", "sort"]):
        self.picker = picker
        self.kind = kind
        super().__init__(self._text, focusable=True, show_cursor=False)

    def _text(self):
        focused = self.picker.active_control == self.kind
        style = "class:resume-picker.control" + (
            " class:resume-picker.control-focused" if focused else ""
        )
        if self.kind == "filter":
            value = "This directory" if self.picker.model.filter_cwd else "All sessions"
            return [(style, f"[Filter: {value}]")]
        value = "Recent activity" if self.picker.model.sort_updated else "Creation date"
        return [(style, f"[Sort: {value}]")]

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.picker.activate_control(self.kind)
            self.picker.change_active_control()
            return None
        return NotImplemented


class ResumePicker:
    CONTROL_ORDER = ("search", "filter", "sort")

    def __init__(self, rows: list[ResumePickerRow], app: Any, cwd: str | None = None):
        self.app = app
        self.model = ResumePickerModel(rows, cwd)
        self.active_control = "search"
        self.buffer = Buffer(multiline=False)
        self.buffer.on_text_changed += lambda _: self._query_changed()
        self.query_control = _PickerSearchControl(self, self.buffer)
        self.query_window = Window(self.query_control, width=Dimension(min=4, weight=1), height=1)
        self.filter_control = _PickerButtonControl(self, "filter")
        self.sort_control = _PickerButtonControl(self, "sort")
        self.search_label_control = FormattedTextControl(self._search_label)
        search = VSplit(
            [
                Window(
                    self.search_label_control,
                    width=9,
                    height=1,
                ),
                self.query_window,
                Window(FormattedTextControl("]"), width=1, height=1),
            ],
            padding=0,
        )
        selectors = VSplit(
            [
                Window(self.filter_control, width=Dimension(min=24, preferred=24), height=1),
                Window(FormattedTextControl(" "), width=1, height=1),
                Window(self.sort_control, width=Dimension(min=23, preferred=23), height=1),
            ],
            padding=0,
        )
        wide_controls = VSplit(
            [search, Window(FormattedTextControl(" "), width=1, height=1), selectors], padding=0
        )
        narrow_controls = HSplit([search, selectors], padding=0)
        controls = DynamicContainer(
            lambda: wide_controls if self.app.output.get_size().columns >= 80 else narrow_controls
        )
        self.list_control = _PickerControl(self, "list")
        self.preview_control = _PickerControl(self, "preview")

        def separator() -> Window:
            return Window(char="─", height=1, style="class:resume-picker.separator")

        self.title_control = FormattedTextControl(self._title)
        self.list_header_control = FormattedTextControl(self._list_header)
        self.preview_header_control = FormattedTextControl(self._preview_header)
        title = Window(self.title_control, height=1)
        list_header = Window(self.list_header_control, height=1)
        preview_header = Window(self.preview_header_control, height=1)
        footer = HSplit(
            [
                Window(
                    FormattedTextControl(
                        "Tab focus · Space change · ↑↓ sessions",
                        style="class:resume-picker.footer",
                    ),
                    height=1,
                ),
                Window(
                    FormattedTextControl(
                        "PgUp/Dn preview · Enter resume · Esc cancel",
                        style="class:resume-picker.footer",
                    ),
                    height=1,
                ),
            ],
            padding=0,
        )
        self.list_window = Window(
            self.list_control, height=Dimension(min=2, weight=3), wrap_lines=False
        )
        self.preview_window = Window(
            self.preview_control,
            height=Dimension(min=1, weight=2),
            wrap_lines=False,
            right_margins=[],
        )
        self._main_container = HSplit(
            [
                title,
                controls,
                separator(),
                list_header,
                self.list_window,
                separator(),
                preview_header,
                self.preview_window,
                separator(),
                footer,
            ],
            padding=0,
        )
        self.small_control = FormattedTextControl(
            [("class:resume-picker.too-small", "Terminal too small")], focusable=True
        )
        self._small_container = HSplit(
            [
                Window(self.small_control, height=1),
                Window(FormattedTextControl(self._small_text), height=1),
            ]
        )
        self.container = DynamicContainer(self._responsive_container)

    def _responsive_container(self):
        size = self.app.output.get_size()
        return (
            self._main_container
            if size.columns >= 48 and size.rows >= 13
            else self._small_container
        )

    def _small_text(self):
        size = self.app.output.get_size()
        return [
            (
                "class:resume-picker.muted",
                f"Need 48 x 13; now {size.columns} x {size.rows}",
            )
        ]

    def _query_changed(self) -> None:
        self.model.set_query(self.buffer.text)
        self.invalidate()

    def _search_label(self):
        style = "class:resume-picker.search-label"
        if self.active_control == "search":
            style += " class:resume-picker.control-focused"
        return [(style, "[Search: ")]

    def _title(self):
        count = len(self.model.matches)
        return [
            (
                "class:resume-picker.title",
                f"Resume a previous session · {count} session{'s' if count != 1 else ''}",
            )
        ]

    def _list_header(self):
        age = "updated" if self.model.sort_updated else "created"
        return [("class:resume-picker.heading", f"Sessions  ·  {age:<7}   state     title")]

    def _preview_header(self):
        row = self.model.current
        title = _single_line(row.title) if row else "No selection"
        return [
            (
                "class:resume-picker.heading",
                _clip(f"Conversation preview · {title}", self.app.output.get_size().columns),
            )
        ]

    def invalidate(self) -> None:
        self.list_control._fragment_cache.clear()
        self.preview_control._fragment_cache.clear()
        self.filter_control._fragment_cache.clear()
        self.sort_control._fragment_cache.clear()
        self.search_label_control._fragment_cache.clear()
        self.title_control._fragment_cache.clear()
        self.list_header_control._fragment_cache.clear()
        self.preview_header_control._fragment_cache.clear()
        self.app.invalidate()

    def focus_initial(self) -> None:
        size = self.app.output.get_size()
        target = (
            self.query_control if size.columns >= 48 and size.rows >= 13 else self.small_control
        )
        self.app.layout.focus(target)

    def activate_control(self, name: str) -> None:
        self.active_control = name
        controls = {
            "search": self.query_control,
            "filter": self.filter_control,
            "sort": self.sort_control,
        }
        self.app.layout.focus(controls[name])
        self.invalidate()

    def focus_next(self) -> None:
        index = (self.CONTROL_ORDER.index(self.active_control) + 1) % len(self.CONTROL_ORDER)
        self.activate_control(self.CONTROL_ORDER[index])

    def focus_previous(self) -> None:
        index = (self.CONTROL_ORDER.index(self.active_control) - 1) % len(self.CONTROL_ORDER)
        self.activate_control(self.CONTROL_ORDER[index])

    def change_active_control(self) -> None:
        if self.active_control == "filter":
            self.model.toggle_filter()
        elif self.active_control == "sort":
            self.model.toggle_sort()
        self.invalidate()

    def move(self, delta: int) -> None:
        self.model.move(delta)
        self.invalidate()

    def scroll_preview(self, delta: int) -> None:
        width, height = self.preview_control.viewport
        lines = _preview_lines(self.model.current, width)
        self.model.scroll_preview(delta, len(lines), height)
        self.invalidate()

    def mouse_scroll(self, pane: Literal["list", "preview"], delta: int) -> None:
        if pane == "list":
            self.move(1 if delta > 0 else -1)
        else:
            self.scroll_preview(delta)

    def selected_id(self) -> str | None:
        return self.model.current.id if self.model.can_select and self.model.current else None
