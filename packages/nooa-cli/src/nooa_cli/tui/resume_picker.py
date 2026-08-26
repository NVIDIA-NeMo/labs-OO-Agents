# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex-inspired, in-application session resume chooser."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import BufferControl, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from rich.cells import cell_len, split_graphemes


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
    preview: str = ""

    @classmethod
    def from_meta(
        cls, meta: Any, *, attached: bool = False, current: bool = False, preview: str = ""
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
            preview,
        )


@dataclass(frozen=True, slots=True)
class FieldMatch:
    row: ResumePickerRow
    field: str | None
    positions: tuple[int, ...]

    def __iter__(self):
        yield self.row
        yield self.positions


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
    SEARCH_FIELDS = ("title", "preview", "id", "model", "agent", "working_directory")

    def __init__(self, rows: list[ResumePickerRow], cwd: str | None = None) -> None:
        self.rows = rows
        default_cwd = rows[0].working_directory if rows and cwd is None else (cwd or os.getcwd())
        self.cwd = os.path.realpath(default_cwd)
        self.query, self.filter_cwd, self.sort_updated = "", True, True
        self.selected = self.offset = 0
        self._matches: list[FieldMatch] = []
        self.set_query("")

    @property
    def matches(self):
        return self._matches

    @property
    def current_match(self):
        return self._matches[self.selected] if self._matches else None

    @property
    def current(self):
        return self.current_match.row if self.current_match else None

    @property
    def can_select(self):
        return self.current is not None and not self.current.attached and not self.current.current

    def set_query(self, query: str) -> None:
        self.query = query
        ranked = []
        for index, row in enumerate(self.rows):
            if self.filter_cwd and os.path.realpath(row.working_directory) != self.cwd:
                continue
            candidates = []
            for field in self.SEARCH_FIELDS:
                result = fuzzy_match(query, getattr(row, field))
                if result is not None:
                    candidates.append((result[0], field, result[1]))
            if candidates:
                score, field, positions = max(candidates, key=lambda x: x[0])
                stamp = row.last_active if self.sort_updated else row.created_at
                ranked.append(
                    (
                        -score if query.strip() else 0,
                        -stamp,
                        index,
                        FieldMatch(row, field if query.strip() else None, positions),
                    )
                )
        ranked.sort(key=lambda x: x[:3])
        self._matches = [x[3] for x in ranked]
        self.selected = min(self.selected, max(0, len(self._matches) - 1))
        self.offset = min(self.offset, self.selected)
        self._skip_unavailable(1)

    def toggle_filter(self):
        self.filter_cwd = not self.filter_cwd
        self.selected = self.offset = 0
        self.set_query(self.query)

    def toggle_sort(self):
        self.sort_updated = not self.sort_updated
        self.selected = self.offset = 0
        self.set_query(self.query)

    def move(self, delta: int) -> None:
        if not self._matches:
            return
        start = self.selected
        for _ in self._matches:
            self.selected = (self.selected + delta) % len(self._matches)
            if self.can_select:
                return
        self.selected = start

    def _skip_unavailable(self, direction: int):
        if self.current is not None and not self.can_select:
            self.move(direction)

    def visible(self, height: int):
        height = max(1, height)
        if self.selected < self.offset:
            self.offset = self.selected
        elif self.selected >= self.offset + height:
            self.offset = self.selected - height + 1
        return [
            (i, self._matches[i])
            for i in range(self.offset, min(len(self._matches), self.offset + height))
        ]


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


def _field_fragments(text, base, positions=()):
    matched = set(positions)
    return [
        (base + (" class:resume-picker.match" if i in matched else ""), c)
        for i, c in enumerate(text)
    ]


def _clip_fragments(fragments, width):
    plain = "".join(t for _, t in fragments)
    if cell_len(plain) <= width:
        return fragments
    if width <= 0:
        return []
    result = []
    used = 0
    for start, stop, cells in split_graphemes(plain)[0]:
        if used + cells > width - 1:
            break
        cluster = fragments[start:stop]
        style = " ".join(dict.fromkeys(p for s, _ in cluster for p in s.split()))
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


def _render_fragments(model: ResumePickerModel, width: int, height: int):
    w = max(1, width - 2)
    if width < 48 or height < 10:
        return [
            [("class:resume-picker.too-small", _clip("Terminal too small", w))],
            [("class:resume-picker.muted", _clip(f"Need 48 x 10; now {width} x {height}", w))],
        ]
    count = f"{len(model.matches)} session" + ("s" if len(model.matches) != 1 else "")
    lines = [[("class:resume-picker.title", f"Resume a previous session  ·  {count}")]]
    budget = max(1, height - 4)
    if not model.matches:
        lines.append([("class:resume-picker.empty", "No matching sessions")])
    for index, match in model.visible(budget):
        row = match.row
        selected = index == model.selected
        base = "class:resume-picker.row" + (" class:resume-picker.selected" if selected else "")
        state = "current" if row.current else ("attached" if row.attached else "")
        stamp = _relative(row.last_active).rjust(8)
        prefix = ("❯ " if selected else "  ") + stamp + "  "
        dominant = row.preview or row.title
        dominant_field = "preview" if row.preview else "title"
        fragments = [(base, prefix)] + _field_fragments(
            dominant, base, match.positions if match.field == dominant_field else ()
        )
        if row.preview and row.title.casefold() not in {"untitled session", "new session"}:
            fragments.append((base + " class:resume-picker.meta", "  — "))
            fragments += _field_fragments(
                row.title,
                base + " class:resume-picker.meta",
                match.positions if match.field == "title" else (),
            )
        if state:
            fragments.append((base + " class:resume-picker.unavailable", f" [{state}]"))
        lines.append(_clip_fragments(fragments, w))
        if selected and match.field and match.field not in {dominant_field, "title"}:
            value = getattr(row, match.field)
            detail = [("class:resume-picker.detail", f"    {match.field.replace('_', ' ')}: ")]
            detail += _field_fragments(value, "class:resume-picker.detail", match.positions)
            lines.append(_clip_fragments(detail, w))
    footer = (
        "Enter resume · Tab filter · F6 sort · Esc cancel"
        if width >= 60
        else "Enter resume · Tab · F6 · Esc cancel"
    )
    lines.append([("class:resume-picker.footer", _clip(footer, w))])
    return lines[:height]


def render_resume_picker(model, width, height):
    return "\n".join(
        "".join(t for _, t in line) for line in _render_fragments(model, width, height)
    )


class _ResumeListControl(FormattedTextControl):
    def __init__(self, picker):
        self.picker = picker
        self._viewport = None
        super().__init__(self._text, focusable=False, show_cursor=False)

    def create_content(self, width, height):
        viewport = (max(1, width), max(1, height or 1))
        if viewport != self._viewport:
            self._viewport = viewport
            self._fragment_cache.clear()
        return super().create_content(width, height)

    def _text(self):
        if self._viewport is None:
            size = self.picker.app.output.get_size()
            width, height = size.columns, size.rows - 1
        else:
            width, height = self._viewport
        out = []
        for n, line in enumerate(_render_fragments(self.picker.model, width, height + 1)):
            if n:
                out.append(("", "\n"))
            out.extend(line)
        return out


class ResumePicker:
    def __init__(self, rows, app, cwd=None):
        self.app = app
        self.model = ResumePickerModel(rows, cwd)
        self.buffer = Buffer(multiline=False)
        self.buffer.on_text_changed += lambda _: self.model.set_query(self.buffer.text)
        query = Window(
            BufferControl(
                self.buffer,
                input_processors=[
                    BeforeInput("Search  ", style="class:resume-picker.search-label")
                ],
            ),
            height=1,
        )

        def toolbar_text():
            columns = self.app.output.get_size().columns
            filt = "Cwd" if self.model.filter_cwd else "All"
            sort = "Updated" if self.model.sort_updated else "Created"
            text = f"Filter: {filt}  Sort: {sort}" if columns >= 60 else f"{filt} · {sort}"
            return [("class:resume-picker.muted", text)]

        toolbar = Window(
            FormattedTextControl(toolbar_text),
            width=Dimension(min=13, preferred=28),
            height=1,
            align="RIGHT",
        )
        self.list_control = _ResumeListControl(self)
        listing = Window(self.list_control, wrap_lines=False)
        self.container = HSplit([VSplit([query, toolbar]), listing], padding=0)
        self.query_control = query.content

    def move(self, delta):
        self.model.move(delta)

    def toggle_filter(self):
        self.model.toggle_filter()

    def toggle_sort(self):
        self.model.toggle_sort()

    def selected_id(self):
        return self.model.current.id if self.model.can_select and self.model.current else None
