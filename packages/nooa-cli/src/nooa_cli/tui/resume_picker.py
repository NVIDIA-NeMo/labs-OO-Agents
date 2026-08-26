# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Purpose-built, in-application session resume picker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import BufferControl, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.widgets import Box, Frame
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

    @classmethod
    def from_meta(
        cls, meta: Any, *, attached: bool = False, current: bool = False
    ) -> ResumePickerRow:
        return cls(
            str(meta.id),
            str(meta.name or "Untitled session"),
            str(meta.model or "—"),
            str(meta.agent or "—"),
            str(meta.working_dir or "—"),
            float(meta.last_active),
            int(meta.turn_count),
            attached,
            current,
        )


@dataclass(frozen=True, slots=True)
class FieldMatch:
    """A fuzzy match anchored to source data, never to rendered chrome."""

    row: ResumePickerRow
    field: str | None
    positions: tuple[int, ...]

    def __iter__(self):
        # Preserve the former ``(row, positions)`` test/caller contract.
        yield self.row
        yield self.positions


def fuzzy_match(query: str, text: str) -> tuple[int, tuple[int, ...]] | None:
    """Score a case-insensitive subsequence match and return source positions.

    ``casefold`` can expand one source code point (for example ``ß`` to
    ``ss``), so matching is performed in folded space while highlights are
    explicitly mapped back to indices in the original string.
    """
    folded_query = query.casefold().strip()
    folded_target_parts: list[str] = []
    source_indices: list[int] = []
    for source_index, char in enumerate(text):
        folded = char.casefold()
        folded_target_parts.append(folded)
        source_indices.extend([source_index] * len(folded))
    target = "".join(folded_target_parts)
    if not folded_query:
        return (0, ())
    folded_positions: list[int] = []
    cursor = 0
    for char in folded_query:
        found = target.find(char, cursor)
        if found < 0:
            return None
        folded_positions.append(found)
        cursor = found + 1
    span = folded_positions[-1] - folded_positions[0] + 1
    adjacency = sum(
        b == a + 1 for a, b in zip(folded_positions, folded_positions[1:], strict=False)
    )
    boundary = sum(p == 0 or not target[p - 1].isalnum() for p in folded_positions)
    positions = tuple(dict.fromkeys(source_indices[p] for p in folded_positions))
    return (adjacency * 20 + boundary * 8 - span - folded_positions[0], positions)


class ResumePickerModel:
    """Pure filtering, ranking, and selection state for the picker."""

    SEARCH_FIELDS = ("title", "id", "model", "agent", "working_directory")

    def __init__(self, rows: list[ResumePickerRow]) -> None:
        self.rows = rows
        self.query = ""
        self.selected = 0
        self.offset = 0
        self._matches: list[FieldMatch] = []
        self.set_query("")

    @property
    def matches(self) -> list[FieldMatch]:
        return self._matches

    @property
    def current_match(self) -> FieldMatch | None:
        return self._matches[self.selected] if self._matches else None

    @property
    def current(self) -> ResumePickerRow | None:
        match = self.current_match
        return match.row if match else None

    @property
    def can_select(self) -> bool:
        row = self.current
        return row is not None and not row.attached and not row.current

    def set_query(self, query: str) -> None:
        self.query = query
        ranked: list[tuple[int, int, FieldMatch]] = []
        for index, row in enumerate(self.rows):
            candidates = []
            for field in self.SEARCH_FIELDS:
                result = fuzzy_match(query, getattr(row, field))
                if result is not None:
                    candidates.append((result[0], field, result[1]))
            if candidates:
                score, field, positions = max(candidates, key=lambda item: item[0])
                ranked.append(
                    (-score, index, FieldMatch(row, field if query.strip() else None, positions))
                )
        ranked.sort(key=lambda item: (item[0], item[1]))
        self._matches = [match for _, _, match in ranked]
        self.selected = min(self.selected, max(0, len(self._matches) - 1))
        self.offset = min(self.offset, self.selected)
        self._skip_attached(1)

    def move(self, delta: int) -> None:
        if not self._matches:
            return
        start = self.selected
        for _ in range(len(self._matches)):
            self.selected = (self.selected + delta) % len(self._matches)
            if self.can_select:
                return
        self.selected = start

    def _skip_attached(self, direction: int) -> None:
        if self.current is not None and not self.can_select:
            self.move(direction)

    def visible(self, height: int) -> list[tuple[int, FieldMatch]]:
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
    """Clip to terminal cells without splitting an extended grapheme."""
    width = max(0, width)
    if cell_len(text) <= width:
        return text
    if width == 0:
        return ""
    kept = []
    used = 0
    for start, stop, cells in split_graphemes(text)[0]:
        if used + cells > width - 1:
            break
        kept.append(text[start:stop])
        used += cells
    return "".join(kept) + "…"


def _field_fragments(text: str, base: str, positions: tuple[int, ...] = ()) -> StyleAndTextTuples:
    matched = set(positions)
    return [
        (base + (" class:resume-picker.match" if i in matched else ""), char)
        for i, char in enumerate(text)
    ]


def _clip_fragments(fragments: StyleAndTextTuples, width: int) -> StyleAndTextTuples:
    plain = "".join(text for _, text in fragments)
    if cell_len(plain) <= width:
        return fragments
    if width <= 0:
        return []
    # Rows are assembled as one-character fragments. Grapheme spans let us retain
    # all code points of a cluster and inherit every semantic class it contains.
    result: StyleAndTextTuples = []
    used = 0
    for start, stop, cells in split_graphemes(plain)[0]:
        if used + cells > width - 1:
            break
        cluster = fragments[start:stop]
        style = " ".join(dict.fromkeys(part for item, _ in cluster for part in item.split()))
        result.append((style, plain[start:stop]))
        used += cells
    result.append((fragments[0][0] if fragments else "", "…"))
    return result


def _render_fragments(
    model: ResumePickerModel, width: int, height: int
) -> list[StyleAndTextTuples]:
    content_width = max(1, width - 4)  # Box padding plus the single Frame border.
    if width < 40 or height < 10:
        return [
            [("class:resume-picker.too-small", _clip("Terminal too small", content_width))],
            [("class:resume-picker.muted", _clip("Need at least 40 columns", content_width))],
            [
                (
                    "class:resume-picker.muted",
                    _clip(f"and 10 rows; now {width} x {height}", content_width),
                )
            ],
        ]

    available_lines = max(1, height - 5)  # outer chrome (4) and search input (1)
    count = f"{len(model.matches)} session" + ("s" if len(model.matches) != 1 else "")
    lines: list[StyleAndTextTuples] = [[("class:resume-picker.muted", count)]]
    if not model.matches:
        lines.append([("class:resume-picker.empty", "No matching sessions")])
    else:
        # Selected rows can use a second detail line; reserve it before paging.
        row_budget = max(1, available_lines - 3)
        for index, match in model.visible(row_budget):
            row = match.row
            selected = index == model.selected
            base = "class:resume-picker.row" + (" class:resume-picker.selected" if selected else "")
            state = "current" if row.current else ("attached" if row.attached else "")
            marker = "> " if selected else "  "
            fragments: StyleAndTextTuples = [(base, marker)]
            title_positions = match.positions if match.field == "title" else ()
            fragments += _field_fragments(row.title, base, title_positions)
            if state:
                fragments.append(
                    (
                        base + f" class:resume-picker.unavailable class:resume-picker.{state}",
                        f" [{state}]",
                    )
                )
            if width >= 60:
                fragments.append((base + " class:resume-picker.meta", f"  {row.turn_count} turns"))
            if width >= 72:
                stamp = datetime.fromtimestamp(row.last_active).strftime("%Y-%m-%d %H:%M")
                fragments.append((base + " class:resume-picker.meta", f"  {row.model}  {stamp}"))
            lines.append(_clip_fragments(fragments, content_width))

            if selected and match.field and match.field != "title":
                value = getattr(row, match.field)
                detail_base = "class:resume-picker.detail"
                detail = [(detail_base, f"  {match.field.replace('_', ' ')}: ")]
                detail += _field_fragments(value, detail_base, match.positions)
                lines.append(_clip_fragments(detail, content_width))

    current = model.current
    if current and not model.can_select:
        action = (
            "Already current · Esc cancel" if current.current else "Already attached · Esc cancel"
        )
    elif model.matches:
        action = "Enter resume · ↑↓ move · Esc cancel"
    else:
        action = "Type to search · Esc cancel"
    lines.append([("class:resume-picker.footer", _clip(action, content_width))])
    return lines[:available_lines]


def render_resume_picker(model: ResumePickerModel, width: int, height: int) -> str:
    """Render the same width-aware content used by the native control."""
    return "\n".join(
        "".join(text for _, text in line) for line in _render_fragments(model, width, height)
    )


class _ResumeListControl(FormattedTextControl):
    def __init__(self, picker: ResumePicker) -> None:
        self.picker = picker
        self._viewport: tuple[int, int] | None = None
        super().__init__(self._text, focusable=False, show_cursor=False)

    def create_content(self, width: int, height: int | None):
        # Window allocation, not terminal geometry, is the authoritative budget:
        # Box/Frame chrome and the dialog's max height have already been removed.
        viewport = (max(1, width), max(1, height or 1))
        if viewport != self._viewport:
            self._viewport = viewport
            self._fragment_cache.clear()
        return super().create_content(width, height)

    def _text(self) -> StyleAndTextTuples:
        if self._viewport is None:
            try:
                size = self.picker.app.output.get_size()
                width, height = max(1, size.columns - 4), max(1, size.rows - 5)
            except Exception:
                width, height = 76, 19
        else:
            width, height = self._viewport
        fragments: StyleAndTextTuples = []
        # _render_fragments accepts outer geometry; compensate for chrome because
        # width/height here are the already allocated list viewport.
        for line_no, line in enumerate(_render_fragments(self.picker.model, width + 4, height + 5)):
            if line_no:
                fragments.append(("", "\n"))
            fragments.extend(line)
        return fragments


class ResumePicker:
    """Native prompt_toolkit controls and state hosted by the main Application."""

    def __init__(self, rows: list[ResumePickerRow], app: Any) -> None:
        self.app = app
        self.model = ResumePickerModel(rows)
        self.buffer = Buffer(multiline=False)
        self.buffer.on_text_changed += lambda _buffer: self.model.set_query(self.buffer.text)
        query = Window(
            BufferControl(
                self.buffer,
                input_processors=[
                    BeforeInput("Search  ", style="class:resume-picker.search-label")
                ],
            ),
            height=1,
        )
        self.list_control = _ResumeListControl(self)
        listing = Window(self.list_control, wrap_lines=False)
        self.container = Box(
            Frame(HSplit([query, listing]), title="Resume session"),
            padding=1,
            width=Dimension(min=40, preferred=76, max=92),
            height=Dimension(min=6, preferred=18, max=22),
        )
        self.query_control = query.content

    def move(self, delta: int) -> None:
        self.model.move(delta)

    def selected_id(self) -> str | None:
        return self.model.current.id if self.model.can_select and self.model.current else None
