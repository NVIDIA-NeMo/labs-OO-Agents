# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-application session chooser with separate list and conversation preview panes."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
from rich.cells import cell_len, split_graphemes

from .explorer_base import ExplorerOption
from .fullscreen_browser import (
    ExplorerBrowser,
)
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


# Folded-text memo: casefold expansion of a field is deterministic, and rows
# are immutable, so per-keystroke search reuses the expansion instead of
# rebuilding a per-character map over every conversation for every keystroke.
_FOLD_CACHE: dict[str, tuple[str, list[int]]] = {}
_FOLD_CACHE_MAX = 4096

# Preview transcripts hold a full replay projection each; keep only a few
# alive so revisited rows reuse their transcript without unbounded growth.
_PREVIEW_CACHE_MAX = 4

# Dwell delay before a preview build starts. Fast up/down navigation moves
# past rows in far less than this, so the per-keystroke builds that stacked
# GIL-heavy threads and starved the UI never start at all.
_PREVIEW_DEBOUNCE_SECONDS = 0.12

# Turns rendered per chunk in a cancellable preview build. The chunk size
# only bounds how long a superseded build can keep the GIL after Esc/Enter;
# output is identical to the whole-history build.
_PREVIEW_BUILD_CHUNK_TURNS = 50


def _folded_with_source(text: str) -> tuple[str, list[int]]:
    cached = _FOLD_CACHE.get(text)
    if cached is not None:
        return cached
    parts: list[str] = []
    source: list[int] = []
    for index, char in enumerate(text):
        folded = char.casefold()
        parts.append(folded)
        source.extend([index] * len(folded))
    entry = ("".join(parts), source)
    if len(_FOLD_CACHE) >= _FOLD_CACHE_MAX:
        _FOLD_CACHE.clear()
    _FOLD_CACHE[text] = entry
    return entry


def _term_hits(
    terms: list[str], text: str
) -> tuple[set[str], int, tuple[int, ...]] | None:
    """Locate query terms in one field, in original-text coordinates.

    Returns the distinct terms hit, the earliest hit position, and the
    original-text positions of every hit (for row highlighting).
    """
    target, source = _folded_with_source(text)
    hit: set[str] = set()
    earliest: int | None = None
    positions: list[int] = []
    for term in terms:
        needle = term.casefold()
        cursor = 0
        while (found := target.find(needle, cursor)) >= 0:
            hit.add(needle)
            if earliest is None or found < earliest:
                earliest = found
            positions.extend(source[found : found + len(needle)])
            cursor = found + len(needle)
        if needle not in hit:
            continue
    if not hit:
        return None
    assert earliest is not None
    return hit, earliest, tuple(dict.fromkeys(positions))


class ResumePickerModel:
    def __init__(self, rows: list[ResumePickerRow]) -> None:
        self.rows = rows
        self._search_fields = [self._fields_for(row) for row in rows]
        self.query = ""
        self.state_filter: Literal["detached", "attached", "all"] = "detached"
        self.sort_updated = True
        self.selected = self.list_offset = 0
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
            # Word-AND matching, shared with the explorers: every term must
            # occur somewhere across the row's fields (title, preview, or
            # conversation). The best field — most terms covered, earliest —
            # provides the ranking score and highlight positions. The score
            # keeps the coverage count so rows with all terms in one field
            # outrank rows whose terms are split across fields.
            terms = [term for term in normalized_query.split() if term.strip()]
            needed = {term.casefold() for term in terms}
            query_matches: list[tuple[int, int, str, tuple[int, ...]] | None] = []
            for fields in self._search_fields:
                best: tuple[int, int, str, tuple[int, ...]] | None = None
                covered: set[str] = set()
                for field, value in fields.items():
                    hits = _term_hits(terms, value)
                    if hits is None:
                        continue
                    hit, earliest, positions = hits
                    covered.update(hit)
                    candidate = (len(hit), -earliest, field, positions)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
                if best is not None and covered == needed:
                    query_matches.append((best[0], best[1], best[2], best[3]))
                else:
                    query_matches.append(None)
            self._query_matches = query_matches
        else:
            self._query_matches = [(0, 0, "", ()) for _row in self.rows]
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
            coverage, score, field, positions = query_match
            stamp = row.last_active if self.sort_updated else row.created_at
            ranked.append(
                (
                    -stamp,
                    # Ascending sort: negate coverage so the most-covered
                    # field ranks first, then the earliest match position.
                    (-coverage, -score) if self.query.strip() else (0, 0),
                    index,
                    FieldMatch(row, field or None, positions),
                )
            )
        ranked.sort(key=lambda item: item[:3])
        self._matches = [item[3] for item in ranked]
        ids = [match.row.id for match in self._matches]
        self.selected = ids.index(previous_id) if previous_id in ids else 0
        self.list_offset = min(self.list_offset, self.selected)

    def set_filter(self, value: Literal["detached", "attached", "all"]) -> None:
        """Apply a filter value and restart the list from the top."""
        self.state_filter = value
        self.selected = self.list_offset = 0
        self._rebuild_matches()

    def cycle_filter(self, delta: int = 1) -> None:
        filters: tuple[Literal["detached", "attached", "all"], ...] = (
            "detached",
            "attached",
            "all",
        )
        self.set_filter(filters[(filters.index(self.state_filter) + delta) % len(filters)])

    def toggle_filter(self) -> None:
        self.cycle_filter()

    def set_sort(self, updated: bool) -> None:
        """Apply a sort direction and restart the list from the top."""
        self.sort_updated = updated
        self.selected = self.list_offset = 0
        self._rebuild_matches()

    def toggle_sort(self) -> None:
        self.set_sort(not self.sort_updated)

    def move(self, delta: int) -> None:
        if not self._matches or not delta:
            return
        self.selected = (self.selected + delta) % len(self._matches)

    def jump_home(self) -> None:
        if self._matches:
            self.selected = 0
            self.list_offset = 0

    def jump_end(self) -> None:
        if self._matches:
            self.selected = len(self._matches) - 1
            # Keep the selection on screen: the list renders from
            # list_offset, so End must scroll it into view too.
            self.list_offset = max(0, len(self._matches) - 1)

    def select(self, index: int) -> None:
        if 0 <= index < len(self._matches) and index != self.selected:
            self.selected = index

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


def _match_snippet(match: FieldMatch, available: int) -> tuple[str, tuple[int, ...]] | None:
    """Return preview-column text that makes an invisible match visible.

    The row normally shows the start of the newest agent message, which can
    clip the matched text away — or the match may live in the conversation
    field, which the row never displays. In both cases show the matched line
    so every listed session shows *why* it matched.
    """
    row = match.row
    positions = match.positions
    if not positions:
        return None
    if match.field == "conversation":
        text = "\n".join(sanitize_live_text(turn.content) for turn in row.turns)
    elif match.field == "preview" and positions[0] >= max(available - 2, 0):
        text = row.preview
    else:
        return None
    first = positions[0]
    if match.field == "conversation":
        start = text.rfind("\n", 0, first) + 1
        stop = text.find("\n", first)
        stop = len(text) if stop < 0 else stop
        # Long conversation lines would still clip the match away; shift the
        # window toward the match instead of starting at the line beginning.
        if first - start > 40:
            start = first - 24
    else:
        start = max(0, first - 24)
        stop = len(text)
    return text[start:stop], tuple(p - start for p in positions if start <= p < stop)


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
    available = max(0, width - 15 - title_width - 2)
    snippet = _match_snippet(match, available)
    if snippet is not None:
        preview_text, preview_positions = snippet
    else:
        preview_text = row.preview
        preview_positions = match.positions if match.field == "preview" else ()
    return [
        _clip_fragments(
            [
                (base, prefix),
                *title,
                *_field_fragments(
                    preview_text,
                    base + " class:fullscreen-browser.meta",
                    preview_positions,
                ),
            ],
            width,
        )
    ]


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


class ResumePicker(ExplorerBrowser):
    """Session-specific ExplorerBrowser with asynchronous replay previews.

    Navigation and options dispatch through the inherited
    ``ExplorerBrowser.handle_key`` via the view facade; only the picker's
    lifecycle keys (Escape/Enter/Ctrl-C) stay host-bound because finishing
    the dialog is a host concern, not a view action.
    """

    class _ViewFacade:
        """View contract for the shared ``ExplorerBrowser.handle_key``."""

        config = SimpleNamespace(actions={})
        item_name = "session"
        title = "Resume a previous session"

        def __init__(self, picker: ResumePicker, options: tuple[ExplorerOption, ...]) -> None:
            self._picker = picker
            self.options = options
            self.pending_input: str | None = None

        @property
        def model(self) -> ResumePickerModel:
            # The shared construction seeds the search buffer from
            # ``view.model.query``; the picker's model is the real one.
            return self._picker.model

        def handle_key(self, _action: str, _value: str = "") -> str:
            return "handled"

        def handle_action(self, action: str, row: Any) -> str:
            if action == "enter" and row is not None:
                selected = self._picker.selected_id()
                if selected is not None:
                    self.pending_input = selected
                    return "close"
            return "ignored"

    @property
    def view(self) -> ResumePicker._ViewFacade:
        return self._view_facade

    @view.setter
    def view(self, value: ResumePicker._ViewFacade) -> None:
        # The shared construction assigns the facade passed to
        # super().__init__; keep it in the same slot the property reads.
        self._view_facade = value

    @property
    def model(self) -> ResumePickerModel:
        return self._resume_model

    @model.setter
    def model(self, value: ResumePickerModel) -> None:
        self._resume_model = value

    def _create_list_control(self) -> Any:
        # The picker's list renders its compact FieldMatch rows.
        return _PickerControl(self)

    def _option_window_width(self, index: int) -> Dimension:
        widths = (Dimension(min=17, preferred=20), Dimension(min=10, preferred=18))
        # Future option rows fall back to the shared default width rather
        # than raising on the two-entry tuple.
        return widths[index] if index < len(widths) else Dimension(min=12, preferred=22)

    def __init__(
        self,
        rows: list[ResumePickerRow],
        app: Any,
        *,
        selection_copy_callback: Callable[[str], None] | None = None,
        selection_status: Callable[[], str] | None = None,
    ):
        self.app = app
        self.model = ResumePickerModel(rows)
        # Real option rows drive the shared option controls and the inherited
        # options mode; their callbacks mirror the model's cycle/toggle
        # semantics (restart from the top, refresh the prepared preview).
        self._view_facade = ResumePicker._ViewFacade(self, self._build_view_options())
        # Preview transcripts are large; keep a bounded LRU so revisiting rows
        # reuses their transcript instead of growing without limit.
        self._preview_models: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._preview_tasks: dict[tuple[str, int], Any] = {}
        # Event for the in-flight preview build thread; superseding or
        # closing sets it so a stale build stops between chunks.
        self._preview_build_cancel: threading.Event | None = None
        self.native_selection = False
        super().__init__(
            self._view_facade,
            app,
            selection_copy_callback=selection_copy_callback,
            selection_status=selection_status,
        )

    def _build_view_options(self) -> tuple[ExplorerOption, ...]:
        """The picker's filter/sort controls as real shared option rows.

        The callbacks preserve the model's filter/sort semantics (restart from
        the top of the list, refresh the prepared preview) while the shared
        option controls and options mode handle rendering and navigation.
        """

        def on_filter(value: str) -> None:
            self.model.set_filter(value)  # type: ignore[arg-type]
            # A filter/sort change can select a different row: reset the
            # (possibly cached) current preview's viewport so it follows the
            # tail instead of keeping the previous scroll position.
            self._reset_preview()
            self._prepare_current_preview()
            self.invalidate()

        def on_sort(value: str) -> None:
            self.model.set_sort(value == "updated")
            self._reset_preview()
            self._prepare_current_preview()
            self.invalidate()

        return (
            ExplorerOption(
                key="filter",
                label="Filter",
                choices=(
                    ("detached", "✓ Not attached"),
                    ("attached", "✗ Attached"),
                    ("all", "✓/✗ All"),
                ),
                value=self.model.state_filter,
                on_change=on_filter,
            ),
            ExplorerOption(
                key="sort",
                label="Sort",
                choices=(("updated", "Recent activity"), ("created", "Creation date")),
                value="updated" if self.model.sort_updated else "created",
                on_change=on_sort,
            ),
        )

    def _query_changed(self) -> None:
        self.model.set_query(self.buffer.text)
        # Typing a query can swap the selected row, exactly like the filter
        # and sort handlers: reset the cached current preview so it follows
        # the tail instead of keeping the previous row's scroll position.
        self._reset_preview()
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

    def page(self, delta: int) -> None:
        if self.active_control == "list":
            self.move(delta * max(1, self.list_control.viewport[1]))
        elif self.active_control == "preview":
            self.scroll_preview(delta * max(1, self.preview_control.viewport[1]))

    def _build_preview_model(self, row: ResumePickerRow, width: int, height: int):
        """Render the preview in chunks, stopping early when superseded.

        ``threading.Event`` cancellation works inside ``asyncio.to_thread``:
        ``Task.cancel()`` only interrupts the coroutine awaiting the thread,
        while the thread itself keeps its GIL-bound projection running and
        starves the UI thread (the slow Esc/Enter after browsing). Polling
        ``self._preview_build_cancel`` between small chunks bounds a superseded
        build to one chunk instead of the whole history.

        Chunking is behavior-neutral: rendering per chunk and appending in
        order produces the same projected rows and formatted text as a
        whole-history render (verified against a real 1894-turn session).
        """
        from .fullscreen_transcript import FullscreenTranscriptModel
        from .output import HistoryTurn

        transcript = FullscreenTranscriptModel(show_trailing_blank=False)
        turns = [HistoryTurn(turn.role, turn.content) for turn in row.turns]
        cancel = self._preview_build_cancel
        for start in range(0, len(turns), _PREVIEW_BUILD_CHUNK_TURNS):
            if cancel is not None and cancel.is_set():
                return None
            transcript.append(self._render_preview_chunk(turns, start, row.id, width))
        # Projection is the dominant first-render cost for long histories.
        transcript.formatted_text(width=width, height=max(1, height))
        return transcript

    @staticmethod
    def _render_preview_chunk(turns: list[Any], start: int, row_id: str, width: int) -> str:
        """Render one chunk of the replay to ANSI (pure; safe on any thread)."""
        from .frontend import render_history_replay_to_ansi
        from .output import HistoryReplay

        chunk = turns[start : start + _PREVIEW_BUILD_CHUNK_TURNS]
        return render_history_replay_to_ansi(
            HistoryReplay(
                turns=chunk,
                # The whole-history render prints this header once.
                session_id=row_id[:8] if start == 0 else "",
                show_header=False,
                show_footer=False,
            ),
            width,
        )

    async def _build_preview_progressively(
        self,
        row: ResumePickerRow,
        width: int,
        height: int,
        on_chunk: Callable[[Any], None] | None = None,
    ):
        """Build the preview with the transcript owned by the UI loop.

        Rendering is pure string work and runs on a worker thread, while
        every append (and its incremental projection extension) happens on
        the UI loop. That keeps the transcript single-threaded, so a partial
        can be published and rendered while later chunks are still rendering
        instead of showing "Preparing…" for the whole build. ``on_chunk``
        receives the partial transcript after each chunk; returns None when
        superseded.
        """
        import asyncio

        from .fullscreen_transcript import FullscreenTranscriptModel
        from .output import HistoryTurn

        transcript = FullscreenTranscriptModel(show_trailing_blank=False)
        turns = [HistoryTurn(turn.role, turn.content) for turn in row.turns]
        cancel = self._preview_build_cancel
        for start in range(0, len(turns), _PREVIEW_BUILD_CHUNK_TURNS):
            if cancel is not None and cancel.is_set():
                return None
            ansi = await asyncio.to_thread(
                self._render_preview_chunk, turns, start, row.id, width
            )
            if cancel is not None and cancel.is_set():
                return None
            transcript.append(ansi)
            # Seed the projection on the first chunk so later appends extend
            # it incrementally; otherwise the whole-history projection would
            # land on the UI loop as one multi-second stall at the end.
            transcript.formatted_text(width=max(1, width), height=max(1, height))
            if on_chunk is not None:
                on_chunk(transcript)
        return transcript

    def _preview_model(self, width: int):
        row = self.model.current
        if row is None:
            return None
        key = (row.id, max(1, width))
        transcript = self._preview_models.get(key)
        if transcript is not None:
            # Most-recently-used preview stays alive; older ones evict first.
            self._preview_models.move_to_end(key)
            return transcript
        self._prepare_current_preview()
        # Synchronous callers (primarily deterministic unit tests) have no event
        # loop in which to schedule preparation, so retain a direct fallback.
        if key not in self._preview_tasks:
            # A fresh event mirrors prepare(): a stale *set* event left by an
            # earlier supersede/close must not cancel the fallback build.
            self._preview_build_cancel = threading.Event()
            transcript = self._build_preview_model(row, key[1], self.preview_control.viewport[1])
            if transcript is not None:
                self._remember_preview(key, transcript)
            return transcript
        return None

    def _remember_preview(self, key: tuple[str, int], transcript: Any) -> None:
        """Insert a built preview into the bounded LRU cache."""
        self._preview_models[key] = transcript
        while len(self._preview_models) > _PREVIEW_CACHE_MAX:
            self._preview_models.popitem(last=False)

    def _prepare_current_preview(self) -> None:
        import asyncio

        row = self.model.current
        if row is None or not row.turns:
            return
        width, height = self.preview_control.viewport
        key = (row.id, max(1, width))
        if key in self._preview_models or key in self._preview_tasks:
            return

        # Single-flight: only the row the user currently sees gets built.
        # Superseded tasks are cancelled, so fast up/down navigation cannot
        # stack one build per keystroke. Task cancellation stops the awaiter,
        # but the build thread itself keeps its GIL-bound projection running
        # (the slow Esc/Enter after browsing), so the cooperative event also
        # tells the in-flight build to stop at its next chunk boundary.
        for pending_key, task in list(self._preview_tasks.items()):
            if pending_key != key:
                task.cancel()
                self._preview_tasks.pop(pending_key, None)
                # A cached entry for a still-pending key can only be a partial
                # published mid-build; drop it so returning to that row rebuilds
                # the full history instead of freezing at a prefix.
                self._preview_models.pop(pending_key, None)
        if self._preview_build_cancel is not None:
            self._preview_build_cancel.set()

        async def prepare() -> None:
            try:
                # Dwell gate: navigation away cancels this task during the
                # delay, so only rows the user actually stop on get built.
                await asyncio.sleep(_PREVIEW_DEBOUNCE_SECONDS)
                if self.model.current is not row:
                    return
                self._preview_build_cancel = threading.Event()

                def publish_partial(partial: Any) -> None:
                    # Runs on the UI loop between chunks: the transcript is
                    # owned here, so publishing mid-build is safe. Supersede
                    # drops the partial (see the cancellation loop above), so
                    # a stale prefix can never freeze as a row's preview.
                    if self.model.query.strip():
                        # Recomputing search matches per streamed chunk is
                        # quadratic for long histories; queried rows keep the
                        # "Preparing…" frame until the finished preview lands.
                        return
                    self._remember_preview(key, partial)
                    self.invalidate()

                transcript = await self._build_preview_progressively(
                    row, key[1], max(1, height), on_chunk=publish_partial
                )
            except asyncio.CancelledError:
                return
            except Exception:
                return
            finally:
                self._preview_tasks.pop(key, None)
            if transcript is None:
                return
            if self.model.query.strip():
                # A query typed mid-build computed matches over a partial
                # history; set_search early-returns on an unchanged query, so
                # recompute over the finished transcript before publishing.
                query_width, query_height = self.preview_control.viewport
                transcript.set_search("", width=max(1, query_width), height=max(1, query_height))
                transcript.set_search(
                    self.model.query, width=max(1, query_width), height=max(1, query_height)
                )
            self._remember_preview(key, transcript)
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
        # Escape/Enter can arrive while a build thread is mid-history; the
        # cooperative event stops it at the next chunk boundary so closing
        # (and the resume that follows) is not starved by GIL-bound work.
        if self._preview_build_cancel is not None:
            self._preview_build_cancel.set()

    def selected_id(self) -> str | None:
        return self.model.current.id if self.model.can_select and self.model.current else None
