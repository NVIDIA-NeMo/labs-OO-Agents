# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session explorer model, renderer, and in-app subview."""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from .explorer_base import (
    BAR_STYLE,
    ExplorerInteraction,
    display_line,
    render_markdown_lines,
    search_terms,
    style_bar,
    style_fts_prompt,
    style_mode_label,
    wrap_plain_line,
)
from .session_manager import SessionManager, Turn
from .subapp import SubviewKeyResult


@dataclass
class SessionExplorerRow:
    id: str
    name: str
    model: str
    agent: str
    working_dir: str
    started_at: float
    last_active: float
    turn_count: int
    turns: list[Turn]
    search_text: str

    @property
    def title(self) -> str:
        return self.name or self.id[:8]


class SessionExplorerModel:
    """Searchable, keyboard-navigable session list."""

    def __init__(self, rows: list[SessionExplorerRow]) -> None:
        self.rows = rows
        self.session_query = ""
        self.detail_query = ""
        self.search_scope = "sessions"
        self.matches = list(range(len(rows)))
        self.cursor = 0
        self.detail_offset = 10**9
        self.focus = "list"
        self.search_active = False
        self._last_detail_line_count = 0
        self._last_detail_visible_lines = 0
        self._last_detail_match_lines: list[int] = []
        self.detail_search_cursor = 0
        self._pending_detail_match_delta = 0
        self.detail_tail_skip = 0
        self._last_detail_count_exact = True
        self._detail_cache: dict[tuple[str, int], list[str]] = {}

    @property
    def current_index(self) -> int | None:
        if not self.matches:
            return None
        self.cursor = min(max(self.cursor, 0), len(self.matches) - 1)
        return self.matches[self.cursor]

    @property
    def current(self) -> SessionExplorerRow | None:
        idx = self.current_index
        return None if idx is None else self.rows[idx]

    @property
    def query(self) -> str:
        return self.session_query if self.search_scope == "sessions" else self.detail_query

    def set_query(self, query: str, *, scope: str | None = None) -> None:
        if scope is not None:
            self.search_scope = scope
        words = [w.lower() for w in query.split() if w.strip()]
        if self.search_scope == "sessions":
            self.session_query = query
            if not words:
                self.matches = list(range(len(self.rows)))
            else:
                self.matches = [
                    i
                    for i, row in enumerate(self.rows)
                    if all(word in row.search_text.lower() for word in words)
                ]
            self.cursor = 0
        else:
            self.detail_query = query
        self.detail_offset = 0 if words and self.search_scope == "dialog" else 10**9
        self.detail_search_cursor = 0
        self._pending_detail_match_delta = 0
        self.detail_tail_skip = 0

    def clear_query(self) -> None:
        self.set_query("", scope=self.search_scope)

    def edit_query(self, text: str) -> None:
        self.set_query(text)
        self.search_active = True

    def move(self, delta: int) -> None:
        if not self.matches:
            return
        self.cursor = min(max(self.cursor + delta, 0), len(self.matches) - 1)
        self.detail_offset = 10**9
        self.detail_search_cursor = 0
        self._pending_detail_match_delta = 0
        self.detail_tail_skip = 0

    def move_or_scroll(self, delta: int) -> None:
        if self.search_active and self.search_scope == "dialog" and self.focus == "dialog":
            self.move_detail_match(delta)
        elif self.focus == "list":
            self.move(delta)
        else:
            self.scroll_detail(delta)

    def move_detail_match(self, delta: int) -> None:
        if not self._last_detail_match_lines:
            self._pending_detail_match_delta += delta
            return
        count = len(self._last_detail_match_lines)
        self.detail_search_cursor = min(max(self.detail_search_cursor + delta, 0), count - 1)
        visible = max(self._last_detail_visible_lines, 1)
        line = self._last_detail_match_lines[self.detail_search_cursor]
        self.detail_offset = max(line - visible // 2, 0)
        self.clamp_detail_offset(visible)

    def jump_home(self) -> None:
        self.cursor = 0
        self.detail_offset = 10**9

    def jump_end(self) -> None:
        if self.matches:
            self.cursor = len(self.matches) - 1
            self.detail_offset = 10**9

    def toggle_focus(self) -> None:
        self.focus = "dialog" if self.focus == "list" else "list"

    def scroll_detail(self, delta: int) -> None:
        visible = max(self._last_detail_visible_lines, 1)
        if not self._last_detail_count_exact and self.detail_offset >= 10**8:
            self.detail_tail_skip = max(self.detail_tail_skip - delta, 0)
            return
        max_offset = max(self._last_detail_line_count - visible, 0)
        current = max_offset if self.detail_offset >= 10**8 else self.detail_offset
        self.detail_offset = min(max(current + delta, 0), max_offset)
        self.detail_tail_skip = 0

    def page_detail(self, delta: int) -> None:
        self.scroll_detail(delta)

    def clamp_detail_offset(self, visible_lines: int) -> None:
        if not self._last_detail_count_exact and self.detail_offset >= 10**8:
            return
        max_offset = max(self._last_detail_line_count - max(visible_lines, 1), 0)
        self.detail_offset = min(max(self.detail_offset, 0), max_offset)


class SessionExplorerView(ExplorerInteraction):
    """In-app subview wrapper for browsing stored sessions."""

    title = "sessions"
    detail_focus = "dialog"

    def __init__(self, *, limit: int = 100) -> None:
        self.model = SessionExplorerModel(build_session_rows(limit=limit))
        self.pending_input: str | None = None

    def render(self, width: int, height: int) -> str:
        return render_session_explorer(
            self.model,
            width,
            height,
            ansi=True,
            selection_hint=self.selection_hint(),
        )

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
            row = model.current
            if row is None:
                return "handled"
            self.pending_input = f"/session resume {row.id[:8]}"
            return "close"
        if action == "escape":
            if model.search_active:
                model.search_active = False
            elif model.query:
                model.clear_query()
            else:
                return "ignored"
        elif action == "enter":
            model.search_active = False
        elif action == "slash":
            if model.search_active:
                model.edit_query(model.query + "/")
            else:
                model.search_scope = "dialog" if model.focus == "dialog" else "sessions"
                model.search_active = True
        elif action == "backspace":
            if model.search_active:
                model.edit_query(model.query[:-1])
        elif action == "tab":
            old_query = model.query
            model.toggle_focus()
            if model.search_active:
                model.search_scope = "dialog" if model.focus == "dialog" else "sessions"
                if model.search_scope == "dialog" and not model.detail_query:
                    model.set_query(old_query, scope="dialog")
                elif model.search_scope == "sessions" and not model.session_query:
                    model.set_query(old_query, scope="sessions")
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
            model.focus = "dialog"
            model.scroll_detail(+3)
        elif action == "scroll_up":
            model.focus = "dialog"
            model.scroll_detail(-3)
        elif action == "text":
            if model.search_active and value and value.isprintable():
                model.edit_query(model.query + value)
            else:
                return "ignored"
        else:
            return "ignored"
        return "handled"

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass


def build_session_rows(*, limit: int = 100) -> list[SessionExplorerRow]:
    """Build explorer rows from persisted sessions, newest first."""
    rows: list[SessionExplorerRow] = []
    for meta in SessionManager.list_sessions(limit=limit):
        if meta.turn_count <= 0:
            continue
        turns = SessionManager.load_turns(meta.id)
        search_parts = [
            meta.id,
            meta.name or "",
            meta.model,
            meta.agent,
            meta.working_dir,
            str(meta.turn_count),
            *[turn.content for turn in turns],
        ]
        rows.append(
            SessionExplorerRow(
                id=meta.id,
                name=meta.name or "",
                model=meta.model,
                agent=meta.agent,
                working_dir=meta.working_dir,
                started_at=meta.started_at,
                last_active=meta.last_active,
                turn_count=meta.turn_count,
                turns=turns,
                search_text="\n".join(search_parts),
            )
        )
    return rows


def _format_time(ts: float) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
    except Exception:
        return "?"


def _detail_match_lines(lines: list[str], terms: list[str]) -> list[int]:
    if not terms:
        return []
    lowered = [term.lower() for term in terms]
    matches: list[int] = []
    body_matches: list[int] = []
    in_dialog = False
    for i, line in enumerate(lines):
        if line == "Dialog:":
            in_dialog = True
            continue
        lower = line.lower()
        if any(term in lower for term in lowered):
            matches.append(i)
            if in_dialog:
                body_matches.append(i)
    return body_matches or matches


def _detail_raw_lines(row: SessionExplorerRow) -> list[str]:
    lines: list[str] = [
        f"Session: {row.title} ({row.id[:8]})",
        f"Model: {row.model}",
        f"Agent: {row.agent}",
        f"Working dir: {row.working_dir or '(unknown)'}",
        f"Started: {_format_time(row.started_at)}   Last active: {_format_time(row.last_active)}   Turns: {row.turn_count}",
        "",
        "Dialog:",
    ]
    for turn in row.turns:
        lines.append("You:" if turn.role == "user" else "OO:")
        for raw in turn.content.splitlines() or [""]:
            lines.append(f"  {raw}")
        lines.append("")
    return lines


def wrapped_detail_lines(row: SessionExplorerRow, width: int) -> list[str]:
    width = max(int(width), 20)
    lines: list[str] = []
    for raw in _detail_raw_lines(row):
        lines.extend(wrap_plain_line(raw, width))
    return lines


def _reversed_detail_raw_lines(row: SessionExplorerRow):
    for turn in reversed(row.turns):
        yield ""
        for raw in reversed(turn.content.splitlines() or [""]):
            yield f"  {raw}"
        yield "You:" if turn.role == "user" else "OO:"
    yield "Dialog:"
    yield ""
    yield f"Started: {_format_time(row.started_at)}   Last active: {_format_time(row.last_active)}   Turns: {row.turn_count}"
    yield f"Working dir: {row.working_dir or '(unknown)'}"
    yield f"Agent: {row.agent}"
    yield f"Model: {row.model}"
    yield f"Session: {row.title} ({row.id[:8]})"


def _tail_detail_lines(
    row: SessionExplorerRow, width: int, visible: int, skip: int, *, markdown: bool = False
) -> list[str]:
    if visible <= 0:
        return []
    width = max(int(width), 20)
    remaining_skip = max(skip, 0)
    selected: list[str] = []
    # Render only a small raw window. With markdown enabled, grab a little extra
    # context so Rich can format lists/fences while still avoiding whole-session work.
    target = visible + (8 if markdown else 0)
    for raw in _reversed_detail_raw_lines(row):
        wrapped = wrap_plain_line(raw, width)
        for line in reversed(wrapped):
            if remaining_skip > 0:
                remaining_skip -= 1
                continue
            selected.append(line)
            if len(selected) >= target:
                lines = list(reversed(selected))
                break
        else:
            continue
        break
    else:
        lines = list(reversed(selected))
    if not markdown:
        return lines[:visible]
    rendered: list[str] = []
    chunk: list[str] = []

    def flush_chunk() -> None:
        if chunk:
            rendered.extend(render_markdown_lines("\n".join(chunk), width))
            chunk.clear()

    for line in lines:
        stripped = line.strip()
        if stripped in {"You:", "OO:", "Dialog:"} or stripped.startswith(
            ("Session:", "Model:", "Agent:", "Working dir:", "Started:")
        ):
            flush_chunk()
            rendered.append(line)
        elif line.startswith("  "):
            chunk.append(line[2:])
        else:
            chunk.append(line)
    flush_chunk()
    return rendered[-visible:]


def _cached_detail_lines(
    model: SessionExplorerModel, row: SessionExplorerRow, width: int
) -> list[str]:
    key = (row.id, max(int(width), 20))
    cached = model._detail_cache.get(key)
    if cached is not None:
        return cached
    lines = wrapped_detail_lines(row, width)
    if len(model._detail_cache) > 32:
        model._detail_cache.clear()
    model._detail_cache[key] = lines
    return lines


def render_session_explorer(
    model: SessionExplorerModel,
    width: int,
    height: int,
    *,
    ansi: bool = False,
    selection_hint: str = "F2 select/copy",
) -> str:
    """Render the current session explorer state as text/ANSI."""
    width = max(int(width), 40)
    height = max(int(height), 1)
    row = model.current
    match_count = len(model.matches)
    total = len(model.rows)
    active_query = model.query
    query = f" {model.search_scope}_search={active_query!r}" if active_query else ""
    pos = f" {model.cursor + 1}/{match_count}" if match_count else " 0/0"
    match_label = (
        f" match {model.cursor + 1}/{match_count}"
        if model.session_query and model.search_scope == "sessions" and match_count
        else ""
    )
    header = style_bar(
        f" Session Explorer{pos} of {total}{match_label}{query} ".ljust(width, "─")[:width],
        ansi=ansi,
    )
    pane_label = "sessions" if model.focus == "list" else "session dialog"
    if model.search_active:
        mode_text = "FTS MODE"
        search_prompt = f"FTS {model.search_scope}: {model.query}"
        enter_hint = "enter exit FTS"
    else:
        mode_text = "BROWSE MODE"
        search_prompt = "/ FTS"
        enter_hint = ""
    focus_label = f"{mode_text} · pane={pane_label}"
    footer_parts = [
        focus_label,
        "↑/↓ sessions/scroll",
        selection_hint,
        "tab switch pane",
        search_prompt,
        enter_hint,
        "esc clear",
        "q close",
    ]
    footer_plain = (" " + "  ".join(part for part in footer_parts if part) + " ").ljust(width, "─")[
        :width
    ]
    if ansi:
        before_mode, after_mode = footer_plain.split(mode_text, 1)
        styled_mode = style_mode_label(mode_text, active=model.search_active, ansi=True)
        footer = f"{BAR_STYLE}{before_mode}{styled_mode}{BAR_STYLE}{after_mode}\x1b[0m"
    else:
        footer = footer_plain

    body_height = max(height - 2, 0)
    model._last_divider_y = 0
    if not total:
        body = ["No sessions found."]
    elif row is None:
        body = [f"No matches for {model.query!r}."]
    else:
        list_count = min(10, max(3, body_height // 3))
        half = list_count // 2
        start = max(0, model.cursor - half)
        end = min(match_count, start + list_count)
        start = max(0, end - list_count)
        list_terms = search_terms(model.session_query)
        detail_terms = search_terms(
            model.detail_query if model.search_scope == "dialog" else model.session_query
        )
        body = []
        for visible_i in range(start, end):
            row_i = model.matches[visible_i]
            item = model.rows[row_i]
            marker = (
                "❯"
                if visible_i == model.cursor and model.focus == "list"
                else "•"
                if visible_i == model.cursor
                else " "
            )
            line = (
                f"{marker} {item.id[:8]:<8} {_format_time(item.last_active):<11} "
                f"{str(item.turn_count):>4} {item.model.split('/')[-1][:18]:<18} {item.title}"
            )
            body.append(display_line(line, width, list_terms, ansi=ansi))
        divider_label = (
            f"[FTS {model.search_scope}: {model.query}] "
            if model.search_active or model.query
            else "[/: FTS current pane] "
        )
        divider_plain = divider_label.ljust(width, "─")[:width]
        if ansi and (model.search_active or model.query):
            divider = style_fts_prompt(divider_label, active=model.search_active, ansi=True)
            divider += "─" * max(width - len(divider_label), 0)
            body.append(divider)
        else:
            body.append(divider_plain)
        model._last_divider_y = len(body)
        available = max(body_height - len(body), 0)
        detail_visible = max(available - 1, 0)
        model._last_detail_visible_lines = detail_visible
        if detail_terms or model.detail_offset < 10**8:
            plain_detail_lines = _cached_detail_lines(model, row, width)
            model._last_detail_count_exact = True
            model._last_detail_match_lines = _detail_match_lines(plain_detail_lines, detail_terms)
            if model._last_detail_match_lines:
                cursor = model.detail_search_cursor + model._pending_detail_match_delta
                model.detail_search_cursor = min(
                    max(cursor, 0), len(model._last_detail_match_lines) - 1
                )
                model._pending_detail_match_delta = 0
            else:
                model.detail_search_cursor = 0
                model._pending_detail_match_delta = 0
            model._last_detail_line_count = len(plain_detail_lines)
            if model.search_active and model._last_detail_match_lines:
                visible = max(model._last_detail_visible_lines, 1)
                line = model._last_detail_match_lines[model.detail_search_cursor]
                if (
                    model.detail_offset >= 10**8
                    or (model.search_scope == "dialog" and model.focus == "dialog")
                    and not (model.detail_offset <= line < model.detail_offset + visible)
                ):
                    model.detail_offset = max(line - visible // 2, 0)
            model.clamp_detail_offset(detail_visible)
            visible_detail_lines = plain_detail_lines[
                model.detail_offset : model.detail_offset + detail_visible
            ]
            if detail_terms:
                current_match_line = (
                    model._last_detail_match_lines[model.detail_search_cursor]
                    if model._last_detail_match_lines
                    else None
                )
                visible_detail_lines = [
                    display_line(
                        line,
                        width,
                        detail_terms,
                        ansi=ansi,
                        current_match=model.detail_offset + i == current_match_line,
                    )
                    for i, line in enumerate(visible_detail_lines)
                ]
        else:
            model._last_detail_count_exact = False
            model._last_detail_match_lines = []
            model.detail_search_cursor = 0
            model._last_detail_line_count = 10**9
            visible_detail_lines = _tail_detail_lines(
                row, width, detail_visible, model.detail_tail_skip, markdown=ansi
            )
        detail_marker = "❯" if model.focus == "dialog" else " "
        if available > 0:
            detail_header = f"{detail_marker} dialog: {row.title} ({row.id[:8]})"
            body.append(detail_header[:width])
        for line in visible_detail_lines:
            body.append(line if ansi and "\x1b[" in line else line.ljust(width)[:width])
    body = [
        line if ansi and "\x1b[" in line else line.ljust(width)[:width]
        for line in body[:body_height]
    ]
    while len(body) < body_height:
        body.append("".ljust(width))
    return "\n".join([header, *body, footer])
