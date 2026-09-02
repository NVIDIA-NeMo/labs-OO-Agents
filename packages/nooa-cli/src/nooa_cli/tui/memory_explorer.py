# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory explorer — browse and curate the agent's long-term memory store.

Rows are prebuilt snapshots: ``build_memory_rows`` touches the SQLite store
and resolves references against live agent state, so it must run on the
agent thread (the composition root dispatches it through the local runtime).
The view itself only renders those snapshots; the ``f``/``d`` actions go
through host-supplied callables that route back to the agent thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .explorer_base import (
    ExplorerConfig,
    ExplorerModel,
    ExplorerView,
    render_markdown_lines,
    wrap_plain_line,
)
from .subapp import SubviewKeyResult

_TYPE_GLYPHS = {
    "info": "ℹ",
    "skill": "⚙",
    "episode": "◈",
    "intent": "⚡",
    "reflection": "✦",
    "scratch": "≈",
}
_TODO_GLYPHS = {"open": "○", "done": "✓", "dropped": "⊘"}


def _span(seconds: float) -> str:
    """Compact duration ("42s", "3m", "2h", "5d")."""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def relative_age(ts: float | None, now: float | None = None) -> str:
    """Relative age of a past timestamp; "never" for None."""
    if ts is None:
        return "never"
    now = time.time() if now is None else now
    return _span(now - ts)


@dataclass
class MemoryExplorerRow:
    """One memory record in the explorer list (a prebuilt snapshot)."""

    id: str
    type: str
    status: str | None  # todo lifecycle (open/done/dropped); None otherwise
    title: str | None
    content: str
    owner: str
    importance: str  # verbal band (CRITICAL..TRIVIAL)
    tags: list[str]
    created_at: float
    last_accessed_at: float
    usage: dict  # per_memory_usage(...) output
    reference_lines: list[str]  # pre-rendered via references.resolve/render
    edges: list[tuple[str, str, float]]  # (target_id, edge_type, weight)
    search_text: str

    @property
    def type_tag(self) -> str:
        return f"{self.type}:{self.status}" if self.status is not None else self.type


def build_memory_rows(agent: Any, manager: Any) -> list[MemoryExplorerRow]:
    """Build explorer rows from a ``MemoryManager``'s store, newest first.

    Touches the store and resolves references against the live agent, so it
    MUST run on the agent thread.
    """
    from nooa_memory.observability import per_memory_usage
    from nooa_memory.references import render, resolve

    rows: list[MemoryExplorerRow] = []
    for m in manager.store.all_memories():
        tag = f"{m.type.value}:{m.status}" if m.status is not None else m.type.value
        search_parts = [m.id, tag, m.title or "", m.content, *m.tags, m.owner]
        rows.append(
            MemoryExplorerRow(
                id=m.id,
                type=m.type.value,
                status=m.status,
                title=m.title,
                content=m.content,
                owner=m.owner,
                importance=m.importance_label(),
                tags=list(m.tags),
                created_at=m.created_at,
                last_accessed_at=m.last_accessed_at,
                usage=per_memory_usage(m, forgetting=manager.forgetting),
                reference_lines=[
                    render(resolve(agent, manager.store, ref)) for ref in m.references
                ],
                edges=[
                    (e.target_id, e.type.value, e.weight) for e in manager.store.neighbors(m.id)
                ],
                search_text="\n".join(search_parts),
            )
        )
    rows.sort(key=lambda r: -r.created_at)
    return rows


def last_reflection_summary(manager: Any) -> str | None:
    """One header line for the latest consolidation run (agent thread only).

    ``last reflection: 2m ago — merged 3, +5 edges, pruned 1 (idle, 1.4s)``;
    interrupted runs render ``(idle, interrupted @ form_edges, 0.3s)``.
    """
    history = manager.store.maintenance_history(1)
    if not history or history[0]["kind"] != "reflect":
        return None
    row = history[0]
    r = row["report"]
    when = relative_age(row["ts"])
    trigger = r.get("trigger", "manual")
    duration = f"{r.get('duration_ms', 0.0) / 1000.0:.1f}s"
    if r.get("interrupted"):
        tail = f"({trigger}, interrupted @ {r.get('stopped_in') or '?'}, {duration})"
    else:
        tail = f"({trigger}, {duration})"
    return (
        f"last reflection: {when} — merged {r.get('merged', 0)}, "
        f"+{r.get('edges_added', 0)} edges, pruned {r.get('pruned', 0)} {tail}"
    )


class MemoryExplorerView(ExplorerView):
    """In-app subview for browsing and curating long-term memories."""

    item_name = "item"
    list_heading = "  type/status  importance  usage   title"

    def __init__(
        self,
        rows: list[MemoryExplorerRow],
        *,
        forget: Callable[[str], None],
        mark_done: Callable[[str], None],
        last_reflection: str | None = None,
    ) -> None:
        self._forget = forget
        self._mark_done = mark_done
        # Destructive actions arm on the first press and fire on the second
        # press of the same key on the same row, so no invisible pane/query
        # state can trigger an unconfirmed forget or mark-done.
        self._pending_confirm: tuple[str, str] | None = None
        model = ExplorerModel(rows)
        title = "Memory Explorer"
        if last_reflection:
            title = f"{title} — {last_reflection}"
        config = ExplorerConfig(
            title=title,
            actions={
                "forget": "f×2 forget",
                "done": "d×2 todo done",
            },
        )
        super().__init__(model, config)
        importance_rank = {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "TRIVIAL": 1,
        }
        self.configure_row_options(
            filters=(
                ("all", "All", lambda _row: True),
                ("memory", "Memories", lambda row: row.type != "todo"),
                (
                    "todo-open",
                    "Open todos",
                    lambda row: row.type == "todo" and row.status == "open",
                ),
                (
                    "todo-done",
                    "Done todos",
                    lambda row: row.type == "todo" and row.status == "done",
                ),
            ),
            sorts=(
                ("created", "Created", lambda row: row.created_at, True),
                ("accessed", "Accessed", lambda row: row.last_accessed_at, True),
                (
                    "importance",
                    "Importance",
                    lambda row: (importance_rank.get(row.importance, 0), row.created_at),
                    True,
                ),
            ),
        )

    def format_row(self, row: MemoryExplorerRow, width: int) -> str:
        glyph = (
            _TODO_GLYPHS.get(row.status or "", "?")
            if row.type == "todo"
            else _TYPE_GLYPHS.get(row.type, "?")
        )
        fetches = row.usage["fetches"]
        fetched = relative_age(row.usage["last_ts"]) if fetches else "never"
        head = (row.title or row.content).replace("\n", " ").strip()
        line = f"{glyph} {row.type_tag:<12} {row.importance:<8} {fetches:>3}x {fetched:>6}  {head}"
        return line[:width]

    def detail_lines(self, row: MemoryExplorerRow, width: int) -> list[str]:
        width = max(int(width), 20)
        u = row.usage
        lines: list[str] = [
            f"Memory: [{row.id[:8]}] {row.title or '(untitled)'}",
            f"Id: {row.id}",
            f"Type: {row.type_tag}   Owner: {row.owner or '(unowned)'}   "
            f"Importance: {row.importance}",
            f"Tags: {', '.join(row.tags) if row.tags else '(none)'}",
            f"Created: {relative_age(row.created_at)} ago   "
            f"Last accessed: {relative_age(row.last_accessed_at)} ago",
            "",
        ]
        lines.extend(render_markdown_lines(row.content, width))
        lines.append("")
        lines.append("Usage:")
        lines.append(
            f"  fetches={u['fetches']}  recalled={u['recalled']}  searched={u['searched']}  "
            f"injected={u['injected']}  reinforced={u['reinforced']}  deref={u['deref']}"
        )
        if u["last_channel"] is not None:
            last = f"  last: {u['last_channel']} {relative_age(u['last_ts'])} ago"
            if u["last_session_ref"]:
                last += f" (session {u['last_session_ref'][:8]})"
            lines.append(last)
        rank_bits = []
        if u["mean_rank"] is not None:
            rank_bits.append(f"mean rank {u['mean_rank']}")
        if u["mean_score"] is not None:
            rank_bits.append(f"mean score {u['mean_score']}")
        if rank_bits:
            lines.append("  " + "  ".join(rank_bits))
        if u["injected_never_used"]:
            lines.append("  injected but never used")
        prune = "never" if u["prune_eta"] is None else f"in {_span(u['prune_eta'] - time.time())}"
        lines.append(f"  retention={u['retention']}   strength={u['strength']}   prune: {prune}")
        if row.reference_lines:
            lines.append("")
            lines.append("References:")
            for ref_line in row.reference_lines:
                lines.extend(wrap_plain_line(f"  {ref_line}", width))
        if row.edges:
            lines.append("")
            lines.append("Edges:")
            for target_id, edge_type, weight in row.edges:
                lines.append(f"  → {target_id[:8]} {edge_type} ({weight:.2f})")
        return lines

    def _consume_confirmation(self, action: str, row: Any) -> bool:
        """Return True when this press confirms the pending armed action.

        The first press of an action key arms it; the same key on the same
        row must follow within the gesture window to fire. Any other key
        (including typing "f"/"d" into the search buffer) disarms.
        """
        pending = self._pending_confirm
        self._pending_confirm = None
        return pending == (action, id(row))

    def pending_confirmation_hint(self) -> str:
        pending = self._pending_confirm
        if pending is None:
            return ""
        action, row_id = pending
        row = next((row for row in self.model.rows if id(row) == row_id), None)
        if row is None:
            self._pending_confirm = None
            return ""
        verb = "forget" if action == "text:f" else "mark done"
        title = (getattr(row, "title", None) or row.id)[:40]
        return f"Press {action[-1]} again to {verb} {title!r}"

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        if row is None:
            return "ignored"
        if action == "text:f":
            if not self._consume_confirmation(action, row):
                self._pending_confirm = (action, id(row))
                return "handled"
            self._forget(row.id)
            self.model.rows.remove(row)
            self.model.set_query(self.model.query)  # rebuild matches without the row
            return "handled"
        if action == "text:d":
            if row.type != "todo" or row.status == "done":
                # An action key that cannot apply still interrupts the armed
                # gesture — the next f/d press must arm, not confirm.
                self._pending_confirm = None
                return "ignored"
            if not self._consume_confirmation(action, row):
                self._pending_confirm = (action, id(row))
                return "handled"
            self._mark_done(row.id)
            # The search tag mirrors the CURRENT status (todo:open / todo:dropped
            # — see build_memory_rows), so replace whatever tag the row was
            # built with, not just todo:open.
            old_tag = f"todo:{row.status}"
            row.status = "done"
            row.search_text = row.search_text.replace(old_tag, "todo:done", 1)
            self.model.set_query(self.model.query)
            return "handled"
        self._pending_confirm = None
        return "ignored"
