# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Todo explorer — browse, inspect, and comment on agent todos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .explorer_base import (
    ExplorerConfig,
    ExplorerModel,
    ExplorerView,
    wrap_plain_line,
)
from .subapp import SubviewKeyResult


@dataclass
class TodoExplorerRow:
    """One todo item in the explorer list."""

    id: str
    title: str
    status: str
    deps: list[str]
    created_at: str
    notes: str
    comments: list[Any]
    search_text: str


def build_todo_rows(todo_manager: Any) -> list[TodoExplorerRow]:
    """Build explorer rows from the agent's TodoManager."""
    rows: list[TodoExplorerRow] = []
    if todo_manager is None:
        return rows
    for t in todo_manager.list_todos():
        search_parts = [
            t.id,
            t.title,
            t.status,
            t.notes,
            *[getattr(c, "body", str(c)) for c in t.comments],
        ]
        rows.append(
            TodoExplorerRow(
                id=t.id,
                title=t.title,
                status=t.status,
                deps=list(t.deps),
                created_at=t.created_at,
                notes=t.notes,
                comments=list(t.comments),
                search_text="\n".join(search_parts),
            )
        )
    return rows


class TodoExplorerView(ExplorerView):
    """In-app subview for browsing and interacting with todos."""

    def __init__(self, todo_manager: Any) -> None:
        self._todo_manager = todo_manager
        rows = build_todo_rows(todo_manager)
        model = ExplorerModel(rows)
        config = ExplorerConfig(
            title="Todo Explorer",
            detail_pane_name="todo detail",
            empty_message="No todos.",
            no_match_message="No todos matching {query!r}.",
            list_ratio=0.4,
            actions={},
        )
        super().__init__(model, config)

    def format_row(self, row: TodoExplorerRow, width: int) -> str:
        status_icon = {
            "open": "○",
            "done": "✓",
            "blocked": "●",
        }.get(row.status, "?")
        dep_hint = f" [needs: {', '.join(d[:8] for d in row.deps)}]" if row.deps else ""
        comment_hint = f" 💬{len(row.comments)}" if row.comments else ""
        line = f"{status_icon} [{row.id[:8]}] {row.title}{dep_hint}{comment_hint}"
        return line[:width]

    def detail_lines(self, row: TodoExplorerRow, width: int) -> list[str]:
        width = max(int(width), 20)
        lines: list[str] = [
            f"Todo: [{row.id[:8]}] {row.title}",
            f"Status: {row.status}",
            f"Created: {row.created_at}",
        ]
        if row.deps:
            lines.append(f"Dependencies: {', '.join(d[:8] for d in row.deps)}")
        lines.append("")

        if row.notes:
            lines.append("Notes:")
            for raw in row.notes.splitlines() or [""]:
                lines.extend(wrap_plain_line(f"  {raw}", width))
            lines.append("")

        if row.comments:
            lines.append(f"Comments ({len(row.comments)}):")
            lines.append("")
            for comment in row.comments:
                ts = getattr(comment, "created_at", "")
                body = getattr(comment, "body", str(comment))
                lines.append(f"  [{ts}]")
                for raw in body.splitlines() or [""]:
                    lines.extend(wrap_plain_line(f"    {raw}", width))
                lines.append("")
        else:
            lines.append("(no comments)")

        return lines

    def handle_action(self, action: str, row: Any) -> SubviewKeyResult:
        return "ignored"
