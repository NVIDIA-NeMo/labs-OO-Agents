# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's to-do list: four calls for the agent, the full ``TodoManager`` underneath.

The model sees ``add``, ``activate``, ``comment`` and ``complete`` and a status
block with the plan, the active item's description and its notes. Lookups,
edits, dependencies, metadata, snapshots, delegation copies and the ACP plan
keep working through the inherited (hidden) ``TodoManager`` methods.
"""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Literal

from pydantic import Field

from nooa import hidden
from nooa.agentdoc import spec
from nooa.tools.todo import Todo as _Todo
from nooa.tools.todo import TodoComment, TodoManager

__all__ = ["Todo", "TodoList"]

_TITLE_CHARS = 120
_DESCRIPTION_CHARS = 2_000
_NOTE_CHARS = 400
_NOTES_SHOWN = 6
_ROWS_SHOWN = 12
_DONE_SHOWN = 3


class Todo(_Todo):
    """An item on your plan; the active one's description and notes show in todo_status."""

    id: str = Field(default="", description='"1", "2", …')
    status: Literal["open", "done"] = "open"
    deps: Annotated[list[str], hidden] = Field(default_factory=list)
    created_at: Annotated[str, hidden] = Field(default_factory=lambda: _Todo().created_at)
    comments: Annotated[list[TodoComment], hidden] = Field(default_factory=list)


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


class TodoList(TodoManager):
    """Your plan for multi-step work. Activate the item you work on: its description and notes show in todo_status."""

    # ── the agent's interface ─────────────────────

    def add(self, title: str, description: str = "", *, deps: Any = None, **vars: Any) -> Todo:
        """Add an open item: ``description`` holds the goal, constraints and how to check it."""
        todo = Todo(
            id=self._next_id(),
            title=title,
            description=description,
            deps=[self._todo_id(dep) for dep in deps or []],
            vars=dict(vars),
        )
        self._todos[todo.id] = todo
        self._order.append(todo.id)
        return todo

    def activate(self, todo: Todo | str) -> Todo:
        """Make it the active item."""
        return super().activate(self._todo_id(todo))

    def comment(self, text: str, todo: Any = None) -> None:
        """Add a note to the active item: findings, decisions, checks and their results."""
        if todo is not None and self._is_item(text) and not self._is_item(todo):
            text, todo = todo, text  # comment(todo, text), the TodoManager order
        super().comment(self._target(todo, "comment"), str(text))

    def complete(self, todo: Todo | str | None = None) -> None:
        """Mark the active item done."""
        super().complete(self._target(todo, "complete"))

    # ── everything else stays callable, out of the docs ──

    @hidden
    def from_dict(self, data: dict) -> None:
        super().from_dict(data)
        self._todos = {i: Todo.model_validate(t.model_dump()) for i, t in self._todos.items()}

    @hidden
    def insert_copy(self, todo: _Todo) -> Todo:
        return super().insert_copy(Todo.model_validate(todo.model_dump()))

    @hidden
    def get(self, todo_id: Any) -> Todo | None:
        return super().get(self._todo_id(todo_id))

    @hidden
    def update(self, todo_id: Any, **kwargs: Any) -> Todo | None:
        return super().update(self._todo_id(todo_id), **kwargs)

    @hidden
    def set_var(self, todo_id: Any, key: str, value: Any) -> Todo | None:
        return super().set_var(self._todo_id(todo_id), key, value)

    @hidden
    def get_var(self, todo_id: Any, key: str) -> Any | None:
        return super().get_var(self._todo_id(todo_id), key)

    @hidden
    def comments(self, todo_id: Any) -> list[TodoComment]:
        return super().comments(self._todo_id(todo_id))

    @hidden
    def list_todos(self, status: str | None = None) -> list[_Todo]:
        return super().list_todos(status)

    @hidden
    def status(self, max_items: int = _ROWS_SHOWN, max_chars: int = 4_000) -> str:
        """The todo_status block: the plan, then the active item's description and notes."""
        todos = [self._todos[i] for i in self._order if i in self._todos]
        if not todos:
            return "(empty)"
        active = self.active()
        done = [t for t in todos if t.status == "done"]
        hidden_done = set(t.id for t in done[:-_DONE_SHOWN]) if len(done) > _DONE_SHOWN else set()
        rows = [t for t in todos if t.id not in hidden_done]
        lines = [f"(+{len(hidden_done)} done)"] if hidden_done else []
        for t in rows[:max_items]:
            mark = "▸" if active is not None and t.id == active.id else ("✓" if t.status == "done" else "·")
            lines.append(f"{mark} {t.id}  {_one_line(t.title, _TITLE_CHARS)}")
        if len(rows) > max_items:
            lines.append(f"(+{len(rows) - max_items} more)")
        if active is not None:
            lines += ["", f"Active: {active.id}  {_one_line(active.title, _TITLE_CHARS)}"]
            if active.description.strip():
                text = active.description.strip()
                if len(text) > _DESCRIPTION_CHARS:
                    text = text[: _DESCRIPTION_CHARS - 1].rstrip() + "…"
                lines.append(text)
            notes = active.comments[-_NOTES_SHOWN:]
            if notes:
                skipped = len(active.comments) - len(notes)
                lines.append("Notes:" + (f" (+{skipped} earlier)" if skipped else ""))
                lines += [f"- {_one_line(c.body, _NOTE_CHARS)}" for c in notes]
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[: max_chars - 2].rstrip() + "\n…"

    # ── helpers ───────────────────────────────────

    @staticmethod
    @hidden
    def _todo_id(todo: Any) -> str:
        if isinstance(todo, _Todo):
            return todo.id
        if isinstance(todo, (str, int)) and not isinstance(todo, bool):
            return str(todo)
        raise TypeError(f"expected a Todo or its id, got {type(todo).__name__}")

    @hidden
    def _next_id(self) -> str:
        numbers = [int(i) for i in self._todos if i.isdigit()]
        return str(max(numbers, default=0) + 1)

    @hidden
    def _is_item(self, value: Any) -> bool:
        if isinstance(value, _Todo):
            return True
        return isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) in self._todos

    @hidden
    def _target(self, todo: Any, verb: str) -> str:
        if todo is not None:
            todo_id = self._todo_id(todo)
            if todo_id not in self._todos:
                raise ValueError(f"no item {todo_id!r}; ids are shown in todo_status")
            return todo_id
        active = self.active()
        if active is not None:
            return active.id
        open_items = [t for t in self.list_todos() if t.status != "done"]
        if len(open_items) == 1:
            return open_items[0].id
        raise ValueError(f"nothing is active: self.todo.activate(id) first, then {verb}()")


# The agent reads these signatures; the extra keywords stay accepted, not advertised.
TodoList.add.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("title", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        inspect.Parameter("description", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="", annotation=str),
    ],
    return_annotation=Todo,
)
TodoList.comment.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("text", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
    ],
    return_annotation=None,
)
TodoList.complete.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)],
    return_annotation=None,
)
spec(TodoList, "source_dir", hidden=True)
