# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's to-do list: four calls for the agent, a core ``TodoManager`` inside.

The model sees ``add``, ``activate``, ``comment`` and ``complete`` and a status
block with the plan, the active item's description and its notes. ``TodoList``
holds a ``TodoManager`` rather than extending it, so none of the manager's
methods are documented here; lookups, edits, dependencies, metadata, delegated
brief copies and the ACP plan reach it through hidden methods and attribute
pass-through, and its state is what snapshots save.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Annotated, Any, Literal

from pydantic import Field

from nooa import Skill, hidden
from nooa.agentdoc import spec
from nooa.storage.markers import snapshotable
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

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8], description='"1", "2", … for items you add')
    status: Literal["open", "done"] = "open"
    deps: Annotated[list[str], hidden] = Field(default_factory=list)
    created_at: Annotated[str, hidden] = Field(default_factory=lambda: _Todo().created_at)
    comments: Annotated[list[TodoComment], hidden] = Field(default_factory=list)


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _numeric(value: str) -> bool:
    return value.isascii() and value.isdigit()


@snapshotable
class TodoList(Skill):
    """Your plan for multi-step work. Activate the item you work on: its description and notes show in todo_status."""

    __nosnapshot__ = False  # Override Skill's __nosnapshot__ = True
    context_block = ("todo_status", "self.todo.status()")

    _manager: TodoManager

    def __init__(self, state: dict | None = None, *, manager: TodoManager | None = None) -> None:
        self._manager = manager if manager is not None else TodoManager(state)

    # ── the agent's interface ─────────────────────

    def add(self, title: str, description: str = "", **legacy: Any) -> Todo:
        """Add an open item: ``description`` holds the goal, constraints and how to check it."""
        notes = legacy.pop("notes", None)
        if notes is not None:
            if description:
                raise ValueError("use either description or legacy notes, not both")
            description = notes
        deps = legacy.pop("deps", None)
        m = self._manager
        todo = Todo(
            id=self._next_id(),
            title=title,
            description=description,
            deps=[m._todo_id(self._coerce(dep)) for dep in deps or []],
            vars=dict(legacy),
        )
        m._todos[todo.id] = todo
        m._order.append(todo.id)
        return todo

    def activate(self, todo: Todo | str) -> Todo:
        """Make it the active item."""
        return self._manager.activate(self._coerce(todo))

    def comment(self, text: str, todo: Any = None) -> None:
        """Add a note to the active item: findings, decisions, checks and their results."""
        if todo is None:
            if isinstance(text, _Todo):
                raise TypeError("comment(text): pass the note text; it goes on the active item")
            body, target = text, None
        else:
            body, target = self._legacy_order(text, todo)
        self._manager.comment(self._target(target, "comment"), str(body))

    def complete(self, todo: Todo | str | None = None) -> None:
        """Mark the active item done."""
        self._manager.complete(self._target(todo, "complete"))

    # ── hidden: the manager's own calls ───────────

    def __getattr__(self, name: str) -> Any:
        # Everything else TodoManager offers (update, list_todos, active, set_var,
        # deps, to_dict, ...) stays callable; none of it is documented here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_manager"), name)

    @hidden
    def get(self, todo_id: Any) -> Todo | None:
        return self._manager.get(self._coerce(todo_id))

    @hidden
    def update(self, todo_id: Any, **kwargs: Any) -> Todo | None:
        return self._manager.update(self._coerce(todo_id), **kwargs)

    @hidden
    def insert_copy(self, todo: _Todo) -> Todo:
        """A delegated brief: kept as this list's Todo type, active when open."""
        return self._manager.insert_copy(Todo.model_validate(todo.model_dump()))

    @hidden
    def status(self, max_items: int = _ROWS_SHOWN, max_chars: int = 4_000) -> str:
        """The todo_status block: the plan, then the active item's description and notes."""
        m = self._manager
        todos = [m._todos[i] for i in m._order if i in m._todos]
        if not todos:
            return "(empty)"
        active = m.active()
        done = [t for t in todos if t.status == "done"]
        folded = {t.id for t in done[:-_DONE_SHOWN]} if len(done) > _DONE_SHOWN else set()
        rows = [t for t in todos if t.id not in folded]
        lines = [f"(+{len(folded)} done)"] if folded else []
        for t in rows[:max_items]:
            mark = "▸" if active is not None and t.id == active.id else ("✓" if t.status == "done" else "·")
            lines.append(f"{mark} {t.id}  {_one_line(t.title, _TITLE_CHARS)}")
        if len(rows) > max_items:
            lines.append(f"(+{len(rows) - max_items} more)")
        if active is not None:
            lines += ["", f"Active: {active.id}  {_one_line(active.title, _TITLE_CHARS)}"]
            text = active.description.strip()
            if text:
                lines.append(text if len(text) <= _DESCRIPTION_CHARS else text[: _DESCRIPTION_CHARS - 1].rstrip() + "…")
            notes = active.comments[-_NOTES_SHOWN:]
            if notes:
                earlier = len(active.comments) - len(notes)
                lines.append("Notes:" + (f" (+{earlier} earlier)" if earlier else ""))
                lines += [f"- {_one_line(c.body, _NOTE_CHARS)}" for c in notes]
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[: max_chars - 2].rstrip() + "\n…"

    # ── helpers ───────────────────────────────────

    @staticmethod
    def _coerce(todo: Any) -> Any:
        """An int id is accepted as its string; anything else goes to the manager unchanged."""
        if isinstance(todo, int) and not isinstance(todo, bool):
            return str(todo)
        return todo

    def _known(self, value: Any) -> bool:
        value = self._coerce(value)
        return isinstance(value, str) and value in self._manager._todos

    def _next_id(self) -> str:
        numbers = [int(i) for i in self._manager._todos if _numeric(i)]
        return str(max(numbers, default=0) + 1)

    def _legacy_order(self, first: Any, second: Any) -> tuple[Any, Any]:
        """Return (text, todo) for a two-argument comment: ``(todo, text)`` or ``(text, todo)``."""
        if isinstance(first, _Todo) and isinstance(second, _Todo):
            raise TypeError("comment() got two todos; pass the note text and at most one todo")
        if isinstance(first, _Todo):
            return second, first  # comment(todo, text), the TodoManager order
        if isinstance(second, _Todo):
            return first, second
        if self._known(first) and self._known(second):
            raise ValueError(
                f"comment({first!r}, {second!r}) is ambiguous: both are item ids; "
                "activate the item and call comment(text)"
            )
        if self._known(first):
            return second, first  # comment(id, text)
        return first, second

    def _target(self, todo: Any, verb: str) -> str:
        m = self._manager
        if todo is not None:
            todo_id = m._todo_id(self._coerce(todo))
            if todo_id not in m._todos:
                raise ValueError(f"no item {todo_id!r}; ids are shown in todo_status")
            return todo_id
        active = m.active()
        if active is not None:
            return active.id
        open_items = [t for t in m.list_todos() if t.status != "done"]
        if len(open_items) == 1:
            return open_items[0].id
        raise ValueError(f"nothing is active: self.todo.activate(id) first, then {verb}()")


# The agent reads these signatures; the legacy arguments stay accepted, not advertised.
def _signature(*params: inspect.Parameter, returns: Any) -> inspect.Signature:
    head = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return inspect.Signature([head, *params], return_annotation=returns)


_P = inspect.Parameter.POSITIONAL_OR_KEYWORD
TodoList.add.__signature__ = _signature(  # type: ignore[attr-defined]
    inspect.Parameter("title", _P, annotation=str),
    inspect.Parameter("description", _P, default="", annotation=str),
    returns=Todo,
)
TodoList.comment.__signature__ = _signature(inspect.Parameter("text", _P, annotation=str), returns=None)  # type: ignore[attr-defined]
TodoList.complete.__signature__ = _signature(returns=None)  # type: ignore[attr-defined]
spec(TodoList, "source_dir", hidden=True)
