# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's to-do list: the list keeps items and progress, an item keeps its content.

``self.todo`` adds, finds, deletes, activates and completes items. A ``Todo``
carries its own content, set directly: ``title``, ``description`` and
``comments`` (a list of note strings). The status block shows the plan and the
active item's description and notes.

``TodoList`` holds a core ``TodoManager`` rather than extending it, so none of
the manager's methods are documented here. Lookups by status, dependencies,
metadata, delegated brief copies and the ACP plan reach it through hidden
methods and attribute pass-through, and its state is what snapshots save.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator

from nooa import Skill, hidden
from nooa.agentdoc import spec
from nooa.storage.markers import snapshotable
from nooa.tools.todo import Todo as _Todo
from nooa.tools.todo import TodoManager

__all__ = ["Todo", "TodoList"]

_TITLE_CHARS = 100
_GIST_CHARS = 90  # a done item's latest note, on its plan row
_DESCRIPTION_CHARS = 2_500
_NOTE_CHARS = 400
_NOTES_SHOWN = 8
_ROWS_SHOWN = 15
_DONE_SHOWN = 4
_STATUS_CHARS = 8_000


class Todo(_Todo):
    """An item on your plan: set its fields directly, e.g. ``t.comments.append("found it")``."""

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Shown in todo_status; get(id) finds the item",
    )
    title: str = ""
    status: Literal["open", "done"] = "open"
    description: str = Field(default="", description="The goal, constraints and how to check it")
    deps: Annotated[list[str], hidden] = Field(default_factory=list)
    created_at: Annotated[str, hidden] = Field(default_factory=lambda: _Todo().created_at)
    comments: list[str] = Field(
        default_factory=list, description="Notes: findings, decisions, checks and results"
    )

    @field_validator("comments", mode="before")
    @classmethod
    def _note_text(cls, value: Any) -> Any:
        """Every note loads as text, so a saved list always restores.

        ``comments`` is a plain list the model appends to, so anything can land
        in it: a core ``TodoComment`` or its dict keeps its ``body``; any other
        value (a number, ``None``, a dict without a text body) becomes ``str(v)``.
        """
        if isinstance(value, list):
            return [_note(v) for v in value]
        return value

    @hidden
    def comment(self, text: str) -> str:
        self.comments.append(str(text))
        return str(text)


def _note(value: Any) -> str:
    """One note as text: a string as is, a TodoComment or its dict by its body, anything else by str()."""
    if isinstance(value, str):
        return value
    body = value.get("body") if isinstance(value, dict) else getattr(value, "body", None)
    return body if isinstance(body, str) else str(value)


def _as_todo(todo: _Todo) -> Todo:
    """The same item as this module's ``Todo`` (a core Todo from a snapshot or a hand-built brief)."""
    return todo if isinstance(todo, Todo) else Todo.model_validate(todo.model_dump())


def _restore(raw: Any) -> Todo:
    """One saved item, with the core snapshot migrations: ``notes`` is the description, status is open or done."""
    if not isinstance(raw, dict):
        return _as_todo(raw)
    data = dict(raw)
    if "description" not in data and "notes" in data:
        data["description"] = data.pop("notes")
    data["status"] = "done" if data.get("status") == "done" else "open"
    return Todo.model_validate(data)  # notes saved as TodoComment dicts become their text


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


@snapshotable
class TodoList(Skill):
    """Your plan for multi-step work. Activate the item you work on: its description and notes show in todo_status."""

    __nosnapshot__ = False  # Override Skill's __nosnapshot__ = True
    context_block = ("todo_status", "self.todo.status()")

    _manager: TodoManager

    def __init__(self, state: dict | None = None, *, manager: TodoManager | None = None) -> None:
        self._manager = manager if manager is not None else TodoManager()
        if state:
            self.from_dict(state)

    # ── the agent's interface ─────────────────────

    def add(self, title: str, description: str = "", **legacy: Any) -> Todo:
        """Add an open item."""
        notes = legacy.pop("notes", None)
        if notes is not None:
            if description:
                raise ValueError("use either description or legacy notes, not both")
            description = notes
        deps = legacy.pop("deps", None)
        m = self._held()
        todo = Todo(
            title=title,
            description=description,
            deps=[m._todo_id(self._coerce(dep)) for dep in deps or []],
            vars=dict(legacy),
        )
        m._todos[todo.id] = todo
        m._order.append(todo.id)
        return todo

    def get(self, todo: Todo | str) -> Todo:
        """The item with this id (ids are shown in todo_status)."""
        return self._held()._todos[self._id(todo)]

    def delete(self, todo: Todo | str) -> None:
        """Remove an item from the plan."""
        self._held().remove(self._id(todo))

    def activate(self, todo: Todo | str) -> Todo:
        """Make it the item you work on."""
        return self._held().activate(self._id(todo))

    def complete(self, todo: Todo | str | None = None) -> None:
        """Mark it done; without an argument, the active item."""
        self._held().complete(self._target(todo, "complete"))

    # ── hidden: earlier calls and the manager's own ──

    def __getattr__(self, name: str) -> Any:
        # Everything else TodoManager offers (update, list_todos, active, set_var,
        # deps, to_dict, ...) stays callable; none of it is documented here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._held(), name)

    @hidden
    def comment(self, text: Any, todo: Any = None) -> str:
        """``self.todo.comment(text[, todo])`` and ``self.todo.comment(todo, text)``: earlier forms of ``todo.comment(text)``."""
        if todo is None:
            if isinstance(text, _Todo):
                raise TypeError("comment(text): pass the note text; it goes on the active item")
            body, target = text, None
        else:
            body, target = self._legacy_order(text, todo)
        return self.get(self._target(target, "comment")).comment(str(body))

    @hidden
    def from_dict(self, data: dict) -> None:
        """Restore ``to_dict()`` state, as this module's ``Todo`` items (core snapshots included)."""
        staged = TodoManager()
        staged.from_dict({**data, "todos": []})
        for raw in data.get("todos", []):
            todo = _restore(raw)
            if todo.id in staged._todos:
                raise ValueError(f"duplicate todo id {todo.id!r} in snapshot")
            staged._todos[todo.id] = todo
            staged._order.append(todo.id)
        active = staged._todos.get(data.get("active_id") or "")
        staged._active_id = active.id if active is not None and active.status != "done" else None
        self._manager = staged

    @hidden
    def update(self, todo_id: Any, **kwargs: Any) -> Todo | None:
        return self._held().update(self._coerce(todo_id), **kwargs)

    @hidden
    def insert_copy(self, todo: _Todo) -> Todo:
        """A delegated brief: kept as this list's Todo type, active when open."""
        return self._held().insert_copy(_as_todo(todo))

    @hidden
    def status(self, max_items: int = _ROWS_SHOWN, max_chars: int = _STATUS_CHARS) -> str:
        """The todo_status block: progress, the plan with ids, then the active item's card.

        Plan rows: ``✓`` done (with its latest note, the result in a line), ``▸``
        active, ``·`` open (with its note count). Older done rows fold into one
        line. The card holds the active item's whole description and its notes,
        numbered, the latest ones when there are many.
        """
        m = self._held()
        todos = [m._todos[i] for i in m._order if i in m._todos]
        if not todos:
            return "No items yet."
        active = m.active()
        done = [t for t in todos if t.status == "done"]
        open_ = len(todos) - len(done)
        head = f"{open_} open · {len(done)} done"
        if active is not None:
            head += f" · working on {active.id}"
        lines = [head]

        folded = {t.id for t in done[:-_DONE_SHOWN]} if len(done) > _DONE_SHOWN else set()
        if folded:
            lines.append(f"✓ (+{len(folded)} earlier done)")
        rows = [t for t in todos if t.id not in folded]
        for t in rows[:max_items]:
            title = _one_line(t.title, _TITLE_CHARS)
            if t.status == "done":
                gist = f" — {_one_line(str(t.comments[-1]), _GIST_CHARS)}" if t.comments else ""
                lines.append(f"✓ {t.id}  {title}{gist}")
            else:
                mark = "▸" if active is not None and t.id == active.id else "·"
                count = f"  ({_plural(len(t.comments), 'note')})" if t.comments else ""
                lines.append(f"{mark} {t.id}  {title}{count}")
        if len(rows) > max_items:
            lines.append(f"· (+{len(rows) - max_items} more)")

        lines.append("")
        if active is not None:
            lines.append(f"▸ {active.id}  {_one_line(active.title, _TITLE_CHARS)}")
            text = active.description.strip()
            if len(text) > _DESCRIPTION_CHARS:
                rest = len(text) - _DESCRIPTION_CHARS
                text = text[:_DESCRIPTION_CHARS].rstrip() + f"… (+{rest} chars in its description)"
            lines.append(text or "(no description: set one with the goal and how to check it)")
            notes = active.comments
            if notes:
                first = max(0, len(notes) - _NOTES_SHOWN)
                lines.append("Notes:" + (f" ({first} earlier not shown)" if first else ""))
                lines += [
                    f"{i}. {_one_line(str(n), _NOTE_CHARS)}"
                    for i, n in enumerate(notes[first:], first + 1)
                ]
            else:
                lines.append("Notes: none yet")
        elif open_:
            lines.append(
                "Nothing active: activate the item you work on to see its description and notes here."
            )
        else:
            lines.append("All items are done.")
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[: max_chars - 2].rstrip() + "\n…"

    # ── helpers ───────────────────────────────────

    def _held(self) -> TodoManager:
        """The manager, with every item as this module's ``Todo`` (restored or copied ones included)."""
        m = self._manager
        for todo_id, todo in list(m._todos.items()):
            if not isinstance(todo, Todo):
                m._todos[todo_id] = _as_todo(todo)
        return m

    @staticmethod
    def _coerce(todo: Any) -> Any:
        """An int id is accepted as its string; anything else goes to the manager unchanged."""
        if isinstance(todo, int) and not isinstance(todo, bool):
            return str(todo)
        return todo

    def _id(self, todo: Any) -> str:
        m = self._held()
        todo_id = m._todo_id(self._coerce(todo))
        if todo_id not in m._todos:
            raise ValueError(f"no item {todo_id!r}; ids are shown in todo_status")
        return todo_id

    def _known(self, value: Any) -> bool:
        value = self._coerce(value)
        return isinstance(value, str) and value in self._manager._todos

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
                "use self.todo.get(id).comment(text)"
            )
        if self._known(first):
            return second, first  # comment(id, text)
        return first, second

    def _target(self, todo: Any, verb: str) -> str:
        if todo is not None:
            return self._id(todo)
        m = self._held()
        active = m.active()
        if active is not None:
            return active.id
        open_items = [t for t in m.list_todos() if t.status != "done"]
        if len(open_items) == 1:
            return open_items[0].id
        raise ValueError(f"nothing is active: pass the item, or activate one first, then {verb}()")


# The agent reads this signature; the legacy keywords stay accepted, not advertised.
TodoList.add.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("title", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        inspect.Parameter(
            "description", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="", annotation=str
        ),
    ],
    return_annotation=Todo,
)
spec(TodoList, "source_dir", hidden=True)
