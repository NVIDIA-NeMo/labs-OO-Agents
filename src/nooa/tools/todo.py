# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory todo manager for agent task tracking.

Agents use this to plan and track progress through multi-step tasks.
The TodoManager is exposed to the agent's CodeAct REPL via self.todo,
and its API is published to LLM context via doc(self.todo).
"""

import uuid as _uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nooa import Skill
from nooa.storage.markers import snapshotable
from nooa.storage.snapshot_vars import SnapshotVars

_MISSING = object()


class TodoComment(BaseModel):
    """An append-only note on a todo — use for progress journalling.

    Survives across turns (snapshot-backed like the rest of the todo
    state). Prefer this over mutating ``Todo.notes`` when you want a
    chronological log: what was tried, what was found, why the approach
    changed.
    """

    id: str = Field(default_factory=lambda: _uuid.uuid4().hex[:8])
    body: str
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


class TodoVars:
    """Attribute-access proxy for a todo's vars dict.

    Lets you write ``t.v.commits = [...]`` instead of
    ``t.vars["commits"] = [...]``.  Reads and writes go straight
    through to the underlying ``Todo.vars`` dict, so snapshot
    serialisation is unaffected.
    """

    def __init__(self, todo: "Todo"):
        object.__setattr__(self, "_todo", todo)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._todo.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on todo {self._todo.id}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self._todo.vars[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self._todo.vars[key]
        except KeyError:
            raise AttributeError(f"No var {key!r} on todo {self._todo.id}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._todo.vars

    def __repr__(self) -> str:
        return repr(self._todo.vars)


class Todo(BaseModel):
    """A single todo item."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: _uuid.uuid4().hex[:8])
    title: str = ""
    status: str = "open"  # open | done | blocked
    deps: list[str] = Field(default_factory=list)
    vars: SnapshotVars = Field(default_factory=SnapshotVars)
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    notes: str = ""
    comments: list[TodoComment] = Field(default_factory=list)

    @field_validator("vars", mode="before")
    @classmethod
    def _coerce_vars(cls, value: Any) -> "SnapshotVars":
        if isinstance(value, SnapshotVars):
            return value
        if value is None:
            return SnapshotVars()
        if isinstance(value, dict):
            return SnapshotVars(value)
        raise TypeError(f"Todo.vars must be a dict or SnapshotVars, got {type(value).__name__}")

    @property
    def v(self) -> "TodoVars":
        """Attribute-access proxy for ``self.vars``.

        Usage::

            t = self.todo.add("Fix the bug")
            t.v.commits = ["abc123"]
            print(t.v.commits)   # ['abc123']
            del t.v.commits      # remove
            "commits" in t.v     # False
        """
        return TodoVars(self)

    def is_blocked(self, all_todos: dict[str, "Todo"]) -> bool:
        """Return True if any dependency is still open."""
        for dep_id in self.deps:
            dep = all_todos.get(dep_id)
            if dep and dep.status != "done":
                return True
        return False


@snapshotable
class TodoManager(Skill):
    """In-memory todo manager with dependency and variable support.

    Use this to plan your work before diving in, and to track progress
    through each step. Update status as you complete each item.

    Example workflow:
        t1 = self.todo.add("Explore repo structure")
        t2 = self.todo.add("Reproduce the failing test", deps=[t1])
        t3 = self.todo.add("Implement fix", deps=[t2])
        t4 = self.todo.add("Verify tests pass", deps=[t3])

        self.todo.done(t1)
        print(self.todo.status())
    """

    __nosnapshot__ = False  # Override Skill's __nosnapshot__ = True
    context_block = ("todo_status", "self.todo.status()")

    def __init__(self, state: dict | None = None) -> None:
        self._todos: dict[str, Todo] = {}
        self._order: list[str] = []  # insertion order
        if state:
            self.from_dict(state)

    # ── SERIALIZATION ─────────────────────────────

    def to_dict(self) -> dict:
        """Serialize todo state to a JSON-safe dict."""
        return {
            "todos": [
                t.model_dump() for t in (self._todos[i] for i in self._order if i in self._todos)
            ],
        }

    def from_dict(self, data: dict) -> None:
        """Restore todo state from a dict (e.g. from a snapshot)."""
        self._todos.clear()
        self._order.clear()
        for raw in data.get("todos", []):
            t = Todo.model_validate(raw)
            self._todos[t.id] = t
            self._order.append(t.id)

    # ── CRUD ──────────────────────────────────────

    @staticmethod
    def _todo_id(todo: Todo | str) -> str:
        """Return an id from either a Todo object or an id string."""
        if isinstance(todo, Todo):
            return todo.id
        if isinstance(todo, str):
            return todo
        raise TypeError(f"expected Todo or str, got {type(todo).__name__}")

    def _copy_todo(self, todo: Todo | str) -> Todo:
        """Return an independent copy of a manager-owned todo."""
        todo_id = self._todo_id(todo)
        current = self._todos.get(todo_id)
        if current is None:
            raise ValueError(f"todo {todo_id!r} is not managed by this TodoManager")
        return current.model_copy(deep=True)

    @classmethod
    def _with_todo(cls, todo: Todo) -> "TodoManager":
        """Create a manager containing an independent copy of one todo."""
        manager = cls()
        copied = todo.model_copy(deep=True)
        manager._todos[copied.id] = copied
        manager._order.append(copied.id)
        return manager

    def _merge_todo(self, updated: Todo, *, base: Todo) -> Todo:
        """Atomically merge delegated changes without overwriting concurrent edits."""
        updated = updated.model_copy(deep=True)
        if updated.id != base.id:
            raise ValueError("updated and base todos must have the same id")
        current = self._todos.get(updated.id)
        if current is None:
            raise ValueError(f"todo {updated.id!r} is not managed by this TodoManager")

        candidate = current.model_copy(deep=True)
        for field in ("title", "status", "deps", "notes"):
            before = getattr(base, field)
            after = getattr(updated, field)
            existing = getattr(current, field)
            if after == before:
                continue
            if existing != before and existing != after:
                raise ValueError(f"todo {updated.id!r} has conflicting {field!r} changes")
            setattr(candidate, field, after.copy() if isinstance(after, list) else after)

        base_vars = dict(base.vars)
        updated_vars = dict(updated.vars)
        current_vars = dict(current.vars)
        for key in base_vars.keys() | updated_vars.keys():
            before = base_vars.get(key, _MISSING)
            after = updated_vars.get(key, _MISSING)
            if after == before:
                continue
            existing = current_vars.get(key, _MISSING)
            if existing != before and existing != after:
                raise ValueError(f"todo {updated.id!r} has conflicting variable {key!r}")
            if after is _MISSING:
                candidate.vars.pop(key, None)
            else:
                candidate.vars[key] = after

        if updated.comments[: len(base.comments)] != base.comments:
            raise ValueError(f"todo {updated.id!r} modified existing comments")
        if current.comments[: len(base.comments)] != base.comments:
            raise ValueError(f"todo {updated.id!r} has conflicting comment changes")
        existing_comment_ids = {comment.id for comment in candidate.comments}
        for comment in updated.comments[len(base.comments) :]:
            if comment.id not in existing_comment_ids:
                candidate.comments.append(comment.model_copy(deep=True))
                existing_comment_ids.add(comment.id)

        # Commit only after every conflict check has succeeded, preserving the
        # authoritative Todo object's identity for existing callers.
        for field in ("title", "status", "deps", "notes", "vars", "comments"):
            setattr(current, field, getattr(candidate, field))
        return current

    def add(
        self,
        title: str,
        deps: list[Todo | str] | None = None,
        notes: str = "",
        **vars: Any,
    ) -> Todo:
        """Add a new todo item. Returns the created Todo."""
        t = Todo(
            title=title,
            deps=[self._todo_id(dep) for dep in deps or []],
            notes=notes,
            vars=dict(vars),
        )
        self._todos[t.id] = t
        self._order.append(t.id)
        return t

    def get(self, todo_id: Todo | str) -> Todo | None:
        """Get a todo by object or id."""
        return self._todos.get(self._todo_id(todo_id))

    def done(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo as done. Returns the updated Todo or None if not found."""
        t = self.get(todo_id)
        if t:
            t.status = "done"
        return t

    def reopen(self, todo_id: Todo | str) -> Todo | None:
        """Re-open a done or blocked todo. Returns the updated Todo or None."""
        t = self.get(todo_id)
        if t:
            t.status = "open"
        return t

    def remove(self, todo_id: Todo | str) -> bool:
        """Remove a todo. Returns True if it existed."""
        todo_id = self._todo_id(todo_id)
        if todo_id in self._todos:
            del self._todos[todo_id]
            self._order = [i for i in self._order if i != todo_id]
            return True
        return False

    def clear(self) -> None:
        """Remove every todo. Used by ``/clear`` to reset in-memory
        working state on a new session — the old session's snapshot
        is preserved on disk and can be re-loaded via ``/session <id>``.
        """
        self._todos.clear()
        self._order.clear()

    def update(self, todo_id: Todo | str, **kwargs: Any) -> Todo | None:
        """Update todo fields (title, status, notes). Returns the updated Todo or None."""
        t = self.get(todo_id)
        if t is None:
            return None
        allowed = {"title", "status", "notes"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(t, k, v)
        return t

    # ── DEPENDENCIES ──────────────────────────────

    def add_dep(self, todo_id: Todo | str, dep_id: Todo | str) -> Todo | None:
        """Add a dependency to a todo. Returns the updated Todo or None."""
        t = self.get(todo_id)
        dep_id = self._todo_id(dep_id)
        if t and dep_id not in t.deps:
            t.deps.append(dep_id)
        return t

    def remove_dep(self, todo_id: Todo | str, dep_id: Todo | str) -> Todo | None:
        """Remove a dependency from a todo. Returns the updated Todo or None."""
        t = self.get(todo_id)
        dep_id = self._todo_id(dep_id)
        if t and dep_id in t.deps:
            t.deps.remove(dep_id)
        return t

    # ── VARIABLES ─────────────────────────────────

    def set_var(self, todo_id: Todo | str, key: str, value: Any) -> Todo | None:
        """Set a variable on a todo (arbitrary metadata). Returns the updated Todo or None."""
        t = self.get(todo_id)
        if t:
            t.vars[key] = value
        return t

    def del_var(self, todo_id: Todo | str, key: str) -> Todo | None:
        """Delete a variable from a todo. Returns the updated Todo or None."""
        t = self.get(todo_id)
        if t:
            t.vars.pop(key, None)
        return t

    def get_var(self, todo_id: Todo | str, key: str) -> Any | None:
        """Get a variable from a todo. Returns the value or None."""
        t = self.get(todo_id)
        return t.vars.get(key) if t else None

    # ── COMMENTS ──────────────────────────────────

    def comment(self, todo_id: Todo | str, body: str) -> TodoComment | None:
        """Append a comment to a todo.

        Use for progress journalling — what you did, what you found, why
        you changed approach. Persists across turns (snapshot-backed), so
        the next turn can read the history via ``comments(todo_id)``.

        Returns the created :class:`TodoComment`, or None if the todo
        doesn't exist.

        Example::

            t = self.todo.add("Solve auth bug")
            self.todo.comment(t, "🔍 root cause: session.py:42 races on refresh")
            # ... next turn ...
            for c in self.todo.comments(t):
                print(c.created_at, c.body)
        """
        t = self.get(todo_id)
        if t is None:
            return None
        c = TodoComment(body=body)
        t.comments.append(c)
        return c

    def comments(self, todo_id: Todo | str) -> list[TodoComment]:
        """Return all comments on a todo in chronological order (empty if none)."""
        t = self.get(todo_id)
        return list(t.comments) if t else []

    # ── QUERIES ───────────────────────────────────

    def list_todos(self, status: str | None = None) -> list[Todo]:
        """List todos, optionally filtered by status ('open', 'done', 'blocked').

        Blocked status is computed dynamically from dependencies — a todo with
        unfinished deps is treated as 'blocked' even if its stored status is 'open'.
        """
        todos = [self._todos[i] for i in self._order if i in self._todos]
        if status is None:
            return todos

        def _effective(t: Todo) -> str:
            if t.status == "open" and t.is_blocked(self._todos):
                return "blocked"
            if t.status == "blocked" and not t.is_blocked(self._todos):
                return "open"
            return t.status

        return [t for t in todos if _effective(t) == status]

    # ── STATUS ────────────────────────────────────

    def status(self) -> str:
        """Return a formatted summary of all todos — call this to see current progress."""
        todos = [self._todos[i] for i in self._order if i in self._todos]
        if not todos:
            return "(no todos)"

        icons = {"open": "○", "done": "✓", "blocked": "●"}
        lines = []
        for t in todos:
            effective_status = t.status
            if t.status == "open" and t.is_blocked(self._todos):
                effective_status = "blocked"
            icon = icons.get(effective_status, "?")
            dep_str = f" [needs: {', '.join(t.deps)}]" if t.deps else ""
            var_str = f" {t.vars}" if t.vars else ""
            notes_str = f"\n    note: {t.notes}" if t.notes else ""
            lines.append(f"  {icon} [{t.id}] {t.title}{dep_str}{var_str}{notes_str}")

        done = sum(1 for t in todos if t.status == "done")
        total = len(todos)
        lines.insert(0, f"Todos ({done}/{total} done):")
        return "\n".join(lines)
