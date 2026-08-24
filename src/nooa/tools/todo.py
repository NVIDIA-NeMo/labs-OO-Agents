# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory todo manager for agent task tracking.

Agents use this to plan and track progress through multi-step tasks.
The TodoManager is exposed to the agent's CodeAct REPL via self.todo,
and its API is published to LLM context via doc(self.todo).
"""

import uuid as _uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nooa import Skill, hidden
from nooa.storage.markers import snapshotable
from nooa.storage.snapshot_vars import SnapshotVars

_MISSING = object()
_STATUS_MAX_ITEMS = 10
_STATUS_MAX_CHARS = 2_000
_STATUS_TITLE_CHARS = 120
_STATUS_MAX_DEPS = 3


class TodoComment(BaseModel):
    """An append-only, snapshot-backed progress-journal entry.

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
    """A managed task; blocking is derived from unfinished dependencies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: _uuid.uuid4().hex[:8], description="Stable task ID")
    title: str = Field(default="", description="Short action-oriented description")
    status: str = Field(default="open", description="Stored status: open or done")
    deps: list[str] = Field(default_factory=list, description="IDs of prerequisite todos")
    vars: Annotated[SnapshotVars, hidden] = Field(default_factory=SnapshotVars)
    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"),
        description="Creation time",
    )
    notes: str = Field(default="", description="Current summary or instructions")
    comments: list[TodoComment] = Field(
        default_factory=list,
        description="Append-only chronological progress journal",
    )

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

    @hidden
    def is_blocked(self, all_todos: dict[str, "Todo"]) -> bool:
        """Return True if any dependency is still open."""
        for dep_id in self.deps:
            dep = all_todos.get(dep_id)
            if dep and dep.status != "done":
                return True
        return False


@snapshotable
class TodoManager(Skill):
    """Track multi-step work, dependencies, metadata, and progress notes.

    Every todo argument accepts either a ``Todo`` returned by this manager or
    its ID. Keep active work visible with ``status()``, inspect details with
    ``get()`` and ``comments()``, and prune history with ``clear_done()``.

    Example::

        explore = self.todo.add("Explore the repository")
        fix = self.todo.add("Implement the fix", deps=[explore])
        self.todo.comment(explore, "Found the relevant code in parser.py")
        self.todo.done(explore)
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

    @hidden
    def to_dict(self) -> dict:
        """Return snapshot state for later restoration by ``from_dict()``."""
        return {
            "todos": [
                t.model_dump() for t in (self._todos[i] for i in self._order if i in self._todos)
            ],
        }

    @hidden
    def from_dict(self, data: dict) -> None:
        """Replace current todos with snapshot state produced by ``to_dict()``."""
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

    @hidden
    def copy_todo(self, todo: Todo | str) -> Todo:
        """Return an independent copy of a manager-owned todo for delegation."""
        todo_id = self._todo_id(todo)
        current = self._todos.get(todo_id)
        if current is None:
            raise ValueError(f"todo {todo_id!r} is not managed by this TodoManager")
        return current.model_copy(deep=True)

    @classmethod
    @hidden
    def with_todo(cls, todo: Todo) -> "TodoManager":
        """Create a manager containing an independent copy of one delegated todo."""
        manager = cls()
        copied = todo.model_copy(deep=True)
        manager._todos[copied.id] = copied
        manager._order.append(copied.id)
        return manager

    @hidden
    def merge_todo(self, updated: Todo, *, base: Todo) -> Todo:
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
        """Create and return an open todo.

        ``deps`` accepts todos or IDs. ``notes`` is the current summary; extra
        keywords become durable metadata retrievable with ``get_var()``.
        """
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
        """Return the matching managed todo, or ``None`` if it is missing."""
        return self._todos.get(self._todo_id(todo_id))

    def done(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo done and return it, or ``None`` if it is missing."""
        t = self.get(todo_id)
        if t:
            t.status = "done"
        return t

    def reopen(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo open and return it, or ``None`` if it is missing.

        It remains effectively blocked while a dependency is unfinished.
        """
        t = self.get(todo_id)
        if t:
            t.status = "open"
        return t

    def remove(self, todo_id: Todo | str) -> bool:
        """Remove a todo and its references from dependent todos."""
        todo_id = self._todo_id(todo_id)
        if todo_id not in self._todos:
            return False
        del self._todos[todo_id]
        self._order = [i for i in self._order if i != todo_id]
        for todo in self._todos.values():
            todo.deps = [dep_id for dep_id in todo.deps if dep_id != todo_id]
        return True

    def clear(self) -> None:
        """Remove every todo from the current manager."""
        self._todos.clear()
        self._order.clear()

    def clear_done(self) -> int:
        """Remove completed todos and return how many were removed.

        Their IDs are also removed from remaining dependency lists.
        """
        done_ids = {todo_id for todo_id, todo in self._todos.items() if todo.status == "done"}
        for todo_id in done_ids:
            del self._todos[todo_id]
        self._order = [todo_id for todo_id in self._order if todo_id not in done_ids]
        for todo in self._todos.values():
            todo.deps = [dep_id for dep_id in todo.deps if dep_id not in done_ids]
        return len(done_ids)

    def update(self, todo_id: Todo | str, **kwargs: Any) -> Todo | None:
        """Update ``title``, ``status``, or ``notes`` and return the todo.

        Returns ``None`` if the todo is missing. Other keyword names are ignored.
        """
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
        """Add a dependency and return the todo, or ``None`` if it is missing."""
        t = self.get(todo_id)
        dep_id = self._todo_id(dep_id)
        if t and dep_id not in t.deps:
            t.deps.append(dep_id)
        return t

    def remove_dep(self, todo_id: Todo | str, dep_id: Todo | str) -> Todo | None:
        """Remove a dependency and return the todo, or ``None`` if it is missing."""
        t = self.get(todo_id)
        dep_id = self._todo_id(dep_id)
        if t and dep_id in t.deps:
            t.deps.remove(dep_id)
        return t

    # ── VARIABLES ─────────────────────────────────

    def set_var(self, todo_id: Todo | str, key: str, value: Any) -> Todo | None:
        """Store durable metadata and return the todo, or ``None`` if it is missing.

        Values that cannot be snapshot-serialized are not stored.
        """
        t = self.get(todo_id)
        if t:
            t.vars[key] = value
        return t

    def del_var(self, todo_id: Todo | str, key: str) -> Todo | None:
        """Delete a metadata key and return the todo, or ``None`` if it is missing."""
        t = self.get(todo_id)
        if t:
            t.vars.pop(key, None)
        return t

    def get_var(self, todo_id: Todo | str, key: str) -> Any | None:
        """Return a metadata value, or ``None`` if the todo or key is missing."""
        t = self.get(todo_id)
        return t.vars.get(key) if t else None

    # ── COMMENTS ──────────────────────────────────

    def comment(self, todo_id: Todo | str, body: str) -> TodoComment | None:
        """Append a progress note and return it, or ``None`` if the todo is missing.

        Comments are append-only, snapshot-backed journal entries. Read them with
        ``comments(todo)``.
        """
        t = self.get(todo_id)
        if t is None:
            return None
        c = TodoComment(body=body)
        t.comments.append(c)
        return c

    def comments(self, todo_id: Todo | str) -> list[TodoComment]:
        """Return a chronological copy of comments, or ``[]`` if none exist."""
        t = self.get(todo_id)
        return list(t.comments) if t else []

    # ── QUERIES ───────────────────────────────────

    def _effective_status(self, todo: Todo) -> str:
        """Resolve dependency-derived open/blocked state."""
        if todo.status == "done":
            return "done"
        if todo.status in {"open", "blocked"}:
            return "blocked" if todo.is_blocked(self._todos) else "open"
        return todo.status

    def list_todos(self, status: str | None = None) -> list[Todo]:
        """Return todos in creation order, optionally filtered by effective status.

        Use ``"open"``, ``"blocked"``, or ``"done"``. Effective blocking is
        derived from unfinished dependencies and updates automatically.
        """
        todos = [self._todos[i] for i in self._order if i in self._todos]
        if status is None:
            return todos
        return [todo for todo in todos if self._effective_status(todo) == status]

    # ── STATUS ────────────────────────────────────

    @staticmethod
    def _status_title(title: str) -> str:
        """Collapse a title to one bounded display line."""
        compact = " ".join(title.split()) or "(untitled)"
        if len(compact) <= _STATUS_TITLE_CHARS:
            return compact
        return compact[: _STATUS_TITLE_CHARS - 1].rstrip() + "…"

    def _status_line(self, todo: Todo) -> str:
        """Render one bounded todo row without exposing detail payloads."""
        effective = self._effective_status(todo)
        icon = {"open": "○", "done": "✓", "blocked": "●"}.get(effective, "?")
        unresolved = [
            dep_id
            for dep_id in todo.deps
            if dep_id in self._todos and self._effective_status(self._todos[dep_id]) != "done"
        ]
        dep_text = ""
        if unresolved:
            shown = unresolved[:_STATUS_MAX_DEPS]
            suffix = f", +{len(unresolved) - len(shown)}" if len(unresolved) > len(shown) else ""
            dep_text = f" [needs: {', '.join(shown)}{suffix}]"

        details: list[str] = []
        if todo.notes.strip():
            details.append("note")
        if todo.vars:
            details.append(f"{len(todo.vars)} var{'s' if len(todo.vars) != 1 else ''}")
        if todo.comments:
            details.append(f"{len(todo.comments)} comment{'s' if len(todo.comments) != 1 else ''}")
        detail_text = f" · {' · '.join(details)}" if details else ""
        return f"  {icon} [{todo.id}] {self._status_title(todo.title)}{dep_text}{detail_text}"

    def status(self, max_items: int = _STATUS_MAX_ITEMS, max_chars: int = _STATUS_MAX_CHARS) -> str:
        """Return a compact, bounded progress summary.

        Orders open and blocked work before newest completed history. Detail
        payloads use compact badges; inspect one with ``get()`` or ``comments()``.
        ``max_items`` and ``max_chars`` bound model-context usage.
        """
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")

        todos = [self._todos[i] for i in self._order if i in self._todos]
        if not todos:
            return "(no todos)"

        by_status: dict[str, list[Todo]] = {"open": [], "blocked": [], "done": []}
        other: list[Todo] = []
        for todo in todos:
            effective = self._effective_status(todo)
            if effective in by_status:
                by_status[effective].append(todo)
            else:
                other.append(todo)
        ordered = [*by_status["open"], *by_status["blocked"], *other, *reversed(by_status["done"])]
        selected = ordered[:max_items]

        def render(rows: list[Todo]) -> str:
            shown_ids = {todo.id for todo in rows}
            omitted = [todo for todo in ordered if todo.id not in shown_ids]
            header = f"Todos ({len(by_status['done'])}/{len(todos)} done"
            if omitted:
                header += f"; showing {len(rows)}"
            lines = [header + "):", *(self._status_line(todo) for todo in rows)]
            if omitted:
                counts: dict[str, int] = {}
                for todo in omitted:
                    status = self._effective_status(todo)
                    counts[status] = counts.get(status, 0) + 1
                labels = [
                    f"{count} {status}"
                    for status in ("open", "blocked", "done")
                    if (count := counts.get(status, 0))
                ]
                other_count = len(omitted) - sum(
                    counts.get(s, 0) for s in ("open", "blocked", "done")
                )
                if other_count:
                    labels.append(f"{other_count} other")
                lines.append(f"  … +{len(omitted)} not shown ({', '.join(labels)})")

            hints: list[str] = []
            if omitted:
                hints.append("list all: self.todo.list_todos()")
            if any(todo.notes.strip() or todo.vars or todo.comments for todo in rows):
                hints.append("inspect: self.todo.get(id)")
            if by_status["done"]:
                hints.append("prune done: self.todo.clear_done()")
            if hints:
                lines.append("Hint — " + " · ".join(hints))
            return "\n".join(lines)

        output = render(selected)
        while selected and len(output) > max_chars:
            selected.pop()
            output = render(selected)
        return output
