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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

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
    state). Use comments for meaningful findings, decisions, completed steps,
    and verification results. Keep the Todo description aligned with the current
    objective; comments preserve the chronology of how understanding changed.
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
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "notes"),
        description="Current scope, constraints, approach, and definition of done",
    )
    comments: list[TodoComment] = Field(
        default_factory=list,
        description="Append-only chronological progress journal",
    )

    @property
    @hidden
    def notes(self) -> str:
        """Deprecated compatibility alias for ``description``."""
        return self.description

    @notes.setter
    def notes(self, value: str) -> None:
        self.description = value

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
    """Track multi-step work, dependencies, metadata, and progress comments.

    Every todo argument accepts either a ``Todo`` returned by this manager or
    its ID. Keep work visible with ``status()``, identify the current objective
    with ``activate()``, and inspect details with ``get()``. Keep titles and
    descriptions aligned with the current understanding of the work. Add comments after material
    findings, decisions, completed steps, and verification—not routine narration.

    Example::

        explore = self.todo.add("Explore the repository")
        fix = self.todo.add("Implement the fix", deps=[explore])
        self.todo.comment(explore, "Found the relevant code in parser.py")
        self.todo.complete(explore)
        print(self.todo.status())
    """

    __nosnapshot__ = False  # Override Skill's __nosnapshot__ = True
    context_block = ("todo_status", "self.todo.status()")

    def __init__(self, state: dict | None = None) -> None:
        self._todos: dict[str, Todo] = {}
        self._order: list[str] = []  # insertion order
        self._active_id: str | None = None
        if state:
            self.from_dict(state)

    # ── SERIALIZATION ─────────────────────────────

    @hidden
    def to_dict(self) -> dict:
        """Return snapshot state for later restoration by ``from_dict()``."""
        self.active()  # Normalize direct mutation of the manager-owned Todo.
        return {
            "todos": [
                t.model_dump() for t in (self._todos[i] for i in self._order if i in self._todos)
            ],
            "active_id": self._active_id,
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
        active_id = data.get("active_id")
        active = self._todos.get(active_id) if isinstance(active_id, str) else None
        self._active_id = active_id if active is not None and active.status != "done" else None

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
        manager._active_id = copied.id if copied.status != "done" else None
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
        for field in ("title", "status", "deps", "description"):
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
        for field in ("title", "status", "deps", "description", "vars", "comments"):
            setattr(current, field, getattr(candidate, field))
        if current.status == "done" and current.id == self._active_id:
            self._active_id = None
        return current

    def add(
        self,
        title: str,
        deps: list[Todo | str] | None = None,
        description: str = "",
        **vars: Any,
    ) -> Todo:
        """Create and return an open todo.

        ``deps`` accepts todos or IDs. ``description`` is the current scope,
        constraints, approach, and definition of done; revise it as understanding
        changes. Extra keywords become durable metadata available through the
        returned Todo's ``v`` proxy.
        """
        legacy_notes = vars.pop("notes", None)
        if legacy_notes is not None:
            if description:
                raise ValueError("use either description or legacy notes, not both")
            description = legacy_notes
        t = Todo(
            title=title,
            deps=[self._todo_id(dep) for dep in deps or []],
            description=description,
            vars=dict(vars),
        )
        self._todos[t.id] = t
        self._order.append(t.id)
        return t

    def get(self, todo_id: Todo | str) -> Todo | None:
        """Return the matching managed todo, or ``None`` if it is missing."""
        return self._todos.get(self._todo_id(todo_id))

    @hidden
    def done(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo done and return it, or ``None`` if it is missing.

        Completing the active todo also clears the active selection; completing
        one of its dependencies leaves the active task selected.
        """
        t = self.get(todo_id)
        if t:
            t.status = "done"
            if t.id == self._active_id:
                self._active_id = None
        return t

    def complete(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo done and return it, or ``None`` if it is missing."""
        return self.done(todo_id)

    @hidden
    def reopen(self, todo_id: Todo | str) -> Todo | None:
        """Mark a todo open and return it, or ``None`` if it is missing.

        It remains effectively blocked while a dependency is unfinished.
        """
        t = self.get(todo_id)
        if t:
            t.status = "open"
        return t

    @hidden
    def remove(self, todo_id: Todo | str) -> bool:
        """Remove a todo and its references from dependent todos."""
        todo_id = self._todo_id(todo_id)
        if todo_id not in self._todos:
            return False
        del self._todos[todo_id]
        self._order = [i for i in self._order if i != todo_id]
        if todo_id == self._active_id:
            self._active_id = None
        for todo in self._todos.values():
            todo.deps = [dep_id for dep_id in todo.deps if dep_id != todo_id]
        return True

    @hidden
    def clear(self) -> None:
        """Remove every todo from the current manager and clear the active task."""
        self._todos.clear()
        self._order.clear()
        self._active_id = None

    @hidden
    def clear_done(self) -> int:
        """Remove completed todos and return how many were removed.

        Their IDs are also removed from remaining dependency lists.
        """
        done_ids = {todo_id for todo_id, todo in self._todos.items() if todo.status == "done"}
        for todo_id in done_ids:
            del self._todos[todo_id]
        self._order = [todo_id for todo_id in self._order if todo_id not in done_ids]
        if self._active_id in done_ids:
            self._active_id = None
        for todo in self._todos.values():
            todo.deps = [dep_id for dep_id in todo.deps if dep_id not in done_ids]
        return len(done_ids)

    def update(self, todo_id: Todo | str, **kwargs: Any) -> Todo | None:
        """Update ``title``, ``status``, or ``description`` and return the todo.

        Keep title and description aligned with the current understanding of the
        task. Use ``comment()`` to append material progress and evidence. Returns
        ``None`` if the todo is missing; other keyword names are ignored.
        """
        t = self.get(todo_id)
        if t is None:
            return None
        if "notes" in kwargs and "description" not in kwargs:
            kwargs["description"] = kwargs["notes"]
        allowed = {"title", "status", "description"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(t, k, v)
        if t.status == "done" and t.id == self._active_id:
            self._active_id = None
        return t

    # ── ACTIVE TODO ────────────────────────────────

    def activate(self, todo_id: Todo | str) -> Todo:
        """Make one open todo the active task shown prominently by ``status()``.

        A new call replaces the previous active task. The full workspace remains
        available through ``list_todos()``. Raises ``ValueError`` when the todo
        is missing or already complete.
        """
        todo_id = self._todo_id(todo_id)
        todo = self._todos.get(todo_id)
        if todo is None:
            raise ValueError(f"todo {todo_id!r} is not managed by this TodoManager")
        if todo.status == "done":
            raise ValueError(f"todo {todo_id!r} is already done")
        self._active_id = todo_id
        return todo

    @hidden
    def deactivate(self) -> Todo | None:
        """Clear the active task and return it, if one was active."""
        todo = self.active()
        self._active_id = None
        return todo

    @hidden
    def active(self) -> Todo | None:
        """Return the active todo, or ``None`` when no open todo is active."""
        if self._active_id is None:
            return None
        todo = self._todos.get(self._active_id)
        if todo is None or todo.status == "done":
            self._active_id = None
            return None
        return todo

    # ── DEPENDENCIES ──────────────────────────────

    @hidden
    def add_dep(self, todo_id: Todo | str, dep_id: Todo | str) -> Todo | None:
        """Add a dependency and return the todo, or ``None`` if it is missing."""
        t = self.get(todo_id)
        dep_id = self._todo_id(dep_id)
        if t and dep_id not in t.deps:
            t.deps.append(dep_id)
        return t

    @hidden
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

    @hidden
    def del_var(self, todo_id: Todo | str, key: str) -> Todo | None:
        """Delete a metadata key and return the todo, or ``None`` if it is missing."""
        t = self.get(todo_id)
        if t:
            t.vars.pop(key, None)
        return t

    @hidden
    def get_var(self, todo_id: Todo | str, key: str) -> Any | None:
        """Return a metadata value, or ``None`` if the todo or key is missing."""
        t = self.get(todo_id)
        return t.vars.get(key) if t else None

    # ── COMMENTS ──────────────────────────────────

    def comment(self, todo_id: Todo | str, body: str) -> TodoComment | None:
        """Append material progress and return it, or ``None`` if missing.

        Record meaningful findings, decisions, completed steps, and verification
        results—not routine narration. Comments are an append-only, snapshot-backed
        journal visible on the ``Todo`` returned by ``get(todo)``.
        """
        t = self.get(todo_id)
        if t is None:
            return None
        c = TodoComment(body=body)
        t.comments.append(c)
        return c

    @hidden
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

    # Skill lifecycle hooks are framework plumbing, not task-management operations.

    @hidden
    def attach(self, agent: Any) -> None:
        super().attach(agent)

    @hidden
    def detach(self) -> None:
        super().detach()

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
        if todo.description.strip():
            details.append("description")
        if todo.vars:
            details.append(f"{len(todo.vars)} var{'s' if len(todo.vars) != 1 else ''}")
        if todo.comments:
            details.append(f"{len(todo.comments)} comment{'s' if len(todo.comments) != 1 else ''}")
        detail_text = f" · {' · '.join(details)}" if details else ""
        return f"  {icon} [{todo.id}] {self._status_title(todo.title)}{dep_text}{detail_text}"

    def _dependency_closure(self, root: Todo) -> tuple[list[Todo], list[str]]:
        """Return transitive dependencies before *root*, without recursion."""
        ordered: list[Todo] = []
        state: dict[str, int] = {root.id: 1}  # 1 = visiting, 2 = emitted
        missing: list[str] = []
        stack: list[tuple[Todo, int]] = [(root, 0)]
        while stack:
            todo, dep_index = stack[-1]
            if dep_index >= len(todo.deps):
                stack.pop()
                state[todo.id] = 2
                if todo.id != root.id:
                    ordered.append(todo)
                continue

            dep_id = todo.deps[dep_index]
            stack[-1] = (todo, dep_index + 1)
            dependency = self._todos.get(dep_id)
            if dependency is None:
                if dep_id not in missing:
                    missing.append(dep_id)
                continue
            if state.get(dependency.id, 0) == 0:
                state[dependency.id] = 1
                stack.append((dependency, 0))
        return ordered, missing

    @staticmethod
    def _bounded_status(output: str, max_chars: int) -> str:
        """Apply a final hard bound even when fixed status framing is large."""
        if len(output) <= max_chars:
            return output
        marker = "\n… status truncated"
        return output[: max_chars - len(marker)].rstrip() + marker

    def status(self, max_items: int = _STATUS_MAX_ITEMS, max_chars: int = _STATUS_MAX_CHARS) -> str:
        """Return a compact, bounded progress summary.

        When a todo is active, shows its description and transitive dependencies,
        followed by a compact summary of unrelated work. Otherwise, orders open
        and blocked work before newest completed history. ``list_todos()`` always
        returns the full workspace.
        """
        if max_items < 0:
            raise ValueError("max_items must be non-negative")
        if max_chars < 200:
            raise ValueError("max_chars must be at least 200")

        todos = [self._todos[i] for i in self._order if i in self._todos]
        if not todos:
            return "(no todos)"

        active = self.active()
        if active is not None:
            dependencies, missing = self._dependency_closure(active)
            # The active task is always represented by its detail card; max_items
            # limits the additional dependency rows.
            selected = dependencies[:max_items]

            def render_active(rows: list[Todo]) -> str:
                omitted_count = len(dependencies) - len(rows)
                active_title = self._status_title(active.title)
                effective = self._effective_status(active)
                details = [effective]
                if dependencies or missing:
                    details.append(
                        f"{len(dependencies) + len(missing)} "
                        f"dependenc{'y' if len(dependencies) + len(missing) == 1 else 'ies'}"
                    )
                if active.vars:
                    details.append(f"{len(active.vars)} var{'s' if len(active.vars) != 1 else ''}")
                if active.comments:
                    details.append(
                        f"{len(active.comments)} comment{'s' if len(active.comments) != 1 else ''}"
                    )
                lines = [f"Active [{active.id}] {active_title}", f"Status: {' · '.join(details)}"]
                note = " ".join(active.description.split())
                lines.append(f"Description: {note or '(none)'}")
                if active.comments:
                    recent = " ".join(active.comments[-1].body.split())
                    lines.append(f"Recent activity: {recent}")
                if rows or omitted_count or missing:
                    lines.append("")
                    lines.append("Dependencies:")
                    lines.extend(self._status_line(todo) for todo in rows)
                if omitted_count:
                    lines.append(f"  … +{omitted_count} dependencies not shown")
                if missing:
                    lines.append(f"  ! missing dependencies: {', '.join(missing)}")

                dependency_ids = {todo.id for todo in dependencies}
                other = [
                    todo for todo in todos if todo.id != active.id and todo.id not in dependency_ids
                ]
                if other:
                    counts: dict[str, int] = {}
                    for todo in other:
                        status = self._effective_status(todo)
                        counts[status] = counts.get(status, 0) + 1
                    summary = " · ".join(
                        f"{count} {status}"
                        for status in ("open", "blocked", "done")
                        if (count := counts.get(status, 0))
                    )
                    other_count = len(other) - sum(
                        counts.get(status, 0) for status in ("open", "blocked", "done")
                    )
                    if other_count:
                        summary += (" · " if summary else "") + f"{other_count} other"
                    lines.extend(("", f"Other Todos: {summary}"))
                hints = ["clear active: self.todo.deactivate()"]
                if not note:
                    hints.append(
                        f'refine task: self.todo.update("{active.id}", description="scope and next step")'
                    )
                if not active.comments:
                    hints.append(
                        f'record progress: self.todo.comment("{active.id}", "what changed or was learned")'
                    )
                else:
                    hints.append("record material progress with self.todo.comment(id, ...)")
                hints.append("show all: self.todo.list_todos()")
                lines.append("Hint — " + " · ".join(hints))
                return self._bounded_status("\n".join(lines), max_chars)

            output = render_active(selected)
            while selected and output.endswith("… status truncated"):
                selected.pop()
                output = render_active(selected)
            return output

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
            if any(todo.description.strip() or todo.vars or todo.comments for todo in rows):
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
