# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``TodoManager`` CRUD, dependency management, variable management,
``list_todos`` filtering, and ``status()`` rendering.

Covers the methods that were previously untested:
  - get / done / reopen / remove / update   (CRUD)
  - add_dep / remove_dep                    (dependencies)
  - set_var / del_var / get_var             (variables)
  - list_todos(status=...)                  (filtered queries)
  - status()                               (display rendering)
"""

from nooa.tools.todo import Todo, TodoManager


# =============================================================================
# Phase 1 — CRUD: get, done, reopen, remove, update
# =============================================================================


class TestGet:
    def test_get_returns_todo_by_id(self) -> None:
        tm = TodoManager()
        t = tm.add("Task A")
        assert tm.get(t.id) is t

    def test_get_missing_id_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.get("nonexistent") is None

    def test_get_after_add_multiple(self) -> None:
        """get() retrieves the correct todo when several are present."""
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        c = tm.add("C")
        assert tm.get(b.id) is b
        assert tm.get(a.id) is a
        assert tm.get(c.id) is c


class TestDone:
    def test_done_sets_status_to_done(self) -> None:
        tm = TodoManager()
        t = tm.add("Finish feature")
        result = tm.done(t.id)
        assert result is t
        assert t.status == "done"

    def test_done_on_missing_id_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.done("missing") is None

    def test_done_is_idempotent(self) -> None:
        """Calling done() twice on the same todo leaves it done."""
        tm = TodoManager()
        t = tm.add("Deploy")
        tm.done(t.id)
        tm.done(t.id)
        assert t.status == "done"

    def test_done_does_not_affect_other_todos(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        tm.done(a.id)
        assert b.status == "open"


class TestReopen:
    def test_reopen_sets_status_to_open(self) -> None:
        tm = TodoManager()
        t = tm.add("Fix bug")
        tm.done(t.id)
        result = tm.reopen(t.id)
        assert result is t
        assert t.status == "open"

    def test_reopen_on_missing_id_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.reopen("ghost") is None

    def test_reopen_of_already_open_todo_stays_open(self) -> None:
        """reopen() on an already-open todo is a no-op for status."""
        tm = TodoManager()
        t = tm.add("Still open")
        tm.reopen(t.id)
        assert t.status == "open"

    def test_reopen_roundtrip(self) -> None:
        """done → reopen → done transitions work correctly."""
        tm = TodoManager()
        t = tm.add("Toggle")
        tm.done(t.id)
        assert t.status == "done"
        tm.reopen(t.id)
        assert t.status == "open"
        tm.done(t.id)
        assert t.status == "done"


class TestRemove:
    def test_remove_existing_returns_true(self) -> None:
        tm = TodoManager()
        t = tm.add("To be removed")
        assert tm.remove(t.id) is True

    def test_remove_existing_deletes_from_todos(self) -> None:
        tm = TodoManager()
        t = tm.add("Gone")
        tm.remove(t.id)
        assert tm.get(t.id) is None

    def test_remove_existing_deletes_from_order(self) -> None:
        tm = TodoManager()
        t = tm.add("Gone")
        tm.remove(t.id)
        assert t.id not in tm._order

    def test_remove_nonexistent_returns_false(self) -> None:
        tm = TodoManager()
        assert tm.remove("no-such-id") is False

    def test_remove_preserves_other_todos_and_their_order(self) -> None:
        """Removing one todo must not shift or drop its neighbours."""
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        c = tm.add("C")
        tm.remove(b.id)
        remaining = [t.id for t in tm.list_todos()]
        assert remaining == [a.id, c.id]

    def test_remove_twice_returns_false_second_time(self) -> None:
        tm = TodoManager()
        t = tm.add("Once")
        assert tm.remove(t.id) is True
        assert tm.remove(t.id) is False


class TestUpdate:
    def test_update_title(self) -> None:
        tm = TodoManager()
        t = tm.add("Old title")
        result = tm.update(t.id, title="New title")
        assert result is t
        assert t.title == "New title"

    def test_update_status(self) -> None:
        tm = TodoManager()
        t = tm.add("Work")
        tm.update(t.id, status="done")
        assert t.status == "done"

    def test_update_notes(self) -> None:
        tm = TodoManager()
        t = tm.add("Annotated")
        tm.update(t.id, notes="important context")
        assert t.notes == "important context"

    def test_update_multiple_fields_at_once(self) -> None:
        tm = TodoManager()
        t = tm.add("Multi")
        tm.update(t.id, title="Updated", notes="new note", status="done")
        assert t.title == "Updated"
        assert t.notes == "new note"
        assert t.status == "done"

    def test_update_ignores_unknown_fields(self) -> None:
        """Fields not in the allowed set must be silently ignored."""
        tm = TodoManager()
        t = tm.add("Task")
        original_id = t.id
        tm.update(t.id, bogus_field="should not appear", id="hijack")
        assert t.id == original_id  # id is protected by the allowed-set filter

    def test_update_on_missing_id_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.update("ghost", title="X") is None


# =============================================================================
# Phase 2 — Dependency management: add_dep, remove_dep
# =============================================================================


class TestAddDep:
    def test_add_dep_appends_dep_id(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        result = tm.add_dep(b.id, a.id)
        assert result is b
        assert a.id in b.deps

    def test_add_dep_on_missing_todo_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.add_dep("ghost", "other") is None

    def test_add_dep_is_idempotent(self) -> None:
        """Adding the same dep twice must not create a duplicate entry."""
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        tm.add_dep(b.id, a.id)
        tm.add_dep(b.id, a.id)
        assert b.deps.count(a.id) == 1

    def test_add_dep_multiple_deps(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        c = tm.add("C")
        tm.add_dep(c.id, a.id)
        tm.add_dep(c.id, b.id)
        assert set(c.deps) == {a.id, b.id}


class TestRemoveDep:
    def test_remove_dep_removes_existing(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        tm.add_dep(b.id, a.id)
        result = tm.remove_dep(b.id, a.id)
        assert result is b
        assert a.id not in b.deps

    def test_remove_dep_on_missing_todo_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.remove_dep("ghost", "any") is None

    def test_remove_dep_when_dep_not_present_is_noop(self) -> None:
        """remove_dep on a dep that was never added must leave deps unchanged."""
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        result = tm.remove_dep(b.id, a.id)  # a.id was never added as dep
        assert result is b
        assert b.deps == []

    def test_remove_dep_does_not_remove_other_deps(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        c = tm.add("C")
        tm.add_dep(c.id, a.id)
        tm.add_dep(c.id, b.id)
        tm.remove_dep(c.id, a.id)
        assert b.id in c.deps
        assert a.id not in c.deps


# =============================================================================
# Phase 3 — Variable management: set_var, del_var, get_var
# =============================================================================


class TestSetVar:
    def test_set_var_stores_value(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        result = tm.set_var(t.id, "branch", "main")
        assert result is t
        assert t.vars["branch"] == "main"

    def test_set_var_on_missing_todo_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.set_var("ghost", "key", "val") is None

    def test_set_var_overwrites_existing_key(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "status_code", 200)
        tm.set_var(t.id, "status_code", 404)
        assert t.vars["status_code"] == 404

    def test_set_var_accepts_complex_value(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        payload = {"commits": ["abc", "def"], "count": 2}
        tm.set_var(t.id, "payload", payload)
        assert t.vars["payload"] == payload


class TestDelVar:
    def test_del_var_removes_key(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "temp", 42)
        result = tm.del_var(t.id, "temp")
        assert result is t
        assert "temp" not in t.vars

    def test_del_var_on_missing_todo_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.del_var("ghost", "key") is None

    def test_del_var_on_missing_key_is_noop(self) -> None:
        """del_var uses vars.pop — removing a non-existent key must not raise."""
        tm = TodoManager()
        t = tm.add("Task")
        result = tm.del_var(t.id, "never-set")
        assert result is t  # todo is returned even when key absent

    def test_del_var_does_not_remove_other_keys(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "keep", "yes")
        tm.set_var(t.id, "remove", "no")
        tm.del_var(t.id, "remove")
        assert t.vars["keep"] == "yes"


class TestGetVar:
    def test_get_var_returns_stored_value(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "pr", 99)
        assert tm.get_var(t.id, "pr") == 99

    def test_get_var_missing_key_returns_none(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        assert tm.get_var(t.id, "nonexistent") is None

    def test_get_var_missing_todo_returns_none(self) -> None:
        tm = TodoManager()
        assert tm.get_var("ghost", "key") is None

    def test_get_var_reflects_set_var(self) -> None:
        """get_var and set_var operate on the same underlying dict."""
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "x", [1, 2, 3])
        assert tm.get_var(t.id, "x") == [1, 2, 3]


# =============================================================================
# Phase 4 — list_todos filtering & status() rendering
# =============================================================================


class TestListTodos:
    def test_list_todos_no_filter_returns_all_in_insertion_order(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        c = tm.add("C")
        ids = [t.id for t in tm.list_todos()]
        assert ids == [a.id, b.id, c.id]

    def test_list_todos_empty_manager_returns_empty_list(self) -> None:
        tm = TodoManager()
        assert tm.list_todos() == []

    def test_list_todos_filter_open(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        tm.done(b.id)
        open_todos = tm.list_todos(status="open")
        assert len(open_todos) == 1
        assert open_todos[0] is a

    def test_list_todos_filter_done(self) -> None:
        tm = TodoManager()
        a = tm.add("A")
        b = tm.add("B")
        tm.done(a.id)
        done_todos = tm.list_todos(status="done")
        assert len(done_todos) == 1
        assert done_todos[0] is a

    def test_list_todos_filter_blocked_dynamic(self) -> None:
        """A todo with an open dependency must appear as 'blocked' even
        though its stored status is 'open' — blocking is computed dynamically."""
        tm = TodoManager()
        dep = tm.add("Prerequisite")  # still open
        task = tm.add("Blocked task", deps=[dep.id])
        blocked = tm.list_todos(status="blocked")
        assert task in blocked
        assert dep not in blocked

    def test_list_todos_blocked_unblocked_after_dep_done(self) -> None:
        """Once the dependency is marked done the task must appear 'open' again."""
        tm = TodoManager()
        dep = tm.add("Dep")
        task = tm.add("Task", deps=[dep.id])
        assert task in tm.list_todos(status="blocked")
        tm.done(dep.id)
        assert task in tm.list_todos(status="open")
        assert task not in tm.list_todos(status="blocked")

    def test_list_todos_stored_blocked_with_deps_resolved_appears_open(self) -> None:
        """A todo whose stored status is 'blocked' but all deps are done
        must be re-classified as 'open' by _effective()."""
        tm = TodoManager()
        dep = tm.add("Dep")
        task = tm.add("Task", deps=[dep.id])
        task.status = "blocked"  # set stored status explicitly
        tm.done(dep.id)  # resolve the dep
        open_todos = tm.list_todos(status="open")
        assert task in open_todos

    def test_list_todos_unknown_status_returns_empty(self) -> None:
        """Querying an unrecognised status string must return an empty list."""
        tm = TodoManager()
        tm.add("A")
        assert tm.list_todos(status="in-progress") == []


class TestStatus:
    def test_status_empty_manager(self) -> None:
        tm = TodoManager()
        assert tm.status() == "(no todos)"

    def test_status_summary_line_counts(self) -> None:
        """Header shows (done/total done)."""
        tm = TodoManager()
        a = tm.add("A")
        tm.add("B")
        tm.done(a.id)
        output = tm.status()
        assert "1/2 done" in output

    def test_status_contains_todo_title(self) -> None:
        tm = TodoManager()
        tm.add("Deploy to prod")
        assert "Deploy to prod" in tm.status()

    def test_status_open_icon(self) -> None:
        tm = TodoManager()
        tm.add("Open task")
        assert "○" in tm.status()

    def test_status_done_icon(self) -> None:
        tm = TodoManager()
        t = tm.add("Finished")
        tm.done(t.id)
        assert "✓" in tm.status()

    def test_status_blocked_icon_for_unresolved_dep(self) -> None:
        """A todo with an open dependency must render with the ● blocked icon."""
        tm = TodoManager()
        dep = tm.add("Dep")
        tm.add("Blocked", deps=[dep.id])
        assert "●" in tm.status()

    def test_status_shows_dep_info(self) -> None:
        tm = TodoManager()
        dep = tm.add("Dep")
        tm.add("Task", deps=[dep.id])
        output = tm.status()
        assert "needs:" in output
        assert dep.id in output

    def test_status_shows_notes(self) -> None:
        tm = TodoManager()
        tm.add("Annotated", notes="check logs first")
        assert "check logs first" in tm.status()

    def test_status_shows_vars_when_set(self) -> None:
        tm = TodoManager()
        t = tm.add("Task")
        tm.set_var(t.id, "pr", 42)
        assert "pr" in tm.status()

    def test_status_insertion_order_preserved_in_output(self) -> None:
        """Todos must appear in insertion order in the rendered output."""
        tm = TodoManager()
        tm.add("First")
        tm.add("Second")
        tm.add("Third")
        output = tm.status()
        assert output.index("First") < output.index("Second") < output.index("Third")
