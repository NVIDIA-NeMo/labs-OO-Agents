# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compact model-facing TodoManager status summary."""

import pytest

from nooa.tools.todo import TodoManager


def test_empty_status_is_minimal() -> None:
    assert TodoManager().status() == "(no todos)"


def test_status_hides_payloads_and_advertises_inspection() -> None:
    manager = TodoManager()
    todo = manager.add(
        "Investigate\nmultiline failure",
        notes="secret note payload",
        secret={"large": "value"},
    )
    manager.comment(todo, "secret comment payload")

    output = manager.status()

    assert "Investigate multiline failure" in output
    assert "· note · 1 var · 1 comment" in output
    assert "secret note payload" not in output
    assert "secret comment payload" not in output
    assert "large" not in output
    assert "inspect: self.todo.get(id)" in output
    assert "list all" not in output
    assert "prune done" not in output


def test_status_prioritizes_active_work_and_recent_done_history() -> None:
    manager = TodoManager()
    old_done = manager.add("old done")
    manager.done(old_done)
    manager.add("first open")
    blocker = manager.add("blocker")
    manager.add("blocked", deps=[blocker])
    recent_done = manager.add("recent done")
    manager.done(recent_done)

    output = manager.status(max_items=3)

    assert "Todos (2/5 done; showing 3):" in output
    assert output.index("first open") < output.index("blocker") < output.index("blocked")
    assert "old done" not in output
    assert "recent done" not in output
    assert "… +2 not shown (2 done)" in output
    assert "list all: self.todo.list_todos()" in output
    assert "prune done: self.todo.clear_done()" in output


def test_status_uses_recent_completed_items_when_capacity_remains() -> None:
    manager = TodoManager()
    oldest = manager.add("oldest")
    newest = manager.add("newest")
    manager.done(oldest)
    manager.done(newest)

    output = manager.status(max_items=1)

    assert "newest" in output
    assert "oldest" not in output


def test_status_at_exact_limit_is_not_marked_truncated() -> None:
    manager = TodoManager()
    for index in range(10):
        manager.add(f"item {index}")

    output = manager.status()

    assert output.startswith("Todos (0/10 done):")
    assert "showing" not in output
    assert "not shown" not in output
    assert "list all" not in output


def test_status_limit_boundary_and_omitted_status_counts() -> None:
    manager = TodoManager()
    todos = [manager.add(f"item {index}") for index in range(11)]
    manager.done(todos[0])
    manager.done(todos[1])
    manager.add_dep(todos[3], todos[2])

    output = manager.status()

    assert "Todos (2/11 done; showing 10):" in output
    assert "… +1 not shown (1 done)" in output
    assert (
        len([line for line in output.splitlines() if line.startswith("  ") and "[" in line]) == 10
    )


def test_effective_blocked_status_updates_after_dependency_completes() -> None:
    manager = TodoManager()
    dependency = manager.add("dependency")
    dependent = manager.add("dependent", deps=[dependency])

    assert dependent in manager.list_todos("blocked")
    assert "●" in manager.status()

    manager.done(dependency)

    assert dependent in manager.list_todos("open")
    dependent_line = next(line for line in manager.status().splitlines() if "dependent" in line)
    assert "○" in dependent_line
    assert "[needs:" not in dependent_line


def test_clear_done_removes_only_completed_todos() -> None:
    manager = TodoManager()
    completed = manager.add("completed")
    active = manager.add("active", deps=[completed])
    manager.done(completed)

    assert manager.clear_done() == 1
    assert manager.list_todos() == [active]
    assert active.deps == []
    assert manager.clear_done() == 0


def test_remove_prunes_dependency_references() -> None:
    manager = TodoManager()
    dependency = manager.add("dependency")
    dependent = manager.add("dependent", deps=[dependency])

    assert dependent in manager.list_todos("blocked")
    assert manager.remove(dependency) is True
    assert dependent.deps == []
    assert dependent in manager.list_todos("open")


def test_status_respects_character_bound_by_dropping_rows() -> None:
    manager = TodoManager()
    for index in range(10):
        manager.add(f"{index} " + "x" * 200)

    output = manager.status(max_chars=300)

    assert len(output) <= 300
    assert "not shown" in output


def test_zero_item_limit_keeps_summary_and_actionable_hints() -> None:
    manager = TodoManager()
    todo = manager.add("completed")
    manager.done(todo)

    output = manager.status(max_items=0)

    assert output.startswith("Todos (1/1 done; showing 0):\n  … +1 not shown (1 done)")
    assert "list all: self.todo.list_todos()" in output
    assert "prune done: self.todo.clear_done()" in output


def test_equal_todo_values_are_counted_independently_when_truncated() -> None:
    manager = TodoManager()
    first = manager.add("same")
    second = manager.add("same")
    # Match every equality-relevant value except the stable identity.
    second.created_at = first.created_at

    output = manager.status(max_items=1)

    assert f"[{first.id}]" in output
    assert f"[{second.id}]" not in output
    assert "… +1 not shown (1 open)" in output


@pytest.mark.parametrize(
    ("max_items", "max_chars", "message"), [(-1, 2000, "max_items"), (10, 199, "max_chars")]
)
def test_status_rejects_invalid_limits(max_items: int, max_chars: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TodoManager().status(max_items=max_items, max_chars=max_chars)


def test_todo_manager_docs_are_task_focused() -> None:
    from nooa.agentdoc import doc

    output = doc(TodoManager())

    assert "Every todo argument accepts either a ``Todo``" in output
    assert "def add(" in output
    assert "extra\n        keywords become durable metadata" in output
    assert "def clear_done(" in output
    assert "def complete(" in output
    assert "def activate(" in output
    assert "def deactivate(" in output
    assert "def active(" in output
    assert "def status(self, max_items: int = 10, max_chars: int = 2000)" in output
    assert "def to_dict(" not in output
    assert "def from_dict(" not in output
    assert "def is_blocked(" not in output
    assert "def set_var(" in output
    assert "vars: SnapshotVars" not in output
    assert "class SnapshotVars:" not in output
    assert "def pop(" not in output


def test_active_status_shows_notes_dependencies_and_other_summary() -> None:
    manager = TodoManager()
    unrelated_open = manager.add("unrelated open")
    unrelated_done = manager.add("unrelated done")
    manager.complete(unrelated_done)
    leaf = manager.add("leaf dependency")
    middle = manager.add("middle dependency", deps=[leaf])
    root = manager.add(
        "active root",
        deps=[middle],
        notes="Detailed instructions for the current work.",
        owner="controller",
    )
    manager.comment(root, "Implementation started")

    assert manager.activate(root) is root
    assert manager.active() is root

    output = manager.status()

    assert output.startswith(
        f"Active [{root.id}] active root\nStatus: blocked · 2 dependencies · 1 var · 1 comment"
    )
    assert "Notes: Detailed instructions for the current work." in output
    rows = [line for line in output.splitlines() if line.startswith("  ") and "[" in line]
    assert [todo.id for todo in (leaf, middle)] == [
        line.split("[", 1)[1].split("]", 1)[0] for line in rows
    ]
    assert "Other Todos: 1 open · 1 done" in output
    assert "unrelated open" not in output
    assert "unrelated done" not in output
    assert "clear active: self.todo.deactivate()" in output
    assert manager.list_todos() == [unrelated_open, unrelated_done, leaf, middle, root]


def test_activate_replaces_previous_and_deactivate_restores_regular_status() -> None:
    manager = TodoManager()
    first = manager.add("first")
    second = manager.add("second")

    manager.activate(first)
    assert manager.activate(second) is second
    assert manager.active() is second
    assert "Other Todos: 1 open" in manager.status()

    assert manager.deactivate() is second
    assert manager.active() is None
    assert "first" in manager.status()
    assert "second" in manager.status()
    assert manager.deactivate() is None


def test_activate_rejects_unmanaged_or_completed_todo() -> None:
    manager = TodoManager()
    other = TodoManager().add("other")
    completed = manager.add("completed")
    manager.complete(completed)

    with pytest.raises(ValueError, match="not managed"):
        manager.activate(other)
    with pytest.raises(ValueError, match="already done"):
        manager.activate(completed)


def test_complete_alias_clears_active_only_for_active_root() -> None:
    manager = TodoManager()
    dependency = manager.add("dependency")
    root = manager.add("root", deps=[dependency])
    manager.activate(root)

    assert manager.complete(dependency) is dependency
    assert dependency.status == "done"
    assert manager.active() is root

    assert manager.complete(root) is root
    assert root.status == "done"
    assert manager.active() is None


def test_other_completion_paths_clear_active_root() -> None:
    manager = TodoManager()
    via_update = manager.add("updated")
    manager.activate(via_update)
    manager.update(via_update, status="done")
    assert manager.active() is None

    via_merge = manager.add("merged")
    manager.activate(via_merge)
    base = manager.copy_todo(via_merge)
    worker = base.model_copy(deep=True)
    worker.status = "done"
    manager.merge_todo(worker, base=base)
    assert manager.active() is None

    direct = manager.add("direct")
    manager.activate(direct)
    direct.status = "done"
    assert manager.active() is None


def test_removing_or_clearing_active_root_clears_selection() -> None:
    manager = TodoManager()
    removed = manager.add("removed")
    manager.activate(removed)
    assert manager.remove(removed) is True
    assert manager.active() is None

    cleared = manager.add("cleared")
    manager.activate(cleared)
    manager.clear()
    assert manager.active() is None

    done = manager.add("clear done")
    manager.activate(done)
    done.status = "done"
    assert manager.clear_done() == 1
    assert manager.active() is None


def test_active_round_trips_through_snapshot_and_ignores_invalid_ids() -> None:
    manager = TodoManager()
    root = manager.add("root")
    manager.activate(root)

    restored = TodoManager(manager.to_dict())
    assert restored.active() is not None
    assert restored.active().id == root.id

    stale_state = manager.to_dict()
    stale_state["active_id"] = "missing"
    assert TodoManager(stale_state).active() is None

    root.status = "done"
    assert manager.to_dict()["active_id"] is None


def test_active_status_bounds_long_notes_and_missing_dependencies() -> None:
    manager = TodoManager()
    root = manager.add("root", deps=["missing-" + "x" * 500], notes="n" * 500)
    manager.activate(root)

    output = manager.status(max_items=0, max_chars=200)

    assert len(output) <= 200
    assert output.endswith("… status truncated")


def test_active_status_handles_deep_graph_and_cycles_without_recursion() -> None:
    manager = TodoManager()
    dependency = manager.add("dependency 0")
    for index in range(1, 1_101):
        dependency = manager.add(f"dependency {index}", deps=[dependency])
    root = manager.add("root", deps=[dependency])
    manager.add_dep(manager.list_todos()[0], root)
    manager.activate(root)

    output = manager.status(max_items=3)

    assert output.startswith(f"Active [{root.id}] root")
    assert "+1098 dependencies not shown" in output


def test_active_dependency_order_is_dependency_first_for_shared_siblings() -> None:
    manager = TodoManager()
    shared = manager.add("shared")
    dependent = manager.add("dependent", deps=[shared])
    root = manager.add("root", deps=[dependent, shared])
    manager.activate(root)

    output = manager.status()
    rows = [line for line in output.splitlines() if line.startswith("  ") and "[" in line]

    assert [shared.id, dependent.id] == [line.split("[", 1)[1].split("]", 1)[0] for line in rows]


def test_with_todo_makes_open_delegated_todo_active() -> None:
    manager = TodoManager()
    todo = manager.add("delegated")

    worker_manager = TodoManager.with_todo(todo)

    assert worker_manager.active() is not None
    assert worker_manager.active().id == todo.id
