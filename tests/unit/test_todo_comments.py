# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``TodoManager.comment`` / ``comments`` — progress journalling.

Comments are the canonical way an SWE skill (brainstorm / root-cause / tdd /
review / ship) logs what happened at each step so the next turn — or a
different skill — can read the history without re-deriving it.
"""

import pytest

from nooa.tools.todo import TodoComment, TodoManager


def test_comment_returns_stored_instance():
    tm = TodoManager()
    t = tm.add("Fix bug")
    c = tm.comment(t.id, "root cause: races on refresh")
    assert isinstance(c, TodoComment)
    assert c.body == "root cause: races on refresh"
    assert c.created_at  # non-empty timestamp


def test_comment_on_missing_todo_returns_none():
    tm = TodoManager()
    assert tm.comment("does-not-exist", "hello") is None


def test_comments_returns_empty_list_for_new_todo():
    tm = TodoManager()
    t = tm.add("New")
    assert tm.comments(t.id) == []


def test_comments_are_chronological_append_only():
    tm = TodoManager()
    t = tm.add("Multi-step work")
    tm.comment(t.id, "first")
    tm.comment(t.id, "second")
    tm.comment(t.id, "third")
    bodies = [c.body for c in tm.comments(t.id)]
    assert bodies == ["first", "second", "third"]


def test_comments_returns_empty_for_missing_todo():
    tm = TodoManager()
    assert tm.comments("nope") == []


def test_comments_survive_snapshot_round_trip():
    """Snapshot → restore must preserve the comment log verbatim."""
    tm = TodoManager()
    t = tm.add("With history")
    tm.comment(t.id, "one")
    tm.comment(t.id, "two")

    state = tm.to_dict()
    restored = TodoManager()
    restored.from_dict(state)

    preserved = restored.comments(t.id)
    assert [c.body for c in preserved] == ["one", "two"]


def test_comment_does_not_overwrite_notes():
    """Comments and ``notes`` are independent fields — append to one doesn't
    touch the other."""
    tm = TodoManager()
    t = tm.add("Item", notes="static note")
    tm.comment(t.id, "progress log entry")
    assert t.notes == "static note"
    assert [c.body for c in tm.comments(t.id)] == ["progress log entry"]


def test_multiple_todos_keep_comments_separate():
    tm = TodoManager()
    a = tm.add("A")
    b = tm.add("B")
    tm.comment(a.id, "about A")
    tm.comment(b.id, "about B")
    assert [c.body for c in tm.comments(a.id)] == ["about A"]
    assert [c.body for c in tm.comments(b.id)] == ["about B"]


@pytest.mark.parametrize("body", ["", "just some text", "multi\nline\nbody", "🔍 emoji"])
def test_comment_accepts_arbitrary_body(body):
    tm = TodoManager()
    t = tm.add("T")
    c = tm.comment(t.id, body)
    assert c is not None
    assert c.body == body


def test_clear_wipes_all_todos_and_order() -> None:
    """``TodoManager.clear()`` removes every todo and the insertion order."""
    tm = TodoManager()
    tm.add("first")
    tm.add("second")
    tm.add("third")
    assert len(tm._todos) == 3
    assert len(tm._order) == 3

    tm.clear()

    assert tm._todos == {}
    assert tm._order == []


def test_clear_then_add_starts_fresh() -> None:
    """After ``clear()``, new todos are added starting from an empty state —
    no id collisions with pre-clear todos (each ``add`` mints a fresh id)."""
    tm = TodoManager()
    old = tm.add("old")
    tm.clear()
    new = tm.add("new")

    assert new.id != old.id
    assert list(tm._todos.keys()) == [new.id]


def test_manager_methods_accept_todo_objects() -> None:
    tm = TodoManager()
    first = tm.add("first")
    second = tm.add("second", deps=[first])

    assert tm.get(first) is first
    assert tm.get(first.id) is first
    assert second.deps == [first.id]
    assert tm.add_dep(first, second) is first
    assert first.deps == [second.id]
    assert tm.remove_dep(first, second) is first
    assert first.deps == []
    assert tm.update(first, title="renamed", notes="detail") is first
    assert (first.title, first.notes) == ("renamed", "detail")
    assert tm.set_var(first, "answer", 42) is first
    assert tm.get_var(first, "answer") == 42
    assert tm.del_var(first, "answer") is first
    assert tm.get_var(first, "answer") is None
    assert tm.comment(first, "finding") is not None
    assert [comment.body for comment in tm.comments(first)] == ["finding"]
    assert tm.done(first) is first
    assert first.status == "done"
    assert tm.reopen(first) is first
    assert first.status == "open"
    assert tm.remove(second) is True
    assert tm.get(second) is None


def test_manager_rejects_non_todo_non_string_references() -> None:
    tm = TodoManager()

    with pytest.raises(TypeError, match="expected Todo or str"):
        tm.get(42)  # type: ignore[arg-type]


def test_delegation_copy_is_independent_and_merge_preserves_identity() -> None:
    tm = TodoManager()
    original = tm.add("review", notes="start", nested={"values": [1]})
    tm.comment(original, "controller baseline")
    base = tm._copy_todo(original)
    worker = base.model_copy(deep=True)

    worker.notes = "worker notes"
    worker.v.nested["values"].append(2)
    worker.v.result = {"file": "parser.py"}
    worker.comments.append(TodoComment(body="worker finding"))

    merged = tm._merge_todo(worker, base=base)

    assert merged is original
    assert original.notes == "worker notes"
    assert original.v.nested == {"values": [1, 2]}
    assert original.v.result == {"file": "parser.py"}
    assert [comment.body for comment in original.comments] == [
        "controller baseline",
        "worker finding",
    ]


def test_delegation_merge_preserves_unrelated_controller_changes() -> None:
    tm = TodoManager()
    original = tm.add("review", controller="initial")
    base = tm._copy_todo(original)
    worker = base.model_copy(deep=True)

    original.v.controller = "new"
    worker.v.worker = "finding"
    tm.comment(original, "controller note")
    worker.comments.append(TodoComment(body="worker note"))

    tm._merge_todo(worker, base=base)

    assert original.v.controller == "new"
    assert original.v.worker == "finding"
    assert [comment.body for comment in original.comments] == [
        "controller note",
        "worker note",
    ]


def test_delegation_merge_rejects_conflicting_field_changes() -> None:
    tm = TodoManager()
    original = tm.add("review")
    base = tm._copy_todo(original)
    worker = base.model_copy(deep=True)
    original.notes = "controller notes"
    worker.notes = "worker notes"

    with pytest.raises(ValueError, match="conflicting 'notes' changes"):
        tm._merge_todo(worker, base=base)


def test_manager_preserves_id_keyword_compatibility() -> None:
    tm = TodoManager()
    first = tm.add("first")
    second = tm.add("second")

    assert tm.get(todo_id=first.id) is first
    assert tm.update(todo_id=first.id, notes="note") is first
    assert tm.add_dep(todo_id=first.id, dep_id=second.id) is first
    assert tm.remove_dep(todo_id=first.id, dep_id=second.id) is first
    assert tm.set_var(todo_id=first.id, key="value", value=1) is first
    assert tm.get_var(todo_id=first.id, key="value") == 1
    assert tm.del_var(todo_id=first.id, key="value") is first
    assert tm.comment(todo_id=first.id, body="finding") is not None
    assert len(tm.comments(todo_id=first.id)) == 1
    assert tm.done(todo_id=first.id) is first
    assert tm.reopen(todo_id=first.id) is first
    assert tm.remove(todo_id=second.id) is True


def test_delegation_merge_conflict_is_atomic() -> None:
    tm = TodoManager()
    original = tm.add("review", notes="base")
    base = tm._copy_todo(original)
    worker = base.model_copy(deep=True)
    worker.title = "worker title"
    worker.notes = "worker notes"
    worker.v.finding = "worker value"
    worker.comments.append(TodoComment(body="worker comment"))
    original.notes = "controller notes"

    with pytest.raises(ValueError, match="conflicting 'notes' changes"):
        tm._merge_todo(worker, base=base)

    assert original.title == "review"
    assert original.notes == "controller notes"
    assert "finding" not in original.v
    assert original.comments == []


def test_delegation_merge_keeps_equal_concurrent_comments() -> None:
    tm = TodoManager()
    original = tm.add("review")
    base = tm._copy_todo(original)
    worker = base.model_copy(deep=True)
    controller_comment = TodoComment(body="same", created_at="2026-01-01 00:00")
    worker_comment = TodoComment(body="same", created_at="2026-01-01 00:00")
    original.comments.append(controller_comment)
    worker.comments.append(worker_comment)

    tm._merge_todo(worker, base=base)

    assert [comment.id for comment in original.comments] == [
        controller_comment.id,
        worker_comment.id,
    ]
