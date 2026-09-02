# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the terminal memory explorer."""

from __future__ import annotations

from nooa_cli.tui.memory_explorer import (
    MemoryExplorerView,
    build_memory_rows,
    relative_age,
)
from nooa_memory import MemoryConfig, MemoryManager, MemoryType
from nooa_memory.config import ReflectionPolicy, SpontaneousConfig, WritePolicy

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


def _seeded_manager():
    """A real MemoryManager on a bare agent with an in-memory store."""
    llm = FakeLLMClient()

    class MemAgent(Agent, llm=llm):
        pass

    agent = MemAgent()
    cfg = MemoryConfig(
        enabled=True,
        path=":memory:",
        owner="tester",
        instruct=False,
        spontaneous=SpontaneousConfig(enabled=False),
        write=WritePolicy(on_events=(), write_episodic=False),
        reflection=ReflectionPolicy(enabled=False),
    )
    mgr = MemoryManager.install(agent, config=cfg)
    skill_id = mgr.remember(
        "Deploys go through `make ship`.",
        type=MemoryType.SKILL,
        title="Deploy procedure",
        tags=["deploy", "ship"],
    )
    todo_id = mgr.remember("Ship the 1.4 release notes", type=MemoryType.TODO)
    ref_id = mgr.remember(
        "The deploy skill is documented in another memory.",
        references=[f"memory:{skill_id}"],
    )
    mgr.associate(ref_id, skill_id, "derived_from")
    return agent, mgr, {"skill": skill_id, "todo": todo_id, "ref": ref_id}


def _view(agent, mgr):
    rows = build_memory_rows(agent, mgr)
    calls: dict[str, list[str]] = {"forgot": [], "done": []}

    def _forget(memory_id: str) -> None:
        calls["forgot"].append(memory_id)
        mgr.forget(memory_id)

    def _mark_done(memory_id: str) -> None:
        calls["done"].append(memory_id)
        mgr.update(memory_id, status="done")

    return MemoryExplorerView(rows, forget=_forget, mark_done=_mark_done), calls


def test_build_rows_snapshots_store_usage_references_and_edges() -> None:
    agent, mgr, ids = _seeded_manager()

    rows = build_memory_rows(agent, mgr)

    assert {r.id for r in rows} == set(ids.values())
    todo = next(r for r in rows if r.id == ids["todo"])
    assert todo.type_tag == "todo:open"
    assert todo.importance == "MEDIUM"
    for expected in ("todo:open", "Ship the 1.4 release notes", "tester"):
        assert expected in todo.search_text

    skill = next(r for r in rows if r.id == ids["skill"])
    assert "deploy" in skill.search_text and "Deploy procedure" in skill.search_text
    assert skill.usage["fetches"] == 0

    ref = next(r for r in rows if r.id == ids["ref"])
    assert len(ref.reference_lines) == 1
    assert "(LIVE)" in ref.reference_lines[0]
    assert ref.edges == [(ids["skill"], "derived_from", 1.0)]


def test_search_filters_on_type_tags_owner_and_content() -> None:
    agent, mgr, ids = _seeded_manager()
    view, _calls = _view(agent, mgr)

    view.model.set_query("todo:open")
    assert [view.model.rows[i].id for i in view.model.matches] == [ids["todo"]]
    view.model.set_query("deploy")
    assert {view.model.rows[i].id for i in view.model.matches} == {ids["skill"], ids["ref"]}
    view.model.set_query("tester")
    assert len(view.model.matches) == 3


def test_action_d_marks_todo_done_round_trip() -> None:
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)

    todo = next(r for r in view.model.rows if r.id == ids["todo"])
    # Route through the real key path: "d" arrives as a text key.
    view.model.cursor = view.model.matches.index(view.model.rows.index(todo))
    # Destructive actions arm on the first press and fire on the second.
    assert view.handle_key("text", "d") == "handled"
    assert calls["done"] == []
    assert view.handle_key("text", "d") == "handled"

    assert calls["done"] == [ids["todo"]]
    assert mgr.store.get(ids["todo"]).status == "done"
    assert todo.status == "done"
    assert "todo:done" in view.format_row(todo, 120)
    assert "todo:done" in todo.search_text

    # d again (already done) and d on a non-todo row are ignored.
    assert view.handle_action("text:d", todo) == "ignored"
    skill = next(r for r in view.model.rows if r.id == ids["skill"])
    assert view.handle_action("text:d", skill) == "ignored"
    assert calls["done"] == [ids["todo"]]


def test_invalid_action_disarms_the_pending_confirm() -> None:
    """An action key that cannot apply still interrupts the armed gesture.

    Press f (arms), press d on a non-todo row (ignored), press f again: the
    final press must re-arm, not fire.
    """
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)
    skill = next(r for r in view.model.rows if r.id == ids["skill"])

    assert view.handle_action("text:f", skill) == "handled"  # arm
    assert view.handle_action("text:d", skill) == "ignored"  # invalid action
    assert calls["forgot"] == []
    assert view.handle_action("text:f", skill) == "handled"  # re-arm, not fire
    assert calls["forgot"] == []
    assert view.handle_action("text:f", skill) == "handled"  # now fires
    assert calls["forgot"] == [ids["skill"]]


def test_stale_arm_expires_instead_of_firing(monkeypatch) -> None:
    """An arm older than the gesture window re-arms, never fires.

    The docstring always promised a gesture window; without one, an f armed
    minutes earlier still fired on the next f.
    """
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)
    skill = next(r for r in view.model.rows if r.id == ids["skill"])

    # Arm, then shrink the window to zero: the stale arm does not fire,
    # it re-arms fresh.
    assert view.handle_action("text:f", skill) == "handled"
    monkeypatch.setattr(view, "_CONFIRM_WINDOW_SECONDS", 0.0)
    assert view.handle_action("text:f", skill) == "handled"
    assert calls["forgot"] == []

    # A prompt second press (fresh window restored) fires.
    monkeypatch.setattr(view, "_CONFIRM_WINDOW_SECONDS", 10.0)
    assert view.handle_action("text:f", skill) == "handled"
    assert calls["forgot"] == [ids["skill"]]

    # The boundary is inclusive: an arm exactly at the window edge still
    # confirms.
    todo = next(r for r in view.model.rows if r.id == ids["todo"])
    assert view.handle_action("text:d", todo) == "handled"
    monkeypatch.setattr(view, "_CONFIRM_WINDOW_SECONDS", 10.0)
    assert view.handle_action("text:d", todo) == "handled"
    assert calls["done"] == [ids["todo"]]


def test_armed_confirm_does_not_leak_across_rows() -> None:
    """Arming on row A must never fire on row B.

    The arm is keyed to (action, id(row)); moving to another row and
    pressing the same key re-arms the new row instead of firing.
    """
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)
    rows = list(view.model.rows)

    first = rows[0]
    second = rows[1]
    # Arm on the first row.
    assert view.handle_action("text:f", first) == "handled"
    assert calls["forgot"] == []
    # Press f on a different row: re-arms that row, never fires on either.
    assert view.handle_action("text:f", second) == "handled"
    assert calls["forgot"] == []
    # A second press on the second row fires exactly there.
    assert view.handle_action("text:f", second) == "handled"
    assert calls["forgot"] == [second.id]
    assert first.id in {r.id for r in view.model.rows}


def test_destructive_action_disarms_on_any_other_key() -> None:
    """A first press only arms; a different key must disarm, never fire.

    Regression for the unconfirmed destructive fire: one 'f' press used to
    permanently forget a memory with no confirmation.
    """
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)

    skill = next(r for r in view.model.rows if r.id == ids["skill"])
    view.model.cursor = view.model.matches.index(view.model.rows.index(skill))
    assert view.handle_key("text", "f") == "handled"
    assert calls["forgot"] == []
    assert "again to forget" in view.pending_confirmation_hint()

    # A different key disarms instead of firing.
    assert view.handle_key("text", "x") == "ignored"
    assert view.handle_key("text", "f") == "handled"  # re-arms
    assert calls["forgot"] == []
    assert view.handle_key("text", "f") == "handled"  # now fires
    assert calls["forgot"] == [ids["skill"]]


def test_action_f_forgets_memory_round_trip() -> None:
    agent, mgr, ids = _seeded_manager()
    view, calls = _view(agent, mgr)

    skill = next(r for r in view.model.rows if r.id == ids["skill"])
    view.model.cursor = view.model.matches.index(view.model.rows.index(skill))
    assert view.handle_key("text", "f") == "handled"
    assert calls["forgot"] == []
    assert view.handle_key("text", "f") == "handled"

    assert calls["forgot"] == [ids["skill"]]
    assert len(view.model.rows) == 2
    assert all(r.id != ids["skill"] for r in view.model.rows)
    assert len(view.model.matches) == 2
    # Archived in the store: gone from the active listing.
    assert {m.id for m in mgr.store.all_memories()} == {ids["todo"], ids["ref"]}


def test_relative_age_bands() -> None:
    now = 1_000_000.0
    assert relative_age(None) == "never"
    assert relative_age(now - 5, now) == "5s"
    assert relative_age(now - 300, now) == "5m"
    assert relative_age(now - 7200, now) == "2h"
    assert relative_age(now - 3 * 86400, now) == "3d"


def test_last_reflection_summary_lines():
    """Header line renders completed and interrupted maintenance rows."""
    from nooa_cli.tui.memory_explorer import last_reflection_summary

    _agent, mgr, _ids = _seeded_manager()
    assert last_reflection_summary(mgr) is None  # nothing yet

    mgr.store.log_maintenance(
        "reflect",
        {
            "trigger": "idle",
            "interrupted": False,
            "duration_ms": 1400.0,
            "merged": 3,
            "edges_added": 5,
            "pruned": 1,
        },
    )
    line = last_reflection_summary(mgr)
    assert "merged 3" in line and "+5 edges" in line and "pruned 1" in line
    assert "(idle, 1.4s)" in line

    mgr.store.log_maintenance(
        "reflect",
        {
            "trigger": "idle",
            "interrupted": True,
            "stopped_in": "form_edges",
            "duration_ms": 300.0,
            "merged": 0,
            "edges_added": 0,
            "pruned": 0,
        },
    )
    line = last_reflection_summary(mgr)
    assert "interrupted @ form_edges" in line


def test_view_title_carries_reflection_line():
    agent, mgr, _ids = _seeded_manager()
    rows = build_memory_rows(agent, mgr)
    view = MemoryExplorerView(
        rows,
        forget=lambda _id: None,
        mark_done=lambda _id: None,
        last_reflection="last reflection: 2m ago — merged 3, +5 edges, pruned 1 (idle, 1.4s)",
    )
    assert "last reflection: 2m ago" in view.config.title


def test_action_d_on_dropped_todo_keeps_search_text_consistent() -> None:
    agent, mgr, ids = _seeded_manager()
    mgr.update(ids["todo"], status="dropped")
    view, calls = _view(agent, mgr)

    todo = next(r for r in view.model.rows if r.id == ids["todo"])
    assert "todo:dropped" in todo.search_text  # row built from the dropped state
    # Destructive actions arm on the first press and fire on the second.
    assert view.handle_action("text:d", todo) == "handled"
    assert calls["done"] == []
    assert view.handle_action("text:d", todo) == "handled"

    assert calls["done"] == [ids["todo"]]
    assert mgr.store.get(ids["todo"]).status == "done"
    assert todo.status == "done"
    # The search tag follows the transition: a "done" filter finds the row,
    # and the stale dropped tag is gone.
    assert "todo:done" in todo.search_text
    assert "todo:dropped" not in todo.search_text
