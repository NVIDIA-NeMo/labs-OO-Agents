# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the todo memory type: lifecycle, injection modes, forgetting guard."""

import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
)
from nooa_memory.config import EmbeddingConfig, SpontaneousConfig
from nooa_memory.descriptors import to_status
from nooa_memory.schema import Memory, _now

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def agent():
    class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
        pass

    return MemAgent()


def _install(agent, **cfg_kw):
    cfg = MemoryConfig(enabled=True, path=":memory:", **cfg_kw)
    return MemoryManager.install(agent, config=cfg)


# --------------------------------------------------------------------------
# lifecycle / schema validation
# --------------------------------------------------------------------------
def test_todo_defaults_to_open():
    m = Memory(content="update the README", type=MemoryType.TODO)
    assert m.status == "open"


def test_todo_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid todo status"):
        Memory(content="x", type=MemoryType.TODO, status="bogus")


def test_status_on_non_todo_raises():
    with pytest.raises(ValueError, match="only valid on todo"):
        Memory(content="x", type=MemoryType.INFO, status="open")


def test_to_status_verbal_roundtrip():
    assert to_status("DONE") == "done"
    assert to_status("open") == "open"
    with pytest.raises(ValueError, match="OPEN | DONE | DROPPED"):
        to_status("FINISHED")


# --------------------------------------------------------------------------
# tool surface
# --------------------------------------------------------------------------
def test_remember_todo_via_tool(agent):
    mgr = _install(agent)
    mid = agent.remember("ship the migration guide", type="todo", importance="HIGH")
    got = mgr.store.get(mid)
    assert got.type is MemoryType.TODO
    assert got.status == "open"


def test_update_memory_closes_todo(agent):
    mgr = _install(agent)
    mid = agent.remember("ship the migration guide", type="todo")
    assert agent.update_memory(mid, status="DONE") is True
    assert mgr.store.get(mid).status == "done"


def test_update_status_on_non_todo_raises(agent):
    _install(agent)
    mid = agent.remember("a plain fact", type="info")
    with pytest.raises(ValueError, match="applies only to todo"):
        agent.update_memory(mid, status="DONE")


def test_type_switch_opens_and_clears_status(agent):
    mgr = _install(agent)
    info = agent.remember("could become a commitment", type="info")
    agent.update_memory(info, type="todo")
    assert mgr.store.get(info).status == "open"
    agent.update_memory(info, type="info")
    assert mgr.store.get(info).status is None


# --------------------------------------------------------------------------
# spontaneous injection modes
#
# Retrieval returns top-k with no absolute relevance floor, so an unrelated
# todo only stays out of the associative block when other memories outrank
# it — the fillers below simulate that realistic store. dim=2048 keeps the
# hashing embedder's collision noise from randomly boosting the todo.
# --------------------------------------------------------------------------
_HIDIM = EmbeddingConfig(dim=2048)


def _fill_baking(agent):
    agent.remember("preheat the oven to 180C before baking the chocolate cake", type="info")
    agent.remember("use dark cocoa and soft butter for the chocolate cake batter", type="info")
    agent.remember("whisk the eggs with sugar until fluffy for the cake", type="info")
    agent.remember("line the cake tin with parchment paper before pouring", type="info")
    agent.remember("let the chocolate cake cool fully before slicing it", type="info")


def test_always_mode_injects_unrelated_open_todo(agent):
    mgr = _install(
        agent,
        embedding=_HIDIM,
        spontaneous=SpontaneousConfig(inject_open_todos="always"),
    )
    agent.remember("update the README after the migration lands", type="todo")
    _fill_baking(agent)  # newer + more relevant -> outrank the todo associatively
    agent.event_manager.add(Task(prompt="bake a chocolate cake"))
    text = mgr.recall_for_context()
    assert "## Open todos" in text
    assert "update the README" in text
    assert "[todo:open#" in text


def test_always_mode_drops_closed_todos(agent):
    mgr = _install(
        agent,
        embedding=_HIDIM,
        spontaneous=SpontaneousConfig(inject_open_todos="always"),
    )
    mid = agent.remember("update the README after the migration lands", type="todo")
    agent.update_memory(mid, status="DONE")
    _fill_baking(agent)
    agent.event_manager.add(Task(prompt="bake a chocolate cake"))
    text = mgr.recall_for_context()
    # A done todo is ordinary history: it may still surface associatively
    # (update touched it, boosting ACT-R frequency) — but the unconditional
    # open-todo section must be gone and nothing may render as open.
    assert "## Open todos" not in text
    assert "[todo:open#" not in text


def test_relevant_mode_skips_unrelated_todo(agent):
    mgr = _install(
        agent,
        embedding=_HIDIM,
        spontaneous=SpontaneousConfig(inject_open_todos="relevant"),
    )
    agent.remember("update the README after the migration lands", type="todo")
    _fill_baking(agent)  # newer + more relevant -> outrank the todo associatively
    agent.event_manager.add(Task(prompt="bake a chocolate cake"))
    text = mgr.recall_for_context()
    assert "cake" in text  # the block itself is present
    assert "[todo" not in text
    assert "## Open todos" not in text


def test_relevant_mode_surfaces_similar_todo(agent):
    mgr = _install(agent, spontaneous=SpontaneousConfig(inject_open_todos="relevant"))
    agent.remember("update the README after the migration lands", type="todo")
    agent.event_manager.add(Task(prompt="the migration landed, update the README"))
    text = mgr.recall_for_context()
    assert "[todo:open#" in text
    assert "## Open todos" not in text  # no unconditional section in relevant mode


def test_off_mode_excludes_matching_todo(agent):
    mgr = _install(agent, spontaneous=SpontaneousConfig(inject_open_todos="off"))
    agent.remember("deploy the new release to production", type="todo")
    agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="deploy the new release"))
    text = mgr.recall_for_context()
    assert "make ship" in text  # ordinary memories still inject
    assert "todo" not in text
    # ...but deliberate recall still reaches todos
    res = agent.recall("deploy the new release", k=3)
    assert any(m.type is MemoryType.TODO for m in res)


def test_always_mode_caps_todo_section(agent):
    mgr = _install(
        agent,
        spontaneous=SpontaneousConfig(inject_open_todos="always", open_todos_k=2),
    )
    _fill_baking(agent)
    for i in range(4):
        agent.remember(f"todo number {i} about very distinct topic {i}", type="todo")
    agent.event_manager.add(Task(prompt="bake a chocolate cake"))
    text = mgr.recall_for_context()
    assert text.count("[todo:open#") == 2


# --------------------------------------------------------------------------
# forgetting guard
# --------------------------------------------------------------------------
def test_open_todo_is_never_pruned(agent):
    mgr = _install(agent)
    old = _now() - 10_000 * 3600  # ancient
    m = Memory(
        content="still not done",
        type=MemoryType.TODO,
        importance=2.0,
        created_at=old,
        last_accessed_at=old,
        access_log=[old],
    )
    assert mgr.forgetting.is_protected(m) is True
    assert mgr.forgetting.should_prune(m) is False


def test_done_todo_decays_normally(agent):
    mgr = _install(agent)
    old = _now() - 10_000 * 3600
    m = Memory(
        content="finished ages ago",
        type=MemoryType.TODO,
        status="done",
        importance=2.0,
        created_at=old,
        last_accessed_at=old,
        access_log=[old],
    )
    assert mgr.forgetting.is_protected(m) is False
    assert mgr.forgetting.should_prune(m) is True
