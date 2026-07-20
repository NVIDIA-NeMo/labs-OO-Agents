# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MemoryManager + MemoryToolsMixin on a real Agent.

Uses ``FakeLLMClient`` so no network/LLM is needed — the hooks, store and
consolidation are exercised directly.
"""

from pathlib import Path

import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
)
from nooa_memory.config import ReflectionPolicy, SpontaneousConfig, WritePolicy

from nooa import Agent
from nooa.events import Error, Task
from nooa.runtime.middleware import AgentCallContext
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def agent(fake_llm):
    class MemAgent(MemoryToolsMixin, Agent, llm=fake_llm):
        pass

    return MemAgent()


def _install(agent, **cfg_kw):
    cfg = MemoryConfig(enabled=True, path=":memory:", **cfg_kw)
    return MemoryManager.install(agent, config=cfg)


# --------------------------------------------------------------------------
# install / uninstall / additive guarantee
# --------------------------------------------------------------------------
def test_install_wires_hooks_and_stores_manager(agent):
    mgr = _install(agent)
    assert agent._memory is mgr
    assert len(mgr._unsubs) >= 1
    assert len(agent.event_manager._middleware["agent_call"]) == 1


def test_uninstall_removes_all_hooks(agent):
    mgr = _install(agent)
    mgr.uninstall()
    assert mgr._unsubs == []
    assert len(agent.event_manager._middleware["agent_call"]) == 0
    # handler lists no longer reference our callbacks
    for handlers in agent.event_manager._handlers.values():
        for h in handlers:
            assert getattr(h, "__self__", None) is not mgr


def test_disabled_install_is_inert(agent):
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=False, path=":memory:"))
    assert mgr._unsubs == []
    assert len(agent.event_manager._middleware["agent_call"]) == 0


def test_install_replaces_existing_manager_without_leaking_hooks(agent):
    first = _install(agent)
    second = _install(agent)

    assert agent._memory is second
    assert first._unsubs == []
    assert len(agent.event_manager._middleware["agent_call"]) == 1
    for handlers in agent.event_manager._handlers.values():
        for h in handlers:
            assert getattr(h, "__self__", None) is not first


def test_uninstall_removes_agent_memory_reference(agent):
    mgr = _install(agent)
    mgr.uninstall()
    assert not hasattr(agent, "_memory")


def test_pathless_memory_uses_legacy_memory_db_not_agent_storage(tmp_path, fake_llm, monkeypatch):
    class StoredAgent(MemoryToolsMixin, Agent, llm=fake_llm):
        pass

    monkeypatch.chdir(tmp_path)
    storage = SQLiteStorageManager(tmp_path / "agent.sqlite")
    agent = StoredAgent(storage=storage)
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True))

    agent.remember("deploy uses make ship", type="skill", importance="HIGH")
    assert mgr.store.path == str(Path(".nooa") / "memory" / "memory.sqlite")
    assert mgr.store.count() == 1
    with pytest.raises(Exception, match="no such table: memories"):
        storage.event_backend._conn.execute("SELECT count(*) FROM memories").fetchone()


def test_associate_unknown_relation_raises(agent):
    mgr = _install(agent)
    a = agent.remember("a", type="info")
    b = agent.remember("b", type="info")
    with pytest.raises(ValueError):
        agent.associate(a, b, relation="nearby-ish")  # no silent RELATED fallback
    assert mgr.store.neighbors(a) == []


def test_tools_raise_when_not_installed(fake_llm):
    class Bare(MemoryToolsMixin, Agent, llm=fake_llm):
        pass

    a = Bare()
    with pytest.raises(RuntimeError, match="not installed"):
        a.remember("x")


def test_tools_raise_when_disabled(agent):
    MemoryManager.install(agent, config=MemoryConfig(enabled=False, path=":memory:"))
    with pytest.raises(RuntimeError):
        agent.recall("x")


def test_individual_tool_can_be_disabled(agent):
    _install(agent, tools=("recall", "search"))  # remember/associate disabled
    with pytest.raises(RuntimeError, match="disabled"):
        agent.remember("x")
    # enabled tool still works
    assert agent.recall("nothing yet") == []


# --------------------------------------------------------------------------
# conscious tools + dedup-on-write
# --------------------------------------------------------------------------
def test_remember_and_recall_via_tools(agent):
    _install(agent)
    agent.remember("the deploy command is make ship", type="info")
    res = agent.recall("how do I deploy", k=1)
    assert res and "make ship" in res[0].content


def test_remember_dedups_on_write(agent):
    mgr = _install(agent)
    id1 = agent.remember("identical fact about shipping releases", type="info")
    id2 = agent.remember("identical fact about shipping releases", type="info")
    assert id1 == id2  # NOOP: reinforced the existing memory
    assert mgr.store.count() == 1
    assert mgr.store.get(id1).reinforcement_count >= 1


# --------------------------------------------------------------------------
# spontaneous association (pre-turn injection)
# --------------------------------------------------------------------------
def test_recall_for_context_after_task(agent):
    mgr = _install(agent)
    agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="please deploy the service"))
    text = mgr.recall_for_context()
    assert "make ship" in text


def test_before_turn_sets_context_block(agent):
    mgr = _install(agent)
    agent.remember("rollback uses make undeploy", type="skill")
    agent.event_manager.add(Task(prompt="we need to rollback the release"))
    mgr._on_before_turn(None)
    key = mgr.config.spontaneous.context_block_key
    assert key in agent.context_manager
    assert "undeploy" in agent.context_manager[key]


def test_injection_respects_char_budget(agent):
    mgr = _install(agent, spontaneous=SpontaneousConfig(context_char_budget=40))
    for i in range(5):
        agent.remember(f"a fairly long memory about deploying service number {i}", type="info")
    agent.event_manager.add(Task(prompt="deploy the service please"))
    text = mgr.recall_for_context()
    assert len(text) <= 40 + 5  # budget + ellipsis slack


def test_spontaneous_disabled_returns_empty(agent):
    mgr = _install(agent, spontaneous=SpontaneousConfig(enabled=False))
    agent.remember("something", type="info")
    agent.event_manager.add(Task(prompt="anything"))
    assert mgr.recall_for_context() == ""


def test_self_gated_cadence_skips_unchanged_query(agent):
    mgr = _install(agent, spontaneous=SpontaneousConfig(inject_cadence="self_gated"))
    agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="deploy please"))
    mgr._on_before_turn(None)
    h1 = mgr._last_query_hash
    mgr._on_before_turn(None)  # same query -> early-return, hash unchanged
    assert mgr._last_query_hash == h1


# --------------------------------------------------------------------------
# write-on-event
# --------------------------------------------------------------------------
def test_error_event_is_written(agent):
    mgr = _install(agent, write=WritePolicy(on_events=("Error",)))
    agent.event_manager.add(Error(content="Traceback: connection refused to database"))
    assert mgr.store.count() == 1
    mems = mgr.store.all_memories()
    assert "connection refused" in mems[0].content


def test_salience_gate_blocks_low_salience(agent):
    # Error salience is 0.9; gate at 0.95 blocks it.
    mgr = _install(agent, write=WritePolicy(on_events=("Error",), salience_min=0.95))
    agent.event_manager.add(Error(content="some error"))
    assert mgr.store.count() == 0


# --------------------------------------------------------------------------
# post-task reflection via middleware
# --------------------------------------------------------------------------
async def test_reflect_middleware_writes_episode_top_level(agent):
    mgr = _install(agent, reflection=ReflectionPolicy(only_top_level=True))

    async def inner(ctx):
        ctx.result = "shipped"
        return ctx

    ctx = AgentCallContext(agent=agent, method_name="work")
    await mgr._reflect_middleware(ctx, inner)
    episodes = [m for m in mgr.store.all_memories() if m.type == MemoryType.EPISODE]
    assert len(episodes) == 1
    assert "work" in episodes[0].content


async def test_nested_calls_do_not_each_reflect(agent):
    mgr = _install(agent, reflection=ReflectionPolicy(only_top_level=True))

    async def leaf(ctx):
        ctx.result = "leaf"
        return ctx

    async def outer(ctx):
        # a nested agent_call happens inside the top-level call
        nested = AgentCallContext(agent=agent, method_name="child")
        await mgr._reflect_middleware(nested, leaf)
        ctx.result = "parent"
        return ctx

    top = AgentCallContext(agent=agent, method_name="parent")
    await mgr._reflect_middleware(top, outer)

    episodes = [m for m in mgr.store.all_memories() if m.type == MemoryType.EPISODE]
    # only the top-level call consolidated -> exactly one episode
    assert len(episodes) == 1
    assert "parent" in episodes[0].content


async def test_manual_reflect_trigger_does_not_register_middleware(agent):
    mgr = _install(agent, reflection=ReflectionPolicy(trigger="manual"))
    assert len(agent.event_manager._middleware["agent_call"]) == 0
    # manual reflect still runs
    agent.remember("a", type="info")
    report = mgr.reflect()
    assert report is not None
