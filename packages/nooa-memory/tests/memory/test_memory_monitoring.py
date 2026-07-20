# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-usage monitoring: counters, events, and logging."""

import logging

import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryRecalled,
    MemoryToolsMixin,
    MemoryWritten,
    ReflectionCompleted,
)

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def agent():
    class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
        pass

    return MemAgent()


def _install(agent, **kw):
    return MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:", **kw))


def test_write_increments_counter_and_emits_event(agent):
    mgr = _install(agent)
    seen = []
    agent.event_manager.on("MemoryWritten", seen.append)
    agent.remember("a brand new fact about deploys", type="info")
    assert mgr.stats.writes == 1
    assert len(seen) == 1 and isinstance(seen[0], MemoryWritten)
    assert seen[0].op == "add"


def test_reinforce_counter(agent):
    mgr = _install(agent)
    agent.remember("identical reinforced fact", type="info")
    agent.remember("identical reinforced fact", type="info")
    assert mgr.stats.writes == 1
    assert mgr.stats.reinforced == 1


def test_recall_counter_and_event(agent):
    mgr = _install(agent)
    agent.remember("deploy with make ship", type="info")
    events = []
    agent.event_manager.on("MemoryRecalled", events.append)
    res = agent.recall("how to deploy", k=1)
    assert mgr.stats.recalls == 1
    assert mgr.stats.recalled_items == len(res) == 1
    assert events and isinstance(events[0], MemoryRecalled)


def test_reflection_counter_and_event(agent):
    mgr = _install(agent)
    agent.remember("x fact one", type="info")
    events = []
    agent.event_manager.on("ReflectionCompleted", events.append)
    mgr.reflect()
    assert mgr.stats.reflections == 1
    assert events and isinstance(events[0], ReflectionCompleted)


def test_injection_counters(agent):
    from nooa.events import Task

    mgr = _install(agent)
    agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="please deploy now"))
    mgr._on_before_turn(None)
    assert mgr.stats.injections == 1
    assert mgr.stats.injected_chars > 0


def test_memory_stats_snapshot_and_summary(agent):
    mgr = _install(agent)
    agent.remember("one", type="info")
    agent.remember("two distinct", type="info")
    stats = mgr.memory_stats()
    assert stats.store_size == 2
    assert "writes=2" in stats.summary()


def test_runtime_events_stay_out_of_llm_context(agent):
    """MemoryWritten uses RUNTIME_EVENT role, so it must not be recorded as context."""
    mgr = _install(agent)
    before = list(agent.event_manager.keys())
    agent.remember("a fact", type="info")
    after = list(agent.event_manager.keys())
    # No new *recorded* event tags from the MemoryWritten emission.
    assert after == before
    assert mgr.stats.writes == 1


def test_log_summary_emits_info(agent, caplog):
    mgr = _install(agent)
    agent.remember("loggable fact", type="info")
    with caplog.at_level(logging.INFO, logger="nooa_memory"):
        mgr.log_summary()
    assert any("memory stats" in r.message for r in caplog.records)
