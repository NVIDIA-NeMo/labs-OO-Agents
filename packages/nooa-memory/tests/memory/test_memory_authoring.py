# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-authored memory: schema instruction injection + write/refine tools."""

import pytest
from nooa_memory import MemoryConfig, MemoryManager, MemoryToolsMixin, MemoryType

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def agent():
    class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
        pass

    return MemAgent()


def _install(agent, **kw):
    return MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:", **kw))


# --- instruction injection: the agent is TOLD it owns/curates its memory ---
def test_install_injects_schema_instruction(agent):
    _install(agent)
    key = "memory_system"
    assert key in agent.context_manager
    guide = agent.context_manager[key]
    assert "own" in guide.lower() and "curate" in guide.lower()
    # schema types are spelled out for the agent
    for t in ("info", "skill", "episode", "intent", "reflection"):
        assert t in guide


def test_instruct_false_skips_injection(agent):
    _install(agent, instruct=False)
    assert "memory_system" not in agent.context_manager


def test_uninstall_removes_instruction(agent):
    mgr = _install(agent)
    assert "memory_system" in agent.context_manager
    mgr.uninstall()
    assert "memory_system" not in agent.context_manager


# --- write tool honours the schema ---
def test_remember_with_schema_fields(agent):
    mgr = _install(agent)
    mid = agent.remember(
        "deploy with make ship", type="skill", importance="HIGH", tags=["deploy", "ci"]
    )
    m = mgr.store.get(mid)
    assert m.type == MemoryType.SKILL
    assert m.importance == 8.0  # HIGH -> 8.0 (verbal ladder)
    assert "deploy" in m.tags


def test_remember_bad_type_raises(agent):
    _install(agent)
    with pytest.raises(ValueError):  # no silent fallback to INFO
        agent.remember("a fact", type="not-a-type")


def test_remember_bad_importance_label_raises(agent):
    _install(agent)
    with pytest.raises(ValueError):
        agent.remember("a fact", importance="VERY-HIGH")  # not a band


# --- refine tools ---
def test_update_memory_changes_content_and_is_retrievable(agent):
    mgr = _install(agent)
    mid = agent.remember("the staging db is named oldname", type="info")
    assert agent.update_memory(
        mid, content="the staging db is named lighthouse_stg", importance="CRITICAL"
    )
    m = mgr.store.get(mid)
    assert "lighthouse_stg" in m.content and m.importance == 10.0  # CRITICAL -> 10.0
    # re-embedded -> new content retrievable
    hits = agent.recall("staging database name", k=1)
    assert hits and "lighthouse_stg" in hits[0].content


def test_update_missing_returns_false(agent):
    _install(agent)
    assert agent.update_memory("nope", content="x") is False


def test_forget_archives_memory(agent):
    mgr = _install(agent)
    mid = agent.remember("obsolete fact", type="info")
    assert mgr.store.count() == 1
    assert agent.forget(mid) is True
    assert mgr.store.count() == 0
    assert mgr.store.get(mid).archived is True
    assert mgr.stats.pruned == 1


def test_default_tools_include_write_and_refine(agent):
    mgr = _install(agent)
    for t in ("remember", "update_memory", "forget", "associate", "recall", "search"):
        assert t in mgr.config.tools


def test_disabled_refine_tools_raise(agent):
    _install(agent, tools=("recall",))
    with pytest.raises(RuntimeError, match="disabled"):
        agent.update_memory("x", content="y")
    with pytest.raises(RuntimeError, match="disabled"):
        agent.forget("x")
