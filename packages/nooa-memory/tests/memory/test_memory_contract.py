# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent-facing contract: the injected guide matches the host's real API,
recalled lines carry usable ids, and the tool boundary raises instead of
silently recovering."""

import re

import pytest
from nooa_memory import (
    Memory,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
)
from nooa_memory.memory_skill import MemorySkill

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


class BareAgent(Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def agent():
    return MemAgent()


def _install(agent, **cfg_kw):
    cfg = MemoryConfig(enabled=True, path=":memory:", **cfg_kw)
    return MemoryManager.install(agent, config=cfg)


# --------------------------------------------------------------------------
# the guide documents the API the host actually has
# --------------------------------------------------------------------------
def test_guide_uses_direct_prefix_for_mixin_install(agent):
    mgr = _install(agent)
    guide = str(agent.context_manager[mgr.config.instruct_block_key])
    assert "self.remember(" in guide
    assert "self.memory.remember(" not in guide


def test_guide_uses_memory_prefix_for_skill_install():
    agent = BareAgent()
    skill = MemorySkill(MemoryConfig(enabled=True, path=":memory:"))
    skill.attach(agent)
    guide = str(agent.context_manager[skill._config.instruct_block_key])
    assert "self.memory.remember(" in guide
    assert "`self.remember(" not in guide
    skill.detach()
    assert skill._config.instruct_block_key not in agent.context_manager


def test_guide_states_store_and_identity(agent):
    mgr = _install(agent)
    guide = str(agent.context_manager[mgr.config.instruct_block_key])
    assert "Store: :memory:" in guide or "Store: (in-memory)" in guide
    assert "your identity: MemAgent" in guide


def test_guide_documents_relations_dedup_and_autowrites(agent):
    mgr = _install(agent)
    guide = str(agent.context_manager[mgr.config.instruct_block_key])
    assert "contradicts" in guide and "derived_from" in guide  # relation vocabulary
    assert "near-duplicate write reinforces" in guide.lower() or "reinforces the existing" in guide
    assert "auto-writes" in guide  # operational writes are not hidden


# Every `self.foo(` / `self.memory.foo(` call the guide shows must exist on
# the surface it was rendered for — documentation that lies fails here.
_GUIDE_CALL_RE = re.compile(r"`self\.(?:memory\.)?(\w+)\(")


def test_guide_methods_exist_on_mixin_host(agent):
    mgr = _install(agent)
    guide = str(agent.context_manager[mgr.config.instruct_block_key])
    methods = set(_GUIDE_CALL_RE.findall(guide))
    assert methods  # the regex still matches the guide's call examples
    for name in methods:
        assert callable(getattr(agent, name, None)), f"guide documents missing {name}()"


def test_guide_methods_exist_on_skill_host():
    agent = BareAgent()
    skill = MemorySkill(MemoryConfig(enabled=True, path=":memory:"))
    skill.attach(agent)
    guide = str(agent.context_manager[skill._config.instruct_block_key])
    methods = set(_GUIDE_CALL_RE.findall(guide))
    assert methods
    for name in methods:
        assert callable(getattr(skill, name, None)), f"guide documents missing {name}()"
    skill.detach()


def test_configured_tools_all_exist_on_mixin(agent):
    mgr = _install(agent)
    for name in mgr.config.tools:
        assert callable(getattr(agent, name, None)), f"config.tools lists missing {name}()"


# --------------------------------------------------------------------------
# recalled/injected lines carry ids the agent can act on
# --------------------------------------------------------------------------
def test_injected_lines_carry_id_prefix(agent):
    mgr = _install(agent)
    mid = agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="how do we deploy"))
    text = mgr.recall_for_context()
    assert f"#{mid[:8]}]" in text


def test_update_and_forget_accept_id_prefix(agent):
    mgr = _install(agent)
    mid = agent.remember("the retry limit is 3", type="info")
    assert agent.update_memory(mid[:8], content="the retry limit is 5") is True
    assert mgr.store.get(mid).content == "the retry limit is 5"
    assert agent.forget(mid[:8]) is True
    assert mgr.store.get(mid).archived is True


def test_ambiguous_prefix_raises(agent):
    mgr = _install(agent)
    for suffix in ("aa", "bb"):
        m = Memory(id="deadbeef" + suffix * 12, content=f"row {suffix}", owner="MemAgent")
        mgr.store.add(m, mgr.embedder.embed(m.content))
    with pytest.raises(ValueError, match="ambiguous"):
        agent.forget("deadbeef")


def test_too_short_prefix_is_not_found(agent):
    _install(agent)
    agent.remember("something", type="info")
    assert agent.forget("abc") is False  # <6 chars never prefix-matches


# --------------------------------------------------------------------------
# raise-don't-recover at the tool boundary
# --------------------------------------------------------------------------
def test_associate_unknown_relation_raises(agent):
    _install(agent)
    a = agent.remember("a fact", type="info")
    b = agent.remember("b fact", type="info")
    with pytest.raises(ValueError):
        agent.associate(a, b, relation="nearby-ish")


def test_associate_accepts_prefixes_and_known_relation(agent):
    mgr = _install(agent)
    a = agent.remember("cause fact", type="info")
    b = agent.remember("effect fact", type="info")
    agent.associate(a[:8], b[:8], relation="causes")
    edges = mgr.store.neighbors(a)
    assert edges and edges[0].target_id == b and edges[0].type.value == "causes"
