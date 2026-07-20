# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MemorySkill — the in-core skill adapter over MemoryManager (offline round-trip)."""

from importlib.metadata import entry_points

import pytest
from nooa_memory import MemoryConfig, MemoryManager
from nooa_memory.manager import MemoryToolsMixin
from nooa_memory.memory_skill import MemorySkill

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


def _bare_agent() -> Agent:
    """A plain agent with NO memory tools — the skill provides them."""

    class Bare(Agent, llm=FakeLLMClient()):
        pass

    return Bare()


def _skill() -> MemorySkill:
    return MemorySkill(MemoryConfig(enabled=True, path=":memory:"))


def test_attach_installs_manager_and_round_trips():
    agent = _bare_agent()
    skill = _skill()
    skill.attach(agent)
    assert isinstance(agent._memory, MemoryManager)  # manager installed on the agent

    mid = skill.remember("the deploy command is make ship", type="skill", importance="HIGH")
    assert mid
    hits = skill.recall("how do I deploy", k=1)
    assert hits and "make ship" in hits[0].content

    skill.reflect()  # consolidation runs through the manager
    assert skill.stats().writes >= 1

    assert skill._mgr is not None
    # The manager owns guide injection, rendered with the skill's API prefix.
    assert skill._mgr.config.instruct is True
    assert skill._mgr.config.api_prefix == "self.memory."
    assert skill._config.instruct_block_key in agent.context_manager

    skill.detach()
    assert skill._mgr is None  # hooks torn down
    assert not hasattr(agent, "_memory")
    assert skill._config.instruct_block_key not in agent.context_manager


def test_unattached_skill_raises():
    skill = _skill()  # never attached
    with pytest.raises(RuntimeError):  # no silent fallback
        skill.remember("x")


def test_verbal_importance_is_enforced():
    agent = _bare_agent()
    skill = _skill()
    skill.attach(agent)
    with pytest.raises(ValueError):  # unknown band -> raise, no fallback
        skill.remember("x", importance="VERY-HIGH")


def test_inherits_mixin_tool_bodies_not_copies():
    # the skill reuses the mixin's conscious tools verbatim (no duplication)
    assert MemorySkill.remember is MemoryToolsMixin.remember
    assert MemorySkill.recall is MemoryToolsMixin.recall


def test_registered_as_builtin_skill_entrypoint():
    # shipped as a built-in skill: discoverable via the nooa.skills group
    names = {ep.name for ep in entry_points(group="nooa.skills")}
    assert "nemo.memory" in names
