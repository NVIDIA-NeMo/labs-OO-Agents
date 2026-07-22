# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical owners (role@instance): role-scope reads/writes across
instances, exact-instance reads, validation, and legacy bare-role rows."""

import pytest
from nooa_memory import (
    Memory,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
)
from nooa_memory.config import SpontaneousConfig
from nooa_memory.schema import owner_matches, role_of

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def shared_path(tmp_path):
    return str(tmp_path / "shared-memory.sqlite")


def _mgr(path, owner, **cfg_kw):
    agent = MemAgent()
    mgr = MemoryManager.install(
        agent, config=MemoryConfig(enabled=True, path=path, owner=owner, **cfg_kw)
    )
    return agent, mgr


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_role_of_and_owner_matches():
    assert role_of("TUIAgent@04aaac87") == "TUIAgent"
    assert role_of("TUIAgent") == "TUIAgent"
    assert role_of("") == ""
    assert owner_matches("TUIAgent@x", "TUIAgent") is True  # role scope
    assert owner_matches("TUIAgent", "TUIAgent") is True  # bare-role legacy row
    assert owner_matches("Other@x", "TUIAgent") is False
    assert owner_matches("TUIAgent@x", "TUIAgent@x") is True  # exact instance
    assert owner_matches("TUIAgent@y", "TUIAgent@x") is False
    assert owner_matches("", "anything") is True  # unowned matches every scope
    assert owner_matches("TUIAgent@x", None) is True


def test_owner_validation():
    MemoryConfig(owner="TUIAgent@04aaac87")  # ok
    MemoryConfig(owner="planner")  # bare role ok
    MemoryConfig(owner="")  # unowned writer ok
    with pytest.raises(ValueError, match="must not contain"):
        MemoryConfig(owner="bad%role")
    with pytest.raises(ValueError, match="must not contain"):
        MemoryConfig(owner="role@in@stance")
    with pytest.raises(ValueError, match="must start with a role"):
        MemoryConfig(owner="@lonelyinstance")


# --------------------------------------------------------------------------
# cross-instance knowledge continuity (the point of role scope)
# --------------------------------------------------------------------------
def test_instances_share_knowledge_and_curation(shared_path):
    a1_agent, a1 = _mgr(shared_path, "agentx@aaaa1111")
    mid = a1_agent.remember("the deploy command is make ship", type="skill")
    assert a1.store.get(mid).owner == "agentx@aaaa1111"  # full-tag write provenance

    a2_agent, a2 = _mgr(shared_path, "agentx@bbbb2222")
    # default recall: role scope -> yesterday's knowledge is visible
    found = a2_agent.recall("how do we deploy")
    assert any(m.id == mid for m in found)
    # curation across instances: same role may refine and close
    assert a2_agent.update_memory(mid, content="deploy via make ship (v2)") is True
    assert a2_agent.forget(mid) is True


def test_other_roles_stay_isolated(shared_path):
    a_agent, _ = _mgr(shared_path, "agentx@aaaa1111")
    mid = a_agent.remember("agentx private convention", type="info")

    b_agent, _ = _mgr(shared_path, "other@cccc3333")
    assert b_agent.recall("agentx private convention") == []  # role isolation
    found = b_agent.recall("agentx private convention", owner="agentx")  # role read
    assert any(m.id == mid for m in found)
    with pytest.raises(PermissionError):
        b_agent.forget(mid)


def test_exact_instance_read(shared_path):
    a1_agent, _ = _mgr(shared_path, "agentx@aaaa1111")
    a1_agent.remember("written by instance one", type="info")
    a2_agent, _ = _mgr(shared_path, "agentx@bbbb2222")
    a2_agent.remember("written by instance two", type="info")

    reader_agent, _ = _mgr(shared_path, "other@dddd4444")
    exact = reader_agent.recall("written by instance", owner="agentx@aaaa1111", k=5)
    contents = {m.content for m in exact}
    assert "written by instance one" in contents
    assert "written by instance two" not in contents


def test_bare_role_legacy_rows_are_first_class(shared_path):
    _, seeder = _mgr(shared_path, "agentx")  # library-style bare-role writer
    legacy = Memory(content="healed legacy convention", owner="agentx")
    seeder.store.add(legacy, seeder.embedder.embed(legacy.embedding_text()))

    a2_agent, _ = _mgr(shared_path, "agentx@bbbb2222")
    assert any(m.id == legacy.id for m in a2_agent.recall("healed legacy convention"))
    assert a2_agent.update_memory(legacy.id, content="healed and refined") is True


def test_dedup_reinforces_across_instances(shared_path):
    a1_agent, a1 = _mgr(shared_path, "agentx@aaaa1111")
    mid1 = a1_agent.remember("identical fact about shipping releases", type="info")
    a2_agent, _ = _mgr(shared_path, "agentx@bbbb2222")
    mid2 = a2_agent.remember("identical fact about shipping releases", type="info")
    assert mid1 == mid2  # same role: reinforced, not duplicated
    assert a1.store.get(mid1).reinforcement_count == 1
    assert a1.store.get(mid1).owner == "agentx@aaaa1111"  # original writer kept


def test_spread_confined_to_role(shared_path):
    a_agent, a = _mgr(shared_path, "agentx@aaaa1111")
    mine = a_agent.remember("the ingest pipeline config lives in configs/", type="info")
    b_agent, _ = _mgr(shared_path, "other@cccc3333")
    theirs = b_agent.remember("other agent's secret analysis", type="info")
    a.store.add_edge(mine, theirs)

    ids = {m.id for m in a_agent.recall("ingest pipeline config", k=5)}
    assert mine in ids and theirs not in ids  # spread must not cross roles


def test_reflection_merges_across_instances_of_same_role(shared_path):
    _, a1 = _mgr(shared_path, "agentx@aaaa1111")
    for inst in ("aaaa1111", "bbbb2222"):
        m = Memory(content="the cache TTL is 300 seconds", owner=f"agentx@{inst}")
        a1.store.add(m, a1.embedder.embed(m.embedding_text()))

    a1.reflect()
    active = [m for m in a1.store.all_memories(owner="agentx") if m.owner != ""]
    assert len(active) == 1  # instances consolidated at role scope


# --------------------------------------------------------------------------
# the M3 patch-miss regression: open-todo section is role-scoped
# --------------------------------------------------------------------------
def test_always_todo_section_does_not_leak_other_roles(shared_path):
    b_agent, _ = _mgr(shared_path, "other@cccc3333")
    b_agent.remember("other role's commitment to ship docs", type="todo")

    a_agent, a = _mgr(
        shared_path,
        "agentx@aaaa1111",
        spontaneous=SpontaneousConfig(inject_open_todos="always"),
    )
    a_agent.remember("agentx commitment to update the readme", type="todo")
    a_agent.event_manager.add(Task(prompt="completely unrelated request"))
    text = a.recall_for_context()
    assert "update the readme" in text
    assert "ship docs" not in text  # foreign role's todo must not surface
