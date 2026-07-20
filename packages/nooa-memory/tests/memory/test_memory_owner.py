# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for owner identity: stamping, scoped reads, write policy, shared stores."""

import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
)
from nooa_memory.schema import Memory

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def shared_path(tmp_path):
    return str(tmp_path / "shared-memory.sqlite")


def _pair(shared_path):
    """Two agents (alice, bob) sharing one memory store."""
    alice, bob = MemAgent(), MemAgent()
    mgr_a = MemoryManager.install(
        alice, config=MemoryConfig(enabled=True, path=shared_path, owner="alice")
    )
    mgr_b = MemoryManager.install(
        bob, config=MemoryConfig(enabled=True, path=shared_path, owner="bob")
    )
    return alice, mgr_a, bob, mgr_b


# --------------------------------------------------------------------------
# identity resolution + stamping
# --------------------------------------------------------------------------
def test_default_owner_is_class_name():
    agent = MemAgent()
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:"))
    assert mgr.owner == "MemAgent"
    mid = agent.remember("a fact", type="info")
    assert mgr.store.get(mid).owner == "MemAgent"


def test_explicit_empty_owner_writes_unowned():
    agent = MemAgent()
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:", owner=""))
    assert mgr.owner == ""
    mid = agent.remember("a shared convention", type="info")
    assert mgr.store.get(mid).owner == ""


# --------------------------------------------------------------------------
# scoped reads on a shared store
# --------------------------------------------------------------------------
def test_default_scope_isolates_owners(shared_path):
    alice, mgr_a, bob, _ = _pair(shared_path)
    alice.remember("the deploy password hint is stored in vault X", type="info")

    assert bob.recall("deploy password hint") == []  # default: own scope only
    found = bob.recall("deploy password hint", owner="*")
    assert found and found[0].owner == "alice"
    named = bob.recall("deploy password hint", owner="alice")
    assert named and named[0].owner == "alice"


def test_unowned_memories_visible_to_everyone(shared_path):
    alice, mgr_a, bob, mgr_b = _pair(shared_path)
    emb = mgr_a.embedder.embed("team convention: always squash-merge to main")
    mgr_a.store.add(Memory(content="team convention: always squash-merge to main"), emb)  # owner=""

    assert alice.recall("squash-merge convention")
    assert bob.recall("squash-merge convention")


def test_dedup_never_reinforces_foreign_memory(shared_path):
    alice, mgr_a, bob, mgr_b = _pair(shared_path)
    aid = alice.remember("identical fact about shipping releases", type="info")
    bid = bob.remember("identical fact about shipping releases", type="info")
    assert aid != bid  # bob got his own copy, alice's was not touched
    assert mgr_b.store.get(bid).owner == "bob"
    assert mgr_a.store.get(aid).reinforcement_count == 0


# --------------------------------------------------------------------------
# write policy
# --------------------------------------------------------------------------
def test_cross_owner_update_and_forget_raise(shared_path):
    alice, _, bob, _ = _pair(shared_path)
    aid = alice.remember("alice's private note", type="info")

    with pytest.raises(PermissionError, match="belongs to 'alice'"):
        bob.update_memory(aid, content="defaced")
    with pytest.raises(PermissionError, match="belongs to 'alice'"):
        bob.forget(aid)
    # own + unowned writes still work
    assert alice.update_memory(aid, content="alice's corrected note") is True


def test_cross_owner_associate_raises(shared_path):
    alice, _, bob, _ = _pair(shared_path)
    aid = alice.remember("alice's fact", type="info")
    bid = bob.remember("bob's fact", type="info")

    with pytest.raises(PermissionError):
        bob.associate(aid, bid)  # foreign src: mutating alice's neighborhood
    bob.associate(bid, aid)  # own src, foreign target: fine (read-only link)


# --------------------------------------------------------------------------
# spontaneous injection + spread confinement
# --------------------------------------------------------------------------
def test_spontaneous_injection_is_own_scope(shared_path):
    alice, mgr_a, bob, _ = _pair(shared_path)
    bob.remember("rollback uses make undeploy", type="skill")
    alice.event_manager.add(Task(prompt="we need to rollback the release"))
    assert "undeploy" not in mgr_a.recall_for_context()
    assert alice.recall("rollback release", owner="*")  # explicit cross-owner works


def test_spread_does_not_leak_foreign_memories(shared_path):
    alice, mgr_a, bob, mgr_b = _pair(shared_path)
    aid = alice.remember("the ingest pipeline config lives in configs/ingest.yaml", type="info")
    bid = bob.remember("bob's secret analysis of quarterly numbers", type="info")
    mgr_a.store.add_edge(aid, bid)  # graph link across owners (store-level)

    res = alice.recall("ingest pipeline config", k=5)
    ids = {m.id for m in res}
    assert aid in ids
    assert bid not in ids  # 1-hop spread must not surface bob's memory
    res_all = alice.recall("ingest pipeline config", k=5, owner="*")
    assert bid in {m.id for m in res_all}  # ...unless explicitly cross-owner


# --------------------------------------------------------------------------
# reflection stays in its lane
# --------------------------------------------------------------------------
def test_reflection_never_merges_foreign_duplicates(shared_path):
    alice, mgr_a, bob, mgr_b = _pair(shared_path)
    # two near-identical bob memories, written via the store to bypass dedup
    for _ in range(2):
        content = "bob learned that the cache TTL is 300 seconds"
        mgr_b.store.add(Memory(content=content, owner="bob"), mgr_b.embedder.embed(content))

    mgr_a.reflect()  # alice consolidates: bob's duplicates must survive
    bob_active = [m for m in mgr_b.store.all_memories(owner="bob") if m.owner == "bob"]
    assert len(bob_active) == 2

    mgr_b.reflect()  # bob consolidates: now they merge
    bob_active = [m for m in mgr_b.store.all_memories(owner="bob") if m.owner == "bob"]
    assert len(bob_active) == 1
