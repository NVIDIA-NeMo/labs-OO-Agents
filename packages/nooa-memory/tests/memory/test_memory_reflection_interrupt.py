# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interruptible consolidation: per-item stop checks, idempotent resume,
reasoner gating, and the manager's idle-reflection bookkeeping."""

import pytest
from nooa_memory import (
    Memory,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
)
from nooa_memory.config import ForgetPolicy, ReflectionPolicy
from nooa_memory.embeddings import HashingEmbedder
from nooa_memory.forgetting import ForgettingEngine
from nooa_memory.reflection import ReflectionEngine
from nooa_memory.schema import _now
from nooa_memory.store import MemoryStore

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


def _stop_after(n: int):
    """A should_stop probe that fires True from its (n+1)-th call on."""
    calls = {"n": 0}

    def probe() -> bool:
        calls["n"] += 1
        return calls["n"] > n

    return probe


def _seed(store: MemoryStore, emb: HashingEmbedder) -> None:
    """Duplicates (merge fodder), related pairs (edge fodder), episodes."""
    base = _now() - 100.0
    rows = [
        ("the deploy command is make ship", MemoryType.INFO),
        ("the deploy command is make ship", MemoryType.INFO),  # dup of ^
        ("the deploy command is make ship!", MemoryType.INFO),  # near-dup
        ("rollback procedure uses make undeploy", MemoryType.SKILL),
        ("rollback runbook: run make undeploy twice", MemoryType.SKILL),
        ("Episode: shipped release 1", MemoryType.EPISODE),
        ("Episode: shipped release 2", MemoryType.EPISODE),
    ]
    for i, (content, mtype) in enumerate(rows):
        m = Memory(content=content, type=mtype, created_at=base + i, last_accessed_at=base + i)
        store.add(m, emb.embed(m.embedding_text()))


def _engine(store, emb):
    return ReflectionEngine(store, emb, ReflectionPolicy(), ForgetPolicy())


def _active_state(store: MemoryStore) -> set[tuple[str, bool]]:
    return {(m.content, m.archived) for m in store.all_memories(include_archived=True)}


# --------------------------------------------------------------------------
# equivalence + interruption
# --------------------------------------------------------------------------
def test_never_stopping_matches_plain_consolidate():
    emb = HashingEmbedder(dim=256)
    s1, s2 = MemoryStore(":memory:"), MemoryStore(":memory:")
    _seed(s1, emb)
    _seed(s2, emb)

    r_plain = _engine(s1, emb).consolidate()
    r_interruptible = _engine(s2, emb).consolidate_interruptible(should_stop=lambda: False)

    assert r_plain.interrupted is False and r_interruptible.interrupted is False
    for field in ("merged", "edges_added", "pruned", "created", "reconciled"):
        assert getattr(r_plain, field) == getattr(r_interruptible, field)
    assert r_interruptible.duration_ms >= 0.0


@pytest.mark.parametrize("n_probes", [0, 1, 2, 5, 10])
def test_interruption_is_safe_and_resume_converges(n_probes):
    """Stop at various depths; a follow-up full pass converges to the baseline."""
    emb = HashingEmbedder(dim=256)
    baseline = MemoryStore(":memory:")
    _seed(baseline, emb)
    _engine(baseline, emb).consolidate()

    store = MemoryStore(":memory:")
    _seed(store, emb)
    engine = _engine(store, emb)
    partial = engine.consolidate_interruptible(should_stop=_stop_after(n_probes))
    assert partial.interrupted is True
    assert partial.stopped_in in {
        "merge_duplicates",
        "form_edges",
        "rescore_importance",
        "prune",
    }
    # the interrupted store is consistent: every edge target exists
    for m in store.all_memories(include_archived=True):
        for e in m.edges:
            assert store.get(e.target_id) is not None

    engine.consolidate()  # resume to completion
    assert _active_state(store) == _active_state(baseline)


def test_stop_before_anything_commits_nothing():
    emb = HashingEmbedder(dim=256)
    store = MemoryStore(":memory:")
    _seed(store, emb)
    before = _active_state(store)
    report = _engine(store, emb).consolidate_interruptible(should_stop=lambda: True)
    assert report.interrupted is True
    assert report.merged == 0 and report.edges_added == 0 and report.pruned == 0
    assert _active_state(store) == before


# --------------------------------------------------------------------------
# LLM-step gating
# --------------------------------------------------------------------------
def test_reasoner_never_starts_when_stopping():
    emb = HashingEmbedder(dim=256)
    store = MemoryStore(":memory:")
    _seed(store, emb)
    calls = []

    def reasoner(episodes):
        calls.append(len(episodes))
        return [Memory(content="an abstraction", type=MemoryType.REFLECTION)]

    _engine(store, emb).consolidate_interruptible(should_stop=lambda: True, reasoner=reasoner)
    assert calls == []  # the stop pre-check prevented the LLM call


def test_inflight_reasoner_result_is_discarded_on_stop():
    emb = HashingEmbedder(dim=256)
    store = MemoryStore(":memory:")
    _seed(store, emb)
    flag = {"stop": False}

    def reasoner(episodes):
        flag["stop"] = True  # the stop arrives WHILE the LLM call runs
        return [Memory(content="a late abstraction", type=MemoryType.REFLECTION)]

    report = _engine(store, emb).consolidate_interruptible(
        should_stop=lambda: flag["stop"], reasoner=reasoner
    )
    assert report.created == 0
    assert all("late abstraction" not in m.content for m in store.all_memories())
    assert report.interrupted is True and report.stopped_in == "abstract"


def test_prune_stops_between_items():
    emb = HashingEmbedder(dim=256)
    store = MemoryStore(":memory:")
    old = _now() - 10_000 * 3600
    for i in range(5):
        m = Memory(
            content=f"ancient fact {i}",
            importance=1.0,
            created_at=old,
            last_accessed_at=old,
        )
        store.add(m, emb.embed(m.embedding_text()))
    forgetting = ForgettingEngine(store, ForgetPolicy())
    pruned = forgetting.prune(should_stop=_stop_after(1))
    assert len(pruned) <= 1  # stopped within one item


# --------------------------------------------------------------------------
# manager bookkeeping
# --------------------------------------------------------------------------
def test_manager_reflect_interruptible_events_and_maintenance():
    agent = MemAgent()
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:"))
    agent.remember("identical fact", type="info")
    agent.remember("another fact entirely", type="info")

    seen = []
    agent.event_manager.on("ReflectionStarted", lambda e: seen.append(("started", e.trigger)))
    agent.event_manager.on(
        "ReflectionCompleted", lambda e: seen.append(("done", e.trigger, e.interrupted))
    )

    report = mgr.reflect_interruptible(lambda: False, trigger="idle")
    assert report.interrupted is False
    assert ("started", "idle") in seen
    assert ("done", "idle", False) in seen

    row = mgr.store.maintenance_history(1)[0]
    assert row["kind"] == "reflect"
    assert row["report"]["trigger"] == "idle"
    assert row["report"]["interrupted"] is False
    assert row["report"]["duration_ms"] >= 0.0


def test_manager_interrupted_run_recorded():
    agent = MemAgent()
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:"))
    agent.remember("some fact to consolidate", type="info")
    report = mgr.reflect_interruptible(lambda: True, trigger="idle")
    assert report.interrupted is True
    row = mgr.store.maintenance_history(1)[0]
    assert row["report"]["interrupted"] is True
    assert row["report"]["stopped_in"] == report.stopped_in


def test_plain_reflect_emits_started_with_manual_trigger():
    agent = MemAgent()
    mgr = MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:"))
    seen = []
    agent.event_manager.on("ReflectionStarted", lambda e: seen.append(e.trigger))
    mgr.reflect()
    assert seen == ["manual"]
    assert mgr.store.maintenance_history(1)[0]["report"]["trigger"] == "manual"
