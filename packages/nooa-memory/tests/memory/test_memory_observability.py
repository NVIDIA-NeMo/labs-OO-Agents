# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the observability substrate: structured access log, channels,
maintenance history, explain(), stats extensions, tracing bridge, KPIs."""

import pytest
from nooa_memory import (
    AccessRecord,
    Memory,
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    MemoryType,
    per_memory_usage,
    store_kpis,
)
from nooa_memory.config import ObservabilityConfig
from nooa_memory.retrieval import base_level_activation
from nooa_memory.schema import _now

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def agent():
    return MemAgent()


def _install(agent, **cfg_kw):
    cfg = MemoryConfig(enabled=True, path=":memory:", **cfg_kw)
    return MemoryManager.install(agent, config=cfg)


# --------------------------------------------------------------------------
# log_access semantics per channel
# --------------------------------------------------------------------------
def test_touching_channel_updates_actr_state():
    m = Memory(content="x")
    before = (m.access_count, m.strength)
    m.log_access(AccessRecord(ts=_now(), channel="recalled"))
    assert m.recalled_count == 1
    assert m.access_count == before[0] + 1
    assert m.strength == before[1] + 1


def test_injected_channel_logs_without_touching():
    m = Memory(content="x")
    before = (m.access_count, m.strength, m.last_accessed_at)
    m.log_access(AccessRecord(ts=_now() + 100, channel="injected"))
    assert m.injected_count == 1
    assert (m.access_count, m.strength, m.last_accessed_at) == before  # invisible to ACT-R
    assert m.access_log[-1].channel == "injected"


def test_injected_entries_excluded_from_activation():
    now = _now()
    only_created = [AccessRecord(ts=now - 1000, channel="created")]
    with_injections = only_created + [
        AccessRecord(ts=now - 1, channel="injected") for _ in range(10)
    ]
    assert base_level_activation(with_injections, now, 0.5) == pytest.approx(
        base_level_activation(only_created, now, 0.5)
    )


def test_reinforced_without_strength_is_update_semantics():
    m = Memory(content="x")
    m.log_access(AccessRecord(ts=_now(), channel="reinforced"), reinforce=False)
    assert m.reinforced_count == 1
    assert m.access_count == 1
    assert m.strength == 1  # unchanged (v1 update path)


def test_unknown_channel_raises():
    with pytest.raises(ValueError, match="unknown access channel"):
        AccessRecord(ts=1.0, channel="teleported")


def test_legacy_float_log_coerces_to_created():
    m = Memory.model_validate({"content": "old row", "access_log": [123.0, 456.0]})
    assert [a.channel for a in m.access_log] == ["created", "created"]
    assert [a.ts for a in m.access_log] == [123.0, 456.0]


def test_access_log_cap_keeps_counters():
    m = Memory(content="x")
    for i in range(100):
        m.log_access(AccessRecord(ts=float(i), channel="recalled"), cap=10)
    assert len(m.access_log) == 10  # ring capped
    assert m.recalled_count == 100  # totals survive rotation


# --------------------------------------------------------------------------
# manager wiring: channels, injection logging, maintenance, explain
# --------------------------------------------------------------------------
def test_recall_logs_rich_access(agent):
    mgr = _install(agent)
    mgr.session_ref = "sess-42"
    mid = agent.remember("deploy uses make ship", type="info")
    agent.recall("how to deploy")
    got = mgr.store.get(mid)
    entry = got.access_log[-1]
    assert entry.channel == "recalled"
    assert entry.query == "how to deploy"
    assert entry.rank == 0 and entry.score is not None
    assert entry.components and set(entry.components) >= {"rel", "rec", "imp", "spread"}
    assert entry.session_ref == "sess-42"


def test_search_logs_searched_channel(agent):
    mgr = _install(agent)
    mid = agent.remember("rollback uses make undeploy", type="skill")
    agent.search("rollback")
    got = mgr.store.get(mid)
    assert got.searched_count == 1
    assert got.access_log[-1].channel == "searched"


def test_injection_logged_and_persisted(agent):
    mgr = _install(agent)
    mid = agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="deploy the service"))
    mgr._on_before_turn(None)
    got = mgr.store.get(mid)
    assert got.injected_count == 1
    assert got.strength == 1  # injection never reinforces
    assert mgr.stats.injection_ms_total > 0


def test_injection_logging_can_be_disabled(agent):
    mgr = _install(agent, observability=ObservabilityConfig(log_injections=False))
    mid = agent.remember("deploy uses make ship", type="info")
    agent.event_manager.add(Task(prompt="deploy the service"))
    mgr._on_before_turn(None)
    assert mgr.store.get(mid).injected_count == 0


def test_dedup_reinforce_logs_channel(agent):
    mgr = _install(agent)
    mid = agent.remember("identical fact about shipping", type="info")
    agent.remember("identical fact about shipping", type="info")
    got = mgr.store.get(mid)
    assert got.reinforced_count == 1
    assert got.access_log[-1].channel == "reinforced"


def test_reflect_appends_maintenance_history(agent):
    mgr = _install(agent)
    agent.remember("something to consolidate", type="info")
    mgr.reflect()
    history = mgr.store.maintenance_history()
    assert history and history[0]["kind"] == "reflect"
    assert "merged" in history[0]["report"]


def test_explain_returns_components_without_touching(agent):
    mgr = _install(agent)
    mid = agent.remember("deploy uses make ship", type="info")
    before = mgr.store.get(mid).access_count
    rows = mgr.explain("how to deploy")
    assert rows and rows[0]["id"] == mid
    assert {"rank", "score", "source", "cos", "rel", "rec", "imp", "spread"} <= set(rows[0])
    assert mgr.store.get(mid).access_count == before  # dry run


def test_stats_snapshot_counts_todos_and_refs(agent):
    mgr = _install(agent)
    agent.vars = {"plan": "v1"}
    agent.remember("open commitment", type="todo")
    done = agent.remember("another commitment about testing", type="todo")
    agent.update_memory(done, status="DONE")
    agent.deref("var:plan")
    agent.deref("var:missing")
    stats = mgr.memory_stats()
    assert stats.todos_open == 1 and stats.todos_done == 1
    assert stats.refs_resolved == 1 and stats.refs_dangling == 1


def test_cross_owner_recall_counter(agent):
    mgr = _install(agent)
    agent.recall("anything", owner="*")
    agent.recall("anything")  # own scope: not counted
    assert mgr.stats.cross_owner_recalls == 1


# --------------------------------------------------------------------------
# tracing bridge
# --------------------------------------------------------------------------
def test_bridge_emits_span_events(agent):
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as ot_trace

    provider = trace_sdk.TracerProvider()
    tracer = provider.get_tracer("test")

    mgr = _install(agent)
    with tracer.start_as_current_span("execute_python") as span:
        agent.remember("traced write", type="info")
        agent.recall("traced write")
    events = {e.name for e in span.events}
    assert {"memory.written", "memory.recalled"} <= events
    written = next(e for e in span.events if e.name == "memory.written")
    assert written.attributes["memory.owner"] == mgr.owner
    assert "memory.db_path" in written.attributes
    recalled = next(e for e in span.events if e.name == "memory.recalled")
    assert recalled.attributes["memory.memory_ids"]
    # trace_ref lands on the access record while a span is active
    with tracer.start_as_current_span("second") as span2:
        res = agent.recall("traced write")
    entry = mgr.store.get(res[0].id).access_log[-1]
    assert entry.trace_ref == format(span2.get_span_context().span_id, "016x")
    del ot_trace  # imported to assert the module is available


# --------------------------------------------------------------------------
# derived KPIs
# --------------------------------------------------------------------------
def test_per_memory_usage_panel(agent):
    mgr = _install(agent)
    mid = agent.remember("deploy uses make ship", type="info")
    agent.recall("deploy")
    usage = per_memory_usage(mgr.store.get(mid), forgetting=mgr.forgetting)
    assert usage["fetches"] == 1 and usage["recalled"] == 1
    assert usage["last_channel"] == "recalled"
    assert usage["mean_rank"] == 0
    assert 0.0 < usage["retention"] <= 1.0
    assert usage["prune_eta"] is None or usage["prune_eta"] > usage["last_ts"]


def test_prune_eta_none_for_protected(agent):
    mgr = _install(agent)
    open_todo = Memory(content="open forever", type=MemoryType.TODO)
    usage = per_memory_usage(open_todo, forgetting=mgr.forgetting)
    assert usage["prune_eta"] is None  # open todos are protected


def test_store_kpis_shape(agent):
    mgr = _install(agent)
    agent.remember("a fact about deploys", type="info")
    agent.remember("an open commitment", type="todo")
    agent.recall("deploys")
    kpis = store_kpis(mgr.store, forgetting=mgr.forgetting)
    assert kpis["total"] == 2
    assert kpis["by_type"]["todo"] == 1
    assert kpis["todos_open"] == 1
    assert kpis["total_fetches"] >= 1
    assert isinstance(kpis["maintenance"], list)


def test_store_kpis_group_owners_by_role(agent):
    mgr = _install(agent)
    for inst in ("aaaa1111", "bbbb2222"):
        m = Memory(content=f"instance fact {inst}", owner=f"agentx@{inst}")
        mgr.store.add(m, mgr.embedder.embed(m.embedding_text()))
    other = Memory(content="another role's fact", owner="other")
    mgr.store.add(other, mgr.embedder.embed(other.embedding_text()))

    kpis = store_kpis(mgr.store, forgetting=mgr.forgetting)
    assert kpis["by_owner"]["agentx"] == 2  # instances fold into one role row
    assert kpis["by_owner"]["other"] == 1

    # same-role, different-instance reads are NOT cross-owner
    row = mgr.store.get(other.id)
    row.log_access(AccessRecord(ts=_now(), channel="recalled", reader_owner="other@cccc3333"))
    row.log_access(AccessRecord(ts=_now(), channel="recalled", reader_owner="agentx@aaaa1111"))
    mgr.store.save(row)
    kpis = store_kpis(mgr.store, forgetting=mgr.forgetting)
    assert kpis["cross_owner_reads"] == 1  # only the agentx read counts


def test_injected_never_used_unverifiable_after_ring_rotation(agent):
    mgr = _install(agent)
    m = Memory(content="hot memory")
    m.log_access(AccessRecord(ts=1.0, channel="injected"), cap=4)
    for i in range(10):  # rotate every injected entry out of the capped ring
        m.log_access(AccessRecord(ts=2.0 + i, channel="recalled"), cap=4)

    assert m.injected_count == 1  # the uncapped counter still remembers
    assert all(e.channel != "injected" for e in m.access_log)
    usage = per_memory_usage(m, forgetting=mgr.forgetting)
    # No surviving injected entry -> the question is unanswerable; never flag.
    assert usage["injected_never_used"] is False


def test_injected_never_used_still_flags_on_surviving_evidence(agent):
    mgr = _install(agent)
    m = Memory(content="surfaced but ignored")
    m.log_access(AccessRecord(ts=1.0, channel="recalled"))
    m.log_access(AccessRecord(ts=2.0, channel="injected"))  # nothing deliberate after

    usage = per_memory_usage(m, forgetting=mgr.forgetting)
    assert usage["injected_never_used"] is True
