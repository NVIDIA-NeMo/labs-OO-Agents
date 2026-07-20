# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pass-by-reference memories: parsing, capture, resolution, rendering."""

import pytest
from nooa_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryRef,
    MemoryToolsMixin,
)
from nooa_memory.references import capture, parse_ref, render, resolve

from nooa import Agent
from nooa.events import Task
from nooa.unifiedllm import FakeLLMClient


class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
    pass


@pytest.fixture
def agent():
    a = MemAgent()
    a.vars = {"plan": "phase 2: ship the owner column"}  # host-provided vars surface
    return a


@pytest.fixture
def mgr(agent):
    return MemoryManager.install(agent, config=MemoryConfig(enabled=True, path=":memory:"))


# --------------------------------------------------------------------------
# parsing + validation
# --------------------------------------------------------------------------
def test_parse_ref_valid_forms():
    assert parse_ref("var:plan") == ("var", "plan")
    assert parse_ref("file:docs/spec.md") == ("file", "docs/spec.md")


@pytest.mark.parametrize("bad", ["plan", "unknown:key", "var:", ":key", ""])
def test_parse_ref_malformed_raises(bad):
    with pytest.raises(ValueError, match="malformed reference|expected"):
        parse_ref(bad)


def test_memory_ref_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown reference kind"):
        MemoryRef(kind="expr", key="1+1")


def test_file_capture_rejects_escaping_paths(mgr, agent):
    with pytest.raises(ValueError, match="escapes the working dir"):
        capture(agent, mgr.store, "file:../../etc/passwd")
    with pytest.raises(ValueError, match="must be relative"):
        capture(agent, mgr.store, "file:/etc/passwd")


def test_foreign_escaping_ref_resolves_dangling_not_raise(mgr, agent):
    # A stored ref may come from another agent: resolution must degrade, not read.
    hostile = MemoryRef(kind="file", key="../../etc/passwd", preview="nope")
    res = resolve(agent, mgr.store, hostile)
    assert res.status == "DANGLING"
    assert res.value_repr == "nope"


# --------------------------------------------------------------------------
# capture + live resolution per kind
# --------------------------------------------------------------------------
def test_var_ref_live_and_preview(mgr, agent):
    ref = capture(agent, mgr.store, "var:plan")
    assert ref.preview and "phase 2" in ref.preview
    agent.vars["plan"] = "phase 3: observability"
    res = resolve(agent, mgr.store, ref)
    assert res.status == "LIVE"
    assert "phase 3" in res.value_repr  # live, not the write-time snapshot


def test_var_ref_dangling_falls_back_to_preview(mgr, agent):
    ref = capture(agent, mgr.store, "var:plan")
    del agent.vars["plan"]
    res = resolve(agent, mgr.store, ref)
    assert res.status == "DANGLING"
    assert "phase 2" in res.value_repr  # the stale snapshot
    assert "DANGLING" in render(res) and "snapshot @" in render(res)


def test_context_ref_resolves_block(mgr, agent):
    agent.context_manager.set_static("current_goal", "finish the memory MR")
    res = resolve(agent, mgr.store, capture(agent, mgr.store, "context:current_goal"))
    assert res.status == "LIVE"
    assert "finish the memory MR" in res.value_repr


def test_file_ref_reads_fresh_content(mgr, agent, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "notes.md"
    f.write_text("v1 of the notes")
    ref = capture(agent, mgr.store, "file:notes.md")
    assert ref.preview == "v1 of the notes"
    f.write_text("v2 of the notes")  # the file moves on...
    res = resolve(agent, mgr.store, ref)
    assert res.status == "LIVE"
    assert "v2" in res.value_repr  # ...and the reference stays current


def test_memory_ref_points_at_record(mgr, agent):
    target = agent.remember("the gateway key lives in secrets.yaml", type="info")
    res = resolve(agent, mgr.store, capture(agent, mgr.store, f"memory:{target}"))
    assert res.status == "LIVE"
    assert "secrets.yaml" in res.value_repr
    gone = resolve(agent, mgr.store, MemoryRef(kind="memory", key="nope"))
    assert gone.status == "DANGLING"


# --------------------------------------------------------------------------
# tool surface + injection rendering
# --------------------------------------------------------------------------
def test_remember_with_references_roundtrips(mgr, agent):
    mid = agent.remember(
        "the current plan lives in self.v.plan",
        type="info",
        references=["var:plan"],
    )
    got = mgr.store.get(mid)
    assert len(got.references) == 1
    assert got.references[0].spec() == "var:plan"
    assert "phase 2" in got.references[0].preview


def test_remember_malformed_reference_raises(mgr, agent):
    with pytest.raises(ValueError, match="malformed reference"):
        agent.remember("bad ref", references=["not-a-ref"])


def test_update_memory_replaces_references(mgr, agent):
    mid = agent.remember("plan pointer", references=["var:plan"])
    agent.context_manager.set_static("goal", "the goal block")
    assert agent.update_memory(mid, references=["context:goal"]) is True
    got = mgr.store.get(mid)
    assert [r.spec() for r in got.references] == ["context:goal"]


def test_deref_tool_reads_live_value(mgr, agent):
    out = agent.deref("var:plan")
    assert "LIVE" in out and "phase 2" in out
    out = agent.deref("var:missing")
    assert "DANGLING" in out


def test_injected_block_renders_resolved_references(mgr, agent):
    agent.remember(
        "the migration plan is tracked in self.v.plan",
        type="info",
        references=["var:plan"],
    )
    agent.event_manager.add(Task(prompt="what is the migration plan?"))
    text = mgr.recall_for_context()
    assert "ref var:plan (LIVE)" in text
    assert "phase 2" in text


def test_deref_memory_prefix_logs_access(mgr, agent):
    """deref('memory:<id8>') must log on the SAME memory it renders LIVE."""
    target = agent.remember("the deploy runbook lives in docs/deploy.md", type="info")
    out = agent.deref(f"memory:{target[:8]}")
    assert "LIVE" in out

    got = mgr.store.get(target)
    assert got.deref_count == 1
    assert got.access_log[-1].channel == "deref"

    # Full-id deref keeps working identically.
    agent.deref(f"memory:{target}")
    assert mgr.store.get(target).deref_count == 2
