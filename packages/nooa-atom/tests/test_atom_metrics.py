# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom's harness counters: the ``atom`` extension of core's HarnessMetrics."""

from __future__ import annotations

import asyncio
import json

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.agent.metrics import AtomHarnessMetrics, atom_metrics
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.interactive import Done
from nooa.runtime.harness_metrics import (
    HarnessMetrics,
    get_span_schema,
    harness_metrics_session,
)
from nooa.strategies import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# Gym/HCB analysis reads these names (namespaced harness.atom.* since #558).
ATOM_KEYS = {
    "harness.atom.shell_queued_behind_running.count": ("Tool Usage", False),
    "harness.atom.shell_busy_rejected.count": ("Tool Usage", False),
    "harness.atom.child_detached_at_deadline.count": ("Tool Usage", False),
    "harness.atom.shell_detached_at_deadline.count": ("Tool Usage", False),
    "harness.atom.shell_long_await.count": ("Tool Usage", False),
    "harness.atom.waiting_on_person_rejected.count": ("Tool Usage", False),
    "harness.atom.waiting_on_person_rejected.roles": ("Tool Usage", True),
    "harness.atom.shell_exit_hint.count": ("Tool Usage", False),
    "harness.atom.shell_exit_hint.codes": ("Tool Usage", True),
    "harness.atom.shell_cwd_fallback.count": ("Tool Usage", False),
    "harness.atom.shell_cwd_fallback.details": ("Tool Usage", True),
    "harness.atom.name_error_hint.count": ("Code Validation", False),
    "harness.atom.name_error_hint.names": ("Code Validation", True),
    "harness.atom.mcp_tool_name_hint.count": ("Code Validation", False),
    "harness.atom.mcp_tool_name_hint.calls": ("Code Validation", True),
    "harness.atom.module_not_found_hint.count": ("Code Validation", False),
    "harness.atom.module_not_found_hint.modules": ("Code Validation", True),
}

RECORD_ALL = (
    "from nooa_atom.agent.metrics import atom_metrics\n"
    "m = atom_metrics()\n"
    "m.shell_queued(); m.shell_busy(); m.child_detached()\n"
    "m.shell_detached(); m.shell_long_await()\n"
    "m.waiting_on_person_rejected('child'); m.shell_exit_hint(5)\n"
    "m.shell_cwd_fallback('/gone -> /ws'); m.name_error_hint('sh')\n"
    "m.mcp_tool_name_hint('srv.tool'); m.module_not_found_hint('cv2')"
)

EXPECTED = {
    "harness.atom.shell_queued_behind_running.count": 1,
    "harness.atom.shell_busy_rejected.count": 1,
    "harness.atom.child_detached_at_deadline.count": 1,
    "harness.atom.shell_detached_at_deadline.count": 1,
    "harness.atom.shell_long_await.count": 1,
    "harness.atom.waiting_on_person_rejected.count": 1,
    "harness.atom.waiting_on_person_rejected.roles": ["child"],
    "harness.atom.shell_exit_hint.count": 1,
    "harness.atom.shell_exit_hint.codes": [5],
    "harness.atom.shell_cwd_fallback.count": 1,
    "harness.atom.shell_cwd_fallback.details": ["/gone -> /ws"],
    "harness.atom.name_error_hint.count": 1,
    "harness.atom.name_error_hint.names": ["sh"],
    "harness.atom.mcp_tool_name_hint.count": 1,
    "harness.atom.mcp_tool_name_hint.calls": ["srv.tool"],
    "harness.atom.module_not_found_hint.count": 1,
    "harness.atom.module_not_found_hint.modules": ["cv2"],
}


@pytest.fixture
def flushed(monkeypatch):
    """Span attributes of every harness metrics session that flushed."""
    seen: list[dict] = []
    real_flush = HarnessMetrics.flush_to_span

    def flush(self):
        seen.append(self.to_span_attributes())
        real_flush(self)

    monkeypatch.setattr(HarnessMetrics, "flush_to_span", flush)
    return seen


def test_schema_lists_atoms_attributes_unchanged():
    schema = {e.key: (e.category, e.is_detail) for e in get_span_schema()}
    assert {k: schema.get(k) for k in ATOM_KEYS} == ATOM_KEYS


def test_recorders_emit_the_same_attributes():
    with harness_metrics_session() as metrics:
        atom_metrics().shell_queued()
        atom_metrics().shell_busy()
        atom_metrics().child_detached()
        atom_metrics().shell_detached()
        atom_metrics().shell_long_await()
        atom_metrics().waiting_on_person_rejected("child")
        atom_metrics().shell_exit_hint(5)
        atom_metrics().shell_cwd_fallback("/gone -> /ws")
        atom_metrics().name_error_hint("sh")
        atom_metrics().mcp_tool_name_hint("srv.tool")
        atom_metrics().module_not_found_hint("cv2")
    assert metrics.extension(AtomHarnessMetrics).shell_exit_hints == [5]
    assert metrics.to_span_attributes() == EXPECTED


def test_atom_counters_live_in_the_atom_namespace():
    from nooa.runtime.harness_metrics import get_harness_metrics

    with harness_metrics_session() as metrics:
        atom_metrics().shell_busy()
        get_harness_metrics().atom.shell_queued()
        assert get_harness_metrics().atom is atom_metrics()
        assert atom_metrics() is metrics.extension(AtomHarnessMetrics)
    assert AtomHarnessMetrics.namespace == "atom"
    assert metrics.atom.shell_busy_rejected == 1
    assert metrics.atom.shell_queued_behind_running == 1
    assert all(k.startswith("harness.atom.") for k in metrics.to_span_attributes())
    assert metrics.model_dump()["atom"]["shell_busy_rejected"] == 1


async def test_an_awaited_shell_command_over_the_limit_counts_as_a_long_await(monkeypatch):
    """Only an inline run longer than the limit counts; bash startup and quick runs don't."""
    from nooa_atom.agent import work_shell

    async with work_shell.Shell() as sh:
        await sh.run("true")  # starts bash outside the session: startup is not timed
        with harness_metrics_session() as metrics:
            await sh.run("true")  # under the default 60 s limit, however slow the runner
            monkeypatch.setattr(work_shell, "_LONG_AWAIT_SECONDS", 1.0)
            await sh.run("sleep 1.5")  # at least 1.5 s on any machine
    assert metrics.atom.shell_long_awaits == 1


def test_outside_a_session_recording_is_a_no_op():
    atom_metrics().shell_queued()  # no session: nothing to record into, no error


async def test_an_atom_turns_harness_span_has_the_nine_counters(tmp_path, flushed):
    models = AtomModels()
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    try:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        options = SessionOptions(workspace=workspace, agent_spec="nooa_atom.agent:AtomAgent")
        models.scripts[None] = [cell(RECORD_ALL), cell("return_result(Done(explanation='ok'))")]
        root = await registry.create(options)
        assert await asyncio.wait_for(root.prompt("go"), 60) == Done(explanation="ok")
    finally:
        await registry.close_all()
    [attrs] = [a for a in flushed if "harness.atom.shell_busy_rejected.count" in a]
    assert {k: v for k, v in attrs.items() if k in ATOM_KEYS} == EXPECTED


async def test_a_plain_core_agents_harness_span_has_none_of_them(flushed):
    call = ToolCall(id="r", name="python_cell", arguments=json.dumps({"code": "return_result(42)"}))
    llm = FakeLLMClient(scripted_responses=[LLMResponse(parts=(call,), finish_reason="tool_calls")])

    class Plain(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> int:
            """Return the answer."""
            ...

    agent = Plain()
    try:
        assert await agent.answer() == 42
    finally:
        await agent.aclose()
    assert flushed
    assert not [k for attrs in flushed for k in attrs if k in ATOM_KEYS]
    for name in AtomHarnessMetrics.model_fields:
        assert name not in HarnessMetrics.model_fields
