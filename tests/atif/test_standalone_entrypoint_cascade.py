# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ATIF capture of standalone generation functions.

Under ``enable_atif()``, a standalone ``@strategy`` generation function
called from a **pure-Python orchestrator** (a method with a real body,
not ``...``) used to be silently dropped: no ``subagent_trajectories[]``
entry and no standalone file of its own, because the standalone-cascade
ContextVar was armed only while an agent *generation turn* was open.

These tests pin the three orderings from the issue, all under one
``enable_atif()`` and deterministic via ``FakeLLMClient``:

- **A** — standalone called from *inside* a generation method (turn open).
- **B** — standalone called *first*, from a pure-Python orchestrator.
- **C** — standalone called *after* a generation method, still from the
  pure-Python orchestrator.

All three must embed the standalone as a ``subagent_trajectories[]``
entry under the orchestrator's single trajectory ("one combined file per
agent.run()"). A crash case pins that the run-scoped ContextVar binding
is released even when the orchestrator raises.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, enable_atif
from nooa.config import CodeActConfig
from nooa.standalone import _atif_exporter_var
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

Category = Literal["bug", "feature", "question"]

_DEFAULT = FakeLLMClient()


# ---------------------------------------------------------------------------
# Helpers (mirror tests/atif/test_enable_atif_isolation.py)
# ---------------------------------------------------------------------------


def _resp(tool_calls: list[ToolCall] | None = None, content: str = "") -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 5, "completion_tokens": 1},
    )


def _exec(code: str, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _ret(result: object, call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


@pytest.fixture
def _isolated_agent_init():
    """Restore ``Agent.__init__`` after each test so ``enable_atif()``
    monkey-patching doesn't bleed into other tests."""
    from nooa import atif as atif_module
    from nooa.agent import Agent

    orig_init = Agent.__init__
    orig_patched = atif_module.install._AGENT_INIT_PATCHED
    orig_cfg = dict(atif_module.install._ENABLE_ATIF_CONFIG)
    try:
        yield
    finally:
        Agent.__init__ = orig_init  # type: ignore[method-assign]
        atif_module.install._AGENT_INIT_PATCHED = orig_patched
        atif_module.install._ENABLE_ATIF_CONFIG.clear()
        atif_module.install._ENABLE_ATIF_CONFIG.update(orig_cfg)


def _only_trajectory(output_dir: Path, agent_cls_name: str) -> Trajectory:
    """Load the single trajectory file written for *agent_cls_name*."""
    agent_dir = output_dir / agent_cls_name
    assert agent_dir.exists(), f"no trajectory dir for {agent_cls_name} under {output_dir}"
    files = list(agent_dir.glob("*.json"))
    assert len(files) == 1, f"expected exactly one {agent_cls_name} trajectory, got {files}"
    return Trajectory.model_validate_json(files[0].read_text())


# ---------------------------------------------------------------------------
# Standalone generation function reused by all cases
# ---------------------------------------------------------------------------


@strategy(PredictStrategy())
async def categorize_ticket(ticket: str) -> Category:
    """Classify a single support ticket {ticket} into its category."""
    ...


# ---------------------------------------------------------------------------
# Case A — standalone called from inside a generation method (turn open).
# This already worked before the fix; it guards against regression.
# ---------------------------------------------------------------------------


class _CaseAAgent(Agent, llm=_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
    async def triage(self, ticket: str) -> str:
        """Triage {ticket}. Call ``categorize_ticket`` then summarize."""
        ...


@pytest.mark.asyncio
async def test_case_a_standalone_inside_generation_method(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Case A (regression guard): a standalone called from *inside* a generation
    method (turn open) is embedded as a subagent_trajectory — the path that
    already worked before the fix."""
    output_dir = tmp_path / "atif-a"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec("c = await categorize_ticket('crash on save')", call_id="a_exec"),
                    _ret("category=bug", call_id="a_ret"),
                ]
            ),
            _resp(content='{"value": "bug"}'),  # standalone PredictStrategy result
        ]
    )
    agent = _CaseAAgent(llm=fake)
    assert await agent.triage("crash on save") == "category=bug"

    traj = _only_trajectory(output_dir, "_CaseAAgent")
    assert_atif_normative(traj)
    names = [c.agent.name for c in (traj.subagent_trajectories or [])]
    assert "categorize_ticket" in names, f"expected standalone embedded; got {names}"

    assert _atif_exporter_var.get() is None


# ---------------------------------------------------------------------------
# Case B — standalone called FIRST from a pure-Python orchestrator.
# ---------------------------------------------------------------------------


class _CaseBAgent(Agent, llm=_DEFAULT):
    async def run(self, ticket: str) -> str:
        # Pure-Python orchestrator: opens NO generation turn of its own.
        category = await categorize_ticket(ticket)  # standalone runs FIRST
        return await self.summarize(ticket, category)

    @strategy(PredictStrategy())
    async def summarize(self, ticket: str, category: str) -> str:
        """Write a one-line triage note for {ticket} given {category}."""
        ...


@pytest.mark.asyncio
async def test_case_b_standalone_first_from_orchestrator(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Case B (the bug): a standalone called *first* from a pure-Python
    orchestrator (before any turn) must still be captured as a
    subagent_trajectory."""
    output_dir = tmp_path / "atif-b"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "bug"}'),  # categorize_ticket (FIRST)
            _resp(content='{"value": "note: a bug report"}'),  # summarize (SECOND)
        ]
    )
    agent = _CaseBAgent(llm=fake)
    assert await agent.run("crash on save") == "note: a bug report"

    traj = _only_trajectory(output_dir, "_CaseBAgent")
    assert_atif_normative(traj)
    names = [c.agent.name for c in (traj.subagent_trajectories or [])]
    assert "categorize_ticket" in names, (
        f"standalone called first from a pure-Python orchestrator was dropped; got {names}"
    )

    assert _atif_exporter_var.get() is None


# ---------------------------------------------------------------------------
# Case C — standalone called AFTER a generation method, still from the
# pure-Python orchestrator.
# ---------------------------------------------------------------------------


class _CaseCAgent(Agent, llm=_DEFAULT):
    async def run(self, ticket: str) -> str:
        note = await self.summarize(ticket)  # generation method first
        category = await categorize_ticket(ticket)  # standalone AFTER
        return f"{note}|{category}"

    @strategy(PredictStrategy())
    async def summarize(self, ticket: str) -> str:
        """Write a one-line triage note for {ticket}."""
        ...


@pytest.mark.asyncio
async def test_case_c_standalone_after_method_from_orchestrator(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Case C (the bug): a standalone called *after* a generation method,
    still from the pure-Python orchestrator (turn already closed), must be
    captured — proving the gap is about an open turn, not ordering."""
    output_dir = tmp_path / "atif-c"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "note"}'),  # summarize (FIRST)
            _resp(content='{"value": "bug"}'),  # categorize_ticket (AFTER)
        ]
    )
    agent = _CaseCAgent(llm=fake)
    assert await agent.run("crash on save") == "note|bug"

    traj = _only_trajectory(output_dir, "_CaseCAgent")
    assert_atif_normative(traj)
    names = [c.agent.name for c in (traj.subagent_trajectories or [])]
    assert "categorize_ticket" in names, (
        f"standalone called after a method from the orchestrator was dropped; got {names}"
    )

    assert _atif_exporter_var.get() is None


# ---------------------------------------------------------------------------
# Crash case — the run-scoped ContextVar binding is released even when the
# orchestrator raises after a standalone call.
# ---------------------------------------------------------------------------


class _CrashAgent(Agent, llm=_DEFAULT):
    async def run(self, ticket: str) -> str:
        await categorize_ticket(ticket)
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_crash_releases_contextvar(tmp_path: Path, _isolated_agent_init) -> None:
    """The run-scoped cascade binding is released even when the orchestrator
    raises after a standalone call (authoritative disarm in the wrapper's
    finally / AfterAgentCall)."""
    enable_atif(output_dir=tmp_path / "atif-crash")

    fake = FakeLLMClient(scripted_responses=[_resp(content='{"value": "bug"}')])
    agent = _CrashAgent(llm=fake)

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run("crash on save")

    assert _atif_exporter_var.get() is None, (
        "entrypoint-armed contextvar leaked after the orchestrator crashed"
    )


# ---------------------------------------------------------------------------
# Concurrency — multiple top-level orchestrator runs gathered in one async
# context must NOT cross-contaminate. This is the invariant the entrypoint
# design pivots on: each ``asyncio`` task copies the context at spawn, so each
# run's cascade binding is isolated to its own task.
# ---------------------------------------------------------------------------


class _ConcurrentAgent(Agent, llm=_DEFAULT):
    async def run(self, ticket: str) -> str:
        # Pure-Python orchestrator (no turn of its own). Yield once so the
        # gathered runs actually interleave around the standalone call.
        await asyncio.sleep(0)
        category = await categorize_ticket(ticket)
        await asyncio.sleep(0)
        return await self.summarize(ticket, category)

    @strategy(PredictStrategy())
    async def summarize(self, ticket: str, category: str) -> str:
        """Write a one-line triage note for {ticket} given {category}."""
        ...


@pytest.mark.asyncio
async def test_concurrent_top_level_runs_do_not_cross_contaminate(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Concurrent top-level runs gathered in one async context each embed
    exactly their own standalone (asyncio tasks copy context at spawn, so each
    run's arm/disarm is isolated)."""
    output_dir = tmp_path / "atif-concurrent"
    enable_atif(output_dir=output_dir)

    n = 4
    agents = []
    for i in range(n):
        fake = FakeLLMClient(
            scripted_responses=[
                _resp(content='{"value": "bug"}'),  # categorize_ticket
                _resp(content=f'{{"value": "note-{i}"}}'),  # summarize
            ]
        )
        agents.append(_ConcurrentAgent(llm=fake))

    results = await asyncio.gather(*(a.run(f"ticket {i}") for i, a in enumerate(agents)))
    assert results == [f"note-{i}" for i in range(n)]

    # Var must be fully unwound after all runs complete.
    assert _atif_exporter_var.get() is None

    files = list((output_dir / "_ConcurrentAgent").glob("*.json"))
    assert len(files) == n, f"expected {n} trajectory files, got {len(files)}"

    # Each run's trajectory must embed EXACTLY its own one standalone — not
    # zero (dropped) and not two (another run's standalone leaked in).
    for f in files:
        traj = Trajectory.model_validate_json(f.read_text())
        assert_atif_normative(traj)
        subs = [c.agent.name for c in (traj.subagent_trajectories or [])]
        assert subs == ["categorize_ticket"], (
            f"concurrent run must embed exactly its own standalone; got {subs}"
        )


# ---------------------------------------------------------------------------
# The cascade is now armed by subscribing to the generic BeforeAgentCall /
# AfterAgentCall events, not by an ATIF-specific poke in
# the method wrapper. These guard the two edges of that event-driven seam.
# ---------------------------------------------------------------------------


class _SyncTopLevelAgent(Agent, llm=_DEFAULT):
    def compute(self, x: int) -> int:
        # A SYNC top-level agent method. It fires BeforeAgentCall with
        # is_top_level=True, so ATIF arms — but a sync body cannot await a
        # standalone, and the arm/disarm is balanced within this synchronous
        # span. The var must be back to None afterward.
        return x * 2


@pytest.mark.asyncio
async def test_sync_top_level_arms_and_releases(tmp_path: Path, _isolated_agent_init) -> None:
    """A sync top-level agent method arms then disarms the cascade within its
    synchronous span (it cannot await a standalone), leaving the var at None."""
    enable_atif(output_dir=tmp_path / "atif-sync")
    agent = _SyncTopLevelAgent()

    assert _atif_exporter_var.get() is None
    assert agent.compute(21) == 42
    assert _atif_exporter_var.get() is None, (
        "sync top-level agent call left the standalone-cascade var armed"
    )


@pytest.mark.asyncio
async def test_agent_call_events_do_not_become_trajectory_steps(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """The agent-call events flow through ATIF's wildcard subscription but must
    NOT be rendered as trajectory steps (RUNTIME_EVENT is skipped). A separate
    capture subscriber confirms they really did fire."""
    output_dir = tmp_path / "atif-noleak"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "bug"}'),
            _resp(content='{"value": "note"}'),
        ]
    )
    agent = _CaseBAgent(llm=fake)

    seen: list[str] = []
    agent.event_manager.on("BeforeAgentCall", lambda e: seen.append(e.event_type))
    agent.event_manager.on("AfterAgentCall", lambda e: seen.append(e.event_type))

    assert await agent.run("crash on save") == "note"

    # The events really fired (wildcard delivered them to ATIF too).
    assert "BeforeAgentCall" in seen and "AfterAgentCall" in seen

    traj = _only_trajectory(output_dir, "_CaseBAgent")
    assert_atif_normative(traj)
    # No trajectory step was synthesized from the agent-call events.
    leaked = [
        s
        for s in traj.steps
        if (s.extra or {}).get("event_type") in {"BeforeAgentCall", "AfterAgentCall"}
    ]
    assert not leaked, f"agent-call RUNTIME_EVENTs leaked into trajectory steps: {leaked}"


# ---------------------------------------------------------------------------
# Referenced (not orphaned) standalone delegations. When a standalone is called
# from a pure-Python orchestrator (no enclosing tool call), the exporter
# synthesizes a deterministic-dispatch agent step (ATIF v1.7 §II,
# llm_call_count=0) whose observation carries a subagent_trajectory_ref to the
# embedded child — so the delegation is ordered, attributed, and navigable
# instead of an orphan in subagent_trajectories[].
# ---------------------------------------------------------------------------


def _dispatch_steps(traj: Trajectory) -> list:
    """Return the synthesized standalone-dispatch steps."""
    return [s for s in traj.steps if (s.extra or {}).get("event_kind") == "standalone_dispatch"]


def _ref_ids(step) -> list[str]:
    """All subagent_trajectory_ref.trajectory_id values on a step's observation."""
    ids: list[str] = []
    for result in step.observation.results if step.observation else []:
        for ref in result.subagent_trajectory_ref or []:
            if ref.trajectory_id:
                ids.append(ref.trajectory_id)
    return ids


@pytest.mark.asyncio
async def test_case_b_standalone_first_is_referenced_before_task(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Standalone-first: the dispatch step references the embedded child and is
    ordered BEFORE the generation method's user task (it ran first)."""
    output_dir = tmp_path / "atif-b-ref"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "bug"}'),  # categorize_ticket (FIRST)
            _resp(content='{"value": "note: a bug report"}'),  # summarize (SECOND)
        ]
    )
    agent = _CaseBAgent(llm=fake)
    assert await agent.run("crash on save") == "note: a bug report"

    traj = _only_trajectory(output_dir, "_CaseBAgent")
    assert_atif_normative(traj)  # includes N9 ref-resolvability

    dispatch = _dispatch_steps(traj)
    assert len(dispatch) == 1, f"expected one standalone-dispatch step; got {dispatch}"
    step = dispatch[0]
    assert step.source == "agent" and step.llm_call_count == 0
    assert step.extra.get("subagent_name") == "categorize_ticket"

    # The ref resolves to the embedded categorize_ticket subagent.
    child_id = next(
        c.trajectory_id for c in traj.subagent_trajectories if c.agent.name == "categorize_ticket"
    )
    assert _ref_ids(step) == [child_id]

    # Ordering: dispatch (standalone ran first) precedes the summarize user task.
    task_step = next(s for s in traj.steps if s.source == "user")
    assert step.step_id < task_step.step_id, (
        "standalone-first dispatch step must be ordered before the generation "
        f"method's task (dispatch={step.step_id}, task={task_step.step_id})"
    )


@pytest.mark.asyncio
async def test_case_c_standalone_after_is_referenced_after_turn(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Standalone-after: the dispatch step references the embedded child and is
    ordered AFTER the generation method's agent turn (it ran later)."""
    output_dir = tmp_path / "atif-c-ref"
    enable_atif(output_dir=output_dir)

    fake = FakeLLMClient(
        scripted_responses=[
            _resp(content='{"value": "note"}'),  # summarize (FIRST)
            _resp(content='{"value": "bug"}'),  # categorize_ticket (AFTER)
        ]
    )
    agent = _CaseCAgent(llm=fake)
    assert await agent.run("crash on save") == "note|bug"

    traj = _only_trajectory(output_dir, "_CaseCAgent")
    assert_atif_normative(traj)

    dispatch = _dispatch_steps(traj)
    assert len(dispatch) == 1
    step = dispatch[0]
    child_id = next(
        c.trajectory_id for c in traj.subagent_trajectories if c.agent.name == "categorize_ticket"
    )
    assert _ref_ids(step) == [child_id]

    # Ordering: the summarize generation turn (an agent step that called the
    # LLM) precedes the dispatch step (standalone ran after).
    turn_step = next(s for s in traj.steps if s.source == "agent" and (s.llm_call_count or 0) >= 1)
    assert turn_step.step_id < step.step_id, (
        "standalone-after dispatch step must be ordered after the generation "
        f"method's turn (turn={turn_step.step_id}, dispatch={step.step_id})"
    )
