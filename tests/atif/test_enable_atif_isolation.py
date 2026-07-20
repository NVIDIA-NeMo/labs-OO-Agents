# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``enable_atif()`` agent isolation.

``install_atif()``'s unconditional ``_atif_exporter_var.set(...)`` would,
in the ``enable_atif()`` auto-install path, leak the most-recently-
constructed agent's exporter into the ContextVar — causing standalone
generation calls inside agent A to be attributed to agent B's trajectory
whenever the two agents shared an async context.

Fix: ``install_atif()`` now takes ``cascade_to_standalones=False`` as
the default; only :func:`atif_scope` opts in. :func:`enable_atif`'s
patched ``Agent.__init__`` does NOT touch the ContextVar — each agent
owns its own trajectory file, and standalone calls fall through to
their OWN auto-installed trajectory (or no trajectory if
``enable_atif`` wasn't called).

These tests pin both halves of the fix:
- ContextVar is NOT set after ``enable_atif()`` + agent construction.
- Standalone calls from inside agent A do not appear in agent B's
  trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.atif import (
    Trajectory,
    atif_scope,
    enable_atif,
    install_atif,
)
from nooa.config import CodeActConfig
from nooa.standalone import _atif_exporter_var
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

# ---------------------------------------------------------------------------
# Helpers
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


_DEFAULT = FakeLLMClient()


# ---------------------------------------------------------------------------
# Unit-level: install_atif's cascade flag
# ---------------------------------------------------------------------------


class TestCascadeFlag:
    """``install_atif(cascade_to_standalones=False)`` does NOT touch the var."""

    def test_default_does_not_set_contextvar(self, tmp_path: Path) -> None:
        before = _atif_exporter_var.get()

        class _A(Agent, llm=_DEFAULT):
            pass

        a = _A()
        uninstall = install_atif(
            a.event_manager,
            path=tmp_path / "a.json",
            session_id="s",
            agent_name="A",
            agent_version="0",
        )
        try:
            after = _atif_exporter_var.get()
            assert after is before, (
                "install_atif must not touch _atif_exporter_var when "
                "cascade_to_standalones is False (default)."
            )
        finally:
            uninstall()

    def test_cascade_true_does_set_contextvar(self, tmp_path: Path) -> None:
        before = _atif_exporter_var.get()

        class _B(Agent, llm=_DEFAULT):
            pass

        b = _B()
        uninstall = install_atif(
            b.event_manager,
            path=tmp_path / "b.json",
            session_id="s",
            agent_name="B",
            agent_version="0",
            cascade_to_standalones=True,
        )
        try:
            after = _atif_exporter_var.get()
            assert after is not None
            assert after is not before
        finally:
            uninstall()
        # Reset on uninstall.
        assert _atif_exporter_var.get() is before


# ---------------------------------------------------------------------------
# atif_scope still cascades
# ---------------------------------------------------------------------------


@strategy(PredictStrategy())
async def _classify_iso(text: str) -> Literal["pos", "neg"]:
    """Classify {text}."""
    ...


class _ScopedCaller(Agent, llm=_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
    async def run(self, prompt: str) -> str:
        """{prompt}"""
        ...


@pytest.mark.asyncio
async def test_atif_scope_still_cascades_standalone_into_subagent(tmp_path: Path) -> None:
    """``atif_scope`` opts in to the cascade, so standalone calls inside
    the block still attach as ``subagent_trajectories[]``. Existing
    Phase-4 behaviour must not regress.
    """
    fake = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec("r = await _classify_iso('great')", call_id="call_exec"),
                    _ret("pos", call_id="call_ret"),
                ]
            ),
            _resp(content='{"value": "pos"}'),
        ]
    )
    agent = _ScopedCaller(llm=fake)
    out = tmp_path / "traj.json"
    async with atif_scope(
        agent, path=out, session_id="cascade-on", agent_name="_ScopedCaller", agent_version="0.1"
    ):
        assert await agent.run("classify") == "pos"

    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)
    # Standalone classify_iso lives under subagent_trajectories[].
    assert loaded.subagent_trajectories is not None
    assert len(loaded.subagent_trajectories) == 1


# ---------------------------------------------------------------------------
# The regression: enable_atif must not cross-contaminate
# ---------------------------------------------------------------------------


@strategy(PredictStrategy())
async def _classify_a(text: str) -> Literal["pos", "neg"]:
    """Classify {text} (called by AgentA only)."""
    ...


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


@pytest.mark.asyncio
async def test_enable_atif_two_agents_one_combined_file_per_run(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Two agents under ``enable_atif()`` in the same async context.

    Both halves of the contract must hold:

    1. **Single combined file per agent.run()** (the "JSONL-style"
       trace contract): a standalone generation called inside AgentA's
       run lands as a ``subagent_trajectory`` under AgentA's
       trajectory — one file, complete picture.
    2. **No cross-contamination**: the standalone work does NOT appear
       under AgentB's trajectory, even though AgentB was constructed
       AFTER AgentA in the same async context.

    The fix that makes both work: the ContextVar cascade is pushed by
    :meth:`AtifExporter.on_before_turn` at the top-level turn boundary
    (turn_number=1, parent_generation_id is None) and popped by
    :meth:`on_after_turn` at the matching final turn — lifetime scoped
    to one agent_call run, not to agent construction.
    """
    output_dir = tmp_path / "atif-out"
    enable_atif(output_dir=output_dir)

    # Before any run starts, the ContextVar is unset (construction
    # does NOT touch it any more).
    assert _atif_exporter_var.get() is None

    fake_a = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec("r = await _classify_a('hi')", call_id="call_a_exec"),
                    _ret("pos", call_id="call_a_ret"),
                ]
            ),
            _resp(content='{"value": "pos"}'),
        ]
    )
    fake_b = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_ret(0, call_id="call_b_ret")]),
        ]
    )

    class _AgentA(Agent, llm=_DEFAULT):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
        async def run(self, prompt: str) -> str:
            """{prompt}"""
            ...

    class _AgentB(Agent, llm=_DEFAULT):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
        async def run(self, prompt: str) -> int:
            """{prompt}"""
            ...

    # Construct both agents BEFORE either runs: the failure mode is a second
    # construct overwriting the contextvar set by the first.
    agent_a = _AgentA(llm=fake_a)
    _agent_b = _AgentB(llm=fake_b)  # construction matters — see docstring

    # ContextVar still untouched: construction is no longer the binding
    # point.
    assert _atif_exporter_var.get() is None

    # Run agent A — the standalone _classify_a is called inside.
    result_a = await agent_a.run("classify it")
    assert result_a == "pos"

    # After the run completes, the ContextVar must be unset again
    # (on_after_turn popped the run-scoped token).
    assert _atif_exporter_var.get() is None, (
        "Run-scoped contextvar token was not reset on AfterTurn — would "
        "leak into the next agent's run."
    )

    # (1) AgentA has one file, containing the FULL picture: the
    # CodeAct turn AND the standalone _classify_a as a
    # subagent_trajectory.
    a_dir = output_dir / "_AgentA"
    assert a_dir.exists()
    a_files = list(a_dir.glob("*.json"))
    assert len(a_files) == 1, f"expected exactly one AgentA trajectory, got {a_files}"
    a_traj = Trajectory.model_validate_json(a_files[0].read_text())
    assert_atif_normative(a_traj)
    assert a_traj.agent.name == "_AgentA"

    # The standalone classify call is embedded as a subagent.
    assert a_traj.subagent_trajectories is not None, (
        "Expected enable_atif() to cascade the standalone call as a "
        "subagent_trajectory under AgentA's trajectory (one combined "
        "file per agent.run, matching the JSONL-trace contract)."
    )
    subagent_names = [child.agent.name for child in a_traj.subagent_trajectories]
    assert "_classify_a" in subagent_names, (
        f"Expected _classify_a as a subagent of AgentA; got {subagent_names}"
    )

    # (2) No AgentB directory yet — agent_b never ran.  Even if a
    # _classify_a directory exists from prior runs, the FRESH standalone
    # call here must not have been written as a top-level file (it
    # cascaded into AgentA's trajectory instead).
    b_dir = output_dir / "_AgentB"
    if b_dir.exists():
        for bf in b_dir.glob("*.json"):
            b_traj = Trajectory.model_validate_json(bf.read_text())
            # If AgentB has a trajectory at all, it must be empty —
            # AgentB never ran.
            assert len(b_traj.steps) == 0, (
                "AgentB never ran but its trajectory has steps — "
                f"cross-contamination from AgentA. Got {b_traj.steps}"
            )


@pytest.mark.asyncio
async def test_enable_atif_contextvar_stays_unset_between_runs(
    tmp_path: Path, _isolated_agent_init
) -> None:
    """Sequential agent runs under ``enable_atif()`` cleanly push/pop
    the ContextVar — it is bound DURING each run and unset between runs.

    This is the property that prevents cross-agent leakage: the
    ContextVar is scoped to the active agent_call frame, not to the
    Python session.
    """
    enable_atif(output_dir=tmp_path)

    class _Probe(Agent, llm=_DEFAULT):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
        async def run(self, x: int) -> int:
            """double {x}."""
            ...

    # Construction does NOT touch the contextvar.
    assert _atif_exporter_var.get() is None
    for _ in range(5):
        _Probe()
        assert _atif_exporter_var.get() is None, (
            "Constructing an agent under enable_atif() must not bind the "
            "_atif_exporter_var ContextVar (would cause cross-agent "
            "standalone cascade contamination)."
        )

    # Three sequential runs — between each, the ContextVar must reset
    # to None.
    for trial in range(3):
        fake = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_ret(trial * 2, call_id=f"call_ret_{trial}")]),
            ]
        )
        agent = _Probe(llm=fake)
        assert _atif_exporter_var.get() is None, (
            f"Trial {trial}: contextvar leaked from previous run"
        )
        assert await agent.run(trial) == trial * 2
        assert _atif_exporter_var.get() is None, (
            f"Trial {trial}: run-scoped contextvar token was not reset on AfterTurn"
        )
