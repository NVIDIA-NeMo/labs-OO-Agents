# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ATIF export through install_atif / atif_scope.

A real CodeAct agent runs with a stubbed LLM (no network); the
exporter is attached via ``atif_scope``; the resulting on-disk
trajectory passes both Pydantic schema validation and the normative
rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope, install_atif
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _exec_tool_call(code: str, call_id: str = "call_exec") -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _ret_tool_call(result: object, call_id: str = "call_ret") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _resp(
    content: str = "", tool_calls: list | None = None, *, usage: dict | None = None
) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
        usage=usage or {"prompt_tokens": 50, "completion_tokens": 10},
    )


_TEST_LLM_DEFAULT = FakeLLMClient()


class CodeActAgent(Agent, llm=_TEST_LLM_DEFAULT):
    """Minimal CodeAct agent for end-to-end testing."""

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompt: str) -> int:
        """Solve {prompt}."""
        ...


# ---------------------------------------------------------------------------
# Single-turn end-to-end (Case A — but via CodeAct, not PredictStrategy)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_codeact_turn_end_to_end(tmp_path: Path) -> None:
    """A single CodeAct turn (execute_python + return_result) writes a
    valid, normatively-correct ATIF trajectory to disk.
    """
    llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                tool_calls=[
                    _exec_tool_call("x = 1 + 1", call_id="call_exec1"),
                    _ret_tool_call(2, call_id="call_ret1"),
                ]
            ),
        ]
    )
    agent = CodeActAgent(llm=llm)

    out = tmp_path / "trajectory.json"
    async with atif_scope(
        agent,
        path=out,
        session_id="end2end-1",
        agent_name="CodeActAgent",
        agent_version="0.1.0",
    ) as _exporter:
        result = await agent.run("compute 1+1")

    assert result == 2

    # File exists and parses through the schema.
    loaded = Trajectory.model_validate_json(out.read_text())
    assert loaded.agent.name == "CodeActAgent"
    assert loaded.session_id == "end2end-1"
    # At least one user step (the Task) + one or more agent steps.
    sources = [s.source for s in loaded.steps]
    assert "user" in sources
    assert "agent" in sources

    # Normative rules pass.
    assert_atif_normative(loaded)

    # Agent step carries the two tool_calls with canonical ids; observation
    # for execute_python is present.
    agent_step = next(s for s in loaded.steps if s.source == "agent")
    assert agent_step.tool_calls is not None
    tc_ids = [tc.tool_call_id for tc in agent_step.tool_calls]
    assert "call_exec1" in tc_ids
    assert "call_ret1" in tc_ids
    assert agent_step.observation is not None
    obs_ids = {r.source_call_id for r in agent_step.observation.results}
    # Both should appear — execute_python via PythonOutput, return_result via
    # ToolCallEvent.result.content.
    assert "call_exec1" in obs_ids
    assert "call_ret1" in obs_ids

    # Final metrics summed.
    assert loaded.final_metrics is not None
    assert loaded.final_metrics.total_steps == len(loaded.steps)


# ---------------------------------------------------------------------------
# install_atif low-level path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_atif_returns_callable_uninstall(tmp_path: Path) -> None:
    llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_ret_tool_call(42, call_id="call_ret_x")]),
        ]
    )
    agent = CodeActAgent(llm=llm)

    out = tmp_path / "trajectory.json"
    uninstall = install_atif(
        agent.event_manager,
        path=out,
        session_id="lowlvl-1",
        agent_name="CodeActAgent",
        agent_version="0.1.0",
    )
    try:
        assert await agent.run("answer") == 42
    finally:
        uninstall()

    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)


# ---------------------------------------------------------------------------
# Crash safety end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crash_inside_scope_marks_trajectory(tmp_path: Path) -> None:
    """If the wrapped block raises, the on-disk trajectory carries the crash marker."""
    llm = FakeLLMClient(
        scripted_responses=[_resp(tool_calls=[_ret_tool_call(1, call_id="call_ret")])]
    )
    agent = CodeActAgent(llm=llm)

    out = tmp_path / "trajectory.json"

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with atif_scope(
            agent, path=out, session_id="crashy", agent_name="CodeActAgent", agent_version="0.1.0"
        ):
            await agent.run("ok")
            raise _Boom("scope-level error")

    loaded = Trajectory.model_validate_json(out.read_text())
    assert loaded.extra is not None
    assert loaded.extra["crashed"] is True
    assert loaded.extra["exception_type"] == "_Boom"


# ---------------------------------------------------------------------------
# The original observation-dropping regression. Resolved by construction in
# this design, but tested end-to-end here to pin the resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observation_paired_end_to_end(tmp_path: Path) -> None:
    """End-to-end pin: every tool_call has its matching observation result
    (the joinability invariant). The fc_*/call_* bridge is unnecessary;
    LLMComplete + PythonOutput route everything by canonical call_* id.
    """
    llm = FakeLLMClient(
        scripted_responses=[
            _resp(tool_calls=[_exec_tool_call("y = 10", call_id="call_exec")]),
            _resp(tool_calls=[_ret_tool_call(10, call_id="call_ret")]),
        ]
    )
    agent = CodeActAgent(llm=llm)

    out = tmp_path / "trajectory.json"
    async with atif_scope(
        agent, path=out, session_id="e2e-joinable", agent_name="CodeActAgent", agent_version="0.1.0"
    ):
        assert await agent.run("two-turn") == 10

    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)
    # Every tool_call has a matching observation result.
    for step in loaded.steps:
        if not step.tool_calls:
            continue
        assert step.observation is not None, (
            f"step_id={step.step_id} has tool_calls but no observation"
        )
        tc_ids = {tc.tool_call_id for tc in step.tool_calls}
        obs_ids = {r.source_call_id for r in step.observation.results}
        assert tc_ids.issubset(obs_ids), (
            f"step_id={step.step_id}: missing observation result for tool_calls {tc_ids - obs_ids}"
        )


@pytest.mark.asyncio
async def test_canonical_call_id_used_when_both_ids_present(tmp_path: Path) -> None:
    """Regression pin for the original OpenAI Responses trace shape.

    OpenAI Responses produces two ids per tool call: a model-emitted
    ``fc_*`` function-call id and a runtime ``call_*`` runtime id. The
    exporter reads from ``LLMResponse.tool_calls`` whose ``id`` is set to
    the canonical ``call_*`` by ``unifiedllm.ResponsesClient``. This test
    simulates an LLMResponse-shape that carries BOTH ids (the
    ``raw_response`` mirrors the OpenAI Responses output: ``fc_*`` as
    ``id``, ``call_*`` as ``call_id``) and asserts the trajectory shows
    ONLY the canonical ``call_*`` form — no leak of the model-emitted
    ``fc_*``.

    Guards against a regression where the unified wrapper picks ``id``
    (fc_*) instead of ``call_id`` (call_*), or the exporter reads from
    ``raw_response`` directly.
    """
    # The framework converts to ``ToolCall(id=call_id, ...)`` — so the
    # tool_call we hand to FakeLLMClient already uses the canonical id.
    # We additionally stuff a raw_response with BOTH ids to verify
    # neither the strategy nor the exporter reaches into the raw shape
    # and picks up the fc_* id.
    raw_response = {
        "output": [
            {
                "type": "function_call",
                "id": "fc_MODEL_EMITTED",  # model-emitted; SHOULD NOT leak
                "call_id": "call_RUNTIME_ID",  # canonical; SHOULD be used
                "name": "return_result",
                "arguments": '{"result": 42}',
            }
        ]
    }
    response = LLMResponse(
        raw_response=raw_response,
        content="",
        tool_calls=[
            ToolCall(
                id="call_RUNTIME_ID",  # canonical (matches unifiedllm.Responses)
                name="return_result",
                arguments=json.dumps({"result": 42}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
        usage={"prompt_tokens": 5, "completion_tokens": 2},
    )
    llm = FakeLLMClient(scripted_responses=[response])
    agent = CodeActAgent(llm=llm)

    out = tmp_path / "trajectory.json"
    async with atif_scope(
        agent,
        path=out,
        session_id="e2e-both-ids",
        agent_name="CodeActAgent",
        agent_version="0.1.0",
    ):
        assert await agent.run("answer") == 42

    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)

    # No fc_* leak ANYWHERE in the trajectory — every id we look at
    # should be the canonical call_*.
    all_ids: set[str] = set()
    for step in loaded.steps:
        for tc in step.tool_calls or []:
            all_ids.add(tc.tool_call_id)
        if step.observation is not None:
            for result in step.observation.results:
                if result.source_call_id is not None:
                    all_ids.add(result.source_call_id)

    fc_leak = {x for x in all_ids if x.startswith("fc_")}
    assert not fc_leak, f"regression: model-emitted fc_* id leaked into trajectory: {fc_leak}"
    assert "call_RUNTIME_ID" in all_ids, (
        f"Canonical call_* id missing from trajectory; got {all_ids}"
    )
