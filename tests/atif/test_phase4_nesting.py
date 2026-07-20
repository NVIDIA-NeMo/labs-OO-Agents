# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nesting (Case B), standalone subagents (Case C), gather (Case D),
multimodal (images), and crash safety.

Uses end-to-end agent runs with stubbed LLMs (no network).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from nooa import Agent, strategy
from nooa.atif import Trajectory, atif_scope
from nooa.atif.exporter import AtifExporter
from nooa.config import CodeActConfig
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    LLMComplete,
    LLMOutput,
    SystemPrompt,
    Task,
)
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from tests.atif.normative import assert_atif_normative

# ---------------------------------------------------------------------------
# Shared test helpers
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


# ---------------------------------------------------------------------------
# Case B — same-agent nested generation (B-flatten)
# ---------------------------------------------------------------------------


class NestedGenAgent(Agent, llm=_TEST_LLM_DEFAULT):
    """Outer CodeAct method calls an inner generation method on the same agent."""

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def outer(self, x: str) -> str:
        """Process {x} via inner method."""
        ...

    @strategy(PredictStrategy())
    async def inner_classify(self, text: str) -> Literal["a", "b"]:
        """Classify {text}."""
        ...


@pytest.mark.asyncio
async def test_case_b_same_agent_nested_flattens(tmp_path: Path) -> None:
    """Same-agent nested generation produces flattened steps with
    ``extra.parent_generation_id`` linking inner steps to the outer turn."""
    fake = FakeLLMClient(
        scripted_responses=[
            # Outer CodeAct: execute_python that calls self.inner_classify, then return_result.
            _resp(
                tool_calls=[
                    _exec_tool_call(
                        "result = await self.inner_classify('hello')", call_id="call_exec1"
                    ),
                    _ret_tool_call("a", call_id="call_ret1"),
                ]
            ),
            # Inner Predict: classify returns "a".
            _resp(content='{"value": "a"}'),
        ]
    )
    agent = NestedGenAgent(llm=fake)
    out = tmp_path / "traj.json"

    async with atif_scope(
        agent, path=out, session_id="case-b", agent_name="NestedGenAgent", agent_version="0.1"
    ):
        result = await agent.outer("hi")

    assert result == "a"
    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)

    # subagent_trajectories[] should be EMPTY — Case B flattens.
    assert not loaded.subagent_trajectories, (
        "Case B (same-agent nested) should flatten, not embed: "
        f"got subagent_trajectories={loaded.subagent_trajectories}"
    )
    # At least two agent steps in the flat trajectory (outer + inner_classify).
    agent_steps = [s for s in loaded.steps if s.source == "agent"]
    assert len(agent_steps) >= 2
    # Inner step carries parent_generation_id linking back to outer.
    inner_steps = [s for s in agent_steps if s.extra and s.extra.get("parent_generation_id")]
    assert inner_steps, (
        f"Expected at least one inner step with extra.parent_generation_id; "
        f"got agent steps {[(s.step_id, s.extra) for s in agent_steps]}"
    )


# ---------------------------------------------------------------------------
# Case C — standalone generation function spawns child trajectory
# ---------------------------------------------------------------------------


@strategy(PredictStrategy())
async def _classify_one(text: str) -> Literal["pos", "neg"]:
    """Classify the sentiment of {text}."""
    ...


class StandaloneCallerAgent(Agent, llm=_TEST_LLM_DEFAULT):
    """CodeAct method that calls a standalone generation function."""

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompt: str) -> str:
        """{prompt}"""
        ...


@pytest.mark.asyncio
async def test_case_c_standalone_embeds_subagent(tmp_path: Path) -> None:
    """A standalone @strategy function called from inside CodeAct emits a
    subagent_trajectory under root.subagent_trajectories[], referenced by
    a SubagentTrajectoryRef on the enclosing tool_call's observation."""
    fake = FakeLLMClient(
        scripted_responses=[
            # Outer CodeAct: execute_python calls the standalone function, then return_result.
            _resp(
                tool_calls=[
                    _exec_tool_call("result = await _classify_one('great')", call_id="call_exec_c"),
                    _ret_tool_call("pos", call_id="call_ret_c"),
                ]
            ),
            # Inner standalone Predict.
            _resp(content='{"value": "pos"}'),
        ]
    )
    agent = StandaloneCallerAgent(llm=fake)
    out = tmp_path / "traj.json"

    async with atif_scope(
        agent,
        path=out,
        session_id="case-c",
        agent_name="StandaloneCallerAgent",
        agent_version="0.1",
    ):
        result = await agent.run("classify 'great'")

    assert result == "pos"
    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)

    # Exactly one child trajectory under subagent_trajectories[].
    assert loaded.subagent_trajectories is not None
    assert len(loaded.subagent_trajectories) == 1, (
        f"Expected exactly 1 standalone subagent; got {len(loaded.subagent_trajectories)}"
    )
    child = loaded.subagent_trajectories[0]
    assert child.trajectory_id  # required for embedded refs
    assert child.agent.name == "_classify_one"

    # The enclosing execute_python tool_call's observation hosts a ref.
    refs_found: list[str] = []
    for step in loaded.steps:
        if step.observation is None:
            continue
        for result_obj in step.observation.results:
            for ref in result_obj.subagent_trajectory_ref or []:
                if ref.trajectory_id:
                    refs_found.append(ref.trajectory_id)
    assert child.trajectory_id in refs_found, (
        f"Standalone child trajectory_id {child.trajectory_id!r} not found in "
        f"any observation's subagent_trajectory_ref; refs_found={refs_found}"
    )


# ---------------------------------------------------------------------------
# Case D.2 — asyncio.gather over standalone generation functions
# ---------------------------------------------------------------------------


@strategy(PredictStrategy())
async def _classify_gather(text: str) -> Literal["pos", "neg", "neu"]:
    """Classify {text}."""
    ...


class GatherCallerAgent(Agent, llm=_TEST_LLM_DEFAULT):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def run(self, prompts: list[str]) -> list[str]:
        """Run {prompts} via gather."""
        ...


@pytest.mark.asyncio
async def test_case_d2_gather_over_standalones_yields_n_subagents(tmp_path: Path) -> None:
    """asyncio.gather over N standalone calls produces N subagent
    trajectories under root.subagent_trajectories[]."""
    GATHER_CODE = "results = await asyncio.gather(*(_classify_gather(t) for t in ['x', 'y', 'z']))"
    fake = FakeLLMClient(
        scripted_responses=[
            # CodeAct: execute_python with gather, then return_result.
            _resp(
                tool_calls=[
                    _exec_tool_call(GATHER_CODE, call_id="call_exec_d"),
                    _ret_tool_call(["pos", "neg", "neu"], call_id="call_ret_d"),
                ]
            ),
            # Three Predict responses for the three gathered calls.
            _resp(content='{"value": "pos"}'),
            _resp(content='{"value": "neg"}'),
            _resp(content='{"value": "neu"}'),
        ]
    )
    agent = GatherCallerAgent(llm=fake)
    out = tmp_path / "traj.json"

    async with atif_scope(
        agent, path=out, session_id="case-d2", agent_name="GatherCallerAgent", agent_version="0.1"
    ):
        result = await agent.run(["x", "y", "z"])

    assert result == ["pos", "neg", "neu"]
    loaded = Trajectory.model_validate_json(out.read_text())
    assert_atif_normative(loaded)

    # Three concurrent standalones → three subagent trajectories.
    assert loaded.subagent_trajectories is not None
    assert len(loaded.subagent_trajectories) == 3
    # All have distinct trajectory_ids.
    ids = {child.trajectory_id for child in loaded.subagent_trajectories}
    assert len(ids) == 3, f"trajectory_ids must be unique: {ids}"


# ---------------------------------------------------------------------------
# Multimodal — Task.images flows into ContentPart[]
# ---------------------------------------------------------------------------


_TINY_PNG_BASE64 = (
    # 1×1 transparent PNG.
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAICAQCRz3qXAAAAAElFTkSuQmCC"
)


def test_multimodal_task_image_rendered_as_content_parts(tmp_path: Path) -> None:
    """Synthetic event drive: Task(images=[...]) ⇒ message is ContentPart[]
    with an image entry; the image bytes are written under images/."""
    exporter = AtifExporter(
        path=tmp_path / "traj.json",
        session_id="mm",
        agent_name="MmAgent",
        agent_version="0.1",
    )
    exporter.on_system_prompt(SystemPrompt(content="You are a vision agent.", generation_id=""))
    exporter.on_task(
        Task(
            prompt="Describe this picture.",
            images=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_TINY_PNG_BASE64}"},
                }
            ],
        )
    )
    # Open + immediately finalize an empty agent turn so final_metrics is set.
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
        )
    )
    exporter.on_llm_complete(
        LLMComplete(
            model_name="fake-model",
            prompt_tokens=1,
            completion_tokens=1,
            generation_id="gen-1",
        )
    )
    exporter.on_llm_output(LLMOutput(content="A 1x1 transparent pixel."))
    exporter.on_after_turn(
        AfterTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
    )

    loaded = Trajectory.model_validate_json(exporter.path.read_text())
    assert_atif_normative(loaded)

    user_step = next(s for s in loaded.steps if s.source == "user")
    # message is a list[ContentPart] in multimodal form.
    assert isinstance(user_step.message, list), f"expected list, got {type(user_step.message)}"
    types = [cp.type for cp in user_step.message]
    assert "text" in types and "image" in types
    image_cp = next(cp for cp in user_step.message if cp.type == "image")
    assert image_cp.source is not None
    assert image_cp.source.media_type == "image/png"
    # File materialized on disk.
    img_path = tmp_path / image_cp.source.path
    assert img_path.exists(), f"image file not written to {img_path}"
    assert img_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Crash safety — verifies the exporter's atomic writes survive an event
# mid-stream (the end-to-end atif_scope path is covered in
# test_end_to_end_codeact.py).
# ---------------------------------------------------------------------------


def test_crash_mid_turn_writes_partial_then_marks(tmp_path: Path) -> None:
    """Crash between BeforeTurn and AfterTurn: on-disk trajectory is parsable,
    has the user step but not the agent step, and carries the crashed marker
    after finalize_on_exception()."""
    exporter = AtifExporter(
        path=tmp_path / "traj.json",
        session_id="crashy",
        agent_name="CrashyAgent",
        agent_version="0.1",
    )
    exporter.on_task(Task(prompt="risky"))
    exporter.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id="gen-1",
            parent_generation_id=None,
            turn_number=1,
        )
    )
    # Simulate an exception inside the turn — atif_scope wraps this in production.
    exporter.finalize_on_exception(RuntimeError("boom"))

    loaded = Trajectory.model_validate_json(exporter.path.read_text())
    # User step is on disk; agent step is not (turn never completed).
    assert [s.source for s in loaded.steps] == ["user"]
    assert loaded.extra is not None
    assert loaded.extra["crashed"] is True
    assert loaded.extra["exception_type"] == "RuntimeError"
    assert loaded.final_metrics is None
