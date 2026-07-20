# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AtifExporter state machine, driven directly with events.

These tests bypass EventManager and call handlers directly; the
end-to-end install_atif / atif_scope wiring is exercised in
test_end_to_end_codeact.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nooa.atif import Trajectory
from nooa.atif.exporter import AtifExporter
from nooa.context_blocks import ResultStatus
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    Error,
    LLMComplete,
    LLMOutput,
    Notification,
    PythonOutput,
    Reasoning,
    Summary,
    SystemPrompt,
    Task,
)
from tests.atif.normative import assert_atif_normative

# Minimal system-prompt content used by synthetic tests. In a real run the
# runtime fires SystemPrompt right after _build_messages with the rendered
# `messages[0].content`; these tests bypass the runtime so they have to
# emit it explicitly to match the production event sequence (and produce a
# step_id=1 system step).
_TEST_SYSTEM_PROMPT = "You are a test agent."


def _seed_system_prompt(exp: AtifExporter, content: str = _TEST_SYSTEM_PROMPT) -> None:
    """Fire a SystemPrompt event to mirror what the runtime would do.

    Call this BEFORE any on_task/on_before_turn in synthetic tests so the
    trajectory shape matches production: step_id=1 is the system step,
    Tasks land at step_id=2+.
    """
    exp.on_system_prompt(SystemPrompt(content=content, generation_id=""))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter(tmp_path: Path):
    exp = AtifExporter(
        path=tmp_path / "trajectory.json",
        session_id="test-session",
        agent_name="test-agent",
        agent_version="0.1.0",
    )
    try:
        yield exp
    finally:
        # Some tests intentionally fire on_before_turn without a
        # matching on_after_turn (e.g. partial-trajectory-on-disk).
        # Without explicit close(), the run-scoped ContextVar token
        # would leak across tests.
        exp.close()


def _drive_basic_codeact_turn(
    exp: AtifExporter,
    *,
    generation_id: str = "gen-1",
    code: str = "print('hello')",
    stdout: str = "hello\n",
    is_final_after: bool = True,
    fire_system_prompt: bool = True,
) -> None:
    """Push a complete BeforeTurn → LLMComplete → ToolCallEvent → PythonOutput → AfterTurn sequence.

    By default also fires a SystemPrompt before BeforeTurn (matching the
    real runtime order: ``_build_messages → SystemPrompt → LLM call``).
    Pass ``fire_system_prompt=False`` if the caller seeded it already.
    """
    if fire_system_prompt and not exp._system_step_emitted:  # type: ignore[attr-defined]
        _seed_system_prompt(exp)
    exp.on_before_turn(
        BeforeTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id=generation_id,
            parent_generation_id=None,
            turn_number=1,
        )
    )
    exp.on_llm_complete(
        LLMComplete(
            model_name="fake-model",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=10,
            cost_usd=0.001,
            tool_calls=[
                {
                    "tool_call_id": "call_alpha",
                    "function_name": "execute_python",
                    "arguments": json.dumps({"code": code}),
                }
            ],
            reasoning_content="thinking...",
            generation_id=generation_id,
        )
    )
    exp.on_llm_output(LLMOutput(content=""))
    exp.on_tool_call_event(
        ToolCallEvent(
            tool_call_id="call_alpha",
            name="execute_python",
            arguments={"code": code},
        )
    )
    exp.on_python_output(
        PythonOutput(
            tool_call_id="call_alpha",
            execution_count=1,
            stdout=stdout,
            stderr="",
            execution_status=ResultStatus.COMPLETE,
        )
    )
    exp.on_after_turn(
        AfterTurn(
            method_name="run",
            strategy="CodeActStrategy",
            generation_id=generation_id,
            parent_generation_id=None,
            turn_number=1,
            is_final=is_final_after,
            success=True,
        )
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestBasicTurn:
    def test_single_codeact_turn_round_trip(self, exporter: AtifExporter) -> None:
        exporter.on_task(Task(prompt="Say hello in Python."))
        _drive_basic_codeact_turn(exporter)

        traj = exporter.get_trajectory()
        # Schema + normative validators both pass.
        Trajectory.model_validate(json.loads(traj.model_dump_json(exclude_none=True)))
        assert_atif_normative(traj)

        # Three steps: system + user task + agent turn.
        assert [s.source for s in traj.steps] == ["system", "user", "agent"]
        agent = traj.steps[2]
        assert agent.tool_calls is not None
        assert agent.tool_calls[0].tool_call_id == "call_alpha"
        assert agent.observation is not None
        assert agent.observation.results[0].source_call_id == "call_alpha"
        assert "hello" in agent.observation.results[0].content  # type: ignore[arg-type]

        # final_metrics populated.
        assert traj.final_metrics is not None
        assert traj.final_metrics.total_prompt_tokens == 100
        assert traj.final_metrics.total_completion_tokens == 20
        assert traj.final_metrics.total_cost_usd == pytest.approx(0.001)

    def test_writes_file_atomically(self, exporter: AtifExporter, tmp_path: Path) -> None:
        exporter.on_task(Task(prompt="hi"))
        _drive_basic_codeact_turn(exporter)
        loaded = Trajectory.model_validate_json(exporter.path.read_text())
        assert loaded.agent.name == "test-agent"
        # No leftover .tmp file.
        assert not (tmp_path / "trajectory.json.tmp").exists()


# ---------------------------------------------------------------------------
# Joinability — the original observation-dropping regression must pass by
# construction
# ---------------------------------------------------------------------------


class TestJoinabilityByConstruction:
    def test_observation_paired_with_tool_call(self, exporter: AtifExporter) -> None:
        """The fc_*/call_* bridge is unnecessary: both come from LLMComplete + PythonOutput."""
        exporter.on_task(Task(prompt="run"))
        _drive_basic_codeact_turn(exporter)

        traj = exporter.get_trajectory()
        agent = traj.steps[-1]
        assert agent.tool_calls is not None
        assert agent.observation is not None
        tc_id = agent.tool_calls[0].tool_call_id
        result_id = agent.observation.results[0].source_call_id
        assert tc_id == result_id == "call_alpha"


# ---------------------------------------------------------------------------
# Return result tool (no PythonOutput — relies on ToolCallEvent.result)
# ---------------------------------------------------------------------------


class TestReturnResultTool:
    def test_return_result_observation_from_tool_call_event(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="solve"))
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
                prompt_tokens=50,
                completion_tokens=5,
                cost_usd=0.0001,
                tool_calls=[
                    {
                        "tool_call_id": "call_ret",
                        "function_name": "return_result",
                        "arguments": json.dumps({"result": 42}),
                    }
                ],
                generation_id="gen-1",
            )
        )
        # Strategy creates a ToolCallEvent and mutates its .result via update().
        # We simulate that by passing a ToolCallEvent whose result is already set.
        tce = ToolCallEvent(
            tool_call_id="call_ret",
            name="return_result",
            arguments={"result": 42},
            result=ToolResult(
                tool_call_id="call_ret",
                content="Result accepted.",
                result_status=ResultStatus.COMPLETE,
            ),
        )
        exporter.on_tool_call_event(tce)
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

        traj = exporter.get_trajectory()
        # Steps: [0]=system, [1]=user task, [2]=agent
        agent = traj.steps[2]
        assert agent.observation is not None
        assert agent.observation.results[0].source_call_id == "call_ret"
        assert agent.observation.results[0].content == "Result accepted."
        assert_atif_normative(traj)

    def test_tool_call_event_result_mutated_after_capture(self, exporter: AtifExporter) -> None:
        """The event_manager.update() mutates the same object reference; the
        exporter captures the reference at .on_tool_call_event() time and reads
        .result at AfterTurn time, picking up the post-mutation state.
        """
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="solve"))
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
                tool_calls=[
                    {
                        "tool_call_id": "call_mut",
                        "function_name": "return_result",
                        "arguments": "{}",
                    }
                ],
                generation_id="gen-1",
            )
        )
        # ToolCallEvent captured with NO result yet
        tce = ToolCallEvent(
            tool_call_id="call_mut",
            name="return_result",
            arguments={},
        )
        exporter.on_tool_call_event(tce)
        # Strategy mutates via event_manager.update() — simulated by direct assign.
        tce.result = ToolResult(
            tool_call_id="call_mut",
            content="Final answer recorded.",
            result_status=ResultStatus.COMPLETE,
        )
        # AfterTurn now reads the mutated state.
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

        traj = exporter.get_trajectory()
        # Steps: [0]=system, [1]=user task, [2]=agent
        assert traj.steps[2].observation is not None
        assert traj.steps[2].observation.results[0].content == "Final answer recorded."


# ---------------------------------------------------------------------------
# Multi-turn linear sequence
# ---------------------------------------------------------------------------


class TestMultiTurn:
    def test_two_codeact_turns_linear(self, exporter: AtifExporter) -> None:
        exporter.on_task(Task(prompt="task"))
        _drive_basic_codeact_turn(
            exporter, generation_id="gen-1", code="x = 1", stdout="", is_final_after=False
        )
        _drive_basic_codeact_turn(
            exporter, generation_id="gen-2", code="print(x)", stdout="1\n", is_final_after=True
        )

        traj = exporter.get_trajectory()
        sources = [s.source for s in traj.steps]
        assert sources == ["system", "user", "agent", "agent"]
        # Sequential step_ids.
        assert [s.step_id for s in traj.steps] == [1, 2, 3, 4]
        # Totals aggregate across both agent turns.
        assert traj.final_metrics is not None
        assert traj.final_metrics.total_prompt_tokens == 200  # 100 + 100
        assert traj.final_metrics.total_completion_tokens == 40  # 20 + 20
        assert_atif_normative(traj)


# ---------------------------------------------------------------------------
# Compaction (Summary event)
# ---------------------------------------------------------------------------


class TestCompaction:
    def test_summary_event_marks_prior_steps_copied(self, exporter: AtifExporter) -> None:
        exporter.on_task(Task(prompt="long task"))
        _drive_basic_codeact_turn(
            exporter, generation_id="gen-1", code="x = 1", stdout="", is_final_after=False
        )
        # Mid-trajectory compaction.
        exporter.on_summary(
            Summary(
                summary_tag="2..3",
                replaced_range=(2, 3),
                summary_text="Earlier steps computed x=1.",
                children_tags=["2", "3"],
            )
        )
        _drive_basic_codeact_turn(
            exporter, generation_id="gen-2", code="print(x)", stdout="1\n", is_final_after=True
        )

        traj = exporter.get_trajectory()
        # Compaction step is "system" with replace-boundary. (The SystemPrompt
        # at step_id=1 is also source=system, so filter on context_management.)
        compaction = next(
            s
            for s in traj.steps
            if s.source == "system" and s.extra and "context_management" in s.extra
        )
        assert compaction.extra is not None
        cm = compaction.extra["context_management"]
        assert cm["boundary"] == "replace"
        assert cm["type"] == "compaction"
        # Steps before the boundary are marked as copied.
        for prior in traj.steps:
            if prior.step_id < compaction.step_id:
                assert prior.is_copied_context is True
        # The post-compaction step is NOT marked.
        post = [s for s in traj.steps if s.step_id > compaction.step_id]
        assert post and post[0].is_copied_context is None
        assert_atif_normative(traj)


# ---------------------------------------------------------------------------
# Crash safety
# ---------------------------------------------------------------------------


class TestCrashSafety:
    def test_finalize_on_exception_writes_crashed_marker(self, exporter: AtifExporter) -> None:
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
        # Pretend the agent_call raised mid-turn — finalize hook fires.
        exporter.finalize_on_exception(RuntimeError("boom"))

        loaded = Trajectory.model_validate_json(exporter.path.read_text())
        assert loaded.extra is not None
        assert loaded.extra["crashed"] is True
        assert loaded.extra["exception_type"] == "RuntimeError"
        assert "boom" in loaded.extra["exception_message"]
        # final_metrics intentionally absent on crash.
        assert loaded.final_metrics is None

    def test_partial_trajectory_on_disk_is_valid(self, exporter: AtifExporter) -> None:
        """Even mid-turn, the on-disk trajectory parses cleanly."""
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="hi"))
        exporter.on_before_turn(
            BeforeTurn(
                method_name="run",
                strategy="CodeActStrategy",
                generation_id="gen-1",
                parent_generation_id=None,
                turn_number=1,
            )
        )
        # No AfterTurn yet — the system + user steps are on disk; the pending agent step is not.
        loaded = Trajectory.model_validate_json(exporter.path.read_text())
        assert [s.source for s in loaded.steps] == ["system", "user"]


# ---------------------------------------------------------------------------
# Error / Notification side-channel events
# ---------------------------------------------------------------------------


class TestSideChannelEvents:
    def test_error_event_emits_user_step(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="hi"))
        exporter.on_error(Error(content="Retry: invalid output"))
        traj = exporter.get_trajectory()
        sources = [s.source for s in traj.steps]
        assert sources == ["system", "user", "user"]
        assert traj.steps[2].extra is not None
        assert traj.steps[2].extra["event_kind"] == "error"

    def test_notification_event_emits_user_step(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="hi"))
        exporter.on_notification(Notification(source="queue:x", description="new msg"))
        traj = exporter.get_trajectory()
        # Steps: [0]=system, [1]=user task, [2]=user notification
        assert traj.steps[2].source == "user"
        assert traj.steps[2].message == "new msg"
        assert traj.steps[2].extra is not None
        assert traj.steps[2].extra["source"] == "queue:x"


# ---------------------------------------------------------------------------
# Reasoning attachment
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_reasoning_event_attaches_to_pending_step(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="hi"))
        exporter.on_before_turn(
            BeforeTurn(
                method_name="run",
                strategy="CodeActStrategy",
                generation_id="gen-1",
                parent_generation_id=None,
                turn_number=1,
            )
        )
        # Reasoning fires inside the turn (mid execute_python).
        exporter.on_reasoning(Reasoning(content="Step A. "))
        exporter.on_llm_complete(
            LLMComplete(
                model_name="fake-model",
                prompt_tokens=1,
                completion_tokens=1,
                reasoning_content="Initial CoT.",
                generation_id="gen-1",
            )
        )
        # Reasoning appended even though LLMComplete already set reasoning_content.
        exporter.on_reasoning(Reasoning(content="Step B."))
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
        traj = exporter.get_trajectory()
        # Steps: [0]=system, [1]=user task, [2]=agent
        agent = traj.steps[2]
        assert agent.reasoning_content is not None
        assert "Step A." in agent.reasoning_content
        assert "Initial CoT." in agent.reasoning_content
        assert "Step B." in agent.reasoning_content


# ---------------------------------------------------------------------------
# SystemPrompt event + dynamic_context + drift
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """The runtime fires SystemPrompt AFTER Task (render-at-call-time, see
    runtime/actor.py). The exporter buffers Tasks until the first
    SystemPrompt arrives, then flushes them so the on-disk shape is
    always [system, user, ...]."""

    def test_task_buffered_then_flushed_on_system_prompt(self, exporter: AtifExporter) -> None:
        # Task arrives BEFORE SystemPrompt.
        exporter.on_task(Task(prompt="hello"))
        # Nothing on disk yet — Task is buffered.
        traj = exporter.get_trajectory()
        assert traj.steps == []

        # SystemPrompt fires → system step at step_id=1, Task flushes at step_id=2.
        _seed_system_prompt(exporter, content="You are a helpful assistant.")
        traj = exporter.get_trajectory()
        assert [s.source for s in traj.steps] == ["system", "user"]
        assert [s.step_id for s in traj.steps] == [1, 2]
        assert traj.steps[0].message == "You are a helpful assistant."
        assert traj.steps[1].message == "hello"

    def test_multiple_tasks_buffered_then_flushed_in_order(self, exporter: AtifExporter) -> None:
        exporter.on_task(Task(prompt="first"))
        exporter.on_task(Task(prompt="second"))
        _seed_system_prompt(exporter)
        traj = exporter.get_trajectory()
        assert [s.source for s in traj.steps] == ["system", "user", "user"]
        assert traj.steps[1].message == "first"
        assert traj.steps[2].message == "second"

    def test_task_after_system_prompt_emits_directly(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter)
        exporter.on_task(Task(prompt="hello"))
        traj = exporter.get_trajectory()
        assert [s.source for s in traj.steps] == ["system", "user"]

    def test_identical_system_prompt_not_re_emitted(self, exporter: AtifExporter) -> None:
        _seed_system_prompt(exporter, content="Same content")
        _seed_system_prompt(exporter, content="Same content")
        traj = exporter.get_trajectory()
        # Only one system step.
        assert [s.source for s in traj.steps] == ["system"]

    def test_system_prompt_drift_annotates_next_agent_step(self, exporter: AtifExporter) -> None:
        """A later SystemPrompt with DIFFERENT content is treated as drift;
        the next agent step's extra carries system_prompt_changed=True."""
        _seed_system_prompt(exporter, content="Original system prompt")
        exporter.on_task(Task(prompt="hi"))
        # New LLM call sees a different system prompt (e.g. a dynamic static
        # block mutated). The runtime fires SystemPrompt again with the new
        # content right before LLMComplete.
        _seed_system_prompt(exporter, content="Drifted system prompt")
        _drive_basic_codeact_turn(exporter, fire_system_prompt=False)

        traj = exporter.get_trajectory()
        agent = traj.steps[-1]
        assert agent.source == "agent"
        assert agent.extra is not None
        assert agent.extra.get("system_prompt_changed") is True
        assert agent.extra.get("system_prompt") == "Drifted system prompt"

    def test_no_system_prompt_ever_flushes_tasks_on_close(self, tmp_path: Path) -> None:
        """If SystemPrompt never arrives (synthetic tests, agents that bypass
        the LLM entirely), close() still flushes buffered Tasks so the
        on-disk trajectory records the user prompt."""
        exp = AtifExporter(
            path=tmp_path / "no-system.json",
            session_id="no-sys",
            agent_name="NoSysAgent",
            agent_version="0.1.0",
        )
        exp.on_task(Task(prompt="hello"))
        exp.close()
        traj = exp.get_trajectory()
        assert [s.source for s in traj.steps] == ["user"]
        assert traj.steps[0].message == "hello"
