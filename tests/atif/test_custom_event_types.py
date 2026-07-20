# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF exporter handles custom EventBase subclasses via wildcard dispatch.

A hardcoded allow-list of event_type strings would silently drop any
user-defined events. The exporter instead subscribes via ``on("*", ...)``
and routes unknown event types by ``_role``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

import pytest
from pydantic import Field

from nooa.atif import Trajectory
from nooa.atif.exporter import AtifExporter
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.roles import Role
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    LLMComplete,
    LLMOutput,
    SystemPrompt,
    Task,
)
from tests.atif.normative import assert_atif_normative

# ---------------------------------------------------------------------------
# Custom event definitions (would normally live in user code)
# ---------------------------------------------------------------------------


class _CustomUserEvent(EventBase):  # type: ignore[misc]
    """User-role custom event — should appear in the trajectory."""

    _role: ClassVar[Role] = Role.USER
    payload: Annotated[str, Field(description="Custom user payload")] = ""


class _CustomAssistantEvent(EventBase):  # type: ignore[misc]
    """Assistant-role custom event — should appear as an agent step."""

    _role: ClassVar[Role] = Role.ASSISTANT
    summary: Annotated[str, Field(description="Custom assistant summary")] = ""


class _CustomToolEvent(EventBase):  # type: ignore[misc]
    """Tool-role custom event — closest spec analog is a user step."""

    _role: ClassVar[Role] = Role.TOOL
    output: Annotated[str, Field(description="Custom tool output")] = ""


class _CustomMetadataEvent(EventBase):  # type: ignore[misc]
    """Metadata-role custom event — internal, MUST NOT appear in trajectory."""

    _role: ClassVar[Role] = Role.METADATA
    note: Annotated[str, Field(description="Internal metadata, not LLM-visible")] = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _setup_exporter(tmp_path: Path, *, seed_system: bool = True) -> AtifExporter:
    """Build an exporter ready for synthetic event dispatch.

    By default fires a :class:`SystemPrompt` immediately so subsequent
    :class:`Task` events emit user steps directly (not buffered).
    """
    exporter = AtifExporter(
        path=tmp_path / "trajectory.json",
        session_id="custom-events",
        agent_name="CustomEventAgent",
        agent_version="0.1.0",
    )
    if seed_system:
        exporter.on_system_prompt(SystemPrompt(content="You are a test agent.", generation_id=""))
    return exporter


class TestCustomEventDispatch:
    """``EventManager.on('*', ...)`` wildcard + role-based fallback."""

    def test_custom_user_event_renders_as_user_step(self, tmp_path: Path) -> None:
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="root task"))
        exporter._dispatch_event(_CustomUserEvent(payload="hello from a custom event"))

        traj = exporter.get_trajectory()
        assert [s.source for s in traj.steps] == ["system", "user", "user"]
        custom = traj.steps[-1]
        assert "hello from a custom event" in custom.message
        assert custom.extra is not None
        assert custom.extra["event_type"] == "_CustomUserEvent"
        assert custom.extra["event_role"] == "user"
        # No metrics / tool_calls on a generic user step.
        assert custom.metrics is None
        assert custom.tool_calls is None

    def test_custom_assistant_event_renders_as_agent_step_with_zero_llm_calls(
        self, tmp_path: Path
    ) -> None:
        """Assistant-role generic events get source='agent' + llm_call_count=0
        (deterministic-dispatch per ATIF v1.7 §II) since no LLM call ran.
        """
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="root"))
        exporter._dispatch_event(_CustomAssistantEvent(summary="agent broadcast — not an LLM call"))

        traj = exporter.get_trajectory()
        agent_step = next(s for s in traj.steps if s.source == "agent")
        assert agent_step.llm_call_count == 0
        assert agent_step.metrics is None  # spec §II: must be absent when llm_call_count=0
        assert agent_step.reasoning_content is None
        assert "agent broadcast" in agent_step.message
        assert agent_step.extra is not None
        assert agent_step.extra["event_type"] == "_CustomAssistantEvent"

    def test_custom_tool_event_renders_as_user_step(self, tmp_path: Path) -> None:
        """ATIF has no TOOL source — closest analog is a user step."""
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="root"))
        exporter._dispatch_event(_CustomToolEvent(output="<sys>tool output</sys>"))

        traj = exporter.get_trajectory()
        assert [s.source for s in traj.steps] == ["system", "user", "user"]
        tool_step = traj.steps[-1]
        assert "tool output" in tool_step.message
        assert tool_step.extra is not None
        assert tool_step.extra["event_role"] == "tool"

    def test_metadata_role_event_skipped(self, tmp_path: Path) -> None:
        """Internal metadata events MUST NOT appear in the trajectory."""
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="root"))
        exporter._dispatch_event(_CustomMetadataEvent(note="agent state snapshot"))

        traj = exporter.get_trajectory()
        # System + Task — the metadata event was filtered out.
        assert [s.source for s in traj.steps] == ["system", "user"]

    def test_known_event_types_still_route_to_specific_handlers(self, tmp_path: Path) -> None:
        """The wildcard dispatcher must not duplicate or skip framework events.

        Pin that Task, BeforeTurn, LLMComplete, AfterTurn still produce
        the same trajectory shape as before — specific handlers ran, no
        generic-event fallback triggered.
        """
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="solve it"))
        exporter._dispatch_event(
            BeforeTurn(
                method_name="run",
                strategy="CodeActStrategy",
                generation_id="gen-1",
                parent_generation_id=None,
                turn_number=1,
            )
        )
        exporter._dispatch_event(
            LLMComplete(
                model_name="fake",
                prompt_tokens=10,
                completion_tokens=2,
                generation_id="gen-1",
            )
        )
        exporter._dispatch_event(LLMOutput(content="answered"))
        exporter._dispatch_event(
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
        sources = [s.source for s in traj.steps]
        assert sources == ["system", "user", "agent"]
        agent_step = traj.steps[2]
        # Specific LLMComplete handler ran ⇒ metrics populated, llm_call_count=1.
        assert agent_step.metrics is not None
        assert agent_step.metrics.prompt_tokens == 10
        assert agent_step.llm_call_count == 1
        assert agent_step.message == "answered"
        # No "event_type" key — that's only set by the generic fallback.
        assert agent_step.extra is None or "event_type" not in (agent_step.extra or {})

    def test_trajectory_with_custom_events_is_normatively_valid(self, tmp_path: Path) -> None:
        """A trajectory containing custom events still passes schema + normative rules."""
        exporter = _setup_exporter(tmp_path)
        exporter._dispatch_event(Task(prompt="root"))
        exporter._dispatch_event(_CustomUserEvent(payload="custom user input"))
        exporter._dispatch_event(_CustomAssistantEvent(summary="custom agent reply"))
        exporter._dispatch_event(_CustomToolEvent(output="custom tool result"))
        exporter._dispatch_event(_CustomMetadataEvent(note="should be skipped"))

        # Schema validation.
        loaded = Trajectory.model_validate_json(exporter.path.read_text())
        # Normative rules (N1 step_id sequencing, N7 message presence, etc.).
        assert_atif_normative(loaded)
        # System + Task + 3 custom events (metadata filtered) = 5 steps.
        assert len(loaded.steps) == 5


class TestWildcardSubscription:
    """End-to-end: install_atif's wildcard subscription picks up custom events."""

    @pytest.mark.asyncio
    async def test_custom_event_on_event_manager_reaches_trajectory(self, tmp_path: Path) -> None:
        # Hermetic CodeAct agent that immediately emits a custom event,
        # then returns via return_result.
        import json

        from nooa import Agent, strategy
        from nooa.atif import atif_scope
        from nooa.config import CodeActConfig
        from nooa.strategies import CodeActStrategy
        from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

        def _resp(tool_calls: list[ToolCall] | None = None) -> LLMResponse:
            return LLMResponse(
                raw_response=None,
                content="",
                tool_calls=tool_calls or [],
                finish_reason="tool_calls" if tool_calls else "stop",
                assistant_message={"role": "assistant", "content": ""},
                usage={"prompt_tokens": 5, "completion_tokens": 1},
            )

        fake = FakeLLMClient(
            scripted_responses=[
                _resp(
                    tool_calls=[
                        ToolCall(
                            id="call_ret",
                            name="return_result",
                            arguments=json.dumps({"result": 0}),
                        )
                    ]
                ),
            ]
        )

        class _CustomEmittingAgent(Agent, llm=fake):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
            async def run(self) -> int:
                """do."""
                ...

        agent = _CustomEmittingAgent(llm=fake)

        # User pushes a custom event onto the event manager BEFORE the run
        # — the wildcard subscription should pick it up.
        out = tmp_path / "trajectory.json"
        async with atif_scope(
            agent,
            path=out,
            session_id="custom-end2end",
            agent_name="_CustomEmittingAgent",
            agent_version="0.1.0",
        ):
            agent.event_manager.add(_CustomUserEvent(payload="hello from user code"))
            await agent.run()

        loaded = Trajectory.model_validate_json(out.read_text())
        assert_atif_normative(loaded)

        # Find the custom step.
        custom_steps = [
            s
            for s in loaded.steps
            if s.extra is not None and s.extra.get("event_type") == "_CustomUserEvent"
        ]
        assert len(custom_steps) == 1, (
            f"Expected exactly one _CustomUserEvent step, got "
            f"{[(s.step_id, s.extra) for s in loaded.steps]}"
        )
        assert "hello from user code" in custom_steps[0].message
