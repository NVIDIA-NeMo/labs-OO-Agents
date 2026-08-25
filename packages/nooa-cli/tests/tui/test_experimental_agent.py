# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the single-tool TUI agent."""

import json

import pytest
from nooa_cli.tui.config import load_agent_class
from nooa_cli.tui.experimental_agent import ExperimentalCodingWorker, ExperimentalTUIAgent

from nooa.context_blocks import DynamicContext, ToolCallEvent
from nooa.interactive import RespondReason
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _response(code: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="python_cell",
                arguments=json.dumps({"code": code}),
            )
        ],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


def test_cli_agent_spec_loads_experimental_tui_agent():
    loaded = load_agent_class("nooa_cli.tui.experimental_agent:ExperimentalTUIAgent")
    assert loaded is ExperimentalTUIAgent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_agent", "expected_type"),
    [(False, ExperimentalTUIAgent), (True, None)],
)
async def test_bootstrap_selects_default_or_legacy_agent(
    legacy_agent, expected_type, tmp_path, monkeypatch
):
    from nooa_cli.tui import session_manager as session_manager_module
    from nooa_cli.tui.agent import TUIAgent
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_cli.tui.config import Config

    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path / "sessions")
    config = Config(legacy_agent=legacy_agent)
    config.tui.default_model = "test-model"
    config.agent.summarization.policy = "none"

    result = await bootstrap(config)
    try:
        expected = TUIAgent if legacy_agent else expected_type
        assert type(result.agent) is expected
    finally:
        await result.agent.close()
        result.session_manager.close()


@pytest.mark.asyncio
async def test_experimental_tui_agent_uses_only_python_cell(tmp_path):
    llm = FakeLLMClient(
        scripted_responses=[
            _response(
                "return_result(RespondResult(kind=RespondReason.DONE, "
                "explanation='request completed'))"
            )
        ]
    )
    agent = ExperimentalTUIAgent(llm=llm, cwd=tmp_path)
    try:
        result = await agent.handle({"user_messages": ["inspect the repository"]})
        assert result.kind is RespondReason.DONE
        assert result.explanation == "request completed"
        assert [tool.name for tool in llm.last_tools or []] == ["python_cell"]
        system_prompt = "\n".join(
            str(message.get("content", ""))
            for message in llm.last_messages
            if message.get("role") == "system"
        )
        assert "<state" not in system_prompt
        assert "<execution_context" not in system_prompt
        rendered_context = "\n".join(
            str(message.get("content", "")) for message in llm.last_messages
        )
        assert "<python_cell_state" in rendered_context
        state_block = rendered_context.split("<python_cell_state", 1)[1].split(
            "</python_cell_state>", 1
        )[0]
        assert "Imports: none" in state_block
        assert "Cell locals (includes method inputs): notification (dict)" in state_block
        assert "`self.v`: none" in rendered_context
        assert f"`self.shell.cwd`: {tmp_path}" in rendered_context
        completion_events = [
            event
            for event in agent.event_manager.values()
            if isinstance(event, ToolCallEvent) and event.name == "return_result"
        ]
        assert len(completion_events) == 1
        assert completion_events[0].metadata["synthetic_type"] == "codeact_inline_return"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_experimental_tui_agent_delegates_to_experimental_worker(tmp_path):
    agent = ExperimentalTUIAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert agent._worker_type is ExperimentalCodingWorker
        worker_context = ExperimentalCodingWorker.investigate._strategy_context
        assert isinstance(worker_context["state"], DynamicContext)
        assert isinstance(worker_context["execution_context"], DynamicContext)
        assert not hasattr(ExperimentalCodingWorker, "delegate")
        assert not hasattr(ExperimentalCodingWorker, "spawn")
        assert (ExperimentalTUIAgent.__doc__ or "").startswith(
            "You are a careful software-development agent working in one local repository."
        )
        prompt = ExperimentalTUIAgent.__doc__ or ""
        normalized_prompt = " ".join(prompt.split())
        policy = (
            "Use an RLM-style controller policy: complete requests directly when they fit "
            "in a few turns. For larger requests, decompose only when there are distinct, "
            "context-heavy, independently verifiable subtasks; keep tightly coupled or "
            "small sequential work local. The top-level controller may spawn bounded, "
            "non-recursive workers for subtasks that benefit from separate context."
        )
        assert policy in normalized_prompt
        assert "prefer it over awaiting" in prompt
        assert "immediately finish that turn" in prompt
        assert "durable cross-task identity" in normalized_prompt
        assert "task-specific plans" in normalized_prompt
        assert "Todo's ``v`` proxy" in prompt
        assert "do not use either persistent store as an uncurated dump" in normalized_prompt
        assert "Never poll a spawned handle" in prompt
        assert "asyncio.sleep()" in prompt
        assert "will invoke a new turn when the report arrives" in prompt
        assert "RespondReason.WAIT" in prompt
        assert 'notification["delegates"]' in prompt
        assert "serialize mutations" in prompt
    finally:
        await agent.close()
