# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the single-tool TUI agent."""

import json

import pytest
from nooa_cli.tui.config import load_agent_class
from nooa_cli.tui.experimental_agent import ExperimentalCodingWorker, ExperimentalTUIAgent

from nooa.context_blocks import ToolCallEvent
from nooa.interactive import RespondReason
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _response(code: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="execute_python",
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
async def test_experimental_tui_agent_uses_only_execute_python(tmp_path):
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
        assert [tool.name for tool in llm.last_tools or []] == ["execute_python"]
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
        assert not hasattr(ExperimentalCodingWorker, "delegate")
        assert not hasattr(ExperimentalCodingWorker, "spawn")
        assert (ExperimentalTUIAgent.__doc__ or "").startswith(
            "You are a careful software-development agent working in one local repository."
        )
        assert "prefer it over awaiting" in (ExperimentalTUIAgent.__doc__ or "")
        assert "RespondReason.WAIT" in (ExperimentalTUIAgent.__doc__ or "")
        assert 'notification["delegates"]' in (ExperimentalTUIAgent.__doc__ or "")
        assert "serialize mutations" in (ExperimentalTUIAgent.__doc__ or "")
    finally:
        await agent.close()
