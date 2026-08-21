# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in experimental TUI agent."""

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
        assert not any(
            isinstance(event, ToolCallEvent) and event.name == "return_result"
            for event in agent.event_manager.values()
        )
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_experimental_tui_agent_delegates_to_experimental_worker(tmp_path):
    agent = ExperimentalTUIAgent(llm=FakeLLMClient(), cwd=tmp_path)
    try:
        assert agent._worker_type is ExperimentalCodingWorker
        assert "prefer it over awaiting" in (ExperimentalTUIAgent.__doc__ or "")
        assert "finish that turn with ``WAIT``" in (ExperimentalTUIAgent.__doc__ or "")
    finally:
        await agent.close()
