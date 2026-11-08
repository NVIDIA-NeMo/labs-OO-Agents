# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider context failures can stop without rewriting existing conversation events."""

import json
from unittest.mock import AsyncMock

import pytest

from nooa import Agent
from nooa.events import Error, Message, Summary
from nooa.runtime.actor import _current_llm_var, _current_method_var
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage, ToolCall


class ProviderError(Exception):
    status_code = 400
    body = {"error": {"code": "context_length_exceeded"}}
    headers = {"x-request-id": "offline-overflow-1"}


def overflow():
    return ProviderError(
        "This model's maximum context length is 32768 tokens. "
        "However, your request has 34000 input tokens."
    )


class RespondingAgent(Agent):
    async def respond(self, prompt: str) -> str:
        """Answer the prompt."""
        ...


@pytest.fixture(autouse=True)
def no_viewer_probe(monkeypatch):
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)


async def passthrough(ctx, call_next):
    return await call_next(ctx)


def make_agent(monkeypatch, policy, middleware):
    llm = FakeLLMClient()
    llm._context_window = 32768
    agent = RespondingAgent(llm=llm, **({"context_error_policy": policy} if policy else {}))
    for i in range(8):
        agent.event_manager.add(Message(content=f"preserved message {i} " * 40))
    if middleware:
        agent.event_manager.intercept("llm_call", passthrough)
    calls = AsyncMock()
    monkeypatch.setattr(llm, "acall", calls)
    return agent, llm, calls


def history(agent):
    return {event.tag: event.model_dump() for event in agent.event_manager.all_events()}


async def generate(agent, llm):
    llm_token = _current_llm_var.set(llm)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        return await agent.runtime.generate(tools=[], max_tokens=1024)
    finally:
        _current_method_var.reset(method_token)
        _current_llm_var.reset(llm_token)


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("warm", [False, True])
async def test_raise_preserves_history_and_original_error(monkeypatch, middleware, warm):
    agent, llm, calls = make_agent(monkeypatch, "raise", middleware)
    try:
        if warm:
            calls.return_value = LLMResponse(
                raw_response=None,
                content="prior answer",
                finish_reason="stop",
                usage=LLMUsage(input_tokens=1000, output_tokens=10, total_tokens=1010),
            )
            await generate(agent, llm)
            calls.reset_mock()
        before = history(agent)
        active = agent.event_manager.keys()
        error = overflow()
        cause = ValueError("provider rejected request")
        error.__cause__ = cause
        calls.side_effect = error
        with pytest.raises(ProviderError) as caught:
            await generate(agent, llm)
        assert caught.value is error
        assert caught.value.__cause__ is cause
        assert caught.value.status_code == 400
        assert caught.value.body == {"error": {"code": "context_length_exceeded"}}
        assert caught.value.headers == {"x-request-id": "offline-overflow-1"}
        assert calls.await_count == 1
        assert calls.await_args.kwargs["max_tokens"] == 1024
        assert history(agent) == before
        assert agent.event_manager.keys() == active
        assert not any(isinstance(e, Summary) for e in agent.event_manager.all_events())
    finally:
        await agent.aclose()


@pytest.mark.parametrize("middleware", [False, True])
async def test_default_still_archives_before_reraising(monkeypatch, middleware):
    agent, llm, calls = make_agent(monkeypatch, None, middleware)
    try:
        assert agent.context_error_policy == "recover"
        active = agent.event_manager.keys()
        error = overflow()
        calls.side_effect = error
        with pytest.raises(ProviderError) as caught:
            await generate(agent, llm)
        assert caught.value is error
        assert calls.await_count == 1  # No reply budget remains, but default recovery archives.
        assert agent.event_manager.keys() != active
        assert any(isinstance(e, Summary) for e in agent.event_manager.all_events())
    finally:
        await agent.aclose()


@pytest.mark.parametrize("middleware", [False, True])
async def test_codeact_does_not_retry_or_add_api_error_feedback(monkeypatch, middleware):
    agent, _, calls = make_agent(monkeypatch, "raise", middleware)
    try:
        before = history(agent)
        active = agent.event_manager.keys()
        error = overflow()
        calls.side_effect = error
        with pytest.raises(ProviderError) as caught:
            await agent.respond("Stop on overflow")
        assert caught.value is error
        assert calls.await_count == 1
        assert all(history(agent)[tag] == event for tag, event in before.items())
        assert set(active) <= set(agent.event_manager.keys())
        assert not any(isinstance(e, Summary | Error) for e in agent.event_manager.all_events())
    finally:
        await agent.aclose()


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="context_error_policy"):
        RespondingAgent(llm=FakeLLMClient(), context_error_policy="ignore")


@pytest.mark.parametrize("middleware", [False, True])
async def test_default_recovery_still_rebuilds_and_retries(monkeypatch, middleware):
    agent, llm, calls = make_agent(monkeypatch, None, middleware)
    error = ProviderError(
        "This model's maximum context length is 32768 tokens. "
        "However, your request has 30000 input tokens."
    )
    reply = LLMResponse(raw_response=None, content="recovered", finish_reason="stop")
    calls.side_effect = [error, reply]
    llm_token = _current_llm_var.set(llm)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        result, _ = await agent.runtime.generate(tools=[], max_tokens=4096)
        assert result is reply
        assert calls.await_count == 2
        first, retry = calls.await_args_list
        assert first.kwargs["max_tokens"] == 4096
        assert 1024 <= retry.kwargs["max_tokens"] < 4096
        assert first.args[0] != retry.args[0]
        assert any(isinstance(e, Summary) for e in agent.event_manager.all_events())
    finally:
        _current_method_var.reset(method_token)
        _current_llm_var.reset(llm_token)
        await agent.aclose()


@pytest.mark.parametrize("policy,context_error", [(None, True), ("raise", False)])
async def test_codeact_keeps_other_retry_behavior(monkeypatch, policy, context_error):
    agent, _, calls = make_agent(monkeypatch, policy, False)
    error = overflow() if context_error else RuntimeError("temporary endpoint error")
    calls.side_effect = [
        error,
        LLMResponse(
            raw_response=None,
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="return-1",
                    name="execute_python",
                    arguments=json.dumps({"code": "return_result('done')"}),
                )
            ],
        ),
    ]
    try:
        assert await agent.respond("Try the task") == "done"
        assert calls.await_count == 2
        events = agent.event_manager.all_events()
        assert any(isinstance(e, Error) for e in events)
        assert any(isinstance(e, Summary) for e in events) is context_error
    finally:
        await agent.aclose()
