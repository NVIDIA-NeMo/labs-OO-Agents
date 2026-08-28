# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context recovery accepts only the normalized UnifiedLLM signal."""

import pytest

from nooa.runtime.actor import _is_context_window_error
from nooa.unifiedllm import LLMContextLengthError


def test_normalized_context_error_is_recognized():
    assert _is_context_window_error(LLMContextLengthError("too large"))


def test_error_text_is_not_parsed_or_classified():
    assert not _is_context_window_error(Exception("context_length_exceeded: 999 tokens"))


def test_wrapped_normalized_error_is_not_runtime_classified():
    outer = RuntimeError("wrapper")
    outer.__cause__ = LLMContextLengthError("too large")
    assert not _is_context_window_error(outer)


@pytest.mark.asyncio
async def test_normalized_error_archives_rebuilds_and_retries_once_without_changing_max_tokens():
    from unittest.mock import patch

    from nooa import Agent
    from nooa.events import Message
    from nooa.runtime.actor import _current_llm_var, _current_method_var
    from nooa.unifiedllm import FakeLLMClient, LLMResponse

    llm = FakeLLMClient()

    class A(Agent, llm=llm):
        async def respond(self) -> str:
            """Respond."""
            ...

    agent = A()
    for i in range(12):
        agent.event_manager.add(Message(content=f"history-{i}"))

    calls = []

    async def acall(messages, **kwargs):
        calls.append((messages, kwargs.get("max_tokens")))
        if len(calls) == 1:
            raise LLMContextLengthError("provider rejected context")
        return LLMResponse(
            content="ok",
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "ok"},
            usage=None,
        )

    llm_token = _current_llm_var.set(llm)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        with patch.object(llm, "acall", side_effect=acall):
            response, _ = await agent.runtime.generate(tools=[], max_tokens=777)
    finally:
        _current_method_var.reset(method_token)
        _current_llm_var.reset(llm_token)

    assert response.content == "ok"
    assert len(calls) == 2
    assert [max_tokens for _, max_tokens in calls] == [777, 777]
    assert calls[1][0] != calls[0][0]  # request was rebuilt after archival
    assert agent.event_manager.keys()[0] == "1..10"


@pytest.mark.asyncio
async def test_recovery_is_bounded_to_one_retry():
    from unittest.mock import patch

    from nooa import Agent
    from nooa.events import Message
    from nooa.runtime.actor import _current_llm_var, _current_method_var
    from nooa.unifiedllm import FakeLLMClient

    llm = FakeLLMClient()

    class A(Agent, llm=llm):
        async def respond(self) -> str:
            """Respond."""
            ...

    agent = A()
    for i in range(25):
        agent.event_manager.add(Message(content=f"history-{i}"))
    attempts = 0

    async def fail(messages, **kwargs):
        nonlocal attempts
        attempts += 1
        raise LLMContextLengthError("still too large")

    llm_token = _current_llm_var.set(llm)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        with patch.object(llm, "acall", side_effect=fail):
            with pytest.raises(LLMContextLengthError):
                await agent.runtime.generate(tools=[], max_tokens=777)
    finally:
        _current_method_var.reset(method_token)
        _current_llm_var.reset(llm_token)
    assert attempts == 2
