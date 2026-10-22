# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from nooa import Agent
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.default_context_view import DefaultAgentView
from nooa.events import Message
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CacheBoundary, FakeLLMClient, LLMResponse, LLMUsage, ToolCall


@pytest.mark.parametrize("mode", ["default", "custom", "rewritten"])
async def test_summary_request_mode_preserves_prefix_or_uses_state(mode):
    class CustomView(DefaultAgentView):
        pass

    agent = Agent(llm=FakeLLMClient())
    for i in range(3):
        agent.event_manager.add(Message(content=f"fact {i}"))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=1, preserve_recent=1)
    )
    summarizer.summarize = AsyncMock(return_value="independent")
    agent.llm.acall = AsyncMock(return_value=LLMResponse(content="cached"))
    prefix = [{"role": "user", "content": "facts"}, CacheBoundary()]
    params = {"tools": [{"name": "execute_python"}], "prompt_cache_key": "key"}
    ctx = LLMCallContext(
        messages=prefix,
        params=params,
        client=agent.llm,
        context_view=CustomView() if mode == "custom" else DefaultAgentView(),
    )

    async def dispatch(request):
        if mode == "rewritten":
            request.messages = [{"role": "user", "content": "filtered"}]
        request.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=100))
        return request

    try:
        result = await agent.event_manager.run_middleware("llm_call", ctx, dispatch)
        assert result.response.content == "parent"
        assert summarizer._pending_task.done()
        assert agent.event_manager.keys() == ["1", "2", "3"]
        if mode == "default":
            messages = agent.llm.acall.call_args.args[0]
            assert messages[:-1] == prefix
            assert agent.llm.acall.call_args.kwargs == {**params, "output_model": None}
            assert len(ctx.messages) == 2
            summarizer.summarize.assert_not_awaited()
        else:
            agent.llm.acall.assert_not_awaited()
            assert "fact 0" in summarizer.summarize.call_args.args[0]
        summarizer._apply_pending_summary()
        assert agent.event_manager.keys() == ["1..2", "3"]
    finally:
        await agent.aclose()


async def test_cached_summary_never_executes_tools_or_collapses_on_tool_reply():
    agent = Agent(llm=FakeLLMClient())
    for i in range(3):
        agent.event_manager.add(Message(content=str(i)))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=1, preserve_recent=1)
    )
    agent.llm.acall = AsyncMock(
        return_value=LLMResponse(
            tool_calls=[
                ToolCall(id="tool", name="execute_python", arguments='{"code":"raise Exception()"}')
            ]
        )
    )
    ctx = LLMCallContext(messages=[], client=agent.llm, context_view=DefaultAgentView())

    async def dispatch(request):
        request.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=100))
        return request

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, dispatch)
        summarizer._apply_pending_summary()
        assert agent.event_manager.keys() == ["1", "2", "3"]
    finally:
        await agent.aclose()
