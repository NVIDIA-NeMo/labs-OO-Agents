# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from nooa import Agent
from nooa.agents import TokenBudgetSummarizer
from nooa.agents.summarization import copy_request_structure, summary_fork_active
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.default_context_view import CodeActView
from nooa.events import Message
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CacheBoundary, FakeLLMClient, LLMResponse, LLMUsage, ToolCall


def test_copy_request_structure_detaches_mutable_data_only():
    response = LLMResponse(content="native")
    boundary = CacheBoundary()
    original = [{"content": [{"text": "parent"}]}, response, boundary]
    copied = copy_request_structure(original)
    copied[0]["content"][0]["text"] = "summary"
    assert original[0]["content"][0]["text"] == "parent"
    assert copied[1] is response and copied[2] is boundary


@pytest.mark.parametrize("placement", ["before", "after"])
@pytest.mark.parametrize("mutation", ["replace", "in_place"])
async def test_summary_middleware_edits_reach_dispatch_without_changing_parent(placement, mutation):
    agent = Agent(llm=FakeLLMClient())
    for i in range(3):
        agent.event_manager.add(Message(content=f"fact {i}"))
    seen = []

    async def middleware(request, nxt):
        if summary_fork_active():
            seen.append("summary")
            if mutation == "replace":
                request.messages = [*request.messages, {"role": "user", "content": "summary-only"}]
                request.params = {**request.params, "temperature": 0.25}
            else:
                request.messages.append({"role": "user", "content": "summary-only"})
                request.params["temperature"] = 0.25
            request.messages[0]["content"][0]["text"] = "edited summary"
            request.params["extra_body"]["nested"] = "edited summary"
        else:
            seen.append("parent")
            request.messages.insert(0, {"role": "system", "content": [{"text": "policy"}]})
        return await nxt(request)

    if placement == "before":
        agent.event_manager.intercept("llm_call", middleware)
    TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=1, preserve_recent=1))
    if placement == "after":
        agent.event_manager.intercept("llm_call", middleware)
    agent.llm.acall = AsyncMock(return_value=LLMResponse(content="summary"))
    ctx = LLMCallContext(
        messages=[{"role": "user", "content": "facts"}],
        params={"extra_body": {"nested": "parent"}},
        client=agent.llm,
        context_view=CodeActView(),
    )

    async def dispatch(request):
        request.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=100))
        return request

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, dispatch)
        sent = agent.llm.acall.call_args
        assert seen == ["parent", "summary"]
        assert sent.args[0][-1]["content"] == "summary-only"
        assert sent.args[0][0]["content"][0]["text"] == "edited summary"
        assert sent.kwargs["temperature"] == 0.25
        assert sent.kwargs["extra_body"]["nested"] == "edited summary"
        assert ctx.messages[0]["content"][0]["text"] == "policy"
        assert len(ctx.messages) == 2
        assert ctx.params == {"extra_body": {"nested": "parent"}}
        assert sum(m["role"] == "system" for m in sent.args[0]) == 1
    finally:
        await agent.aclose()


@pytest.mark.parametrize("mode", ["default", "custom", "rewritten"])
async def test_summary_request_mode_preserves_prefix_or_uses_state(mode):
    class CustomView(CodeActView):
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
        context_view=CustomView() if mode == "custom" else CodeActView(),
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
        if mode != "custom":
            messages = agent.llm.acall.call_args.args[0]
            assert messages[:-1] == ctx.messages
            assert agent.llm.acall.call_args.kwargs == {**params, "output_model": None}
            assert len(ctx.messages) == (1 if mode == "rewritten" else 2)
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
    ctx = LLMCallContext(messages=[], client=agent.llm, context_view=CodeActView())

    async def dispatch(request):
        request.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=100))
        return request

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, dispatch)
        summarizer._apply_pending_summary()
        assert agent.event_manager.keys() == ["1", "2", "3"]
    finally:
        await agent.aclose()
