# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent shutdown drains installed background summaries before releasing resources."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from nooa_bench.bench_agent import BenchAgent
from nooa_bench.rlm_bench_agent import RLMBenchAgent

from nooa.events import Message
from nooa.interactive import SummarizationConfig
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [BenchAgent, RLMBenchAgent])
async def test_close_drains_pending_summary_before_shell_and_shared_client(agent_type, tmp_path):
    """Exercise the installed summarizer's real middleware and cancellation hook."""
    entered, cancelled = asyncio.Event(), asyncio.Event()
    llm = FakeLLMClient()
    agent = agent_type(
        llm=llm,
        working_dir=str(tmp_path),
        summarization=SummarizationConfig(max_tokens=100, preserve_recent=1),
    )
    original_shell_close = agent.shell.close
    closed = []

    async def summary_call(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Include asynchronous cleanup, not only immediate cancellation.
            await asyncio.sleep(0)
            cancelled.set()

    async def close_shell():
        assert cancelled.is_set(), "shell closed before summary task drained"
        closed.append("shell")
        await original_shell_close()

    async def close_client():
        assert cancelled.is_set(), "shared client closed before summary task drained"
        closed.append("client")

    llm.acall = summary_call
    llm.aclose = AsyncMock(side_effect=close_client)
    agent.shell.close = close_shell
    for i in range(4):
        agent.event_manager.add(Message(content=f"fact {i}"))
    ctx = LLMCallContext(
        agent=agent,
        runtime=agent.runtime,
        client=llm,
        messages=[{"role": "user", "content": "task"}],
        params={"tools": []},
    )

    async def complete(request):
        request.response = LLMResponse(content="parent", usage=LLMUsage(input_tokens=1000))
        return request

    try:
        await agent.event_manager.run_middleware("llm_call", ctx, complete)
        await asyncio.wait_for(entered.wait(), 2)
        await agent.close()
        assert cancelled.is_set()
        llm.aclose.assert_not_awaited()  # Only the caller owns the shared client.
        await llm.aclose()
        assert closed == ["shell", "client"]
    finally:
        await agent.aclose()
        await original_shell_close()
