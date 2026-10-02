# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated state summaries do not inherit the parent projection or tools."""

import json

import httpx
import pytest

from nooa import Agent
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.context_blocks.events import UserEvent
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CompletionClient, ResponsesClient, Tool
from nooa.unifiedllm.unifiedllm import _ClientHttp
from tests.integration.test_cache_resume_live import _render


def forbidden_tool(code: str) -> str:
    raise AssertionError("Summary must not execute tools")


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["openai", "anthropic"])
async def test_state_summary_wire_is_independent_of_parent(family, monkeypatch):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        if family == "openai":
            data = {
                "id": "resp",
                "created_at": 0,
                "model": "gpt-5.6",
                "status": "completed",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"value":"summary"}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 1000, "output_tokens": 5, "total_tokens": 1005},
            }
        else:
            data = {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": '{"value":"summary"}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 5},
            }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )

    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    client = (
        ResponsesClient(
            "openai/gpt-5.6",
            api_key="test",
            api_base="https://example.test/v1",
            cache_breakpoint="openai",
            reasoning={"effort": "medium"},
        )
        if family == "openai"
        else CompletionClient(
            "anthropic/claude-sonnet-4-5",
            api_key="test",
            api_base="https://example.test",
            cache_breakpoint="anthropic",
            thinking={"type": "enabled", "budget_tokens": 1024},
            max_tokens=2048,
        )
    )
    async with client:
        agent = Agent(llm=client)
        for text in ("stable facts", "more facts", "recent facts"):
            agent.event_manager.add(UserEvent(content=text))
        summarizer = TokenBudgetSummarizer.install(
            agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
        )
        messages = _render(family, agent.event_manager.values(), "fixed instructions", "live=1")
        ctx = LLMCallContext(
            client=client,
            agent=agent,
            runtime=agent.runtime,
            messages=messages,
            params={
                "tools": [
                    Tool(name="execute_python", callable=forbidden_tool, description="Run Python")
                ],
                "tool_choice": "auto",
                "prompt_cache_key": "parent-shard",
            },
        )

        async def core(request):
            request.response = await client.acall(request.messages, **request.params)
            return request

        await agent.event_manager.run_middleware("llm_call", ctx, core)
        assert summarizer._pending_task is not None
        await summarizer._pending_task
        assert summarizer._pending_summary == "summary"
        summarizer._uninstall()
    assert len(bodies) == 2
    parent, summary = bodies
    key = "input" if family == "openai" else "messages"
    assert "stable facts" in json.dumps(summary[key])
    assert "more facts" in json.dumps(summary[key])
    assert "recent facts" not in json.dumps(summary[key])
    assert "live=1" not in json.dumps(summary)
    assert "fixed instructions" not in json.dumps(summary)
    assert not summary.get("tools")
    assert summary.get("prompt_cache_key") != "parent-shard"
