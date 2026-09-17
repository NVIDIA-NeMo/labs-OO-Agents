# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-sharing forks preserve the actual SDK body, not just message kwargs."""

import json

import httpx
import pytest

from nooa import Agent
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.context_blocks.events import UserEvent
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CacheBoundary, CompletionClient, ResponsesClient, RetryConfig, Tool
from nooa.unifiedllm.direct import DirectTransport
from nooa.unifiedllm.unifiedllm import _ClientHttp
from tests.integration.test_cache_resume_live import _render


def forbidden_tool(code: str) -> str:
    raise AssertionError("Summary must not execute tools")


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["openai", "anthropic"])
async def test_fork_wire_prefix_and_settings_are_identical(family, monkeypatch):
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
                        "content": [{"type": "output_text", "text": "summary", "annotations": []}],
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
                "content": [{"type": "text", "text": "summary"}],
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
    parent, fork = bodies
    key = "input" if family == "openai" else "messages"
    assert {k: v for k, v in parent.items() if k != key} == {
        k: v for k, v in fork.items() if k != key
    }
    if family == "openai":
        assert fork[key][:-1] == parent[key]
        assert "prompt_cache_breakpoint" in json.dumps(parent[key][:-1])
    else:
        # Anthropic coalesces adjacent user messages into one content list.
        assert fork[key][:-1] == parent[key][:-1]
        assert fork[key][-1]["content"][:-1] == parent[key][-1]["content"]
        assert "cache_control" in json.dumps(parent[key])
    assert "live=1" in json.dumps(fork[key])
    assert "Background memory compaction" in json.dumps(fork[key][-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("system_cache", [False, True], ids=["history-cache", "system-cache"])
@pytest.mark.parametrize("explicit_auto", [False, True], ids=["default-choice", "explicit-auto"])
@pytest.mark.parametrize("tool_reply", [False, True], ids=["text-summary", "tool-summary"])
async def test_anthropic_summary_fork_matches_across_transports(
    monkeypatch, tmp_path, explicit_auto, tool_reply, system_cache
):
    """Compare full fork bodies with adaptive thinking, signed history and cache markers."""
    requests = {}
    summary = "Keep ticket-4242, owner alex.moreau, launch 2026-11-04."
    thinking = {
        "type": "thinking",
        "thinking": "Check the recorded facts.",
        "signature": "test-signature",
    }

    def no_network(*args, **kwargs):
        raise AssertionError("Test escaped the mocked HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)

    def return_result(result: str) -> str:
        raise AssertionError("The summary tool reply is data, never executed")

    for transport_name in ("litellm", "direct"):
        captured = []
        requests[transport_name] = captured

        def respond(request, captured=captured, transport_name=transport_name):
            assert request.url == "https://models.example/v1/messages"
            captured.append(bytes(request.content))
            (tmp_path / f"{transport_name}-{len(captured)}.json").write_bytes(request.content)
            if len(captured) == 1:
                content = [
                    thinking,
                    {"type": "text", "text": "Checking the facts."},
                    {
                        "type": "tool_use",
                        "id": "call-test",
                        "name": "execute_python",
                        "input": {"code": "print(42)"},
                    },
                ]
            elif tool_reply:
                content = [
                    {
                        "type": "tool_use",
                        "id": "summary-test",
                        "name": "return_result",
                        "input": {"result": summary},
                    }
                ]
            else:
                content = [{"type": "text", "text": summary}]
            return httpx.Response(
                200,
                json={
                    "id": "msg-test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": content,
                    "stop_reason": "tool_use"
                    if any(b["type"] == "tool_use" for b in content)
                    else "end_turn",
                    "usage": {"input_tokens": 1000, "output_tokens": 30},
                },
            )

        mock = httpx.MockTransport(respond)
        monkeypatch.setattr(
            _ClientHttp, "_httpx_hardening", staticmethod(lambda mock=mock: {"transport": mock})
        )
        monkeypatch.setattr(
            DirectTransport, "_http_settings", lambda *args, mock=mock: {"transport": mock}
        )
        client = CompletionClient(
            "anthropic/claude-sonnet-4-6",
            transport=transport_name,
            api_base="https://models.example",
            api_key="test",
            cache_breakpoint="anthropic",
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            max_tokens=2048,
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
        async with client:
            tools = [
                Tool(name="execute_python", callable=forbidden_tool, description="Run Python"),
                Tool(name="return_result", callable=return_result, description="Return the result"),
            ]
            seed = await client.acall(
                [{"role": "user", "content": "Check the facts."}], tools=tools
            )
            assert seed.reasoning == thinking["thinking"]
            agent = Agent(llm=client)
            for text in (summary, "The identifiers must be kept verbatim.", "Recent facts."):
                agent.event_manager.add(UserEvent(content=text))
            summarizer = TokenBudgetSummarizer.install(
                agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1, target_chars=600)
            )
            try:
                messages = _render(
                    "anthropic", agent.event_manager.values(), "Keep accurate notes.", "live=1"
                )
                if system_cache:
                    messages[0]["content"] = [
                        {
                            "type": "text",
                            "text": messages[0]["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                boundary = next(i for i, m in enumerate(messages) if isinstance(m, CacheBoundary))
                messages[boundary:boundary] = [
                    seed,
                    {"role": "tool", "tool_call_id": "call-test", "content": "42"},
                ]
                params = {"tools": tools, "prompt_cache_key": "same-parent-shard"}
                if explicit_auto:
                    params["tool_choice"] = "auto"
                ctx = LLMCallContext(
                    client=client,
                    agent=agent,
                    runtime=agent.runtime,
                    messages=messages,
                    params=params,
                )

                async def core(request):
                    request.response = await request.client.acall(
                        request.messages, **request.params
                    )
                    return request

                await agent.event_manager.run_middleware("llm_call", ctx, core)
                assert summarizer._pending_task is not None
                await summarizer._pending_task
                assert summarizer._pending_summary == summary
            finally:
                summarizer._uninstall()
                await agent.aclose()

        assert len(captured) == 3  # seed, parent, actual background fork
        parent, fork = [json.loads(raw) for raw in captured[1:]]
        assert fork["thinking"] == {"type": "adaptive"}
        assert fork["output_config"] == {"effort": "high"}
        assert fork["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert fork["max_tokens"] == 2048
        system = {"type": "text", "text": "<instructions>\nKeep accurate notes.\n</instructions>"}
        if system_cache:
            system["cache_control"] = {"type": "ephemeral"}
        assert fork["system"] == [system]
        assert {k: v for k, v in parent.items() if k != "messages"} == {
            k: v for k, v in fork.items() if k != "messages"
        }
        assert parent["messages"][:-1] == fork["messages"][:-1]
        assert parent["messages"][-1]["content"] == fork["messages"][-1]["content"][:-1]
        assert "Background memory compaction" in fork["messages"][-1]["content"][-1]["text"]
        blocks = [b for m in fork["messages"] for b in m["content"]]
        assert thinking in blocks
        assert any(b["type"] == "tool_use" and b["id"] == "call-test" for b in blocks)
        assert any(b["type"] == "tool_result" and b["tool_use_id"] == "call-test" for b in blocks)
        if not system_cache:
            assert "cache_control" in json.dumps(blocks)
        assert "live=1" in json.dumps(fork["messages"][-1])

    # SDKs order JSON object keys differently; preserve array order and every
    # value, including the exact text/signature strings, when comparing bodies.
    assert json.loads(requests["litellm"][2]) == json.loads(requests["direct"][2])
