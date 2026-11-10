# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LiteLLM adapters: one attempt, the same wire body as the clients."""

import json

import litellm
import pytest

from nooa.unifiedllm import CompletionClient, LLMRequest, ResponsesClient, Tool
from nooa.unifiedllm.adapters import Adapter, Capabilities
from nooa.unifiedllm.adapters.litellm_chat import LiteLLMChatAdapter
from nooa.unifiedllm.adapters.litellm_responses import LiteLLMResponsesAdapter
from nooa.unifiedllm.retry_config import RetryConfig


def lookup(query: str) -> str:
    """Look something up."""
    raise AssertionError("never executed")


TOOL = Tool(name="lookup", description="Look something up.", callable=lookup)
MESSAGES = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]


def chat_reply():
    return litellm.ModelResponse(
        choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )


def responses_reply():
    return litellm.ResponsesAPIResponse(
        id="resp_1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-5.6",
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _body(kwargs):
    return json.dumps({k: v for k, v in kwargs.items() if k != "client"}, sort_keys=True)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def completion(**kwargs):
        seen.append(_body(kwargs))
        return chat_reply()

    async def acompletion(**kwargs):
        return completion(**kwargs)

    def responses(**kwargs):
        seen.append(_body(kwargs))
        return responses_reply()

    async def aresponses(**kwargs):
        return responses(**kwargs)

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "responses", responses)
    monkeypatch.setattr(litellm, "aresponses", aresponses)
    return seen


def chat_client(**kwargs):
    return CompletionClient(
        "openai/gpt-4o-mini", api_key="sk-test", api_base="https://gateway.test/v1", **kwargs
    )


def responses_client(**kwargs):
    return ResponsesClient("openai/gpt-5.6", api_key="sk-test", **kwargs)


@pytest.mark.parametrize("make", [chat_client, responses_client])
def test_adapters_implement_the_protocol(make):
    client = make(media_inputs=["image"], context_window=1000)
    adapter = client._adapter
    assert isinstance(adapter, Adapter)
    assert adapter.capabilities == Capabilities(
        media_kinds=frozenset({"image"}), reasoning_levels=None, context_window=1000
    )
    client.close()


def test_capabilities_are_frozen():
    with pytest.raises(AttributeError):
        Capabilities().tools = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("make", "adapter_type"),
    [(chat_client, LiteLLMChatAdapter), (responses_client, LiteLLMResponsesAdapter)],
)
@pytest.mark.asyncio
async def test_send_matches_the_client_wire_body(calls, make, adapter_type):
    client = make()
    adapter = adapter_type(client)
    request = LLMRequest(messages=MESSAGES, tools=[TOOL], tool_choice="required")
    client.call(MESSAGES, [TOOL], tool_choice="required")
    adapter.send_sync(request)
    await client.acall(MESSAGES, [TOOL], tool_choice="required")
    result = await adapter.send(request)
    assert result.content == "ok"
    assert len(calls) == 4 and len(set(calls)) == 1
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_send_makes_one_attempt(monkeypatch, make):
    attempts = []

    def failing(**kwargs):
        attempts.append(1)
        raise litellm.InternalServerError("boom", llm_provider="openai", model="m")

    async def afailing(**kwargs):
        return failing(**kwargs)

    for name in ("completion", "responses"):
        monkeypatch.setattr(litellm, name, failing)
        monkeypatch.setattr(litellm, "a" + name, afailing)
    client = make(
        retry_config=RetryConfig(max_retries=2, base_delay=0, max_delay=0, jitter_factor=0)
    )
    request = LLMRequest(messages=MESSAGES)
    with pytest.raises(litellm.InternalServerError):
        client._adapter.send_sync(request)
    with pytest.raises(litellm.InternalServerError):
        await client._adapter.send(request)
    assert len(attempts) == 2
    await client.aclose()


def test_request_keeps_messages_and_tools_unchanged():
    messages = [{"role": "user", "content": "Hi"}]
    tools = [TOOL]
    request = LLMRequest(messages=messages, tools=tools)
    assert request.messages is messages
    assert request.tools is tools
    assert request.tool_choice == "auto" and request.output_type is None


def test_client_builds_request_from_call(monkeypatch, calls):
    client = chat_client(
        reasoning_levels={"low": {"reasoning_effort": "low"}}, reasoning_level="low"
    )
    seen = []
    prepare = client._adapter.prepare

    def spy(request, *args, **kwargs):
        seen.append(request)
        return prepare(request, *args, **kwargs)

    monkeypatch.setattr(client._adapter, "prepare", spy)
    client.call(MESSAGES, [TOOL], tool_choice="required", max_tokens=50)
    [request] = seen
    assert request.messages is MESSAGES
    assert request.tools == [TOOL]
    assert request.tool_choice == "required"
    assert request.reasoning_level == "low"
    assert request.max_output_tokens == 50
    client.close()
