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
    # Equal bodies alone would hide a choice both paths drop.
    assert json.loads(calls[0])["tool_choice"] == "required"
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


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.parametrize(
    "options",
    [
        {"max_completion_tokens": 512},
        {"max_output_tokens": 512},
        {"extra_body": {"max_tokens": 512}},
    ],
)
def test_option_reply_cap_wins_over_the_request_cap(calls, make, options):
    """A cap in the options, under any spelling, keeps the request's cap off the wire."""
    client = make()
    client._adapter.send_sync(LLMRequest(messages=MESSAGES, max_output_tokens=256), **options)
    [body] = [json.loads(call) for call in calls]
    caps = {
        k: v
        for k, v in body.items()
        if k in {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    }
    caps.update({k: v for k, v in (body.get("extra_body") or {}).items() if k.startswith("max_")})
    assert list(caps.values()) == [512]
    client.close()


@pytest.mark.parametrize("make", [chat_client, responses_client])
def test_request_reply_cap_fills_in_when_options_have_none(calls, make):
    client = make()
    client._adapter.send_sync(LLMRequest(messages=MESSAGES, max_output_tokens=256))
    [body] = [json.loads(call) for call in calls]
    assert 256 in (body.get("max_tokens"), body.get("max_output_tokens"))
    client.close()


class ProtocolOnlyAdapter:
    """Only what ``Adapter`` declares: capabilities, send and send_sync."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities()

    def _next(self, request, options):
        self.seen.append((request, options))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def send_sync(self, request, **options):
        return self._next(request, options)

    async def send(self, request, **options):
        return self._next(request, options)


def _turn(text="", reasoning=None):
    from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, LLMUsage

    parts = []
    if reasoning:
        parts.append(AssistantReasoning(text=reasoning))
    if text:
        parts.append(AssistantText(text=text))
    return LLMResponse(
        parts=tuple(parts),
        finish_reason="stop",
        usage=LLMUsage(input_tokens=3, output_tokens=1),
    )


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_client_dispatches_through_a_protocol_only_adapter(make):
    """Clients use only the declared protocol; retries stay in the client."""
    client = make(
        retry_config=RetryConfig(max_retries=2, base_delay=0, max_delay=0, jitter_factor=0)
    )
    boom = litellm.InternalServerError("boom", llm_provider="openai", model="m")
    adapter = ProtocolOnlyAdapter([boom, _turn("one"), boom, _turn("two")])
    assert isinstance(adapter, Adapter)
    client._adapter = adapter
    assert client.call(MESSAGES, [TOOL], tool_choice="required", temperature=0).content == "one"
    assert (await client.acall(MESSAGES)).content == "two"
    assert len(adapter.seen) == 4
    request, options = adapter.seen[1]
    assert request.messages is MESSAGES and request.tools == [TOOL]
    assert request.tool_choice == "required"
    assert options == {"tool_choice": "required", "temperature": 0}
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_content_retry_checks_the_returned_response():
    client = chat_client(
        retry_config=RetryConfig(
            max_retries=2, base_delay=0, max_delay=0, jitter_factor=0, retry_on_empty_content=True
        )
    )
    client._adapter = ProtocolOnlyAdapter(
        [_turn(reasoning="thinking"), _turn("done"), _turn(reasoning="again"), _turn("ok")]
    )
    assert client.call(MESSAGES).content == "done"
    assert (await client.acall(MESSAGES)).content == "ok"
    assert len(client._adapter.seen) == 4
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.parametrize("choice", ["auto", "required", "none"])
def test_request_tool_choice_reaches_the_wire(calls, make, choice):
    client = make()
    client._adapter.send_sync(LLMRequest(messages=MESSAGES, tools=[TOOL], tool_choice=choice))
    [body] = [json.loads(call) for call in calls]
    # Chat leaves "auto" implicit beside parallel_tool_calls=False.
    expected = None if choice == "auto" and make is chat_client else choice
    assert body.get("tool_choice") == expected
    client.close()


@pytest.mark.parametrize(
    ("make", "wire"),
    [
        (chat_client, {"type": "function", "function": {"name": "lookup"}}),
        (responses_client, {"type": "function", "name": "lookup"}),
    ],
)
@pytest.mark.asyncio
async def test_named_tool_choice_uses_the_wire_form(calls, make, wire):
    client = make()
    request = LLMRequest(messages=MESSAGES, tools=[TOOL], tool_choice="lookup")
    client._adapter.send_sync(request)
    await client._adapter.send(request)
    client.call(MESSAGES, [TOOL], tool_choice="lookup")
    assert [json.loads(call)["tool_choice"] for call in calls] == [wire] * 3
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.parametrize("tools", [[TOOL], []])
@pytest.mark.asyncio
async def test_named_tool_choice_must_name_a_request_tool(calls, make, tools):
    client = make()
    request = LLMRequest(messages=MESSAGES, tools=tools, tool_choice="search")
    with pytest.raises(ValueError, match="tool_choice 'search' names no tool"):
        client._adapter.send_sync(request)
    with pytest.raises(ValueError, match="tool_choice 'search' names no tool"):
        await client._adapter.send(request)
    assert calls == []
    await client.aclose()
