# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LiteLLM adapters: one attempt, the same wire body as the clients."""

import asyncio
import inspect
import json

import litellm
import pytest
from pydantic import BaseModel

from nooa.unifiedllm import CompletionClient, LLMRequest, ResponsesClient, Tool
from nooa.unifiedllm.adapters import Adapter, Capabilities
from nooa.unifiedllm.adapters.legacy import configure_legacy_adapter
from nooa.unifiedllm.adapters.litellm_chat import LiteLLMChatAdapter
from nooa.unifiedllm.adapters.litellm_responses import LiteLLMResponsesAdapter
from nooa.unifiedllm.errors import FailureKind, LLMFailure
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
    # Strict sends now force the SDK/library retry controls off; these are
    # transport settings, not serialized request-body fields. Legacy golden
    # kwargs remain unchanged in test_wire_snapshots.
    bodies = [json.loads(call) for call in calls]
    for body in bodies:
        body.pop("max_retries", None)
        body.pop("num_retries", None)
    assert len(calls) == 4 and all(body == bodies[0] for body in bodies)
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
    # Direct request-only sends now expose typed failures but still never retry.
    with pytest.raises(LLMFailure) as caught:
        client._adapter.send_sync(request)
    assert caught.value.kind == FailureKind.TRANSIENT
    assert isinstance(caught.value.__cause__, litellm.InternalServerError)
    with pytest.raises(LLMFailure) as caught:
        await client._adapter.send(request)
    assert caught.value.kind == FailureKind.TRANSIENT
    assert isinstance(caught.value.__cause__, litellm.InternalServerError)
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
    assert set(type(request).model_fields) == {"messages", "tools", "tool_choice", "output_type"}
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
def test_legacy_option_reply_cap_is_preserved(calls, make, options):
    """A cap in the private options survives the four-field content request."""
    client = make()
    adapter = configure_legacy_adapter(client._adapter, options)
    adapter.send_sync(LLMRequest(messages=MESSAGES))
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
def test_bound_reply_cap_fills_in_when_options_have_none(calls, make):
    from nooa.unifiedllm.adapters._binding import _bind_adapter

    client = make()
    _bind_adapter(client._adapter, max_output_tokens=256).send_sync(LLMRequest(messages=MESSAGES))
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

    def _next(self, request):
        self.seen.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def send_sync(self, request):
        return self._next(request)

    async def send(self, request):
        return self._next(request)


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
        retry_config=RetryConfig(max_retries=2, base_delay=0, max_delay=0, jitter_factor=0),
        reasoning_levels={"low": {"reasoning_effort": "low"}},
    )
    boom = litellm.InternalServerError("boom", llm_provider="openai", model="m")
    adapter = ProtocolOnlyAdapter([boom, _turn("one"), boom, _turn("two")])
    assert isinstance(adapter, Adapter)
    client._adapter = adapter

    class Output(BaseModel):
        answer: str

    options = {"tool_choice": "required"}
    with pytest.raises(ValueError, match="unsupported.*temperature"):
        client.call(MESSAGES, [TOOL], temperature=0)
    with pytest.raises(ValueError, match="unsupported.*temperature"):
        await client.acall(MESSAGES, temperature=0)
    assert adapter.seen == []
    assert client.call(MESSAGES, [TOOL], output_model=Output, **options).content == "one"
    assert (await client.acall(MESSAGES, [TOOL], output_model=Output, **options)).content == "two"
    assert len(adapter.seen) == 4
    for request in adapter.seen:
        assert request.messages is MESSAGES and request.tools == [TOOL]
        assert request.tool_choice == "required"
        assert set(type(request).model_fields) == {
            "messages",
            "tools",
            "tool_choice",
            "output_type",
        }
        assert request.output_type is Output
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
    with pytest.raises(LLMFailure, match="tool_choice 'search' names no tool") as caught:
        client._adapter.send_sync(request)
    assert caught.value.kind == FailureKind.BAD_REQUEST
    assert isinstance(caught.value.__cause__, ValueError)
    with pytest.raises(LLMFailure, match="tool_choice 'search' names no tool") as caught:
        await client._adapter.send(request)
    assert caught.value.kind == FailureKind.BAD_REQUEST
    assert isinstance(caught.value.__cause__, ValueError)
    assert calls == []
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
def test_send_signatures_are_genuinely_request_only(make):
    client = make()
    for name in ("send", "send_sync"):
        assert list(inspect.signature(getattr(client._adapter, name)).parameters) == ["request"]
    client.close()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0},
        {"top_p": 0.9},
        {"prompt_cache_key": "session"},
        {"timeout": 4},
        {"model": "openai/other"},
        {"extra_body": {"reasoning_effort": "low"}},
    ],
)
@pytest.mark.asyncio
async def test_strict_adapter_rejects_wire_settings_before_dispatch(make, override):
    client = make()
    client._adapter = ProtocolOnlyAdapter([])
    with pytest.raises(ValueError, match="unsupported by a request-only adapter"):
        client.call(MESSAGES, **override)
    with pytest.raises(ValueError, match="unsupported by a request-only adapter"):
        await client.acall(MESSAGES, **override)
    assert client._adapter.seen == []
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_strict_adapter_rejects_inherited_settings_instead_of_dropping(make):
    client = make(
        tool_choice="required",
        extra_body={"max_tokens": 80},
        reasoning_levels={"low": {"reasoning_effort": "low", "max_tokens": 40}},
        reasoning_level="low",
    )
    adapter = ProtocolOnlyAdapter([])
    client._adapter = adapter
    try:
        with pytest.raises(ValueError, match="unsupported.*(extra_body|reasoning)"):
            client.call(MESSAGES, [TOOL])
        with pytest.raises(ValueError, match="unsupported.*(extra_body|reasoning)"):
            await client.acall(MESSAGES, [TOOL])
        assert adapter.seen == []
    finally:
        await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
def test_strict_adapter_rejects_inherited_sampling(make):
    client = make(temperature=0.5)
    client._adapter = ProtocolOnlyAdapter([])
    with pytest.raises(ValueError, match="unsupported.*temperature"):
        client.call(MESSAGES)
    assert client._adapter.seen == []
    client.close()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_litellm_call_local_options_are_isolated(monkeypatch, make):
    """Concurrent calls retain real production knobs and the original transport."""
    seen = []
    arrived = asyncio.Event()
    release = asyncio.Event()

    async def provider(**kwargs):
        seen.append(kwargs)
        if len(seen) == 2:
            arrived.set()
        await release.wait()
        return chat_reply() if make is chat_client else responses_reply()

    monkeypatch.setattr(litellm, "acompletion" if make is chat_client else "aresponses", provider)
    client = make(temperature=0.7, top_p=0.95)
    adapter = client._adapter
    owned_http = client._http
    config = dict(client.config)
    one = {
        "temperature": 0.1,
        "top_p": 0.2,
        "prompt_cache_key": "one",
        "timeout": 3,
        "extra_body": {"custom": {"value": "one"}},
    }
    two = {
        "temperature": 0.8,
        "top_p": 0.9,
        "prompt_cache_key": "two",
        "timeout": 7,
        "api_base": "https://replacement.test/v1",
        "extra_body": {"custom": {"value": "two"}},
    }
    tasks = [asyncio.create_task(client.acall(MESSAGES, **options)) for options in (one, two)]
    try:
        await asyncio.wait_for(arrived.wait(), timeout=5)
        one["extra_body"]["custom"]["value"] = "mutated"
        assert client.config == config
        assert client._adapter is adapter and adapter._legacy_options is None
        assert client._http is owned_http
        by_key = {body["prompt_cache_key"]: body for body in seen}
        assert [
            (by_key[k]["temperature"], by_key[k]["top_p"], by_key[k]["timeout"])
            for k in ("one", "two")
        ] == [
            (0.1, 0.2, 3),
            (0.8, 0.9, 7),
        ]
        assert by_key["one"]["extra_body"]["custom"]["value"] == "one"
        assert by_key["two"]["extra_body"]["custom"]["value"] == "two"
        assert by_key["one"]["client"] is not None
        if make is chat_client:
            # Chat SDK clients bind URL/key; replacement routes bypass them.
            assert by_key["two"].get("client") is None
        else:
            # Responses uses an unbound HTTP handler, safe across route changes.
            assert by_key["two"]["client"] is by_key["one"]["client"]
        assert by_key["two"]["api_base"] == "https://replacement.test/v1"
    finally:
        release.set()
        await asyncio.gather(*tasks)
        await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_legacy_reasoning_cap_and_recovery_preserve_exact_settings(calls, make):
    client = make(
        extra_body={"max_tokens": 80},
        reasoning_levels={"low": {"reasoning_effort": "low", "max_completion_tokens": 40}},
        reasoning_level="low",
    )
    client.call(MESSAGES)
    await client.acall(MESSAGES)
    reduced = client._with_reduced_reply_limit({}, 20)
    assert reduced["reasoning_level"] is None
    client.call(MESSAGES, **reduced)
    await client.acall(MESSAGES, **reduced)
    bodies = [json.loads(body) for body in calls]
    for body, cap in zip(bodies, (40, 40, 20, 20), strict=True):
        assert body["reasoning_effort"] == "low"
        assert [
            body[k]
            for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
            if k in body
        ] == [cap]
        assert not any(k.startswith("max_") for k in body.get("extra_body", {}))
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.parametrize("cap", ["50", True, 0, -1])
def test_strict_adapter_does_not_silently_drop_invalid_caps(make, cap):
    client = make()
    client._adapter = ProtocolOnlyAdapter([])
    with pytest.raises(ValueError, match="unsupported.*max_tokens"):
        client.call(MESSAGES, max_tokens=cap)
    assert client._adapter.seen == []
    client.close()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_legacy_sync_async_preserve_constructor_and_call_options(calls, make):
    client = make(temperature=0.7, top_p=0.95, max_completion_tokens=80, timeout=9)
    options = {
        "temperature": 0.1,
        "top_p": 0.2,
        "prompt_cache_key": "session",
        "timeout": 3,
        "extra_body": {"max_tokens": 20},
        "tool_choice": "none",
    }
    client.call(MESSAGES, [TOOL])
    await client.acall(MESSAGES, [TOOL])
    client.call(MESSAGES, [TOOL], **options)
    await client.acall(MESSAGES, [TOOL], **options)
    bodies = [json.loads(body) for body in calls]
    assert bodies[0] == bodies[1] and bodies[2] == bodies[3]
    for body in bodies[:2]:
        assert (body["temperature"], body["top_p"], body["timeout"]) == (0.7, 0.95, 9)
        assert body.get("max_completion_tokens", body.get("max_output_tokens")) == 80
    for body in bodies[2:]:
        assert (body["temperature"], body["top_p"], body["timeout"]) == (0.1, 0.2, 3)
        assert body["prompt_cache_key"] == "session" and body["tool_choice"] == "none"
        assert body.get("max_tokens", body.get("max_output_tokens")) == 20
    await client.aclose()


@pytest.mark.parametrize("make", [chat_client, responses_client])
@pytest.mark.asyncio
async def test_public_media_does_not_borrow_legacy_sticky_exclusions(calls, make):
    from nooa.unifiedllm import LLM
    from nooa.unifiedllm.media_inputs import media_occurrences

    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://image.test/i.png"}}],
        }
    ]
    client = make(media_inputs=["image"])
    try:
        rejected = client._rejected_media(client._prepare_call_config({}))
        rejected.add(media_occurrences(messages, ("image",))[0][1])
        client.call(messages)
        assert "could not read" in calls[-1]
        llm = LLM(client._adapter)
        llm.call(LLMRequest(messages=messages))
        await llm.acall(LLMRequest(messages=messages))
        assert "https://image.test/i.png" in calls[-1]
        assert "could not read" not in calls[-1]
        assert len(rejected) == 1
        assert messages[0]["content"][0]["type"] == "image_url"
    finally:
        await client.aclose()
