# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable-prefix boundaries from dynamic context to provider wire payload."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest

from nooa._llm_state import ReplayCarryingMessage, carried_cache_boundary, carry_replay_batch
from nooa.context_blocks.events import UserEvent
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import BlockMetadata, RenderedMessage, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.replay_state import capture_chat_state, prepare_chat_messages, replay_scope
from nooa.unifiedllm.unifiedllm import _extract_usage


def _render(dynamic: str) -> list[dict]:
    event = UserEvent(content="solve this", tag="1")
    blocks = [
        ResolvedBlock(
            key="instructions",
            content="stable instructions",
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=True),
        ),
        ResolvedBlock(
            key="event_1",
            content=event.content,
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
            event=event,
        ),
        ResolvedBlock(
            key="live_state",
            content=dynamic,
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=False, user_block=True),
        ),
    ]
    return render_context(
        blocks,
        block_formatter=CachedBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output


def _responses_output() -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        output_text="ok",
        status="completed",
        usage=None,
    )


def test_cached_renderer_marks_only_the_dynamic_suffix_outside_json() -> None:
    messages = _render("state-a")

    assert not carried_cache_boundary(messages[-2])
    assert carried_cache_boundary(messages[-1])
    assert "state-a" in messages[-1]["content"]
    assert "cache_boundary" not in json.dumps(messages)


def test_rendered_message_serialization_excludes_private_transport_fields() -> None:
    message = RenderedMessage(
        role=Role.ASSISTANT,
        content="public",
        llm_state={"encrypted_content": "opaque"},
        reasoning="private reasoning",
        cache_boundary_before=True,
    )

    dumped = message.model_dump()
    assert "llm_state" not in dumped
    assert "reasoning" not in dumped
    assert "cache_boundary_before" not in dumped
    assert "opaque" not in message.model_dump_json()


def test_cache_mapping_must_match_the_client_api_style() -> None:
    with pytest.raises(ValueError, match="CompletionClient.*'anthropic'"):
        CompletionClient(model="openai/gpt-5.6", cache_breakpoint="openai")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ResponsesClient.*'openai'"):
        ResponsesClient(
            model="anthropic/claude-sonnet-4",
            cache_breakpoint="anthropic",  # type: ignore[arg-type]
        )


def test_cache_marker_copy_preserves_private_replay_metadata() -> None:
    state = {"version": 1, "scope": "chat:anthropic:test"}
    original = ReplayCarryingMessage(
        {"role": "system", "content": "stable"},
        llm_state=state,
        reasoning="private reasoning",
        replay_batch_id="batch",
        replay_batch_size=1,
    )
    with CompletionClient(model="anthropic/claude-sonnet-4-5") as client:
        prepared = client._inject_cache_control(
            [original], [{"role": "system"}], model=client.model
        )

    assert isinstance(prepared[0], ReplayCarryingMessage)
    assert prepared[0].llm_state is state
    assert prepared[0].reasoning == "private reasoning"
    assert prepared[0].replay_batch_id == "batch"
    assert prepared[0].replay_batch_size == 1
    assert "cache_control" not in original


def test_overlapping_cache_injection_does_not_mutate_original_content() -> None:
    original = {
        "role": "tool",
        "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "last"}],
    }
    before = json.dumps(original)
    with CompletionClient(model="anthropic/claude-sonnet-4-5") as client:
        prepared = client._inject_cache_control(
            [original], [{"role": "tool"}, {"role": "tool", "position": "last"}]
        )

    assert json.dumps(original) == before
    assert prepared[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert prepared[0]["content"][0] is original["content"][0]


def test_dropped_opaque_assistant_preserves_its_cache_boundary() -> None:
    messages = [
        {"role": "user", "content": "stable"},
        ReplayCarryingMessage(
            {"role": "assistant", "content": ""},
            llm_state={
                "version": 1,
                "scope": "chat:anthropic:test",
                "format": "litellm-chat",
                "payload": {
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "private", "signature": "sig"}
                    ],
                    "carrier": {"content": "", "tool_calls": []},
                    "state_only": True,
                },
            },
            cache_boundary_before=True,
        ),
        {"role": "user", "content": "live state"},
    ]
    prepared = prepare_chat_messages(messages, None)
    with CompletionClient(
        model="anthropic/claude-sonnet-4-5", cache_breakpoint="anthropic"
    ) as client:
        wire, _, _ = client._prepare_cache_boundary(prepared, responses=False)

    assert wire == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": "live state"},
    ]


@pytest.mark.parametrize("static_prefix", [False, True])
def test_dynamic_system_messages_remain_after_the_cache_boundary(static_prefix: bool) -> None:
    prefix = [{"role": "system", "content": "stable instructions"}] if static_prefix else []
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        transformed, instructions = client._transform_messages(
            [
                *prefix,
                ReplayCarryingMessage(
                    {"role": "system", "content": "live state"}, cache_boundary_before=True
                ),
                {"role": "system", "content": "more live state"},
            ]
        )
        assert instructions == ("stable instructions" if static_prefix else None)
        wire, instructions, enabled = client._prepare_cache_boundary(
            transformed, responses=True, instructions=instructions
        )

    assert enabled is True
    assert instructions is None
    assert wire[-2:] == [
        {"role": "system", "content": "live state"},
        {"role": "system", "content": "more live state"},
    ]
    if static_prefix:
        assert wire[0]["content"][0] == {
            "type": "input_text",
            "text": "stable instructions",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    else:
        assert "prompt_cache_breakpoint" not in repr(wire)


def test_boundary_consumption_preserves_ordinary_empty_messages_and_private_state() -> None:
    state = {"version": 1, "scope": "responses:openai:test"}
    boundary = ReplayCarryingMessage(
        {"role": "user", "content": "live state"},
        llm_state=state,
        reasoning="private reasoning",
        cache_boundary_before=True,
    )
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        messages, _, enabled = client._prepare_cache_boundary([{}, boundary], responses=True)

    assert enabled is True
    assert messages[0] == {}
    assert isinstance(messages[1], ReplayCarryingMessage)
    assert messages[1].llm_state is state
    assert messages[1].reasoning == "private reasoning"
    assert not carried_cache_boundary(messages[1])


def test_multiple_cache_boundaries_fail_loudly() -> None:
    messages: list[dict] = [
        ReplayCarryingMessage({"role": "user", "content": "one"}, cache_boundary_before=True),
        ReplayCarryingMessage({"role": "user", "content": "two"}, cache_boundary_before=True),
    ]
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        with pytest.raises(ValueError, match="more than one cache boundary"):
            client._prepare_cache_boundary(messages, responses=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("api_style", ["chat", "responses"])
async def test_cache_mapping_rejects_per_call_model_overrides(api_style: str) -> None:
    if api_style == "chat":
        client = CompletionClient(model="anthropic/claude-sonnet-4-5", cache_breakpoint="anthropic")
        target = "litellm.acompletion"
    else:
        client = ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai")
        target = "litellm.aresponses"

    try:
        with patch(target, new_callable=AsyncMock) as request:
            with pytest.raises(ValueError, match="per-call model override"):
                await client.acall(_render("state-a"), model="openai/a-different-model")
        request.assert_not_awaited()
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_body", "message"),
    [
        ([], "extra_body must be a mapping"),
        (
            {"prompt_cache_options": "explicit"},
            "extra_body.prompt_cache_options must be a mapping",
        ),
    ],
)
async def test_openai_cache_config_rejects_malformed_mappings(
    extra_body: object, message: str
) -> None:
    client = ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai")
    try:
        with patch("litellm.aresponses", new_callable=AsyncMock) as request:
            with pytest.raises(ValueError, match=message):
                await client.acall(_render("state-a"), extra_body=extra_body)
        request.assert_not_awaited()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_openai_marks_the_stable_prefix_before_dynamic_context() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="test", cache_breakpoint="openai")
    try:
        with patch("litellm.aresponses", new_callable=AsyncMock) as request:
            request.return_value = _responses_output()
            await client.acall(_render("state-a"))
            await client.acall(_render("state-b"))

        first, second = (call.kwargs for call in request.await_args_list)
        assert first["extra_body"]["prompt_cache_options"] == {"mode": "explicit"}
        assert first["input"][:-1] == second["input"][:-1]
        assert first["input"][-1] != second["input"][-1]
        assert first["input"][-2]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert "prompt_cache_breakpoint" not in repr(first["input"][-1]["content"])
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_boundary_is_inert_without_capability_opt_in() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="test")
    try:
        with patch("litellm.aresponses", new_callable=AsyncMock) as request:
            request.return_value = _responses_output()
            await client.acall(_render("state-a"))
        assert request.await_args is not None
        sent = request.await_args.kwargs
        assert "extra_body" not in sent
        assert "prompt_cache_breakpoint" not in repr(sent["input"])
        assert not any(carried_cache_boundary(item) for item in sent["input"])
    finally:
        await client.aclose()


def test_openai_can_mark_a_system_only_stable_prefix() -> None:
    rendered = _render("state-a")
    rendered.pop(1)  # no history yet: stable instructions + volatile suffix
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        transformed, instructions = client._transform_messages(rendered)
        messages, instructions, enabled = client._prepare_cache_boundary(
            transformed, responses=True, instructions=instructions
        )

    assert enabled is True
    assert instructions is None
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "state-a" in messages[-1]["content"]


def test_openai_falls_back_to_instructions_behind_ineligible_output() -> None:
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        messages, instructions, enabled = client._prepare_cache_boundary(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "stable output"}],
                },
                ReplayCarryingMessage({}, cache_boundary_before=True),
                {"role": "user", "content": "live state"},
            ],
            responses=True,
            instructions="stable instructions",
        )

    assert enabled is True
    assert instructions is None
    assert messages[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert messages[1]["content"][0] == {"type": "output_text", "text": "stable output"}


@pytest.mark.asyncio
async def test_openai_fields_reach_the_serialized_http_body() -> None:
    bodies: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.6",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "store": False,
                "tools": [],
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {
                        "cached_tokens": 500,
                        "cache_write_tokens": 250,
                    },
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 1100,
                },
            },
        )

    client = ResponsesClient(
        model="openai/gpt-5.6",
        api_key="test",
        base_url="https://example.test/v1",
        cache_breakpoint="openai",
    )
    assert client._http is not None
    await client._http.httpx_async.aclose()
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._http.httpx_async = transport
    client._http.async_client.client = transport
    try:
        response = await client.acall(_render("state-a"))
    finally:
        await client.aclose()

    assert bodies[0]["prompt_cache_options"] == {"mode": "explicit"}
    assert bodies[0]["input"][-2]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "cache_boundary" not in repr(bodies[0])
    assert response.usage is not None
    assert response.usage.cached_input_tokens == 500
    assert response.usage.cache_write_input_tokens == 250
    hidden_cost = response.raw_response._hidden_params["response_cost"]
    assert isinstance(hidden_cost, (int, float)) and hidden_cost > 0
    assert response.usage.cost_usd == hidden_cost


@pytest.mark.parametrize("bad_cost", [True, "unknown", float("nan"), -1.0])
def test_malformed_litellm_response_cost_warns_without_losing_usage(
    bad_cost: object, caplog: pytest.LogCaptureFixture
) -> None:
    raw_response = SimpleNamespace(
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        _hidden_params={"response_cost": bad_cost},
    )

    usage = _extract_usage(raw_response)

    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.cost_usd == 0.0
    assert "Ignoring malformed LiteLLM response_cost metadata" in caplog.text


@pytest.mark.asyncio
async def test_anthropic_breakpoint_survives_user_message_coalescing() -> None:
    bodies: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    client = CompletionClient(
        model="anthropic/claude-sonnet-4-5",
        api_key="test",
        api_base="https://example.test",
        cache_breakpoint="anthropic",
    )
    assert client._http is not None
    await client._http.httpx_async.aclose()
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._http.httpx_async = transport
    client._http.async_client.client = transport
    try:
        await client.acall(_render("state-a"))
        system_only = _render("state-b")
        system_only.pop(1)
        await client.acall(system_only)
    finally:
        await client.aclose()

    content = bodies[0]["messages"][0]["content"]
    assert "solve this" in content[0]["text"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "state-a" in content[1]["text"]
    assert "cache_control" not in content[1]
    assert bodies[1]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "state-b" in bodies[1]["messages"][0]["content"][0]["text"]
    assert "cache_control" not in bodies[1]["messages"][0]["content"][0]


def test_openai_skips_assistant_output_and_marks_latest_input() -> None:
    boundary = ReplayCarryingMessage({}, cache_boundary_before=True)
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        messages, _, enabled = client._prepare_cache_boundary(
            [
                {"role": "user", "content": "stable input"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}],
                },
                boundary,
                {"role": "user", "content": "live state"},
            ],
            responses=True,
        )

    assert enabled is True
    assert messages[0]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "prompt_cache_breakpoint" not in messages[1]["content"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_tool_call", [False, True])
async def test_anthropic_boundary_skips_assistants_without_public_content(with_tool_call) -> None:
    bodies = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    client = CompletionClient(
        model="anthropic/claude-sonnet-4-5",
        api_key="test",
        api_base="https://example.test",
        cache_breakpoint="anthropic",
        cache_control_injection_points=[],
    )
    assistant = {"role": "assistant", "content": None if with_tool_call else ""}
    suffix = {"role": "user", "content": "live state"}
    if with_tool_call:
        assistant["tool_calls"] = [
            {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ]
        suffix = {"role": "tool", "tool_call_id": "c1", "content": "live result"}
    thinking = [{"type": "thinking", "thinking": "Check the inputs.", "signature": "sig"}]
    state = capture_chat_state(
        {**assistant, "thinking_blocks": thinking}, replay_scope(client.model, "chat", {})
    )
    assert client._http is not None
    await client._http.httpx_async.aclose()
    transport = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client._http.httpx_async = transport
    client._http.async_client.client = transport
    try:
        await client.acall(
            [
                {"role": "user", "content": "stable input"},
                ReplayCarryingMessage(assistant, llm_state=state),
                ReplayCarryingMessage(suffix, cache_boundary_before=True),
            ]
        )
    finally:
        await client.aclose()

    messages = bodies[0]["messages"]
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in repr(messages[1:])
    assert messages[1]["content"][0] == thinking[0]


def test_openai_can_mark_a_stable_function_result() -> None:
    boundary = ReplayCarryingMessage({}, cache_boundary_before=True)
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        messages, _, enabled = client._prepare_cache_boundary(
            [
                {"type": "function_call_output", "call_id": "c1", "output": "done"},
                boundary,
                {"role": "user", "content": "live state"},
            ],
            responses=True,
        )

    assert enabled is True
    assert messages[0]["output"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_replay_expansion_stays_inside_the_stable_prefix() -> None:
    scope = "responses:openai:sha256:test"
    state = {
        "version": 1,
        "scope": scope,
        "format": "openai-responses",
        "payload": {
            "items": [{"type": "reasoning", "encrypted_content": "opaque"}],
            "order": [
                {"type": "reasoning", "index": 0},
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "run",
                    "arguments": "{}",
                },
            ],
        },
    }
    with ResponsesClient(model="openai/gpt-5.6", cache_breakpoint="openai") as client:
        transformed, instructions = client._transform_messages(
            [
                {"role": "user", "content": "run it"},
                *carry_replay_batch(
                    [
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "run",
                            "arguments": "{}",
                        }
                    ],
                    state,
                    None,
                ),
                {"type": "function_call_output", "call_id": "c1", "output": "done"},
                ReplayCarryingMessage(
                    {"role": "user", "content": "live state"}, cache_boundary_before=True
                ),
            ],
            scope,
        )
        messages, _, enabled = client._prepare_cache_boundary(
            transformed, responses=True, instructions=instructions
        )

    assert enabled is True
    assert [item.get("type", item.get("role")) for item in messages] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
        "user",
    ]
    assert messages[-2]["output"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert messages[-1] == {"role": "user", "content": "live state"}


def test_gemini_gets_no_invented_inline_cache_field() -> None:
    with CompletionClient(
        model="gemini/gemini-2.5-pro", cache_control_injection_points=[]
    ) as client:
        messages, _, enabled = client._prepare_cache_boundary(_render("state-a"), responses=False)

    assert enabled is False
    assert "cache_control" not in repr(messages)
    assert "prompt_cache_breakpoint" not in repr(messages)
    assert not any(carried_cache_boundary(item) for item in messages)


@pytest.mark.asyncio
async def test_gemini_boundary_is_inert_on_the_actual_call_path() -> None:
    """The neutral boundary must not become a native Gemini wire field.

    ``cache_control_injection_points`` is an older, separate gateway-level
    extension point. Disable it here so this regression pins only the automatic
    dynamic-context boundary introduced by this change.
    """
    client = CompletionClient(model="gemini/gemini-2.5-pro", cache_control_injection_points=[])
    response = litellm.ModelResponse(
        model="gemini-2.5-pro",
        choices=[
            litellm.Choices(
                index=0,
                finish_reason="stop",
                message=litellm.Message(role="assistant", content="ok"),
            )
        ],
    )
    try:
        with patch("litellm.acompletion", new_callable=AsyncMock) as request:
            request.return_value = response
            await client.acall(_render("state-a"))
        assert request.await_args is not None
        sent = request.await_args.kwargs["messages"]
        assert "cache_control" not in repr(sent)
        assert "prompt_cache_breakpoint" not in repr(sent)
        assert not any(carried_cache_boundary(item) for item in sent)
    finally:
        await client.aclose()
