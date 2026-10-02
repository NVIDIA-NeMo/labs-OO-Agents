# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bad public input is rejected before the provider SDK is called."""

import pytest

from nooa.unifiedllm.direct import anthropic_request


@pytest.mark.parametrize("limit", [None, False, 0, -1, "10"])
def test_anthropic_requires_positive_reply_limit(limit):
    with pytest.raises(ValueError, match="max_tokens"):
        anthropic_request({"messages": [], "max_tokens": limit})


@pytest.mark.parametrize(
    "fmt",
    [{"type": "text"}, {"type": "json_object"}, {"type": "json_schema", "json_schema": {}}, 1],
)
def test_response_format_has_actionable_error(fmt):
    with pytest.raises(ValueError, match="response_format.*json_schema"):
        anthropic_request({"messages": [], "max_tokens": 10, "response_format": fmt})


def test_null_response_format_is_absent():
    assert "output_config" not in anthropic_request(
        {"messages": [], "max_tokens": 10, "response_format": None}
    )


@pytest.mark.parametrize(
    "message,field",
    [
        ({"content": "x"}, "role"),
        ({"role": "tool", "content": "x"}, "tool_call_id"),
        ({"role": "assistant", "thinking_blocks": "oops"}, "thinking_blocks"),
        ({"role": "assistant", "tool_calls": [{}]}, "tool_calls"),
        (
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "not json"}}],
            },
            "arguments",
        ),
        *[
            ({"role": "user", "content": [block]}, "content")
            for block in [
                {"type": "video"},
                {"text": "x"},
                {"type": "input_image"},
                {"type": "image_url"},
                {"type": "image_url", "image_url": {}},
                {"type": "image_url", "image_url": {"url": None}},
                {"type": "image_url", "image_url": "data:image/png,abc"},
                {"type": "file"},
                {"type": "file", "file": {"file_data": "abc"}},
            ]
        ],
    ],
)
def test_invalid_message_names_index_and_field(message, field):
    with pytest.raises(ValueError, match=rf"message\[0\].*{field}"):
        anthropic_request({"messages": [message], "max_tokens": 10})


def test_tool_arguments_may_already_be_a_dictionary():
    result = anthropic_request(
        {
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": {"x": 1}}}],
                }
            ],
        }
    )
    assert result["messages"][0]["content"][0]["input"] == {"x": 1}


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"tool_choice": 1}, "tool_choice"),
        ({"tool_choice": {"type": "function", "function": {}}}, "tool_choice"),
        ({"tool_choice": {"type": []}}, "tool_choice"),
        ({"tool_choice": {"type": "auto", "unknown": True}}, "tool_choice"),
        ({"tools": [{}]}, "tools"),
        ({"tools": 1}, "tools"),
        ({"tools": [{"type": "function", "function": 1}]}, "function"),
        (
            {"tools": [{"type": "function", "function": {"name": "f", "parameters": []}}]},
            "parameters",
        ),
        (
            {
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {}, "strict": 1}}
                ]
            },
            "strict",
        ),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {}, "unsupported": True},
                    }
                ]
            },
            "unsupported",
        ),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {}},
                        "unsupported": True,
                    }
                ]
            },
            "unsupported",
        ),
        *[
            (
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "f", "parameters": {}},
                            "cache_control": cache,
                        }
                    ]
                },
                "cache_control",
            )
            for cache in (
                None,
                1,
                {},
                {"type": "permanent"},
                {"type": "ephemeral", "ttl": "10m"},
                {"type": "ephemeral", "ttl": []},
                {"type": "ephemeral", "unknown": True},
            )
        ],
        ({"output_config": None}, "output_config"),
        ({"output_config": 1}, "output_config"),
    ],
)
def test_anthropic_malformed_nested_options_raise_value_error(patch, field):
    with pytest.raises(ValueError, match=field):
        anthropic_request({"messages": [], "max_tokens": 10, **patch})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "patch,field",
    [
        (
            {
                "tool_choice": 1,
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            },
            "tool_choice",
        ),
        ({"tools": [{}]}, "tools"),
        ({"output_config": None}, "output_config"),
        ({"extra_body": {"output_config": None}}, "output_config"),
    ],
)
async def test_anthropic_nested_options_rejected_before_http(
    monkeypatch, asynchronous, patch, field
):
    import httpx

    from nooa.unifiedllm import CompletionClient

    def no_network(*args, **kwargs):
        raise AssertionError("Invalid input reached HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    # Dictionary schemas are transport/config tools, not the call's list[Tool]
    # callable API. Keep tools in config to exercise the native translator.
    options = dict(patch)
    tools = options.pop("tools", None)
    async with CompletionClient(
        "anthropic/test",
        direct=True,
        api_key="test",
        max_tokens=10,
        **({"tools": tools} if tools is not None else {}),
    ) as llm:
        with pytest.raises(ValueError, match=field):
            history = [{"role": "user", "content": "test"}]
            await llm.acall(history, **options) if asynchronous else llm.call(history, **options)


def test_anthropic_conflicting_output_formats_rejected():
    with pytest.raises(ValueError, match="output_config.format"):
        anthropic_request(
            {
                "messages": [],
                "max_tokens": 10,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
                "output_config": {"format": {"type": "text"}},
            }
        )


@pytest.mark.parametrize("scope", [None, "chat:openai:sha256:test", "chat:gemini:sha256:test"])
def test_raw_google_signature_requires_canonical_turn(scope):
    from nooa.unifiedllm.errors import ReasoningReplayError
    from nooa.unifiedllm.replay_state import reject_native_message

    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call",
                "function": {"name": "f", "arguments": "{}"},
                "extra_content": {"google": {"thought_signature": "SECRET"}},
            }
        ],
    }
    with pytest.raises(ReasoningReplayError, match="canonical LLMResponse"):
        reject_native_message(message, scope, reject_google_signature=True)
    message["tool_calls"][0]["extra_content"] = {"unrelated": "value"}
    reject_native_message(message, scope, reject_google_signature=True)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize(
    "fmt", [None, 1, [], {}, {"type": "text"}, {"type": "json_schema", "schema": []}]
)
async def test_malformed_native_output_format_never_reaches_http(
    monkeypatch, asynchronous, extra, fmt
):
    import httpx

    from nooa.unifiedllm import CompletionClient

    def no_network(*args, **kwargs):
        raise AssertionError("Malformed output_config.format reached HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    patch = {"output_config": {"format": fmt}}
    if extra:
        patch = {"extra_body": patch}
    async with CompletionClient(
        "anthropic/test", direct=True, api_key="test", max_tokens=10
    ) as llm:
        with pytest.raises(ValueError, match="output_config.format"):
            history = [{"role": "user", "content": "test"}]
            await llm.acall(history, **patch) if asynchronous else llm.call(history, **patch)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("choice", [1, {}, {"type": []}, {"type": "function", "function": {}}])
async def test_malformed_native_choice_without_tools_is_not_silently_dropped(asynchronous, choice):
    from nooa.unifiedllm import CompletionClient

    async with CompletionClient(
        "anthropic/test", direct=True, api_key="test", max_tokens=10
    ) as llm:
        with pytest.raises(ValueError, match="tool_choice"):
            history = [{"role": "user", "content": "test"}]
            await llm.acall(history, tool_choice=choice) if asynchronous else llm.call(
                history, tool_choice=choice
            )


def test_google_dictionary_guard_is_direct_only_for_default_compatibility():
    from nooa.unifiedllm.replay_state import reject_native_message

    reject_native_message(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call",
                    "function": {"name": "f", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "SECRET"}},
                }
            ],
        },
        None,
    )
