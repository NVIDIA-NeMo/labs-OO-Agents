# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit cache-boundary transport in ResponsesClient."""

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from nooa.context_blocks.models import CACHE_BOUNDARY_MESSAGE_KEY
from nooa.unifiedllm import ResponsesClient


def make_mock_responses_response(content: str = "ok"):
    """Create a minimal litellm.ResponsesAPIResponse for testing."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.output = [MagicMock(type="message", content=[MagicMock(type="output_text", text=content)])]
    resp.output_text = content
    resp.usage = None
    return resp


class TestResponsesClientCacheControlInjection:
    """Tests that cache_control is injected and preserved through _transform_messages."""

    @pytest.fixture
    def client(self):
        return ResponsesClient(model="test-model")

    def test_system_cache_control_not_in_output(self, client):
        """System messages are extracted to instructions; cache_control on system is harmless."""
        messages = [
            {
                "role": "system",
                "content": "System prompt",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "Hi"},
        ]
        prepared = client._apply_cache_boundaries(messages)
        input_msgs, instructions = client._transform_messages(prepared)
        # System extracted to instructions
        assert instructions == "System prompt"
        # Input should just have the user message
        assert len(input_msgs) == 1
        assert input_msgs[0]["role"] == "user"

    def test_tool_cache_control_preserved_in_native_format(self, client):
        """cache_control on tool messages is preserved as function_call_output."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result 1",
                "tool_call_id": "tc1",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "What next?"},
        ]
        prepared = client._apply_cache_boundaries(messages)
        input_msgs, _ = client._transform_messages(prepared)

        # Find the function_call_output item
        fco_items = [m for m in input_msgs if m.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        # Should have cache_control preserved
        assert "cache_control" in fco_items[0]
        assert fco_items[0]["cache_control"] == {"type": "ephemeral"}

    def test_native_format_function_call_output_gets_cache_control(self, client):
        """When messages are already in native format, function_call_output gets marked."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do something"},
            {"type": "function_call", "call_id": "tc1", "name": "run", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "tc1",
                "output": "result",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "Next"},
        ]
        prepared = client._apply_cache_boundaries(messages)
        # The function_call_output item should have cache_control
        fco = [m for m in prepared if m.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert fco[0].get("cache_control") == {"type": "ephemeral"}

    def test_user_message_cache_control_preserved(self, client):
        """cache_control on user messages is preserved in native format."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello", CACHE_BOUNDARY_MESSAGE_KEY: True},
        ]
        prepared = client._apply_cache_boundaries(messages)
        input_msgs, _ = client._transform_messages(prepared)
        user_msgs = [m for m in input_msgs if m.get("role") == "user"]
        assert user_msgs[0].get("cache_control") == {"type": "ephemeral"}

    def test_direct_nested_assistant_annotation_is_not_moved(self, client):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "working",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
            }
        ]
        original = deepcopy(messages)

        transformed, _ = client._transform_messages(messages)

        assert transformed[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in transformed[1]
        assert messages == original


class TestToolOutputNotCorrupted:
    """Ensure tool message output stays a string after boundary translation."""

    def test_tool_output_remains_string_after_boundary(self):
        client = ResponsesClient(model="test-model")
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "tool output text",
                "tool_call_id": "tc1",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "Next"},
        ]
        prepared = client._apply_cache_boundaries(messages)
        input_msgs, _ = client._transform_messages(prepared)

        fco = [m for m in input_msgs if m.get("type") == "function_call_output"]
        assert len(fco) == 1
        # output MUST be a string, not a list
        assert isinstance(fco[0]["output"], str)
        assert fco[0]["output"] == "tool output text"
        # cache_control should also be present
        assert fco[0].get("cache_control") == {"type": "ephemeral"}


class TestResponsesClientEndToEnd:
    """End-to-end tests that cache_control reaches litellm.aresponses on Anthropic models.

    The end-to-end pipeline is gated on _is_anthropic_model so that OpenAI/Azure/NIM
    Responses calls don't ship cache_control keys (the Responses API rejects them).
    These tests pin a Claude alias so the pipeline runs and we can assert the marker
    survives _transform_messages.
    """

    ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_acall_injects_cache_control(self):
        """acall() injects cache_control on messages before calling litellm."""
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Do something"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "tool output",
                        "tool_call_id": "tc1",
                        CACHE_BOUNDARY_MESSAGE_KEY: True,
                    },
                    {"role": "user", "content": "Current turn"},
                ],
            )

            call_kwargs = mock_aresponses.call_args[1]
            input_items = call_kwargs["input"]

            # Find function_call_output (tool result)
            fco_items = [m for m in input_items if m.get("type") == "function_call_output"]
            assert len(fco_items) == 1
            # Last tool should have cache_control
            assert fco_items[0].get("cache_control") == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_acall_without_boundary_has_no_cache_control(self):
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hi"},
                ],
            )

            call_kwargs = mock_aresponses.call_args[1]
            input_items = call_kwargs["input"]
            # No items should have cache_control
            for item in input_items:
                assert "cache_control" not in item

    @pytest.mark.asyncio
    async def test_acall_rejects_legacy_option(self):
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        with pytest.raises(TypeError, match="cache_control_injection_points was removed"):
            await client.acall(
                [{"role": "user", "content": "Hello"}],
                cache_control_injection_points=[{"role": "user", "position": "last"}],
            )

    @pytest.mark.asyncio
    async def test_explicit_boundary_maps_to_exact_native_item(self):
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()
        messages = [
            {"role": "system", "content": "System"},
            {
                "type": "function_call_output",
                "call_id": "tc1",
                "output": "first",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "next"},
        ]

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response
            await client.acall(messages)

        sent = mock_aresponses.call_args.kwargs["input"]
        assert sent[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in sent[1]
        assert CACHE_BOUNDARY_MESSAGE_KEY in messages[1]

    @pytest.mark.asyncio
    async def test_tool_call_boundary_follows_complete_expansion(self):
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()
        messages = [
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            }
        ]

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response
            await client.acall(messages)

        sent = mock_aresponses.call_args.kwargs["input"]
        assert sent[0]["role"] == "assistant"
        assert not _has_cache_control_anywhere(sent[0])
        assert sent[1]["type"] == "function_call"
        assert sent[1]["cache_control"] == {"type": "ephemeral"}
        assert CACHE_BOUNDARY_MESSAGE_KEY in messages[0]

    @pytest.mark.asyncio
    async def test_explicit_system_boundary_is_content_preserving_noop(self):
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()
        messages = [
            {
                "role": "system",
                "content": "System",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "user", "content": "next"},
        ]

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response
            await client.acall(messages)

        sent = mock_aresponses.call_args.kwargs
        assert sent["instructions"] == "System"
        assert not any(_has_cache_control_anywhere(item) for item in sent["input"])


def _has_cache_control_anywhere(item: dict) -> bool:
    """Recursively check whether `cache_control` appears anywhere in an input[] item."""
    if not isinstance(item, dict):
        return False
    if "cache_control" in item:
        return True
    for v in item.values():
        if isinstance(v, dict) and _has_cache_control_anywhere(v):
            return True
        if isinstance(v, list):
            for sub in v:
                if isinstance(sub, dict) and _has_cache_control_anywhere(sub):
                    return True
    return False


def _make_non_anthropic_messages() -> list[dict]:
    """Fresh message payload for each non-Anthropic regression test.

    A factory (rather than a shared class-level constant) so that even if a
    future change to _transform_messages or _apply_cache_boundaries starts
    mutating the input list, parametrized cases stay independent.
    """
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Do something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "run", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "tool output", "tool_call_id": "tc1"},
        {"role": "user", "content": "Current turn"},
    ]


class TestNonAnthropicResponsesPath:
    """Regression: non-Anthropic Responses calls must NOT ship cache_control.

    litellm.aresponses passes input[] through verbatim — unlike the Chat Completions
    path, there's no OpenAIGPTConfig strip — so a stray cache_control key on any item
    triggers a 400 'Unknown parameter: input[N].cache_control' at the OpenAI/Azure/NIM
    gateway. ResponsesClient.{call,acall} must gate the inject on _is_anthropic_model.
    """

    @pytest.mark.parametrize(
        "model",
        [
            # Direct OpenAI route via NVIDIA gateway (the bug reported in 24bbe09f)
            "openai/openai/openai/gpt-5.5",
            # Azure-routed OpenAI
            "openai/azure/openai/gpt-5.5",
            # NVIDIA Nemotron
            "openai/nvidia/nemotron-3-super-v3",
        ],
    )
    @pytest.mark.asyncio
    async def test_acall_no_cache_control_on_non_anthropic(self, model: str):
        """acall() must not inject cache_control for non-Anthropic Responses models.

        litellm.aresponses passes input[] verbatim; a stray cache_control key
        triggers a 400 'Unknown parameter: input[N].cache_control' at the gateway.
        """
        client = ResponsesClient(model=model)
        mock_response = make_mock_responses_response()

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response
            await client.acall(_make_non_anthropic_messages())

            call_kwargs = mock_aresponses.call_args[1]
            input_items = call_kwargs["input"]

            for i, item in enumerate(input_items):
                assert not _has_cache_control_anywhere(item), (
                    f"cache_control leaked to input[{i}] for non-Anthropic model {model!r}: "
                    f"{item!r}"
                )

    def test_call_no_cache_control_on_non_anthropic(self):
        """Sync variant — same gate."""
        client = ResponsesClient(model="openai/openai/openai/gpt-5.5")
        mock_response = make_mock_responses_response()

        with patch("litellm.responses") as mock_responses:
            mock_responses.return_value = mock_response
            client.call(_make_non_anthropic_messages())

            call_kwargs = mock_responses.call_args[1]
            input_items = call_kwargs["input"]
            for i, item in enumerate(input_items):
                assert not _has_cache_control_anywhere(item), (
                    f"cache_control leaked to input[{i}] (sync path): {item!r}"
                )

    @pytest.mark.asyncio
    async def test_explicit_boundary_is_stripped_for_non_anthropic(self):
        client = ResponsesClient(model="openai/openai/openai/gpt-5.5")
        mock_response = make_mock_responses_response()
        messages = [
            {
                "role": "user",
                "content": "Hi",
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            }
        ]

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response
            await client.acall(messages)

        assert mock_aresponses.call_args.kwargs["input"] == [{"role": "user", "content": "Hi"}]
        assert CACHE_BOUNDARY_MESSAGE_KEY in messages[0]
