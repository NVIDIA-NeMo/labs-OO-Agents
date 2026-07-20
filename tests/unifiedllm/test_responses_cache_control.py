# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cache_control injection in ResponsesClient."""

from unittest.mock import AsyncMock, patch

import pytest

from nooa.unifiedllm import ResponsesClient


def make_mock_responses_response(content: str = "ok"):
    """Create a minimal litellm.ResponsesAPIResponse for testing."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.output = [MagicMock(type="message", content=[MagicMock(type="output_text", text=content)])]
    resp.output_text = content
    resp.usage = None
    return resp


class TestResponsesClientCacheControlDefaults:
    """ResponsesClient should have cache_control_injection_points by default."""

    def test_default_has_system_and_last_tool(self):
        """ResponsesClient gets the default injection points from UnifiedLLM."""
        client = ResponsesClient(model="test-model")
        assert {"role": "system"} in client.cache_control_injection_points
        assert {"role": "tool", "position": "last"} in client.cache_control_injection_points

    def test_inject_cache_control_on_system(self):
        """_inject_cache_control marks system messages."""
        client = ResponsesClient(model="test-model")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = client._inject_cache_control(messages, [{"role": "system"}])
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]


class TestResponsesClientCacheControlInjection:
    """Tests that cache_control is injected and preserved through _transform_messages."""

    @pytest.fixture
    def client(self):
        return ResponsesClient(model="test-model")

    def test_system_cache_control_not_in_output(self, client):
        """System messages are extracted to instructions; cache_control on system is harmless."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ]
        prepared = client._inject_cache_control(messages, [{"role": "system"}])
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
            {"role": "tool", "content": "result 1", "tool_call_id": "tc1"},
            {"role": "user", "content": "What next?"},
        ]
        # Inject cache_control on last tool
        prepared = client._inject_cache_control(messages, [{"role": "tool", "position": "last"}])
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
            {"type": "function_call_output", "call_id": "tc1", "output": "result"},
            {"role": "user", "content": "Next"},
        ]
        # The injection should find function_call_output as equivalent to "tool"
        prepared = client._inject_cache_control(messages, [{"role": "tool", "position": "last"}])
        # The function_call_output item should have cache_control
        fco = [m for m in prepared if m.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert fco[0].get("cache_control") == {"type": "ephemeral"}

    def test_user_message_cache_control_preserved(self, client):
        """cache_control on user messages is preserved in native format."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        prepared = client._inject_cache_control(messages, [{"role": "user"}])
        input_msgs, _ = client._transform_messages(prepared)
        user_msgs = [m for m in input_msgs if m.get("role") == "user"]
        assert user_msgs[0].get("cache_control") == {"type": "ephemeral"}


class TestToolOutputNotCorrupted:
    """Ensure tool message output stays a string after position-based injection."""

    def test_tool_output_remains_string_after_position_injection(self):
        """When last-tool injection converts content to blocks, output must stay a string."""
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
            {"role": "tool", "content": "tool output text", "tool_call_id": "tc1"},
            {"role": "user", "content": "Next"},
        ]
        # Position-based injection converts tool content to list of blocks
        prepared = client._inject_cache_control(messages, [{"role": "tool", "position": "last"}])
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
                    {"role": "tool", "content": "tool output", "tool_call_id": "tc1"},
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
    async def test_acall_no_injection_when_empty(self):
        """No cache_control when injection_points is empty."""
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        client.cache_control_injection_points = []
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
    async def test_acall_custom_injection_points(self):
        """Custom injection points override defaults."""
        client = ResponsesClient(model=self.ANTHROPIC_MODEL)
        mock_response = make_mock_responses_response()

        with patch("litellm.aresponses", new_callable=AsyncMock) as mock_aresponses:
            mock_aresponses.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "Bye"},
                ],
                cache_control_injection_points=[{"role": "user", "position": "last"}],
            )

            call_kwargs = mock_aresponses.call_args[1]
            input_items = call_kwargs["input"]
            # Last user message should have content-block-level cache_control
            last_user = [m for m in input_items if m.get("role") == "user"][-1]
            assert isinstance(last_user["content"], list)
            assert last_user["content"][0]["cache_control"] == {"type": "ephemeral"}


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
    future change to _transform_messages or _inject_cache_control starts
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
