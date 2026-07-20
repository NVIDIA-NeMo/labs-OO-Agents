# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ResponsesProviderFormatter and ResponsesClient._transform_messages.

Tests the formatting pipeline without requiring a live API.
"""

import json

import pytest

from nooa.context_blocks.formatter import ResponsesProviderFormatter
from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo


class TestResponsesProviderFormatter:
    """Test that ResponsesProviderFormatter emits correct wire format."""

    def test_simple_user_message(self):
        messages = [
            RenderedMessage(role=Role.USER, content="Hello"),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_message_preserved_in_list(self):
        """System messages stay in the list for downstream budget clamping."""
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="You are helpful."),
            RenderedMessage(role=Role.USER, content="Hi"),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

    def test_tool_call_format(self):
        """Tool calls become function_call items."""
        messages = [
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(
                    id="call_123", name="execute_python", arguments={"code": "print(1)"}
                ),
            ),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "execute_python",
                "arguments": json.dumps({"code": "print(1)"}),
            }
        ]

    def test_tool_result_format(self):
        """Tool results become function_call_output items."""
        messages = [
            RenderedMessage(role=Role.TOOL, content="status: complete", tool_call_id="call_123"),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "status: complete",
            }
        ]

    def test_multi_turn_tool_calling(self):
        """Full multi-turn conversation with tool calls renders correctly."""
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="You are a coding assistant."),
            RenderedMessage(role=Role.USER, content="Add 2+2"),
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(id="tc_1", name="execute_python", arguments={"code": "2+2"}),
            ),
            RenderedMessage(role=Role.TOOL, content="4", tool_call_id="tc_1"),
            RenderedMessage(role=Role.USER, content="Now multiply by 3"),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Add 2+2"},
            {
                "type": "function_call",
                "call_id": "tc_1",
                "name": "execute_python",
                "arguments": json.dumps({"code": "2+2"}),
            },
            {"type": "function_call_output", "call_id": "tc_1", "output": "4"},
            {"role": "user", "content": "Now multiply by 3"},
        ]

    def test_skips_metadata_and_runtime_event_roles(self):
        messages = [
            RenderedMessage(role=Role.RUNTIME_EVENT, content="internal"),
            RenderedMessage(role=Role.METADATA, content="meta"),
            RenderedMessage(role=Role.USER, content="visible"),
        ]
        formatter = ResponsesProviderFormatter()
        result = formatter.format(messages)

        assert result == [{"role": "user", "content": "visible"}]


class TestResponsesClientTransformMessages:
    """Test _transform_messages handles both native and legacy formats."""

    @pytest.fixture
    def client(self):
        from nooa.unifiedllm import ResponsesClient

        return ResponsesClient(model="test-model", api_key="fake")

    def test_native_format_passthrough(self, client):
        """Messages already in Responses format pass through unchanged."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"type": "function_call", "call_id": "tc1", "name": "foo", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "tc1", "output": "bar"},
        ]
        input_msgs, instructions = client._transform_messages(messages)

        assert instructions == "Be helpful."
        assert input_msgs == [
            {"role": "user", "content": "Hi"},
            {"type": "function_call", "call_id": "tc1", "name": "foo", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "tc1", "output": "bar"},
        ]

    def test_legacy_openai_format_conversion(self, client):
        """Legacy OpenAI chat format is converted to Responses format."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "execute_python", "arguments": '{"code": "1+1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "2"},
        ]
        input_msgs, instructions = client._transform_messages(messages)

        assert instructions == "System prompt"
        assert input_msgs == [
            {"role": "user", "content": "Do something"},
            {
                "type": "function_call",
                "call_id": "tc1",
                "name": "execute_python",
                "arguments": '{"code": "1+1"}',
            },
            {"type": "function_call_output", "call_id": "tc1", "output": "2"},
        ]

    def test_multiple_system_messages_concatenated(self, client):
        """Multiple system messages are joined with double newline."""
        messages = [
            {"role": "system", "content": "Part 1"},
            {"role": "system", "content": "Part 2"},
            {"role": "user", "content": "Hi"},
        ]
        input_msgs, instructions = client._transform_messages(messages)

        assert instructions == "Part 1\n\nPart 2"
        assert input_msgs == [{"role": "user", "content": "Hi"}]

    def test_no_system_messages_returns_none(self, client):
        """No system messages → instructions is None."""
        messages = [{"role": "user", "content": "Hi"}]
        input_msgs, instructions = client._transform_messages(messages)

        assert instructions is None
        assert input_msgs == [{"role": "user", "content": "Hi"}]
