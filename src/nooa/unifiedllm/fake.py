# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake LLM client for deterministic testing."""

import asyncio
import json
from collections import deque
from typing import Any

from pydantic import BaseModel

from nooa.unifiedllm.unifiedllm import LLMResponse, Tool, ToolCall, UnifiedLLM


class FakeLLMClient(UnifiedLLM):
    """
    Fake LLM client that returns scripted responses.

    Useful for hermetic testing without network calls.
    Thread-safe for concurrent calls.
    """

    def __init__(
        self,
        scripted_responses: list[LLMResponse] | None = None,
    ):
        """
        Initialize fake client.

        Args:
            scripted_responses: Pre-defined responses to return (in order).
        """
        super().__init__(model="fake-model")
        self._response_queue = deque(scripted_responses or [])
        self._lock = asyncio.Lock()
        self.call_count = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[Tool] | None = None
        self._context_window = 128_000

    @property
    def context_window(self) -> int | None:
        """Return fake context window size for testing."""
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Fake token counter - rough estimate of 4 chars per token."""
        return len(text) // 4 + 1

    async def acall(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Return scripted response.

        Captures call arguments for assertions.
        Thread-safe: uses asyncio.Lock to ensure concurrent calls get responses in order.
        """
        async with self._lock:
            self.call_count += 1
            self.last_messages = messages
            self.last_tools = tools

            # Return next response from queue, or empty response if none left
            if self._response_queue:
                return self._response_queue.popleft()
            else:
                # Return empty response if we've exhausted scripted responses
                return LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": ""},
                    reasoning=None,
                    usage=None,
                )

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous version of acall for UnifiedLLM compatibility."""
        # For sync call, we don't need locking since tests are usually single-threaded
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools

        if self._response_queue:
            return self._response_queue.popleft()
        else:
            return LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": ""},
                reasoning=None,
                usage=None,
            )

    def reset(self) -> None:
        """Reset call history."""
        self.call_count = 0
        self.last_messages = []
        self.last_tools = None

    @classmethod
    def with_code_responses(cls, code_strings: list[str]) -> "FakeLLMClient":
        """
        Create a fake client that returns multiple code generation responses.

        Useful for testing agent code generation with retries.

        Args:
            code_strings: List of code strings to return (in order)

        Returns:
            FakeLLMClient configured with code responses
        """
        responses = []
        for code in code_strings:
            responses.append(
                LLMResponse(
                    raw_response=None,
                    content=code,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": code},
                    reasoning=None,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": len(code.split()),
                        "total_tokens": 10 + len(code.split()),
                    },
                )
            )
        return cls(scripted_responses=responses)

    @classmethod
    def simple_message(cls, message: str) -> "FakeLLMClient":
        """
        Create a fake client that returns a simple message.

        Args:
            message: Message content to return

        Returns:
            FakeLLMClient configured with message response
        """
        words = message.split()
        return cls(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=message,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": message},
                    reasoning=None,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": len(words),
                        "total_tokens": 10 + len(words),
                    },
                )
            ]
        )

    @classmethod
    def with_tool_call(
        cls,
        tool_name: str,
        tool_args: dict[str, Any],
        message: str | None = None,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns a tool call.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments (will be JSON-serialized)
            message: Optional message before tool call

        Returns:
            FakeLLMClient configured with tool call
        """
        return cls(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=message or "",
                    tool_calls=[
                        ToolCall(
                            id="call_fake_123",
                            name=tool_name,
                            arguments=json.dumps(tool_args),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={
                        "role": "assistant",
                        "content": message,
                        "tool_calls": [
                            {
                                "id": "call_fake_123",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    },
                    reasoning=None,
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )
            ]
        )

    @classmethod
    def with_reasoning(
        cls,
        reasoning: str,
        message: str,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns reasoning + message (o1-style).

        Args:
            reasoning: Internal reasoning text
            message: Final message

        Returns:
            FakeLLMClient configured with reasoning
        """
        return cls(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=message,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": message},
                    reasoning=reasoning,
                    usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
                )
            ]
        )
