# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake LLM client for deterministic testing."""

import asyncio
import copy
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

from nooa.llm_types import CacheBoundary, LLMResponse, LLMUsage, ToolCall
from nooa.unifiedllm.unifiedllm import Tool, UnifiedLLM

from .cache_policy import apply_cache_policy
from .replay_state import prepare_chat_messages


class FakeLLMResponseExhaustedError(RuntimeError):
    """Raised when strict fake-LLM execution has no scripted response left."""


@dataclass(frozen=True, slots=True)
class FakeLLMToolSnapshot:
    """Immutable model-facing metadata captured for one fake LLM tool."""

    name: str
    description: str
    parameters_model: type[BaseModel] | None


@dataclass(frozen=True, slots=True)
class FakeLLMCall:
    """Read-only snapshot of one call made through :class:`FakeLLMClient`."""

    index: int
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Any, ...] | None
    output_model: type[BaseModel] | None
    kwargs: Mapping[str, Any]
    response: LLMResponse | None
    error: Exception | None


def _snapshot(value: Any) -> Any:
    """Detach mutable call inputs and freeze their container structure."""
    if isinstance(value, Tool):
        return FakeLLMToolSnapshot(
            name=value.name,
            description=value.description,
            parameters_model=value.parameters_model,
        )
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_snapshot(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        # Some test doubles deliberately wrap non-copyable runtime objects. Their
        # surrounding container is still frozen; retain the opaque leaf by identity.
        return value


def _response_snapshot(response: LLMResponse) -> LLMResponse:
    """Detach a recorded outcome without making fake calls reject opaque raw responses."""
    try:
        return cast(LLMResponse, response.model_copy(deep=True))  # type: ignore[no-untyped-call]
    except Exception:
        return cast(LLMResponse, response.model_copy())  # type: ignore[no-untyped-call]


class FakeLLMClient(UnifiedLLM):
    """
    Fake LLM client that returns scripted responses.

    Useful for hermetic testing without network calls.
    Thread-safe for concurrent calls.
    """

    def __init__(
        self,
        scripted_responses: list[LLMResponse] | None = None,
        *,
        strict_exhaustion: bool = False,
    ):
        """
        Initialize fake client.

        Args:
            scripted_responses: Pre-defined responses to return (in order).
            strict_exhaustion: Raise when a call has no scripted response instead
                of returning the compatibility empty response.
        """
        super().__init__(model="fake-model")
        # Each provider call owns one canonical response/event. Tests often use
        # ``[response] * n`` as shorthand; materialize those aliases as distinct
        # event objects while preserving the first response by identity.
        responses: list[LLMResponse] = []
        seen: set[int] = set()
        for response in scripted_responses or []:
            if id(response) in seen:
                response = response.model_copy(  # type: ignore[no-untyped-call]
                    update={
                        "id": str(uuid4()),
                        "metadata": dict(response.metadata),
                        "tag": None,
                        "timestamp": datetime.now(),
                    }
                )
            seen.add(id(response))
            responses.append(response)
        self._response_queue = deque(responses)
        self._strict_exhaustion = strict_exhaustion
        self._lock = asyncio.Lock()
        self._calls: list[FakeLLMCall] = []
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

    @property
    def calls(self) -> tuple[FakeLLMCall, ...]:
        """Return the ordered, read-only transcript of calls made so far."""
        return tuple(self._calls)

    @property
    def remaining_responses(self) -> int:
        """Return the number of scripted responses that have not been consumed."""
        return len(self._response_queue)

    async def acall(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
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
            return self._scripted_call(messages, tools, output_model, kwargs)

    def call(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None = None,
        output_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous version of acall for UnifiedLLM compatibility."""
        return self._scripted_call(messages, tools, output_model, kwargs)

    def _scripted_call(
        self,
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary],
        tools: list[Tool] | None,
        output_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        """Consume one scripted response and record the normalized call."""
        call_config = self._prepare_call_config(kwargs)
        self.call_count += 1
        self.last_messages, _, _ = apply_cache_policy(
            prepare_chat_messages(messages, None), None, responses=False
        )
        self.last_tools = tools

        if self._response_queue:
            response = self._response_queue.popleft()
            error = None
        elif self._strict_exhaustion:
            response = None
            error = FakeLLMResponseExhaustedError(
                f"FakeLLMClient has no scripted response for call {self.call_count}; "
                "add a response or disable strict_exhaustion"
            )
        else:
            response = LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[],
                finish_reason="stop",
                reasoning=None,
                usage=None,
            )
            error = None

        self._calls.append(
            FakeLLMCall(
                index=self.call_count,
                messages=_snapshot(self.last_messages),
                tools=_snapshot(tools),
                output_model=output_model,
                kwargs=_snapshot(call_config),
                response=_response_snapshot(response) if response is not None else None,
                error=error,
            )
        )
        if error is not None:
            raise error
        assert response is not None
        return response

    def reset(self) -> None:
        """Reset call history."""
        self.call_count = 0
        self.last_messages = []
        self.last_tools = None
        self._calls.clear()

    @classmethod
    def _from_scripted_responses(
        cls,
        responses: list[LLMResponse],
        *,
        strict_exhaustion: bool,
    ) -> "FakeLLMClient":
        """Build through a convenience constructor without changing subclass defaults."""
        if strict_exhaustion:
            return cls(scripted_responses=responses, strict_exhaustion=True)
        return cls(scripted_responses=responses)

    @classmethod
    def with_code_responses(
        cls,
        code_strings: list[str],
        *,
        strict_exhaustion: bool = False,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns multiple code generation responses.

        Useful for testing agent code generation with retries.

        Args:
            code_strings: List of code strings to return (in order)
            strict_exhaustion: Raise after the final code response is consumed.

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
                    reasoning=None,
                    usage=LLMUsage(
                        input_tokens=10,
                        output_tokens=len(code.split()),
                        total_tokens=10 + len(code.split()),
                    ),
                )
            )
        return cls._from_scripted_responses(
            responses,
            strict_exhaustion=strict_exhaustion,
        )

    @classmethod
    def simple_message(
        cls,
        message: str,
        *,
        strict_exhaustion: bool = False,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns a simple message.

        Args:
            message: Message content to return
            strict_exhaustion: Raise after the message response is consumed.

        Returns:
            FakeLLMClient configured with message response
        """
        words = message.split()
        return cls._from_scripted_responses(
            [
                LLMResponse(
                    raw_response=None,
                    content=message,
                    tool_calls=[],
                    finish_reason="stop",
                    reasoning=None,
                    usage=LLMUsage(
                        input_tokens=10,
                        output_tokens=len(words),
                        total_tokens=10 + len(words),
                    ),
                )
            ],
            strict_exhaustion=strict_exhaustion,
        )

    @classmethod
    def with_tool_call(
        cls,
        tool_name: str,
        tool_args: dict[str, Any],
        message: str | None = None,
        *,
        strict_exhaustion: bool = False,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns a tool call.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments (will be JSON-serialized)
            message: Optional message before tool call
            strict_exhaustion: Raise after the tool-call response is consumed.

        Returns:
            FakeLLMClient configured with tool call
        """
        return cls._from_scripted_responses(
            [
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
                    reasoning=None,
                    usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ],
            strict_exhaustion=strict_exhaustion,
        )

    @classmethod
    def with_reasoning(
        cls,
        reasoning: str,
        message: str,
        *,
        strict_exhaustion: bool = False,
    ) -> "FakeLLMClient":
        """
        Create a fake client that returns reasoning + message (o1-style).

        Args:
            reasoning: Internal reasoning text
            message: Final message
            strict_exhaustion: Raise after the reasoning response is consumed.

        Returns:
            FakeLLMClient configured with reasoning
        """
        return cls._from_scripted_responses(
            [
                LLMResponse(
                    raw_response=None,
                    content=message,
                    tool_calls=[],
                    finish_reason="stop",
                    reasoning=reasoning,
                    usage=LLMUsage(input_tokens=15, output_tokens=10, total_tokens=25),
                )
            ],
            strict_exhaustion=strict_exhaustion,
        )
