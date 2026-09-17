# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CodeAct finish_reason='length' handling, including bounded auto-continuation."""

import json
from types import SimpleNamespace
from typing import Literal

import pytest

from nooa import Agent, return_text_as_result, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.events import DebugTrace, Error, TextOnlyReply
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.strategies.codeact import LENGTH_CONTINUATION_PROMPT, CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TEST_LLM = FakeLLMClient()


def _resp(
    content: str,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: Literal["stop", "tool_calls", "length", "error"] | None = None,
) -> LLMResponse:
    """Create a test LLM response."""
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )


def _ret(val: str, cid: str = "c2") -> ToolCall:
    return ToolCall(id=cid, name="return_result", arguments=json.dumps({"result": val}))


class TestMaxTokensExhaustedError:
    """Tests for the max_tokens exhaustion error in CodeAct strategy."""

    @pytest.mark.asyncio
    async def test_finish_reason_length_raises_immediately(self):
        """When model hits max_tokens (finish_reason='length') with no output,
        GenerationError is raised immediately (no retry) with an actionable message."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3, max_iterations=10)))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", finish_reason="length"),  # Reasoning ate all tokens
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        # Verify a DebugTrace was emitted (not an Error event in LLM context)
        all_events = agent_instance.event_manager.values()
        debug_events = [e for e in all_events if isinstance(e, DebugTrace)]
        error_events = [e for e in all_events if isinstance(e, Error)]
        assert any("finish_reason='length'" in e.content for e in debug_events), (
            f"Expected DebugTrace with finish_reason, got: {[e.content for e in debug_events]}"
        )
        # No Error event should be injected into LLM context for this case
        max_tokens_errors = [e for e in error_events if "max_tokens" in e.content]
        assert len(max_tokens_errors) == 0, (
            f"max_tokens error should not be an Error event (LLM-visible), got: {max_tokens_errors}"
        )
        assert [event.content for event in all_events if isinstance(event, LLMResponse)] == [""]

    @pytest.mark.asyncio
    async def test_length_partial_text_stitches_two_segments(self):
        """A truncated text reply is continued and delivered as one stitched answer."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(on_text_only=return_text_as_result))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("Hello, ", finish_reason="length"),
                _resp("world.", finish_reason="stop"),
            ]
        )
        agent_instance = TestAgent(llm=fake_llm)

        assert await agent_instance.my_task() == "Hello, world."
        assert fake_llm.call_count == 2

        events = agent_instance.event_manager.values()
        responses = [event for event in events if isinstance(event, LLMResponse)]
        assert [event.content for event in responses] == ["Hello, ", "world."]
        assert responses[1].metadata.get("output_continued") is True
        assert responses[1].metadata.get("segment_count") == 2
        assert "continued" in str(responses[1].metadata.get("continuation_note", "")).lower()

        debug_events = [event for event in events if isinstance(event, DebugTrace)]
        assert any("Length continuation 1/3" in event.content for event in debug_events)
        assert any(
            isinstance(event, Error) and LENGTH_CONTINUATION_PROMPT in event.content
            for event in events
        )
        assert any(
            LENGTH_CONTINUATION_PROMPT in json.dumps(message, default=str)
            for message in fake_llm.last_messages
        )
        replies = [event for event in events if isinstance(event, TextOnlyReply)]
        assert replies
        assert replies[0].content == "Hello, world."

    @pytest.mark.asyncio
    async def test_truncation_diagnostics_do_not_persist_provider_output(self):
        """Debug metadata records output shape without opaque provider payloads."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_length_continuations=0)))
            async def my_task(self) -> str:
                """A task."""
                ...

        response = _resp("partial", finish_reason="length")
        response.raw_response = SimpleNamespace(
            output=[
                {
                    "type": "reasoning",
                    "encrypted_content": "must-never-enter-debug-events",
                },
                {"type": "message", "content": []},
            ]
        )
        agent_instance = TestAgent(llm=FakeLLMClient(scripted_responses=[response]))

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        debug = next(
            event.content
            for event in agent_instance.event_manager.values()
            if isinstance(event, DebugTrace)
        )
        assert "must-never-enter-debug-events" not in debug
        assert "raw_response.output_count=2; output_types=['reasoning', 'message']" in debug

    @pytest.mark.asyncio
    async def test_empty_response_without_length_retries_normally(self):
        """When empty response has finish_reason != 'length', normal retry logic applies."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=2)))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", finish_reason="stop"),  # Empty but not length
                _resp("", tool_calls=[_ret("hello")], finish_reason="tool_calls"),  # Recovery
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.my_task()
        assert result == "hello"
        assert [
            event.content
            for event in agent_instance.event_manager.values()
            if isinstance(event, LLMResponse)
        ] == ["", ""]
        assert not any(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and not message["content"].strip()
            and not message.get("tool_calls")
            for message in fake_llm.last_messages
        )

    @pytest.mark.asyncio
    async def test_length_continuation_respects_max_three(self):
        """The fourth consecutive truncated text response still raises GenerationError."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(on_text_only=return_text_as_result))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_resp(f"chunk{i}", finish_reason="length") for i in range(4)]
        )
        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        assert fake_llm.call_count == 4
        events = agent_instance.event_manager.values()
        debug = [event.content for event in events if isinstance(event, DebugTrace)]
        assert sum("Length continuation" in content for content in debug) == 3
        assert any("Truncated response:" in content for content in debug)
        assert [event.content for event in events if isinstance(event, LLMResponse)] == [
            "chunk0",
            "chunk1",
            "chunk2",
            "chunk3",
        ]
        assert not any(event.event_type == "TextOnlyReply" for event in events)

    @pytest.mark.asyncio
    async def test_length_with_tool_calls_only_still_raises(self):
        """A length-truncated tool call is not continued or executed."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy())
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_ret("hello")], finish_reason="length"),
            ]
        )
        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        assert fake_llm.call_count == 1
        events = agent_instance.event_manager.values()
        assert not any(
            isinstance(event, Error) and LENGTH_CONTINUATION_PROMPT in event.content
            for event in events
        )
        assert not any(event.event_type == "ToolCallEvent" for event in events)

    @pytest.mark.asyncio
    async def test_length_then_tool_call_resets_counter(self):
        """A completed tool call after a text continuation is progress, not a bound error."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy())
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("working on it", finish_reason="length"),
                _resp("", tool_calls=[_ret("hello")], finish_reason="tool_calls"),
            ]
        )
        agent_instance = TestAgent(llm=fake_llm)

        assert await agent_instance.my_task() == "hello"
        assert fake_llm.call_count == 2
        events = agent_instance.event_manager.values()
        assert any(
            isinstance(event, Error) and LENGTH_CONTINUATION_PROMPT in event.content
            for event in events
        )
        responses = [event for event in events if isinstance(event, LLMResponse)]
        assert responses[-1].metadata.get("output_continued") is True
        assert responses[-1].metadata.get("segment_count") == 2

    @pytest.mark.asyncio
    async def test_length_continuation_records_per_model_metric(self, monkeypatch):
        metrics = HarnessMetrics()
        monkeypatch.setattr("nooa.strategies.codeact.get_harness_metrics", lambda: metrics)

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(on_text_only=return_text_as_result))
            async def my_task(self) -> str:
                """A task."""
                ...

        agent_instance = TestAgent(
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp("Hello, ", finish_reason="length"),
                    _resp("world.", finish_reason="stop"),
                ]
            )
        )

        assert await agent_instance.my_task() == "Hello, world."
        assert metrics.length_continuation_count == 1
        assert metrics.length_continuation_models == ["fake-model"]
