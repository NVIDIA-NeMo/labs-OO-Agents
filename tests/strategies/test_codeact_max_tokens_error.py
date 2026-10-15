# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replies cut off at max_tokens get bounded recovery nudges, then an actionable error."""

import json
from types import SimpleNamespace
from typing import Literal

import pytest

from nooa import Agent, return_text_as_result, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.events import DebugTrace, Error
from nooa.strategies.codeact import (
    MAX_LENGTH_RECOVERIES,
    OUTPUT_LIMIT_NUDGE,
    CodeActStrategy,
)
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


class TestMaxTokensExhaustedError:
    """Tests for the max_tokens exhaustion error in CodeAct strategy."""

    @pytest.mark.asyncio
    async def test_finish_reason_length_raises_after_bounded_nudges(self):
        """Repeated max_tokens cut-offs (finish_reason='length') get MAX_LENGTH_RECOVERIES
        nudges, then GenerationError with an actionable message."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3, max_iterations=10)))
            async def my_task(self) -> str:
                """A task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", finish_reason="length"),  # Reasoning ate all tokens
                _resp("", finish_reason="length"),
                _resp("", finish_reason="length"),
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
        assert [event.content for event in all_events if isinstance(event, LLMResponse)] == [""] * (
            MAX_LENGTH_RECOVERIES + 1
        )
        nudges = [e for e in error_events if e.content == OUTPUT_LIMIT_NUDGE]
        assert len(nudges) == MAX_LENGTH_RECOVERIES

    @pytest.mark.asyncio
    async def test_finish_reason_length_does_not_return_partial_text(self):
        """A text-only handler cannot accept output truncated at max_tokens."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(on_text_only=return_text_as_result))
            async def my_task(self) -> str:
                """A task."""
                ...

        agent_instance = TestAgent(
            llm=FakeLLMClient(
                scripted_responses=[
                    _resp("truncated partial", finish_reason="length"),
                    _resp("truncated partial", finish_reason="length"),
                    _resp("truncated partial", finish_reason="length"),
                ]
            )
        )

        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()

        events = agent_instance.event_manager.values()
        assert [event.content for event in events if isinstance(event, LLMResponse)] == [
            "truncated partial"
        ] * (MAX_LENGTH_RECOVERIES + 1)
        assert not any(event.event_type == "TextOnlyReply" for event in events)

    @pytest.mark.asyncio
    async def test_truncation_diagnostics_do_not_persist_provider_output(self):
        """Debug metadata records output shape without opaque provider payloads."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy())
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
        agent_instance = TestAgent(llm=FakeLLMClient(scripted_responses=[response] * 3))

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

        def _ret(val: str, cid: str = "c2") -> ToolCall:
            return ToolCall(id=cid, name="return_result", arguments=json.dumps({"result": val}))

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
    async def test_cut_off_reply_recovers_after_nudge(self):
        """One cut-off reply is not fatal: the nudge lets the next reply finish."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def my_task(self) -> str:
                """A task."""
                ...

        ret = ToolCall(id="c1", name="return_result", arguments=json.dumps({"result": "ok"}))
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("loop loop loop", finish_reason="length"),
                _resp("", tool_calls=[ret], finish_reason="tool_calls"),
            ]
        )
        agent_instance = TestAgent(llm=fake_llm)
        assert await agent_instance.my_task() == "ok"
        nudges = [
            e
            for e in agent_instance.event_manager.values()
            if isinstance(e, Error) and e.content == OUTPUT_LIMIT_NUDGE
        ]
        assert len(nudges) == 1
        assert "cut off at the output-token limit" in OUTPUT_LIMIT_NUDGE

    @pytest.mark.asyncio
    async def test_cut_off_nudges_respect_max_retries(self):
        """With max_retries=1 the first cut-off already exhausts the call: no nudge."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def my_task(self) -> str:
                """A task."""
                ...

        agent_instance = TestAgent(
            llm=FakeLLMClient(scripted_responses=[_resp("", finish_reason="length")])
        )
        with pytest.raises(GenerationError, match="max_tokens"):
            await agent_instance.my_task()
