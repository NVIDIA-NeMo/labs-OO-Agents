# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replies cut off at max_tokens get bounded recovery nudges, then an actionable error."""

import json
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest

from nooa import Agent, return_text_as_result, strategy
from nooa.config import CodeActConfig
from nooa.errors import GenerationError
from nooa.events import DebugTrace, Error, Feedback, PythonOutput
from nooa.strategies import codeact as codeact_module
from nooa.strategies.codeact import (
    CONTINUED_TEXT_NOTE,
    LENGTH_CONTINUATION_NUDGE,
    MAX_LENGTH_RECOVERIES,
    OUTPUT_LIMIT_NUDGE,
    OUTPUT_TOKENS_EXHAUSTED_MESSAGE,
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
        # Reasoning cut off with no tool call: the model is asked to continue.
        nudges = [
            e
            for e in all_events
            if isinstance(e, Feedback) and e.content == LENGTH_CONTINUATION_NUDGE
        ]
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
            if isinstance(e, Feedback) and e.content == LENGTH_CONTINUATION_NUDGE
        ]
        assert len(nudges) == 1
        assert "cut off at the output-token limit" in LENGTH_CONTINUATION_NUDGE

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


def _call(name: str, arguments: str, cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments=arguments)


def _ret(value: str, cid: str = "r1") -> ToolCall:
    return _call("return_result", json.dumps({"result": value}), cid)


def _agent_cls(**strategy_kwargs):
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(max_retries=3, max_iterations=10), **strategy_kwargs
            )
        )
        async def my_task(self) -> str:
            """A task."""
            ...

    return TestAgent


def _events(agent, kind):
    return [e for e in agent.event_manager.values() if isinstance(e, kind)]


class TestLengthContinuation:
    """Cut-off text is continued; a cut-off tool call is never run."""

    @pytest.mark.asyncio
    async def test_cut_off_text_is_continued_not_joined(self):
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("Part A, ", finish_reason="length"),
                _resp("part B, ", finish_reason="length"),
                _resp("part C."),
                _resp("", tool_calls=[_ret("Part A, part B, part C.")]),
            ]
        )
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)

        # The pieces are not joined (they are all in history), and the tail
        # alone is not accepted as the result: the model returns the whole answer.
        assert await agent.my_task() == "Part A, part B, part C."
        corrections = [e.content for e in _events(agent, Error)]
        assert len(corrections) == 1
        assert CONTINUED_TEXT_NOTE in corrections[0]
        # Each reply stays in history as generated; nothing is rewritten.
        assert [e.content for e in _events(agent, LLMResponse)][:3] == [
            "Part A, ",
            "part B, ",
            "part C.",
        ]
        # The continue prompt is plain feedback, not an Error event.
        assert [e.content for e in _events(agent, Feedback)] == [LENGTH_CONTINUATION_NUDGE] * 2
        assert not any(LENGTH_CONTINUATION_NUDGE in e.content for e in _events(agent, Error))
        assert _events(agent, LLMResponse)[2].metadata["output_continued"] is True
        # The model saw its partial reply, then the continue prompt.
        seen = [(m.get("role"), str(m.get("content"))) for m in llm.last_messages]
        first = seen.index(("assistant", "Part A, "))
        assert seen[first + 1][0] == "user"
        assert "Continue exactly where you left off" in seen[first + 1][1]
        assert "Error(" not in seen[first + 1][1]
        assert seen[first + 2] == ("assistant", "part B, ")

    @pytest.mark.asyncio
    async def test_tail_only_completion_is_not_accepted(self):
        """Severin's #356 case: the tail of a continued reply is not the result."""
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("The account number is ", finish_reason="length"),
                _resp("1234."),
                _resp("", tool_calls=[_ret("The account number is 1234.")]),
            ]
        )
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        assert await agent.my_task() == "The account number is 1234."
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_uncontinued_text_is_still_accepted(self):
        llm = FakeLLMClient(scripted_responses=[_resp("The answer is 42.")])
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        assert await agent.my_task() == "The answer is 42."

    @pytest.mark.asyncio
    async def test_cut_off_code_cell_is_not_executed(self):
        cut_code = '{"code": "rows = []\\nfor i in range(1000):\\n    rows.append(i*'
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_call("execute_python", cut_code)], finish_reason="length"),
                _resp("", tool_calls=[_ret("done")]),
            ]
        )
        agent = _agent_cls()(llm=llm)
        assert await agent.my_task() == "done"
        assert _events(agent, PythonOutput) == []
        assert [e.content for e in _events(agent, Feedback)] == [OUTPUT_LIMIT_NUDGE]
        assert "smaller steps" in OUTPUT_LIMIT_NUDGE
        # The cut-off call stays visible in history.
        assert _events(agent, LLMResponse)[0].tool_calls[0].name == "execute_python"

    @pytest.mark.asyncio
    async def test_cut_off_return_result_is_not_continued(self):
        cut = '{"result": "Section 1 ... Section 2 ... Sect'
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_call("return_result", cut)], finish_reason="length"),
                _resp("brief answer"),
            ]
        )
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        # The partial return_result is neither accepted nor joined with the next reply.
        assert await agent.my_task() == "brief answer"
        assert [e.content for e in _events(agent, Feedback)] == [OUTPUT_LIMIT_NUDGE]

    @pytest.mark.asyncio
    async def test_bound_then_exhausted_message(self):
        llm = FakeLLMClient(scripted_responses=[_resp("seg ", finish_reason="length")] * 8)
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent.my_task()
        # Hosts map this exact message to stop_reason="max_tokens".
        assert str(exc_info.value) == OUTPUT_TOKENS_EXHAUSTED_MESSAGE
        assert llm.call_count == MAX_LENGTH_RECOVERIES + 1

    @pytest.mark.asyncio
    async def test_default_handler_keeps_continued_text(self):
        """The default correction does not ask for the continued text again in one call."""
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("long prose ", finish_reason="length"),
                _resp("end."),
                _resp("", tool_calls=[_ret("ok")]),
            ]
        )
        agent = _agent_cls()(llm=llm)
        assert await agent.my_task() == "ok"
        reply = next(e for e in agent.event_manager.values() if e.event_type == "TextOnlyReply")
        assert reply.content == "end."
        corrections = [e.content for e in _events(agent, Error)]
        assert len(corrections) == 1
        assert CONTINUED_TEXT_NOTE in corrections[0]
        assert "Re-issue your response" not in corrections[0]

    @pytest.mark.asyncio
    async def test_tool_call_after_cut_off_text_clears_the_mark(self):
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("thinking about it ", finish_reason="length"),
                _resp("", tool_calls=[_call("execute_python", json.dumps({"code": "x = 1"}))]),
                _resp("final text"),
            ]
        )
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        assert await agent.my_task() == "final text"

    @pytest.mark.asyncio
    async def test_every_cut_off_is_counted_with_model(self, monkeypatch):
        metrics = MagicMock()
        monkeypatch.setattr(codeact_module, "get_harness_metrics", lambda: metrics)
        llm = FakeLLMClient(
            scripted_responses=[
                _resp("a ", finish_reason="length"),
                _resp(
                    "", tool_calls=[_call("return_result", '{"result": "x')], finish_reason="length"
                ),
                _resp("b"),
            ]
        )
        agent = _agent_cls(on_text_only=return_text_as_result)(llm=llm)
        await agent.my_task()
        calls = metrics.length_continuation.call_args_list
        assert len(calls) == 2
        # FakeLLMClient stamps its model name on every response.
        assert [c.args for c in calls] == [("fake-model",), ("fake-model",)]
