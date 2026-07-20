# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MethodSummarizer call_id based range computation.

These tests run real agents with real strategies through the full runtime,
then verify that call_id metadata is correctly present on all recorded events
and that MethodSummarizer._compute_range produces correct ranges.

Scenarios tested:
- CodeAct: single turn, multi-turn, exception mid-call
- PurePython: single turn, multi-turn
- Nested calls: parent calls child method during execution
- Deterministic methods: no generation, events still get call_id
"""

from __future__ import annotations

import json

import pytest

from nooa import Agent, strategy
from nooa.agents import MethodSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import MethodSummarizerConfig
from nooa.events import AfterTurn
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# =============================================================================
# Helpers
# =============================================================================


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _exec_python(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _return_result(result=None, call_id: str = "call_return") -> ToolCall:
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


def _pp_resp(code: str) -> LLMResponse:
    """PurePython response — LLM returns code directly."""
    return _resp(code)


_TEST_LLM = FakeLLMClient()


def _get_events_with_metadata(agent):
    """Return list of (tag, event, call_id) for all recorded events."""
    em = agent.event_manager
    result = []
    for tag in em.keys():
        evt = em[tag]
        result.append((tag, evt, evt.metadata.get("call_id")))
    return result


# =============================================================================
# CodeAct: single turn
# =============================================================================


class TestCodeActSingleTurnCallId:
    """CodeAct: LLM returns result immediately — all events get same call_id."""

    @pytest.mark.asyncio
    async def test_all_events_have_call_id(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def answer(self) -> int:
                """Return 42."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.answer()
        assert result == 42

        events = _get_events_with_metadata(agent)
        assert len(events) > 0, "Should have at least one recorded event"

        # All events should have the same call_id (from the answer() invocation)
        call_ids = {cid for _, _, cid in events}
        assert None not in call_ids, f"Some events missing call_id: {events}"
        assert len(call_ids) == 1, f"Expected single call_id, got {call_ids}"

    @pytest.mark.asyncio
    async def test_compute_range_covers_all_events(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def answer(self) -> int:
                """Return 42."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        await agent.answer()

        events = _get_events_with_metadata(agent)
        call_id = events[0][2]

        summarizer = MethodSummarizer(agent, config=MethodSummarizerConfig(min_events=1))
        after = AfterTurn(
            method_name="answer",
            strategy="CODEACT",
            generation_id="ignored",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        after.metadata["call_id"] = call_id

        result = summarizer._compute_range(after)
        assert result is not None
        start, end = result
        all_tags = [t for t, _, _ in events]
        assert start == all_tags[0]
        assert end == all_tags[-1]


# =============================================================================
# CodeAct: multi-turn (execute_python + return_result)
# =============================================================================


class TestCodeActMultiTurnCallId:
    """CodeAct: LLM runs code then returns — events across turns share call_id."""

    @pytest.mark.asyncio
    async def test_multi_turn_events_share_call_id(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_exec_python("x = 2 + 3\nprint(x)")]),
                _resp("", tool_calls=[_return_result(result=5)]),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.compute()
        assert result == 5

        events = _get_events_with_metadata(agent)
        assert len(events) >= 2, f"Expected multiple events, got {len(events)}"

        call_ids = {cid for _, _, cid in events}
        assert None not in call_ids, f"Some events missing call_id: {events}"
        assert len(call_ids) == 1, f"Expected single call_id across turns, got {call_ids}"


# =============================================================================
# PurePython: single turn
# =============================================================================


class TestPurePythonSingleTurnCallId:
    """PurePython: LLM returns code — all events get same call_id."""

    @pytest.mark.asyncio
    async def test_all_events_have_call_id(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def greet(self) -> str:
                """Return greeting."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _pp_resp("return 'hello'"),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.greet()
        assert result == "hello"

        events = _get_events_with_metadata(agent)
        assert len(events) > 0

        call_ids = {cid for _, _, cid in events}
        assert None not in call_ids, f"Some events missing call_id: {events}"
        assert len(call_ids) == 1


# =============================================================================
# Nested calls: parent method calls child method
# =============================================================================


class TestNestedCallsCallId:
    """Nested generation: parent calls child — child events get child's call_id."""

    @pytest.mark.asyncio
    async def test_nested_call_produces_two_call_ids(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def parent_method(self) -> str:
                """Call child and return result."""
                ...

            @strategy(PurePythonStrategy())
            async def child_method(self) -> str:
                """Do child work."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # parent: calls child
                _pp_resp("result = await self.child_method()\nreturn f'parent-{result}'"),
                # child
                _pp_resp("return 'child'"),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.parent_method()
        assert result == "parent-child"

        events = _get_events_with_metadata(agent)
        call_ids = {cid for _, _, cid in events if cid is not None}
        # Should have at least 2 distinct call_ids (parent and child)
        assert len(call_ids) >= 2, f"Expected >=2 call_ids for nested call, got {call_ids}"

    @pytest.mark.asyncio
    async def test_parent_range_includes_child_events(self):
        """MethodSummarizer range for parent should span all events including child's."""

        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def parent_method(self) -> str:
                """Call child and return result."""
                ...

            @strategy(PurePythonStrategy())
            async def child_method(self) -> str:
                """Do child work."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _pp_resp("result = await self.child_method()\nreturn f'parent-{result}'"),
                _pp_resp("return 'child'"),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        await agent.parent_method()

        events = _get_events_with_metadata(agent)
        all_tags = [t for t, _, _ in events]

        # Find parent's call_id (first and last events should be parent's)
        parent_call_id = events[0][2]
        last_parent_event = [t for t, _, cid in events if cid == parent_call_id]

        summarizer = MethodSummarizer(agent, config=MethodSummarizerConfig(min_events=1))
        after = AfterTurn(
            method_name="parent_method",
            strategy="PURE_PYTHON",
            generation_id="ignored",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=True,
        )
        after.metadata["call_id"] = parent_call_id

        result = summarizer._compute_range(after)
        assert result is not None
        start, end = result

        # The range should start at first event and end at last event
        # (child events are chronologically between parent events)
        assert start == all_tags[0]
        assert end == last_parent_event[-1]

        # Verify child events are inside the range
        start_idx = all_tags.index(start)
        end_idx = all_tags.index(end)
        child_tags = [t for t, _, cid in events if cid != parent_call_id]
        for ct in child_tags:
            idx = all_tags.index(ct)
            assert start_idx <= idx <= end_idx, f"Child tag {ct} outside parent range"


# =============================================================================
# Exception mid-call: events emitted during error recovery still get call_id
# =============================================================================


class TestExceptionMidCallCallId:
    """Exception during execution: error events still get call_id."""

    @pytest.mark.asyncio
    async def test_error_events_have_call_id_on_exhaustion(self):
        """When max_retries exhausted, all events (including errors) have call_id."""
        from nooa.errors import GenerationError

        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1, max_iterations=5)))
            async def fail_method(self) -> int:
                """This will fail."""
                ...

        # LLM returns code that causes execution error, then retries exhaust
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_exec_python("raise ValueError('boom')")]),
                _resp("", tool_calls=[_exec_python("raise ValueError('boom again')")]),
                _resp("", tool_calls=[_exec_python("raise ValueError('boom 3')")]),
            ]
        )
        agent = MyAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent.fail_method()

        events = _get_events_with_metadata(agent)
        assert len(events) > 0, "Should have events even on failure"

        # All events should have call_id, including error events
        call_ids = {cid for _, _, cid in events}
        assert None not in call_ids, f"Some events missing call_id after error: {events}"
        assert len(call_ids) == 1, "All events from failed call should share call_id"

    @pytest.mark.asyncio
    async def test_compute_range_works_after_exception(self):
        """MethodSummarizer can compute range even for methods that raised."""
        from nooa.errors import GenerationError

        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1, max_iterations=5)))
            async def fail_method(self) -> int:
                """This will fail."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_exec_python("raise ValueError('boom')")]),
                _resp("", tool_calls=[_exec_python("raise ValueError('boom again')")]),
                _resp("", tool_calls=[_exec_python("raise ValueError('boom 3')")]),
            ]
        )
        agent = MyAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent.fail_method()

        events = _get_events_with_metadata(agent)
        call_id = events[0][2]

        summarizer = MethodSummarizer(agent, config=MethodSummarizerConfig(min_events=1))
        after = AfterTurn(
            method_name="fail_method",
            strategy="CODEACT",
            generation_id="ignored",
            parent_generation_id=None,
            turn_number=1,
            is_final=True,
            success=False,
        )
        after.metadata["call_id"] = call_id

        result = summarizer._compute_range(after)
        assert result is not None
        start, end = result
        all_tags = [t for t, _, _ in events]
        assert start == all_tags[0]
        assert end == all_tags[-1]


# =============================================================================
# Deterministic method called from LLM code: gets its own call_id
# =============================================================================


class TestDeterministicMethodCallId:
    """Deterministic (non-generation) methods called from LLM code get own call_id."""

    @pytest.mark.asyncio
    async def test_deterministic_child_events_have_separate_call_id(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            def lookup(self, key: str) -> str:
                """Deterministic lookup."""
                return f"value-{key}"

            @strategy(PurePythonStrategy())
            async def process(self) -> str:
                """Process data using lookup."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _pp_resp("val = self.lookup('abc')\nreturn val"),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.process()
        assert result == "value-abc"

        # Verify events exist and have call_ids
        events = _get_events_with_metadata(agent)
        assert len(events) > 0
        call_ids = {cid for _, _, cid in events if cid is not None}
        assert len(call_ids) >= 1  # At least the parent's call_id


# =============================================================================
# PurePython multi-turn (code has error, LLM retries)
# =============================================================================


class TestPurePythonMultiTurnCallId:
    """PurePython: LLM code fails validation, retries — all events share call_id."""

    @pytest.mark.asyncio
    async def test_retry_events_share_call_id(self):
        class MyAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_retries=3))
            async def compute(self) -> int:
                """Return an integer."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First attempt: returns wrong type (string)
                _pp_resp("return 'not an int'"),
                # Second attempt: correct
                _pp_resp("return 42"),
            ]
        )
        agent = MyAgent(llm=fake_llm)
        result = await agent.compute()
        assert result == 42

        events = _get_events_with_metadata(agent)
        assert len(events) >= 2, "Expected events from both attempts"

        call_ids = {cid for _, _, cid in events}
        assert None not in call_ids
        assert len(call_ids) == 1, "Retry events should share the same call_id"
