# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproduction of nested agent history ordering bug.

This test demonstrates a bug where nested agent calls cause tool_call_id
ordering issues in the message history.

Bug: When outer execute_python code calls a nested agent method, the tool
result for execute_python is added AFTER the nested agent's events. This
violates OpenAI/Gemini's requirement that tool results reference tool_call_ids
from the immediately preceding assistant message.
"""

import json

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    """Create a test LLM response."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _execute_python(code: str, call_id: str = "call_1") -> ToolCall:
    """Create an execute_python tool call."""
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _return_result(result, call_id: str = "call_return") -> ToolCall:
    """Create a return_result tool call."""
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


# Module-level test LLM placeholder
_TEST_LLM = FakeLLMClient()


class TestNestedAgentHistoryBug:
    """Tests demonstrating the nested agent history ordering bug."""

    @pytest.mark.asyncio
    async def test_nested_call_causes_history_ordering_issue(self):
        """Reproduce the bug: nested agent call causes tool_call_id mismatch.

        Scenario:
        1. outer_method calls execute_python with code that calls inner_method
        2. inner_method uses return_result to complete
        3. outer execute_python result is added to history (AFTER inner's events)
        4. outer tries to continue → FAILS because tool result references wrong tool_call_id

        Expected history (correct):
        [1] assistant: execute_python (id=outer_1)
        [2] tool: result for outer_1
        [3] user: inner task (if separate session)
        [4] assistant: return_result (id=inner_1)
        [5] tool: result for inner_1

        Actual history (buggy):
        [1] assistant: execute_python (id=outer_1)
        [2] user: inner task  ← nested events inserted here
        [3] assistant: return_result (id=inner_1)
        [4] tool: result for inner_1
        [5] tool: result for outer_1  ← references outer_1 but last assistant is inner_1!
        """

        class NestedAgent(Agent, llm=_TEST_LLM):
            """Agent with two methods - outer calls inner."""

            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
            async def outer_method(self) -> str:
                """Outer method that delegates to inner_method."""
                ...

            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
            async def inner_method(self) -> str:
                """Inner method that returns a result."""
                ...

        # Scripted responses:
        # 1. outer_method turn 1: execute_python calling inner_method, then print result
        # 2. inner_method turn 1: return_result with "inner_done"
        # 3. outer_method turn 2: return_result with combined result (SHOULD FAIL due to bug)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # outer_method turn 1: call inner_method and capture result
                _resp(
                    content="I'll call the inner method.",
                    tool_calls=[
                        _execute_python(
                            code="result = await self.inner_method()\nprint(f'Got: {result}')",
                            call_id="call_outer_exec_1",
                        )
                    ],
                ),
                # inner_method turn 1: return result directly
                _resp(
                    content="Returning the result.",
                    tool_calls=[_return_result("inner_done", call_id="call_inner_return")],
                ),
                # outer_method turn 2: use the result and return
                # This LLM call is where the bug manifests - history validation fails
                _resp(
                    content="Now I'll return the final result.",
                    tool_calls=[
                        _return_result("outer_with_inner_done", call_id="call_outer_return")
                    ],
                ),
            ]
        )

        agent = NestedAgent(llm=fake_llm)

        # This should succeed but currently fails with:
        # "Missing corresponding tool call for tool response message"
        # because tool result for call_outer_exec_1 appears after inner agent's events
        result = await agent.outer_method()

        assert result == "outer_with_inner_done"

    @pytest.mark.asyncio
    async def test_single_method_no_nesting_works(self):
        """Baseline: single method without nesting should work fine."""

        class SimpleAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
            async def simple_method(self) -> str:
                """Simple method that computes and returns."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: execute some code
                _resp(
                    content="Computing...",
                    tool_calls=[_execute_python(code="x = 1 + 1\nprint(x)", call_id="call_1")],
                ),
                # Turn 2: return result
                _resp(
                    content="Done.",
                    tool_calls=[_return_result("simple_result", call_id="call_2")],
                ),
            ]
        )

        agent = SimpleAgent(llm=fake_llm)
        result = await agent.simple_method()

        assert result == "simple_result"
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_inner_method_standalone_works(self):
        """Baseline: calling inner method directly should work."""

        class NestedAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
            async def inner_method(self) -> str:
                """Inner method that returns a result."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    content="Returning.",
                    tool_calls=[_return_result("inner_standalone", call_id="call_inner")],
                ),
            ]
        )

        agent = NestedAgent(llm=fake_llm)
        result = await agent.inner_method()

        assert result == "inner_standalone"


class TestHistoryInspection:
    """Tests that inspect history state to verify the bug."""

    @pytest.mark.asyncio
    async def test_inspect_history_after_nested_call(self):
        """Inspect history to see the ordering issue directly."""

        class InspectAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
            async def outer_method(self) -> str:
                """Outer method."""
                ...

            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
            async def inner_method(self) -> str:
                """Inner method."""
                ...

        # Only script enough for the nested call to happen
        # Then we'll inspect history manually
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # outer: call inner
                _resp(
                    tool_calls=[
                        _execute_python(
                            code="inner_result = await self.inner_method()",
                            call_id="call_outer_1",
                        )
                    ],
                ),
                # inner: return result
                _resp(
                    tool_calls=[_return_result("inner_done", call_id="call_inner_1")],
                ),
                # outer: would continue, but let's just return
                _resp(
                    tool_calls=[_return_result("outer_done", call_id="call_outer_2")],
                ),
            ]
        )

        agent = InspectAgent(llm=fake_llm)

        try:
            result = await agent.outer_method()
            # If we get here, the bug might be fixed!
            print(f"Result: {result}")
        except Exception as e:
            # Expected: tool_call_id mismatch error
            error_msg = str(e)
            print(f"Error (expected): {error_msg[:200]}...")

            # Verify it's the expected error
            assert "tool_call" in error_msg.lower() or "tool call" in error_msg.lower()

        # Inspect history to understand the state
        history_events = agent.event_manager.values()
        print(f"\n=== History has {len(history_events)} events ===")
        for i, event in enumerate(history_events):
            event_type = type(event).__name__
            # Try to extract tool_call_id if present
            tool_call_id = ""
            if hasattr(event, "data"):
                data = event.data
                if hasattr(data, "tool_call_id"):
                    tool_call_id = f" (tool_call_id={data.tool_call_id})"
                elif hasattr(data, "content") and isinstance(data.content, list):
                    # Assistant message with tool calls
                    for item in data.content:
                        if hasattr(item, "id"):
                            tool_call_id += f" (tool_call.id={item.id})"
            print(f"  [{i}] {event_type}{tool_call_id}")


if __name__ == "__main__":
    import asyncio

    print("Running nested agent history bug reproduction tests...\n")

    # Run the inspection test first to see the state
    asyncio.run(TestHistoryInspection().test_inspect_history_after_nested_call())
