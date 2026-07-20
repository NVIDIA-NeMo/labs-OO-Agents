# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for injectable session_locals in CodeAct and PurePython strategies.

Tests that callers can inject initial session_locals into a strategy method call
and read back the final locals after it completes, enabling persistent stack across calls.
"""

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    """Create a test LLM response."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    """Create an execute_python tool call."""
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _return_result(result: Any, call_id: str = "call_return") -> ToolCall:
    """Create a return_result tool call."""
    return ToolCall(id=call_id, name="return_result", arguments=json.dumps({"result": result}))


_TEST_LLM = FakeLLMClient()


class TestCodeActSessionLocalsInjection:
    """Tests for _session_locals kwarg on CodeActStrategy methods."""

    @pytest.mark.asyncio
    async def test_default_no_session_locals(self):
        """Without _session_locals, behavior is unchanged (fresh dict per call)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Return 42."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.compute()
        assert result == 42

    @pytest.mark.asyncio
    async def test_injected_locals_visible_to_llm_code(self):
        """Injected session_locals should be accessible in LLM-generated code."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Use injected_value to compute result."""
                ...

        # LLM code reads `injected_value` from session_locals and returns inline.
        # Single scripted response: if injection fails (NameError), there's no
        # fallback response so the test correctly fails.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return_result(injected_value * 2)")]),
            ]
        )
        stack: dict[str, Any] = {"injected_value": 42}
        agent = TestAgent(llm=fake_llm)
        result = await agent.compute(_session_locals=stack)
        assert result == 84

    @pytest.mark.asyncio
    async def test_captured_locals_written_back(self):
        """After execute(), caller's dict should contain captured locals."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute and store intermediate values."""
                ...

        # LLM defines a variable, then returns
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("computed = 99\nprint(computed)")]),
                _resp("", tool_calls=[_return_result(result=99)]),
            ]
        )
        stack: dict[str, Any] = {}
        agent = TestAgent(llm=fake_llm)
        result = await agent.compute(_session_locals=stack)
        assert result == 99
        assert "computed" in stack
        assert stack["computed"] == 99

    @pytest.mark.asyncio
    async def test_persistent_stack_across_calls(self):
        """Session locals should persist across two successive calls via the same dict."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def step(self) -> str:
                """Do one step of work."""
                ...

        # First call: define counter = 1
        fake_llm1 = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("counter = 1")]),
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )
        stack: dict[str, Any] = {}
        agent = TestAgent(llm=fake_llm1)
        await agent.step(_session_locals=stack)
        assert stack.get("counter") == 1

        # Second call: LLM reads counter (injected) and increments
        fake_llm2 = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("counter = counter + 1")]),
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )
        agent2 = TestAgent(llm=fake_llm2)
        await agent2.step(_session_locals=stack)
        assert stack.get("counter") == 2

    @pytest.mark.asyncio
    async def test_out_accessor_excluded_from_writeback(self):
        """The internal Out accessor should not leak into caller's dict."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Return 1."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=1)]),
            ]
        )
        stack: dict[str, Any] = {}
        agent = TestAgent(llm=fake_llm)
        await agent.compute(_session_locals=stack)
        assert "Out" not in stack


def _pure_resp(code: str) -> LLMResponse:
    """Create a PurePython-style LLM response (raw code as content)."""
    return LLMResponse(
        raw_response=None,
        content=code,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": code},
    )


class TestPurePythonSessionLocalsInjection:
    """Tests for _session_locals kwarg on PurePythonStrategy methods."""

    @pytest.mark.asyncio
    async def test_injected_locals_visible(self):
        """Injected session_locals should be accessible in PurePython strategy."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Use injected_value."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _pure_resp("return injected_value + 10"),
            ]
        )
        stack: dict[str, Any] = {"injected_value": 42}
        agent = TestAgent(llm=fake_llm)
        result = await agent.compute(_session_locals=stack)
        assert result == 52

    @pytest.mark.asyncio
    async def test_captured_locals_written_back(self):
        """After PurePython execute(), caller's dict should contain captured locals."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _pure_resp("my_var = 123"),
                _pure_resp("return my_var"),
            ]
        )
        stack: dict[str, Any] = {}
        agent = TestAgent(llm=fake_llm)
        await agent.compute(_session_locals=stack)
        assert "my_var" in stack
        assert stack["my_var"] == 123
