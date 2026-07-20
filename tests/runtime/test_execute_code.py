# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ActorRuntime.execute_code() method."""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM
_TEST_LLM = FakeLLMClient()


@pytest.fixture
def test_agent():
    """Create a test agent."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


class TestExecuteCodeFenceStripping:
    """Tests for the fence-stripping safety net in execute_code()."""

    @pytest.mark.asyncio
    async def test_fenced_code_executes(self, test_agent):
        """execute_code() is the safety-net fence-stripping intercept point.
        Fenced code passed directly (bypassing strategies) must still run."""
        result = await test_agent.runtime.execute_code('```python\nprint("hello")\n```')
        assert result.success
        assert result.stdout == "hello\n"

    @pytest.mark.asyncio
    async def test_bare_fenced_code_executes(self, test_agent):
        """Bare fences (no language tag) are stripped by the safety net."""
        result = await test_agent.runtime.execute_code('```\nprint("bare")\n```')
        assert result.success
        assert result.stdout == "bare\n"


class TestExecuteCodeStdout:
    """Tests for stdout capture in execute_code()."""

    @pytest.mark.asyncio
    async def test_captures_print_output(self, test_agent):
        """print() output should be captured in stdout."""
        result = await test_agent.runtime.execute_code('print("hello world")')

        assert result.success
        assert result.stdout == "hello world\n"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_captures_multiple_prints(self, test_agent):
        """Multiple prints should all be captured."""
        code = """
print("line 1")
print("line 2")
print("line 3")
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout
        assert "line 3" in result.stdout

    @pytest.mark.asyncio
    async def test_empty_code_no_stdout(self, test_agent):
        """Empty code should produce no stdout."""
        result = await test_agent.runtime.execute_code("x = 1")

        assert result.success
        assert result.stdout == ""


class TestExecuteCodeDefinedMethods:
    """Tests for defined_methods capture in execute_code()."""

    @pytest.mark.asyncio
    async def test_captures_method_definition(self, test_agent):
        """Method definitions (functions with self) should be captured."""
        code = """


def my_method(self, x):
    return x * 2
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "my_method" in result.defined_methods
        # Method should be bound
        method = result.defined_methods["my_method"]
        assert callable(method)

    @pytest.mark.asyncio
    async def test_captured_method_is_callable(self, test_agent):
        """Captured method should be callable.: captured helpers are plain functions (not bound to the agent),
        so callers pass ``self`` explicitly.
        """
        code = """


def double(self, x):
    return x * 2
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        method = result.defined_methods["double"]
        assert method(test_agent, 5) == 10

    @pytest.mark.asyncio
    async def test_captures_async_method(self, test_agent):
        """Async method definitions should be captured."""
        code = """


async def async_method(self, x):
    return x + 1
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "async_method" in result.defined_methods

    @pytest.mark.asyncio
    async def test_ignores_non_method_functions(self, test_agent):
        """Functions without self should not be in defined_methods."""
        code = """


def helper(x):
    return x * 2


def my_method(self, x):
    return helper(x)
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "my_method" in result.defined_methods
        assert "helper" not in result.defined_methods


class TestExecuteCodeErrors:
    """Tests for error handling in execute_code()."""

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self, test_agent):
        """Syntax errors should be captured in result."""
        result = await test_agent.runtime.execute_code("def broken(")

        assert not result.success
        assert result.error is not None
        # Validation wraps SyntaxError in RestrictedCodeError
        assert "syntax" in str(result.error).lower() or "SyntaxError" in str(type(result.error))

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self, test_agent):
        """Runtime errors should be captured in result."""
        result = await test_agent.runtime.execute_code("x = 1 / 0")

        assert not result.success
        assert result.error is not None
        assert isinstance(result.error, ZeroDivisionError)

    @pytest.mark.asyncio
    async def test_partial_stdout_on_error(self, test_agent):
        """Stdout before error should still be captured."""
        code = """
print("before")
x = 1 / 0
print("after")
"""
        result = await test_agent.runtime.execute_code(code)

        assert not result.success
        assert "before" in result.stdout


class TestExecuteCodeBuiltins:
    """Tests for builtins parameter in execute_code()."""

    @pytest.mark.asyncio
    async def test_builtins_available_in_code(self, test_agent):
        """Builtins should be available in executed code."""
        captured = []

        def my_builtin(x):
            captured.append(x)

        result = await test_agent.runtime.execute_code(
            'my_builtin("hello")',
            builtins={"my_builtin": my_builtin},
        )

        assert result.success
        assert captured == ["hello"]

    @pytest.mark.asyncio
    async def test_builtins_override_globals(self, test_agent):
        """Builtins should override existing globals."""
        result = await test_agent.runtime.execute_code(
            "print(custom_value)",
            builtins={"custom_value": 42},
        )

        assert result.success
        assert "42" in result.stdout


class TestExecuteCodeAsync:
    """Tests for async code execution."""

    @pytest.mark.asyncio
    async def test_await_in_code(self, test_agent):
        """Code with await should execute correctly."""
        # asyncio is already in the namespace (pre-imported by execute_code)
        code = """
await asyncio.sleep(0)
print("done")
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "done" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
