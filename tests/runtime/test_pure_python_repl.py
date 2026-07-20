# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test PURE_PYTHON REPL behavior via RuntimeServices.execute_code().

Tests that stdout (print statements) is properly captured and returned
to the agent as feedback.

Note: Unlike IPython, bare expressions are NOT captured - only print() output.
This is by design (simpler implementation, explicit is better than implicit).

These tests complement test_execute_code.py with specific REPL-style behavior tests.
"""

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


class TestReplStyleBehavior:
    """Test REPL-style behavior: what gets captured vs what doesn't."""

    @pytest.mark.asyncio
    async def test_bare_expression_not_captured(self, test_agent):
        """Bare expressions (without print) are NOT captured - by design.

        Unlike IPython or Python REPL, we don't capture expression results.
        This is intentional: explicit print() is required.
        """
        result = await test_agent.runtime.execute_code("5 + 3")

        assert result.success
        # Unlike IPython, bare expressions don't produce output
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_variable_expression_not_captured(self, test_agent):
        """Variable at end of code is NOT captured - must use print().

        This differs from IPython where the last expression is auto-displayed.
        """
        code = """
x = 42
x
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        # Must use print(x) to see the value
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_assignment_no_output(self, test_agent):
        """Simple assignments produce no output."""
        result = await test_agent.runtime.execute_code("x = 2 ** 10")

        assert result.success
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_print_required_for_output(self, test_agent):
        """print() is required to see values."""
        # Without print - no output
        result1 = await test_agent.runtime.execute_code("x = 42; x")
        assert result1.success
        assert result1.stdout == ""

        # With print - output captured
        result2 = await test_agent.runtime.execute_code("x = 42; print(x)")
        assert result2.success
        assert "42" in result2.stdout


class TestPrintVsExpression:
    """Test the difference between print() and bare expressions."""

    @pytest.mark.asyncio
    async def test_print_shows_value(self, test_agent):
        """print() shows the value."""
        result = await test_agent.runtime.execute_code('print("hello")')

        assert result.success
        assert result.stdout == "hello\n"

    @pytest.mark.asyncio
    async def test_string_expression_no_output(self, test_agent):
        """String expression without print produces no output."""
        result = await test_agent.runtime.execute_code('"hello"')

        assert result.success
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_list_expression_no_output(self, test_agent):
        """List expression without print produces no output."""
        result = await test_agent.runtime.execute_code("[1, 2, 3]")

        assert result.success
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_print_list_shows_value(self, test_agent):
        """print() on a list shows it."""
        result = await test_agent.runtime.execute_code("print([1, 2, 3])")

        assert result.success
        assert "[1, 2, 3]" in result.stdout


class TestComputationOutput:
    """Test output from computations."""

    @pytest.mark.asyncio
    async def test_computation_only_print_captured(self, test_agent):
        """Only print() output is captured, not computation results."""
        code = """
data = [1, 2, 3, 4, 5]
total = sum(data)
print(f"Total: {total}")
total  # This won't be captured
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "Total: 15" in result.stdout
        # The bare 'total' expression at the end doesn't add anything
        assert result.stdout.strip() == "Total: 15"

    @pytest.mark.asyncio
    async def test_debug_print_pattern(self, test_agent):
        """Common debug pattern: print intermediate values."""
        code = """
x = 10
print(f"x = {x}")
y = x * 2
print(f"y = {y}")
result = x + y
print(f"result = {result}")
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert "x = 10" in result.stdout
        assert "y = 20" in result.stdout
        assert "result = 30" in result.stdout


class TestMethodDefinitionOutput:
    """Test that method definitions don't produce output."""

    @pytest.mark.asyncio
    async def test_method_definition_no_stdout(self, test_agent):
        """Defining a method produces no stdout."""
        code = """


def my_method(self, x):
    return x * 2
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        assert result.stdout == ""
        assert "my_method" in result.defined_methods

    @pytest.mark.asyncio
    async def test_method_with_print_inside(self, test_agent):
        """Print inside method only outputs when method is called."""
        code = """


def my_method(self, x):
    print(f"Processing: {x}")
    return x * 2
"""
        result = await test_agent.runtime.execute_code(code)

        assert result.success
        # No output yet - method wasn't called
        assert result.stdout == ""
        assert "my_method" in result.defined_methods

        # Now call the method.: it's a plain function, so pass self explicitly.
        method = result.defined_methods["my_method"]
        method(test_agent, 5)
        # The print happens but we'd need to capture it in a new execute_code call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
