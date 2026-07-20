# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PurePythonExecutor."""

import pytest

from nooa import Agent, strategy
from nooa.errors import GenerationError
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class TestAwaitDetection:
    """Tests to verify await detection works correctly (indirect via code execution)."""

    def test_await_in_string_not_detected(self):
        """'await' in string literal should not trigger async wrapping."""
        import ast

        # Simulate what the executor does
        code = 'x = "await something"'
        tree = ast.parse(code)
        has_await = any(isinstance(node, ast.Await) for node in ast.walk(tree))
        assert not has_await

    def test_await_in_comment_not_detected(self):
        """'await' in comment should not trigger async wrapping."""
        import ast

        code = "# await something\nx = 1"
        tree = ast.parse(code)
        has_await = any(isinstance(node, ast.Await) for node in ast.walk(tree))
        assert not has_await

    def test_actual_await_detected(self):
        """Actual await expression should be detected."""
        import ast

        code = "result = await foo()"
        # This will fail to parse in regular mode, need async context
        # So test with ast.parse in 'exec' mode won't work for bare await
        # The executor wraps it before parsing, so we test the wrapped version
        wrapped = f"async def __test__():\n    {code}"
        tree = ast.parse(wrapped)
        has_await = any(isinstance(node, ast.Await) for node in ast.walk(tree))
        assert has_await


class TestPurePythonIntegration:
    """Integration tests for PURE_PYTHON executor error handling."""

    @pytest.mark.asyncio
    async def test_max_turns_exceeded_raises_generation_error(self):
        """Test that GenerationError is raised when max_turns exceeded."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.value = 0

            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Create LLM that never defines the target method - just prints
        # Need enough responses for max_turns (default 10)
        # Note: PURE_PYTHON expects raw Python (no fences)
        fake_responses = [_resp(f"print('Turn {i}')") for i in range(15)]
        fake_llm = FakeLLMClient(scripted_responses=fake_responses)

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.compute()

        # Check error message mentions iterations and target method
        assert "10 iterations" in str(exc_info.value)
        assert "compute" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_syntax_error_feedback_allows_retry(self):
        """Test that syntax errors are returned to LLM for fixing."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @strategy(PurePythonStrategy())
            async def greet(self, name: str) -> str:
                """Return a greeting."""
                ...

        # First response has syntax error, second response fixes it
        # Note: PURE_PYTHON expects raw Python (no fences)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: Syntax error (missing colon)
                _resp("return f'Hello {name}"),  # Missing closing quote
                # Turn 2: Fixed syntax, returns result
                _resp("return f'Hello, {name}!'"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.greet("World")

        # Should succeed after retry
        assert result == "Hello, World!"
        # LLM should have been called twice (error + retry)
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_in_exploration_allows_retry(self):
        """Test that runtime errors during exploration are returned to LLM for fixing."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = [1, 2, 3]

            @strategy(PurePythonStrategy())
            async def sum_data(self) -> int:
                """Sum all data values."""
                ...

        # First response has runtime error in exploration code,
        # second response returns the correct result
        # Note: PURE_PYTHON expects raw Python (no fences)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: Runtime error in exploration code
                _resp("# Let me check the data first\nprint(self.data[100])  # This will fail"),
                # Turn 2: Fixed - return the result
                _resp("return sum(self.data)"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.sum_data()

        # Should succeed after retry
        assert result == 6  # sum([1, 2, 3])
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_response_feedback(self):
        """Test that empty responses trigger feedback to LLM."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @strategy(PurePythonStrategy())
            async def say_hello(self) -> str:
                """Return hello."""
                ...

        # First response is empty, second has proper code
        # Note: PURE_PYTHON expects raw Python (no fences)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: Empty response
                _resp(""),
                # Turn 2: Proper raw Python code - just return
                _resp("return 'Hello!'"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.say_hello()

        assert result == "Hello!"
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_message_builtin(self):
        """Test that the message() builtin works correctly."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @strategy(PurePythonStrategy())
            async def greet(self, name: str) -> str:
                """Greet the user by name."""
                ...

        # Response uses the message() builtin
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp('message("Generating greeting...")\nreturn f"Hello, {name}!"'),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.greet("World")

        assert result == "Hello, World!"
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_method_arguments_available_in_scope(self):
        """Test that method arguments are available in generated code scope."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.count = 0

            @strategy(PurePythonStrategy())
            async def double_value(self, value: int):
                """Double the given value and store in count."""
                ...

        # LLM generates code that uses the 'value' parameter
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("self.count = value * 2\nreturn None"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Call with value=21
        await agent_instance.double_value(21)

        # Should have doubled 21 to 42
        assert agent_instance.count == 42
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_method_for_none_return(self):
        """Test that empty/pass method body is allowed for methods returning None."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @strategy(PurePythonStrategy())
            async def acknowledge(self) -> None:
                """Just acknowledge, no action needed."""
                ...

        # LLM generates code that just passes (or returns None implicitly)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("pass"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Should succeed without error
        result = await agent_instance.acknowledge()
        assert result is None
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_multi_turn_generation(self):
        """Test that method can be completed over multiple turns."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.result = ""

            @strategy(PurePythonStrategy())
            async def process(self, text: str) -> str:
                """Process the text."""
                ...

        # First turn: exploration (no return statement)
        # Second turn: return the result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: Exploration only, no return
                _resp("# Let me think about this\nprint('Planning...')"),
                # Turn 2: Now return the result
                _resp("return text.upper()"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.process("hello")

        # Should complete after 2 turns
        assert result == "HELLO"
        assert fake_llm.call_count == 2
