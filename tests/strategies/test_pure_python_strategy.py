# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PurePythonStrategy.

TDD: Write these tests first, then implement pure_python.py to make them pass.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nooa import Agent, strategy
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


class TestPurePythonStrategyProperties:
    """Tests for PurePythonStrategy properties."""

    def test_name_is_pure_python(self):
        """PurePythonStrategy.name should be 'PURE_PYTHON'."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        assert strategy.name == "PURE_PYTHON"

    def test_block_overrides_provides_strategy_prompt(self):
        """PurePythonStrategy.get_block_overrides() should provide strategy_prompt block."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        blocks = strategy.get_block_overrides()
        assert "strategy_prompt" in blocks
        assert blocks["strategy_prompt"] is not None


class TestPurePythonStrategyConfig:
    """Tests for PurePythonStrategy configuration."""

    def test_default_max_iterations(self):
        """Default max_iterations should be 10."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        assert strategy.max_iterations == 10

    def test_default_max_retries(self):
        """Default max_retries should be 3."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        assert strategy.max_retries == 3

    def test_custom_max_iterations(self):
        """Should accept custom max_iterations via config."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy(max_iterations=5)
        assert strategy.max_iterations == 5

    def test_custom_max_retries(self):
        """Should accept custom max_retries via config."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy(max_retries=2)
        assert strategy.max_retries == 2


class TestPurePythonStrategyInheritance:
    """Tests for PurePythonStrategy inheritance."""

    def test_is_generation_strategy(self):
        """PurePythonStrategy should inherit from GenerationStrategy."""
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        assert isinstance(strategy, GenerationStrategy)

    def test_has_execute_method(self):
        """PurePythonStrategy should implement execute()."""
        from nooa.strategies.pure_python import PurePythonStrategy

        strategy = PurePythonStrategy()
        assert hasattr(strategy, "execute")
        assert callable(strategy.execute)


class TestPurePythonStrategyExecute:
    """Tests for PurePythonStrategy.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_simple_return(self, mock_runtime):
        """execute() should handle simple return value generation."""
        from nooa.events import ExecutionResult
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.pure_python import PurePythonStrategy

        # Create a bound method that returns 42
        def get_answer(self):
            return 42

        def bound_method():
            return 42

        # Setup mock to return code with return statement (REPL style)
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content="return 42",
                    reasoning=None,
                    usage={},
                ),
                "event_123",
            )
        )
        # execute_code returns ExecutionResult with returned_value
        mock_runtime.execute_code = AsyncMock(
            return_value=ExecutionResult(
                stdout="",
                error=None,
                defined_methods={},
                returned_value=42,
            )
        )

        strategy = PurePythonStrategy()
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
            signature="(self) -> int",
            docstring="Return the answer.",
            args=(),
            kwargs={},
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_uses_max_iterations(self):
        """execute() should respect custom max_iterations config."""
        from nooa.errors import GenerationError
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.unifiedllm import FakeLLMClient

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_iterations=2, max_retries=10))
            async def never_completes(self) -> str:
                """This method will never complete."""
                ...

        # LLM generates code that never returns (exceeds max_iterations)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("# Iteration 1\nprint('no return')"),
                _resp("# Iteration 2\nprint('still no return')"),
                _resp("# Iteration 3 (should not be reached)\nprint('extra')"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.never_completes()

        # Verify error message mentions iterations
        assert "iterations" in str(exc_info.value).lower()
        assert "never_completes" in str(exc_info.value)
        # Verify we hit max_iterations (2), not max_retries (10)
        assert "2" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_messages_added_to_history_with_content(self):
        """LLM assistant messages should be added to history."""
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.unifiedllm import FakeLLMClient

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_iterations=5))
            async def solve_task(self, x: int) -> int:
                """Add 10 to x and return the result."""
                ...

        # LLM generates exploratory code, then the actual solution
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp('print("calculating")'),  # First turn: exploratory code
                _resp("return x + 10"),  # Second turn: actual solution
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.solve_task(5)

        # Verify result is correct
        assert result == 15

        # Get LLM-generated events (exclude synthetic prefill events)
        history_events = agent_instance.event_manager.values()
        assistant_events = [
            e
            for e in history_events
            if e.event_type == "LLMOutput" and not (e.metadata or {}).get("prefill")
        ]

        # Should have 2 assistant events (one per LLM call)
        assert len(assistant_events) >= 2

        # First assistant message should contain the code
        first_msg = assistant_events[0].content
        assert first_msg, "First assistant message should not be empty"
        assert "print" in first_msg, "First message should contain the print statement"

        # Second assistant message should contain the return statement
        second_msg = assistant_events[1].content
        assert second_msg, "Second assistant message should not be empty"
        assert "return" in second_msg, "Second message should contain return statement"


class TestPurePythonFencedCodeBlocks:
    """Tests for handling fenced code blocks in LLM responses."""

    @pytest.mark.asyncio
    async def test_fenced_code_block_is_accepted(self):
        """LLM returning code in ```python ... ``` fences should work."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def process(self, x: int) -> int:
                """Double the value."""
                ...

        # LLM responds with fenced code block
        fenced_code = """```python
return x * 2
```"""
        fake_llm = FakeLLMClient(scripted_responses=[_resp(fenced_code)])

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process(21)

        assert result == 42

    @pytest.mark.asyncio
    async def test_fenced_code_block_without_language_tag(self):
        """LLM returning code in ``` ... ``` fences (no language) should work."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def greet(self, name: str) -> str:
                """Return greeting."""
                ...

        # LLM responds with fenced code block without language tag
        fenced_code = """```
return f"Hello, {name}!"
```"""
        fake_llm = FakeLLMClient(scripted_responses=[_resp(fenced_code)])

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.greet("World")

        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_fenced_multiline_code_block_is_accepted(self):
        """LLM returning multiline code in fences should work."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.items = [1, 2, 3, 4, 5]

            @strategy(PurePythonStrategy())
            async def process_items(self) -> list[int]:
                """Double all items."""
                ...

        # LLM responds with multiline fenced code
        fenced_code = """```python
result = []
for item in self.items:
    result.append(item * 2)
return result
```"""
        fake_llm = FakeLLMClient(scripted_responses=[_resp(fenced_code)])

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process_items()

        assert result == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_fenced_code_stored_clean_in_history(self):
        """Fenced code should be stored WITHOUT fences in history for LLM learning."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self, x: int) -> int:
                """Compute x + 1."""
                ...

        fenced_code = """```python
return x + 1
```"""
        fake_llm = FakeLLMClient(scripted_responses=[_resp(fenced_code)])

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute(5)

        assert result == 6

        # Check history - the assistant message should have CLEAN code (no fences)
        history_events = agent_instance.event_manager.values()
        assistant_events = [
            e
            for e in history_events
            if e.event_type == "LLMOutput" and not (e.metadata or {}).get("prefill")
        ]

        assert len(assistant_events) >= 1
        stored_code = assistant_events[0].content

        # Should NOT contain fence markers
        assert "```" not in stored_code, f"History should not contain fences: {stored_code}"
        # Should contain the actual code
        assert "return x + 1" in stored_code


class TestPurePythonMalformedOutputs:
    """Tests for handling malformed LLM outputs (XML tags, etc.).

    Some LLMs confuse the expected output format and return code wrapped
    in XML-like tags (e.g., <tool_code>) instead of plain Python.
    The system now strips simple XML wrappers and provides clear error messages.
    """

    @pytest.mark.asyncio
    async def test_simple_xml_wrapper_is_stripped(self):
        """LLM returning code in a single XML wrapper tag should work.

        Simple XML wrappers like <tool_code>...</tool_code> are now stripped,
        allowing the inner Python code to execute successfully.
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def simple_add(self, x: int, y: int) -> int:
                """Add x and y."""
                ...

        # LLM responds with code wrapped in <tool_code> XML tags
        wrapped_response = """<tool_code>
return x + y
</tool_code>"""

        fake_llm = FakeLLMClient(scripted_responses=[_resp(wrapped_response)])

        agent_instance = TestAgent(llm=fake_llm)

        # Should succeed - XML wrapper is stripped
        result = await agent_instance.simple_add(10, 32)

        assert result == 42
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_xml_wrapper_with_attributes_is_stripped(self):
        """LLM returning code in XML tag with attributes should work.

        The real-world example from the bug report includes attributes
        like expr="..." and timestamp="...".
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def calculate_single(self, a: int, b: int, calculation: str) -> int:
                """Perform a calculation based on the description."""
                ...

        # Real-world example with attributes on the XML tag
        wrapped_response = """<tool_code expr="self.calculate_single(a=a, b=b, calculation=calculation)" timestamp="2025-12-19T09:27:49.000380">
result = a + b
return result
</tool_code>"""

        fake_llm = FakeLLMClient(scripted_responses=[_resp(wrapped_response)])

        agent_instance = TestAgent(llm=fake_llm)

        # Should succeed - XML wrapper is stripped
        result = await agent_instance.calculate_single(123, 456, "add these numbers")

        assert result == 579  # 123 + 456
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_nested_xml_tags_cause_clear_error(self):
        """Nested XML tags should produce a clear error message."""
        from nooa.errors import GenerationError
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_retries=1, max_iterations=2))
            async def compute(self, x: int) -> int:
                """Double x."""
                ...

        # LLM responds with nested XML tags
        nested_response = """<outer>
<inner>
return x * 2
</inner>
</outer>"""

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(nested_response),
                _resp(nested_response),  # Still wrong on retry
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.compute(21)

        # Verify the error feedback was clear about XML tags
        history_events = agent_instance.event_manager.values()
        error_events = [e for e in history_events if e.event_type == "Error"]

        assert len(error_events) >= 1
        error_content = error_events[0].content
        # Should mention XML/HTML and plain Python
        assert "xml" in error_content.lower() or "plain python" in error_content.lower()

    @pytest.mark.asyncio
    async def test_xml_error_allows_llm_to_retry_and_succeed(self):
        """LLM can recover from XML format error when given clear feedback."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_retries=3, max_iterations=3))
            async def greet(self, name: str) -> str:
                """Return a greeting."""
                ...

        # First response has nested XML (error), second is clean
        nested_response = """<outer>
<inner>
return f"Hello, {name}!"
</inner>
</outer>"""

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(nested_response),
                _resp('return f"Hello, {name}!"'),  # Fixed response
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Should succeed after retry
        result = await agent_instance.greet("World")

        assert result == "Hello, World!"
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_xml_in_python_string_is_preserved(self):
        """XML within Python string literals should not be affected.

        Code that legitimately uses XML strings like return "<config>value</config>"
        should work fine since we only strip wrapper tags around the entire response.
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def get_xml_config(self) -> str:
                """Return an XML config string."""
                ...

        # Response is plain Python that returns an XML string
        response = 'return "<config><option>value</option></config>"'

        fake_llm = FakeLLMClient(scripted_responses=[_resp(response)])

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.get_xml_config()

        assert result == "<config><option>value</option></config>"

    @pytest.mark.asyncio
    async def test_xml_wrapping_markdown_fences_is_stripped(self):
        """LLM returning XML tags wrapping markdown fences should work.

        Some LLMs combine both formats: <tool_code>```python...```</tool_code>
        Both layers should be stripped.
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def calculate_single(self, a: int, b: int, calculation: str) -> int:
                """Perform a calculation based on the description."""
                ...

        # Real-world example: XML wrapping markdown fences
        wrapped_response = """<tool_code expr="self.calculate_single(a=a, b=b, calculation=calculation)" timestamp="2025-12-19T10:43:20.397120">
```python
result = a + b
return result
```
</tool_code>"""

        fake_llm = FakeLLMClient(scripted_responses=[_resp(wrapped_response)])

        agent_instance = TestAgent(llm=fake_llm)

        # Should succeed - both XML and markdown wrappers are stripped
        result = await agent_instance.calculate_single(100, 50, "add these numbers")

        assert result == 150
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_history_stores_clean_code_after_xml_stripping(self):
        """History should store the clean code (without XML wrapper) for LLM learning."""
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self, x: int) -> int:
                """Return x plus 1."""
                ...

        wrapped_response = """<tool_code>
return x + 1
</tool_code>"""

        fake_llm = FakeLLMClient(scripted_responses=[_resp(wrapped_response)])

        agent_instance = TestAgent(llm=fake_llm)

        result = await agent_instance.compute(5)

        assert result == 6

        # Check history - should have clean code without XML tags
        history_events = agent_instance.event_manager.values()
        assistant_events = [
            e
            for e in history_events
            if e.event_type == "LLMOutput" and not (e.metadata or {}).get("prefill")
        ]

        assert len(assistant_events) >= 1
        stored_code = assistant_events[0].content

        # Should NOT contain XML tags
        assert "<tool_code>" not in stored_code
        assert "</tool_code>" not in stored_code
        # Should contain the actual code
        assert "return x + 1" in stored_code


@pytest.fixture
def mock_runtime():
    """Create mock runtime for strategy tests."""

    class MockRuntime:
        def __init__(self):
            self._agent = MagicMock()
            self._agent.agent_id = "test_agent"
            self._agent.__class__.__name__ = "TestAgent"
            self._agent.event_manager = MagicMock()
            self._agent.events = []
            self._events = MagicMock()
            self._events.add = MagicMock(return_value="event_123")
            self._events.get = MagicMock(return_value=None)
            self._events.update = MagicMock(return_value=True)

        @property
        def agent(self):
            return self._agent

        @property
        def event_manager(self):
            """Event manager."""
            return self._events

        async def generate(self, *, tools=None, **kwargs):
            response = MagicMock(content="", reasoning=None, usage={})
            return response, "event_123"

        async def execute_code(self, code, builtins=None, validate=True, wrap_in_function=False):
            return None

        async def execute_nested(self, strategy, call):
            """Execute nested strategy (for @strategy methods)."""
            return await strategy.execute(self, call)

        @property
        def truncation_config(self):
            """Truncation configuration."""
            from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

            return DEFAULT_TRUNCATION_CONFIG

        def get_generation_id(self) -> str | None:
            """Get the current generation session ID."""
            return "mock-generation-id"

        def get_parent_generation_id(self) -> str | None:
            """Get the parent generation session ID."""
            return None

        async def expand_variables(self, template, extra_context=None, error_mode="raise"):
            """Simple variable expansion for templates."""
            context = extra_context or {}
            result = template

            # Handle simple {variable} patterns
            for key, value in context.items():
                result = result.replace(f"{{{key}}}", str(value))

            # Handle {self.xxx} patterns
            if "self" in context:
                if "{len(self.tools)}" in result:
                    tools = getattr(context["self"], "tools", [])
                    result = result.replace("{len(self.tools)}", str(len(tools)))

            return result

    return MockRuntime()


class TestPurePythonTurnEvents:
    """Tests for turn event emission in PurePythonStrategy.

    Note: Turn events have Role.RUNTIME_EVENT and are emitted but not stored in event_manager.
    We capture them via event_manager.on() event handlers.
    """

    @pytest.mark.asyncio
    async def test_emits_before_and_after_turn_events_on_success(self):
        """Strategy should emit BeforeTurn and AfterTurn on successful execution."""
        from nooa.events import AfterTurn, BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Single turn: return directly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("return 42"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 42

        # Filter captured events
        before_events = [e for e in captured_events if isinstance(e, BeforeTurn)]
        after_events = [e for e in captured_events if isinstance(e, AfterTurn)]

        # Should have exactly one turn (before + after)
        assert len(before_events) == 1, f"Expected 1 BeforeTurn, got {len(before_events)}"
        assert len(after_events) == 1, f"Expected 1 AfterTurn, got {len(after_events)}"

        # Verify BeforeTurn
        assert before_events[0].method_name == "compute"
        assert before_events[0].strategy == "PURE_PYTHON"
        assert before_events[0].turn_number == 1

        # Verify AfterTurn
        assert after_events[0].method_name == "compute"
        assert after_events[0].strategy == "PURE_PYTHON"
        assert after_events[0].turn_number == 1
        assert after_events[0].is_final is True
        assert after_events[0].success is True
        assert after_events[0].exception_type is None

    @pytest.mark.asyncio
    async def test_emits_after_turn_event_on_failure(self):
        """Strategy should emit AfterTurn with success=False on failure."""
        from nooa.errors import GenerationError
        from nooa.events import AfterTurn, BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_retries=2))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Always return empty (trigger error path)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(""),
                _resp(""),
                _resp(""),  # Extra for exhaustion
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        with pytest.raises(GenerationError):
            await agent_instance.compute()

        # Filter captured events
        after_events = [e for e in captured_events if isinstance(e, AfterTurn)]

        # Should have multiple turns (one per retry + final)
        assert len(after_events) >= 2  # At least 2 intermediate turns + final

        # The last AfterTurn should indicate final failure
        final_after = after_events[-1]
        assert final_after.is_final is True
        assert final_after.success is False
        assert final_after.exception_type == "GenerationError"

    @pytest.mark.asyncio
    async def test_turn_number_increments_across_iterations(self):
        """Turn number should increment with each iteration."""
        from nooa.events import BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Two turns: first sets x, second returns x
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("x = 42"),  # First turn: no return
                _resp("return x"),  # Second turn: return
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, BeforeTurn):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 42

        # Filter captured events
        before_events = [e for e in captured_events if isinstance(e, BeforeTurn)]

        # Should have two turns
        assert len(before_events) == 2
        assert before_events[0].turn_number == 1
        assert before_events[1].turn_number == 2

    @pytest.mark.asyncio
    async def test_turn_events_have_generation_id(self):
        """Turn events should have a non-empty generation_id."""
        from nooa.events import AfterTurn, BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_resp("return 42")])
        agent_instance = TestAgent(llm=fake_llm)

        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        await agent_instance.compute()

        # All turn events should have generation_id
        for event in captured_events:
            assert event.generation_id is not None
            assert len(event.generation_id) > 0

        # All events in same method should have same generation_id
        generation_ids = [e.generation_id for e in captured_events]
        assert len(set(generation_ids)) == 1, "All turn events should have same generation_id"

    @pytest.mark.asyncio
    async def test_turn_events_not_recorded_in_history(self):
        """Turn events should NOT appear in event_manager (record=False)."""
        from nooa.events import AfterTurn, BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_resp("return 42")])
        agent_instance = TestAgent(llm=fake_llm)

        # Track emitted events
        emitted_count = {"before": 0, "after": 0}

        def capture_event(event):
            if isinstance(event, BeforeTurn):
                emitted_count["before"] += 1
            elif isinstance(event, AfterTurn):
                emitted_count["after"] += 1

        agent_instance.event_manager.on("*", capture_event)

        await agent_instance.compute()

        # Events were emitted
        assert emitted_count["before"] >= 1
        assert emitted_count["after"] >= 1

        # But they're not in event_manager
        turn_events_in_history = [
            e
            for e in agent_instance.event_manager.values()
            if isinstance(e, (BeforeTurn, AfterTurn))
        ]
        assert len(turn_events_in_history) == 0, "Turn events should not be recorded"

    @pytest.mark.asyncio
    async def test_turn_events_have_matching_before_after_pairs(self):
        """Each BeforeTurn should have a matching AfterTurn."""
        from nooa.events import AfterTurn, BeforeTurn
        from nooa.strategies.pure_python import PurePythonStrategy

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Multi-turn execution
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("x = 1"),
                _resp("y = 2"),
                _resp("return x + y"),
            ]
        )
        agent_instance = TestAgent(llm=fake_llm)

        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 3

        before_events = [e for e in captured_events if isinstance(e, BeforeTurn)]
        after_events = [e for e in captured_events if isinstance(e, AfterTurn)]

        # Should have matching pairs
        assert len(before_events) == len(after_events)
        assert len(before_events) == 3  # 3 turns

        # Turn numbers should match
        for before, after in zip(before_events, after_events, strict=True):
            assert before.turn_number == after.turn_number
            assert before.generation_id == after.generation_id
