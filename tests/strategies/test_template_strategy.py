# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TemplateStrategy - string templating without LLM calls."""

import dataclasses
from unittest.mock import Mock

import pytest

from nooa import strategy
from nooa.strategies import CurrentCall, TemplateStrategy


class MockAgent:
    """Mock agent for testing."""

    def __init__(self):
        self.tools = ["tool1", "tool2", "tool3"]
        self.doc = Mock()
        self.doc.show = Mock(return_value="Available tools: tool1, tool2, tool3")


class MockRuntime:
    """Mock RuntimeServices for testing.

    Implements full RuntimeServices Protocol for isinstance() checks.
    """

    def __init__(self, agent=None):
        self.agent = agent or MockAgent()
        self.event_manager = Mock()
        self._events = Mock()

    @property
    def events(self):
        """Event manager."""
        return self._events

    async def generate(self, *, tools=None, output_model=None, **kwargs):
        """Mock generate - not used by TemplateStrategy but required by Protocol."""
        raise NotImplementedError("MockRuntime.generate() not implemented")

    async def execute_code(self, code, *, builtins=None, validate=True, **kwargs):
        """Mock execute_code - not used by TemplateStrategy but required by Protocol."""
        raise NotImplementedError("MockRuntime.execute_code() not implemented")

    async def execute_nested(self, strategy, call):
        """Mock execute_nested - executes the strategy."""
        return await strategy.execute(self, call)

    async def expand_variables(self, template, extra_context=None, error_mode="raise"):
        """Mock expand_variables that evaluates Python expressions using eval()."""
        import string

        context = extra_context or {}

        # Use Python's string.Formatter to parse template
        formatter = string.Formatter()
        result_parts = []

        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            result_parts.append(literal_text)

            if field_name is not None:
                # Evaluate the expression using eval()
                try:
                    value = eval(field_name, {}, context)

                    # Apply conversion
                    if conversion == "r":
                        value = repr(value)
                    elif conversion == "s":
                        value = str(value)
                    elif conversion == "a":
                        value = ascii(value)

                    # Apply format spec
                    if format_spec:
                        result_parts.append(format(value, format_spec))
                    else:
                        result_parts.append(str(value))
                except Exception as e:
                    if error_mode == "raise":
                        raise
                    result_parts.append(f"{{{field_name} | ERROR: {e}}}")

        return "".join(result_parts)

    @property
    def truncation_config(self):
        """Truncation configuration."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        return DEFAULT_TRUNCATION_CONFIG

    def get_generation_id(self) -> str | None:
        """Get current generation ID."""
        return "mock-generation-id"

    def get_parent_generation_id(self) -> str | None:
        """Get parent generation ID."""
        return None


class TestTemplateStrategy:
    """Tests for TemplateStrategy."""

    def test_is_generation_strategy(self):
        """TemplateStrategy should inherit from GenerationStrategy."""
        from nooa.strategies.base import GenerationStrategy

        strategy = TemplateStrategy()
        assert isinstance(strategy, GenerationStrategy)

    def test_name(self):
        """TemplateStrategy name should be 'TEMPLATE'."""
        strategy = TemplateStrategy()
        assert strategy.name == "TEMPLATE"

    def test_traceable_is_false(self):
        """TemplateStrategy should not be traceable (no LLM call)."""
        strategy = TemplateStrategy()
        assert strategy.traceable is False

    def test_requires_lock_is_false(self):
        """TemplateStrategy should not require lock (no LLM call)."""
        strategy = TemplateStrategy()
        assert strategy.requires_lock is False

    def test_block_overrides_is_empty(self):
        """TemplateStrategy should have empty block overrides (no LLM prompting)."""
        strategy = TemplateStrategy()
        assert strategy.get_block_overrides() == {}

    @pytest.mark.asyncio
    async def test_simple_variable_substitution(self):
        """Test simple variable substitution like {name}."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_1",
            method_name="test",
            decorator="plan",
            docstring="Hello {name}!",
            kwargs={"name": "World"},
        )

        result = await strategy.execute(runtime, call)
        assert result == "Hello World!"

    @pytest.mark.asyncio
    async def test_multiple_variables(self):
        """Test multiple variable substitution."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_2",
            method_name="test",
            decorator="plan",
            docstring="Error in {method} at line {line}",
            kwargs={"method": "analyze", "line": 42},
        )

        result = await strategy.execute(runtime, call)
        assert result == "Error in analyze at line 42"

    @pytest.mark.asyncio
    async def test_self_expression_len(self):
        """Test Python expression like {len(self.tools)}."""
        strategy = TemplateStrategy()
        agent = MockAgent()
        runtime = MockRuntime(agent=agent)

        call = CurrentCall(
            id="test_3",
            method_name="test",
            decorator="plan",
            docstring="Found {len(self.tools)} tools",
            kwargs={},
        )

        result = await strategy.execute(runtime, call)
        assert result == "Found 3 tools"

    @pytest.mark.asyncio
    async def test_self_method_call(self):
        """Test method call expression like {self.doc.show()}."""
        strategy = TemplateStrategy()
        agent = MockAgent()
        runtime = MockRuntime(agent=agent)

        call = CurrentCall(
            id="test_4",
            method_name="test",
            decorator="plan",
            docstring="Available: {self.doc.show()}",
            kwargs={},
        )

        result = await strategy.execute(runtime, call)
        assert "Available tools: tool1, tool2, tool3" in result

    @pytest.mark.asyncio
    async def test_call_attribute_access(self):
        """Test accessing call attributes like {call.method_name}."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_5",
            method_name="my_method",
            decorator="plan",
            docstring="Executing {call.method_name}",
            kwargs={},
        )

        result = await strategy.execute(runtime, call)
        assert result == "Executing my_method"

    @pytest.mark.asyncio
    async def test_empty_template(self):
        """Test empty template returns empty string."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_6", method_name="test", decorator="plan", docstring="", kwargs={}
        )

        result = await strategy.execute(runtime, call)
        assert result == ""

    @pytest.mark.asyncio
    async def test_none_docstring(self):
        """Test None docstring returns empty string."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_7", method_name="test", decorator="plan", docstring=None, kwargs={}
        )

        result = await strategy.execute(runtime, call)
        assert result == ""

    @pytest.mark.asyncio
    async def test_multiline_template(self):
        """Test multiline template with variables."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        template = """Task: {task}
Result: {result}

Evaluation complete."""

        call = CurrentCall(
            id="test_8",
            method_name="test",
            decorator="plan",
            docstring=template,
            kwargs={"task": "analyze data", "result": "success"},
        )

        result = await strategy.execute(runtime, call)
        assert "Task: analyze data" in result
        assert "Result: success" in result
        assert "Evaluation complete." in result

    @pytest.mark.asyncio
    async def test_combines_kwargs_and_self(self):
        """Test that both kwargs and self.xxx work together."""
        strategy = TemplateStrategy()
        agent = MockAgent()
        runtime = MockRuntime(agent=agent)

        call = CurrentCall(
            id="test_9",
            method_name="test",
            decorator="plan",
            docstring="Method: {method}, Tools: {len(self.tools)}",
            kwargs={"method": "analyze"},
        )

        result = await strategy.execute(runtime, call)
        assert result == "Method: analyze, Tools: 3"


class TestTemplateTcContext:
    """tc (TruncationConfig) is injected into TemplateStrategy context.

    tc is injected AFTER kwargs so it always refers to the truncation config,
    even if a method has a parameter also named 'tc'.
    """

    @pytest.mark.asyncio
    async def test_tc_available_in_template(self):
        """tc is accessible in template expressions as runtime.truncation_config."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        strategy = TemplateStrategy()
        runtime = MockRuntime()

        call = CurrentCall(
            id="test_tc_1",
            method_name="test",
            decorator="plan",
            docstring="elements={tc.event_format.max_length}",
            kwargs={},
        )

        result = await strategy.execute(runtime, call)
        assert str(DEFAULT_TRUNCATION_CONFIG.event_format.max_length) in result

    @pytest.mark.asyncio
    async def test_tc_format_parameters_as_code_pattern(self):
        """The canonical usage: {call.format_parameters_as_code(tc=tc)} works."""
        strategy = TemplateStrategy()
        runtime = MockRuntime()

        def test(self, items: list): ...

        call = dataclasses.replace(
            CurrentCall.from_method(test, args=(list(range(5)),)),
            id="test_tc_2",
            docstring="Params:\n{call.format_parameters_as_code(tc=tc)}",
        )

        result = await strategy.execute(runtime, call)
        assert "items" in result  # parameter name present
        assert "0" in result  # values present

    @pytest.mark.asyncio
    async def test_tc_wins_over_kwarg_named_tc(self):
        """tc injected after kwargs — a method param named 'tc' doesn't shadow the config."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        strategy = TemplateStrategy()
        runtime = MockRuntime()

        # Method has a kwarg literally named "tc" with a non-config value
        call = CurrentCall(
            id="test_tc_3",
            method_name="test",
            decorator="plan",
            docstring="elements={tc.event_format.max_length}",
            kwargs={"tc": "this_should_be_overridden"},
        )

        result = await strategy.execute(runtime, call)
        # The injected TruncationConfig wins — the string kwarg was overridden
        assert str(DEFAULT_TRUNCATION_CONFIG.event_format.max_length) in result
        assert "this_should_be_overridden" not in result


class TestPlanDecoratorOnStrategies:
    """Tests for @strategy decorator working on strategy methods."""

    @pytest.mark.asyncio
    async def test_plan_on_strategy_method(self):
        """Test @strategy decorator on strategy method."""
        from nooa.strategies import CompositeStrategy

        class TestStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())
            async def build_prompt(self, runtime, task: str) -> str:
                """Build prompt for {task}"""
                ...

            async def execute(self, runtime, call):
                """Dummy execute implementation."""
                pass

        test_strategy = TestStrategy()
        runtime = MockRuntime()

        # The @strategy decorator should route this through TemplateStrategy
        result = await test_strategy.build_prompt(runtime, task="analyze")
        assert result == "Build prompt for analyze"

    @pytest.mark.asyncio
    async def test_plan_requires_explicit_strategy(self):
        """Test that @strategy with ellipsis on strategy should specify strategy explicitly."""
        from nooa.strategies import CompositeStrategy

        # When @strategy() has ellipsis but no strategy, it defaults to PurePythonStrategy
        # This is OK for now, but ideally we'd want explicit strategy on strategies
        # For this test, we just verify the behavior

        class TestStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())  # Explicit strategy - good practice
            async def build_prompt(self, runtime, task: str) -> str:
                """Template for {task}"""
                ...

            async def execute(self, runtime, call):
                """Dummy execute implementation."""
                pass

        test_strategy = TestStrategy()
        runtime = MockRuntime()

        result = await test_strategy.build_prompt(runtime, task="test")
        assert result == "Template for test"

    @pytest.mark.asyncio
    async def test_multiple_plan_methods_on_strategy(self):
        """Test multiple @strategy methods on same strategy."""
        from nooa.strategies import CompositeStrategy

        class TestStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())
            async def prompt1(self, runtime, x: int) -> str:
                """Value: {x}"""
                ...

            @strategy(TemplateStrategy())
            async def prompt2(self, runtime, y: str) -> str:
                """Text: {y}"""
                ...

            async def execute(self, runtime, call):
                """Dummy execute implementation."""
                pass

        test_strategy = TestStrategy()
        runtime = MockRuntime()

        result1 = await test_strategy.prompt1(runtime, x=42)
        result2 = await test_strategy.prompt2(runtime, y="hello")

        assert result1 == "Value: 42"
        assert result2 == "Text: hello"

    @pytest.mark.asyncio
    async def test_plan_with_multiple_kwargs(self):
        """Test @strategy method with multiple keyword arguments."""
        from nooa.strategies import CompositeStrategy

        class TestStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())
            async def build_message(self, runtime, name: str, age: int, city: str) -> str:
                """Hello {name}, age {age}, from {city}"""
                ...

            async def execute(self, runtime, call):
                """Dummy execute implementation."""
                pass

        test_strategy = TestStrategy()
        runtime = MockRuntime()

        result = await test_strategy.build_message(runtime, name="Alice", age=30, city="NYC")

        assert result == "Hello Alice, age 30, from NYC"

    @pytest.mark.asyncio
    async def test_plan_with_call_object_parameter(self):
        """Test @strategy method that receives a CurrentCall object and accesses its attributes.

        This tests the pattern used in PurePythonStrategy._build_task_message where
        we pass a CurrentCall object and want to access its attributes in the template.
        """
        from nooa.strategies import CompositeStrategy

        class TestStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())
            async def build_task_message(self, runtime, current_call: CurrentCall) -> str:
                """Task: {current_call.method_name}, Args: {current_call.args}"""
                ...

            async def execute(self, runtime, call):
                """Dummy execute implementation."""
                pass

        test_strategy = TestStrategy()
        runtime = MockRuntime()

        # Create a sample call object
        sample_call = CurrentCall(
            id="test_call",
            method_name="analyze_sentiment",
            decorator="plan",
            docstring="Analyze the sentiment of the text.",
            args=("I love this product!",),
            kwargs={},
        )

        # Test with keyword argument (this works correctly)
        result = await test_strategy.build_task_message(runtime, current_call=sample_call)

        # Verify the template was properly rendered
        assert "Task: analyze_sentiment" in result
        assert "Args: ('I love this product!',)" in result

        # Verify no double rendering (template vars shouldn't appear)
        assert "{current_call.method_name}" not in result
        assert "{current_call.args}" not in result
