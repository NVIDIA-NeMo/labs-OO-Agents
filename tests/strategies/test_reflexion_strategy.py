# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ReflexionStrategy.

Tests the composite strategy that wraps a base strategy with reflection.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nooa.config.strategy_config import ReflexionConfig


def make_smart_execute_nested(mock_runtime, base_results):
    """Create execute_nested that handles TemplateStrategy + returns base_results for others.

    Args:
        mock_runtime: The mock runtime fixture
        base_results: List of results or exceptions to return for non-Template strategies
    """
    from nooa.strategies.template import TemplateStrategy

    base_iter = iter(base_results)
    original_execute_nested = mock_runtime.execute_nested

    async def smart_side_effect(strategy, call):
        if isinstance(strategy, TemplateStrategy):
            # Let TemplateStrategy actually execute
            return await original_execute_nested(strategy, call)
        else:
            # Get next base result (could be a value or exception)
            result = next(base_iter)
            if isinstance(result, Exception):
                raise result
            return result

    return smart_side_effect


class TestReflexionStrategyProperties:
    """Tests for ReflexionStrategy properties."""

    def test_name_includes_base_strategy(self):
        """ReflexionStrategy.name should include base strategy name."""
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        assert "REFLEXION" in strategy.name
        assert "PURE_PYTHON" in strategy.name

    def test_name_with_custom_base(self):
        """ReflexionStrategy.name should reflect custom base strategy."""
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.strategies.reflexion import ReflexionStrategy

        base = PurePythonStrategy()
        strategy = ReflexionStrategy(base=base)
        assert strategy.name == "REFLEXION[PURE_PYTHON]"

    def test_block_overrides_delegates_to_base(self):
        """ReflexionStrategy.get_block_overrides should delegate to base strategy."""
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.strategies.reflexion import ReflexionStrategy

        base = PurePythonStrategy()
        strategy = ReflexionStrategy(base=base)
        assert strategy.get_block_overrides() == base.get_block_overrides()

    def test_requires_lock_delegates_to_base(self):
        """ReflexionStrategy.requires_lock should inherit from base."""
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        # Default PurePythonStrategy requires lock
        assert strategy.requires_lock is True


class TestReflexionStrategyConfig:
    """Tests for ReflexionStrategy configuration."""

    def test_default_max_reflections(self):
        """Default max_iterations should be 3."""
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        assert strategy.config.max_iterations == 3

    def test_custom_max_reflections(self):
        """Should accept custom max_iterations via config."""
        from nooa.config.strategy_config import ReflexionConfig
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=5))
        assert strategy.config.max_iterations == 5

    def test_rejects_old_max_reflections_kwarg(self):
        """Should reject old max_reflections kwarg."""
        from nooa.strategies.reflexion import ReflexionStrategy

        with pytest.raises(TypeError):
            ReflexionStrategy(max_reflections=5)  # old API — must fail

    def test_default_base_is_pure_python(self):
        """Default base strategy should be PurePythonStrategy."""
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        assert isinstance(strategy.base, PurePythonStrategy)

    def test_custom_base_strategy(self):
        """Should accept custom base strategy."""
        from nooa.strategies.pure_python import PurePythonStrategy
        from nooa.strategies.reflexion import ReflexionStrategy

        base = PurePythonStrategy(max_iterations=5)
        strategy = ReflexionStrategy(base=base)
        assert strategy.base is base
        assert strategy.base.max_iterations == 5


class TestReflexionStrategyInheritance:
    """Tests for ReflexionStrategy inheritance."""

    def test_is_generation_strategy(self):
        """ReflexionStrategy should inherit from GenerationStrategy."""
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        assert isinstance(strategy, GenerationStrategy)

    def test_has_execute_method(self):
        """ReflexionStrategy should implement execute()."""
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        assert hasattr(strategy, "execute")
        assert callable(strategy.execute)


class TestReflectionOutput:
    """Tests for ReflectionOutput model."""

    def test_reflection_output_fields(self):
        """ReflectionOutput should have expected fields."""
        from nooa.strategies.reflexion import ReflectionOutput

        output = ReflectionOutput(
            is_satisfactory=True,
            issues=["issue1"],
            suggestions=["suggestion1"],
            reasoning="good",
        )
        assert output.is_satisfactory is True
        assert output.issues == ["issue1"]
        assert output.suggestions == ["suggestion1"]
        assert output.reasoning == "good"

    def test_reflection_output_defaults(self):
        """ReflectionOutput should have sensible defaults."""
        from nooa.strategies.reflexion import ReflectionOutput

        output = ReflectionOutput(is_satisfactory=True)
        assert output.issues == []
        assert output.suggestions == []
        assert output.reasoning == ""


class TestReflexionStrategyExecute:
    """Tests for ReflexionStrategy.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_returns_on_satisfactory(self, mock_runtime):
        """execute() should return immediately when reflection is satisfactory."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Setup: mock execute_nested using helper
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [42])
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content=ReflectionOutput(
                        is_satisfactory=True,
                        reasoning="Result looks good",
                    ),
                ),
                "event_456",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
            signature="(self) -> int",
            docstring="Return the answer.",
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_retries_on_unsatisfactory(self, mock_runtime):
        """execute() should retry when reflection says not satisfactory."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Setup: first call unsatisfactory, second call satisfactory
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [10, 42])
        )
        mock_runtime.generate = AsyncMock(
            side_effect=[
                # First reflection: not satisfactory
                (
                    MagicMock(
                        content=ReflectionOutput(
                            is_satisfactory=False,
                            issues=["Needs improvement"],
                            suggestions=["Try harder"],
                        ),
                    ),
                    "event_1",
                ),
                # Second reflection: satisfactory
                (
                    MagicMock(
                        content=ReflectionOutput(
                            is_satisfactory=True,
                            reasoning="Now it's good",
                        ),
                    ),
                    "event_2",
                ),
            ]
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == 42
        # Should run base strategy twice

    @pytest.mark.asyncio
    async def test_execute_respects_max_reflections(self, mock_runtime):
        """execute() should stop after max_reflections even if unsatisfactory."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Setup: always unsatisfactory, provide enough results for max_reflections
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [10, 10, 10])
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content=ReflectionOutput(
                        is_satisfactory=False,
                        issues=["Still not good"],
                    ),
                ),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=2))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        result = await strategy.execute(mock_runtime, call)
        # Should return best result even if not satisfactory
        assert result == 10
        # Should only run max_reflections times


class TestReflexionStrategyErrorHandling:
    """Tests for ReflexionStrategy error handling."""

    @pytest.mark.asyncio
    async def test_execute_handles_base_strategy_error(self, mock_runtime):
        """execute() should handle errors from base strategy and retry."""
        from nooa.errors import GenerationError
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Setup: first call fails, second succeeds
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(
                mock_runtime, [GenerationError("First attempt failed"), 42]
            )
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content=ReflectionOutput(is_satisfactory=True),
                ),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == 42
        # Should have retried after error

    @pytest.mark.asyncio
    async def test_execute_raises_after_all_attempts_fail(self, mock_runtime):
        """execute() should raise GenerationError if all attempts fail."""
        from nooa.errors import GenerationError
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflexionStrategy

        # Setup: all calls fail (provide enough for max_reflections=2)
        error = GenerationError("Always fails")
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [error, error])
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=2))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        with pytest.raises(GenerationError) as exc_info:
            await strategy.execute(mock_runtime, call)

        assert "2 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_adds_feedback_on_error(self, mock_runtime):
        """execute() should add feedback event when base strategy fails."""
        from nooa.errors import GenerationError
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Setup: first call fails, second succeeds
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(
                mock_runtime, [GenerationError("Something went wrong"), 42]
            )
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content=ReflectionOutput(is_satisfactory=True),
                ),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        await strategy.execute(mock_runtime, call)

        # Should have added feedback about the error
        add_calls = mock_runtime.event_manager.add.call_args_list
        assert len(add_calls) >= 1
        # Find the feedback event about the error
        feedback_contents = [str(c[0][0].content) for c in add_calls if hasattr(c[0][0], "content")]
        assert any("Previous attempt failed" in content for content in feedback_contents)


class TestReflexionStrategyReflectionParsing:
    """Tests for reflection response parsing."""

    @pytest.mark.asyncio
    async def test_execute_handles_dict_response(self, mock_runtime):
        """execute() should handle dict response from generate()."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflexionStrategy

        # Setup: reflection returns dict instead of ReflectionOutput
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [42])
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content={
                        "is_satisfactory": True,
                        "issues": [],
                        "suggestions": [],
                        "reasoning": "Looks good",
                    },
                ),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_handles_unparseable_response(self, mock_runtime):
        """execute() should retry when response can't be parsed (assumes not satisfactory)."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflexionStrategy

        # Setup: reflection returns something unparseable every time.
        # With is_satisfactory=False fallback, the strategy retries up to max_iterations.
        # Provide enough base results for all 3 iterations.
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [42, 43, 44])
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(content="Just some text that isn't a ReflectionOutput"),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        # Should not raise — returns last attempt's result after exhausting iterations
        result = await strategy.execute(mock_runtime, call)
        assert result == 44  # Last iteration's result
        # Should call base strategy 3 times (max_iterations) since unparseable = not satisfactory


class TestReflexionStrategyFeedbackFormatting:
    """Tests for reflection feedback formatting."""

    def test_format_reflection_feedback_with_issues(self):
        """_format_reflection_feedback should include issues."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        strategy = ReflexionStrategy()
        reflection = ReflectionOutput(
            is_satisfactory=False,
            issues=["Missing error handling", "No input validation"],
            suggestions=[],
            reasoning="Code is incomplete",
        )
        call = CurrentCall(
            id="call_123",
            method_name="process_data",
            decorator="plan",
        )

        feedback = strategy._format_reflection_feedback(reflection, call)

        assert "process_data" in feedback
        assert "Missing error handling" in feedback
        assert "No input validation" in feedback
        assert "Code is incomplete" in feedback

    def test_format_reflection_feedback_with_suggestions(self):
        """_format_reflection_feedback should include suggestions."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        strategy = ReflexionStrategy()
        reflection = ReflectionOutput(
            is_satisfactory=False,
            issues=[],
            suggestions=["Add try-except block", "Validate input types"],
            reasoning="",
        )
        call = CurrentCall(
            id="call_123",
            method_name="process_data",
            decorator="plan",
        )

        feedback = strategy._format_reflection_feedback(reflection, call)

        assert "Add try-except block" in feedback
        assert "Validate input types" in feedback


class TestReflexionStrategyResultFormatting:
    """Tests for result formatting for reflection."""

    def test_format_result_simple_value(self):
        """_format_result_for_reflection should handle simple values."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        tc = DEFAULT_TRUNCATION_CONFIG

        assert strategy._format_result_for_reflection(42, tc) == "42"
        # truncating_pformat returns raw string (no quotes) for strings
        assert strategy._format_result_for_reflection("hello", tc) == "hello"
        assert strategy._format_result_for_reflection([1, 2, 3], tc) == "[1, 2, 3]"

    def test_format_result_none(self):
        """_format_result_for_reflection should handle None."""
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        result = strategy._format_result_for_reflection(None, DEFAULT_TRUNCATION_CONFIG)
        assert "None" in result
        assert "no value" in result.lower()

    def test_format_result_passes_strings_through(self):
        """Block-level string truncation has been removed; long string results
        now pass through verbatim from `_format_result_for_reflection`. Per-
        value bounds come from cfg.events.* for non-string values."""
        from nooa.config.truncation_config import TruncationConfig
        from nooa.strategies.reflexion import ReflexionStrategy

        strategy = ReflexionStrategy()
        tc = TruncationConfig()
        long_result = "x" * 3000
        formatted = strategy._format_result_for_reflection(long_result, tc)

        assert long_result in formatted


class TestReflexionStrategyHistoryInteraction:
    """Tests for history event interactions."""

    @pytest.mark.asyncio
    async def test_execute_adds_feedback_between_iterations(self, mock_runtime):
        """execute() should add feedback events between iterations."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        # Track events.add calls
        add_calls = []
        mock_runtime.event_manager.add = MagicMock(side_effect=lambda e, **kw: add_calls.append(e))

        # Setup: two iterations
        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [10, 42])
        )
        mock_runtime.generate = AsyncMock(
            side_effect=[
                # First reflection: not satisfactory
                (
                    MagicMock(
                        content=ReflectionOutput(
                            is_satisfactory=False,
                            issues=["Not good enough"],
                            suggestions=["Do better"],
                        ),
                    ),
                    "event_1",
                ),
                # Second reflection: satisfactory
                (
                    MagicMock(
                        content=ReflectionOutput(is_satisfactory=True),
                    ),
                    "event_2",
                ),
            ]
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="get_answer",
            decorator="plan",
        )

        await strategy.execute(mock_runtime, call)

        # Should have added feedback events (reflection prompt + improvement feedback)
        assert len(add_calls) >= 2

    @pytest.mark.asyncio
    async def test_execute_with_complex_result(self, mock_runtime):
        """execute() should handle complex dict results."""
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.reflexion import ReflectionOutput, ReflexionStrategy

        complex_result = {
            "analysis": {"score": 0.95, "confidence": "high"},
            "recommendations": ["item1", "item2"],
            "metadata": {"timestamp": "2024-01-01"},
        }

        mock_runtime.execute_nested = AsyncMock(
            side_effect=make_smart_execute_nested(mock_runtime, [complex_result])
        )
        mock_runtime.generate = AsyncMock(
            return_value=(
                MagicMock(
                    content=ReflectionOutput(
                        is_satisfactory=True,
                        reasoning="Analysis looks complete",
                    ),
                ),
                "event_123",
            )
        )

        strategy = ReflexionStrategy(config=ReflexionConfig(max_iterations=3))
        call = CurrentCall(
            id="call_123",
            method_name="analyze",
            decorator="plan",
            docstring="Analyze the data.",
        )

        result = await strategy.execute(mock_runtime, call)
        assert result == complex_result


@pytest.fixture
def mock_runtime():
    """Create mock runtime for strategy tests."""

    class MockRuntime:
        def __init__(self):
            self._agent = MagicMock()
            self._agent.agent_id = "test_agent"
            self._agent.__class__.__name__ = "TestAgent"
            self._agent.event_manager = MagicMock()
            self._events = MagicMock()
            self._events.add = MagicMock(return_value="event_123")
            self._events.update = MagicMock(return_value=True)

        @property
        def agent(self):
            return self._agent

        @property
        def event_manager(self):
            """Event manager."""
            return self._events

        @property
        def truncation_config(self):
            """Truncation configuration."""
            from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

            return DEFAULT_TRUNCATION_CONFIG

        async def generate(self, *, tools=None, output_model=None, **kwargs):
            response = MagicMock(content="", reasoning=None, usage={})
            return response, "event_123"

        async def execute_code(self, code, *, builtins=None, validate=True):
            from nooa.events import ExecutionResult

            return ExecutionResult(stdout="", error=None, defined_methods={})

        async def execute_nested(self, strategy, call):
            # Let TemplateStrategy actually execute (needed for @strategy methods)
            from nooa.strategies.template import TemplateStrategy

            if isinstance(strategy, TemplateStrategy):
                return await strategy.execute(self, call)
            # For other strategies, let individual tests override
            return await strategy.execute(self, call)

        async def expand_variables(
            self, text: str, extra_context: dict | None = None, error_mode: str = "show"
        ) -> str:
            """Simple variable expansion for template testing."""
            import re

            # Combine agent state with extra context
            context = extra_context or {}

            def replace_var(match):
                expr = match.group(1)
                # Simple variable lookup
                if expr in context:
                    return str(context[expr])
                return match.group(0)  # Return unchanged if not found

            # Replace {var} patterns
            return re.sub(r"\{([^}:]+)\}", replace_var, text)

        def get_generation_id(self) -> str | None:
            """Get current generation ID."""
            return "mock-generation-id"

        def get_parent_generation_id(self) -> str | None:
            """Get parent generation ID."""
            return None

    return MockRuntime()
