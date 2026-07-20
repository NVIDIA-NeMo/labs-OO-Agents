# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GenerationStrategy ABC.

TDD: Write these tests first, then implement base.py to make them pass.
"""

import pytest


class TestGenerationStrategyABC:
    """Tests for GenerationStrategy abstract base class."""

    def test_is_abstract(self):
        """GenerationStrategy should be abstract - cannot instantiate directly."""
        from nooa.strategies.base import GenerationStrategy

        with pytest.raises(TypeError):
            GenerationStrategy()

    def test_name_has_default_implementation(self):
        """GenerationStrategy.name has a default (class name)."""
        from nooa.strategies.base import GenerationStrategy

        class TestStrategy(GenerationStrategy):
            async def execute(self, runtime, call):
                pass

        strategy = TestStrategy()
        assert strategy.name == "TestStrategy"

    def test_get_block_overrides_has_default_implementation(self):
        """GenerationStrategy.get_block_overrides has a default (empty dict)."""
        from nooa.strategies.base import GenerationStrategy

        class TestStrategy(GenerationStrategy):
            async def execute(self, runtime, call):
                pass

        strategy = TestStrategy()
        assert strategy.get_block_overrides() == {}

    def test_requires_execute_method(self):
        """GenerationStrategy subclasses must implement execute()."""
        from nooa.strategies.base import GenerationStrategy

        class IncompleteStrategy(GenerationStrategy):
            @property
            def name(self) -> str:
                return "test"

            async def strategy_prompt(self, runtime):
                return "test"

        with pytest.raises(TypeError):
            IncompleteStrategy()

    @pytest.mark.asyncio
    async def test_complete_implementation_can_instantiate(self, mock_runtime):
        """Complete GenerationStrategy implementation should instantiate."""
        from nooa.strategies.base import GenerationStrategy

        class TestStrategy(GenerationStrategy):
            @property
            def name(self) -> str:
                return "TEST"

            async def execute(self, runtime, call):
                return "result"

        strategy = TestStrategy()
        assert strategy.name == "TEST"


class TestGenerationStrategyTraceable:
    """Tests for GenerationStrategy.traceable property."""

    def test_traceable_default_true(self):
        """GenerationStrategy.traceable should default to True."""
        from nooa.strategies.base import GenerationStrategy

        class TestStrategy(GenerationStrategy):
            async def execute(self, runtime, call):
                pass

        strategy = TestStrategy()
        assert strategy.traceable is True


class TestGenerationStrategyConfig:
    """Tests for strategy configuration via __init__."""

    def test_strategy_accepts_max_iterations(self):
        """Strategy should accept max_iterations config."""
        from nooa.strategies.base import GenerationStrategy

        class ConfigurableStrategy(GenerationStrategy):
            def __init__(self, max_iterations: int = 10, max_retries: int = 3):
                self.max_iterations = max_iterations
                self.max_retries = max_retries

            @property
            def name(self) -> str:
                return "CONFIGURABLE"

            @property
            def strategy_prompt(self) -> str:
                return "Configurable strategy"

            async def execute(self, runtime, call):
                return "result"

        strategy = ConfigurableStrategy(max_iterations=5, max_retries=2)
        assert strategy.max_iterations == 5
        assert strategy.max_retries == 2

    def test_strategy_defaults_when_no_config(self):
        """Strategy should use defaults when no config provided."""
        from nooa.strategies.base import GenerationStrategy

        class ConfigurableStrategy(GenerationStrategy):
            def __init__(self, max_iterations: int = 10, max_retries: int = 3):
                self.max_iterations = max_iterations
                self.max_retries = max_retries

            @property
            def name(self) -> str:
                return "CONFIGURABLE"

            @property
            def strategy_prompt(self) -> str:
                return "Configurable strategy"

            async def execute(self, runtime, call):
                return "result"

        strategy = ConfigurableStrategy()
        assert strategy.max_iterations == 10
        assert strategy.max_retries == 3
