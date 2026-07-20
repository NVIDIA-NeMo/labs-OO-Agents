# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for @strategy decorator with strategy instances.

TDD: Write tests first, then implement to make them pass.
"""

from nooa import strategy
from nooa.strategies.pure_python import PurePythonStrategy


class TestStrategyDecoratorWithInstances:
    """Tests for @strategy decorator accepting strategy instances."""

    def test_strategy_accepts_strategy_instance(self):
        """@strategy should accept a strategy instance."""
        strat = PurePythonStrategy(max_iterations=5)

        @strategy(strat)
        async def my_method(self) -> str:
            """Do something."""
            ...

        # @strategy decorator sets _strategy_override, metaclass reads it
        assert my_method._strategy_override == strat

    def test_strategy_instance_used_directly(self):
        """Strategy instance should be stored directly (not converted to enum)."""
        strat = PurePythonStrategy(max_retries=2)

        @strategy(strat)
        async def task(self) -> int: ...

        # Should be the exact same instance
        assert task._strategy_override is strat

    def test_no_decorator_defaults_to_none(self):
        """Methods without @strategy decorator have no _strategy_override (resolved at runtime)."""

        async def default_method(self) -> str: ...

        # Without @strategy decorator, no _strategy_override attribute
        assert not hasattr(default_method, "_strategy_override")

    def test_strategy_config_preserved(self):
        """Strategy's configuration should be preserved."""
        strat = PurePythonStrategy(max_iterations=7, max_retries=4)

        @strategy(strat)
        async def configured(self) -> dict: ...

        stored = configured._strategy_override
        assert stored.max_iterations == 7
        assert stored.max_retries == 4


class TestStrategyDecoratorValidation:
    """Tests for @strategy decorator validation."""

    def test_strategy_on_implemented_method_is_allowed(self):
        """@strategy on implemented method is allowed (acts as entry point marker)."""

        # This should NOT raise - @strategy on implemented methods is valid
        @strategy(PurePythonStrategy())
        async def has_body(self) -> str:
            return "implemented"

        # Should have the decorator metadata
        assert has_body._strategy_override is not None
