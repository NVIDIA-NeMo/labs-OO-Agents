# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation strategies for nooa.

One class = one strategy. Each strategy owns its configuration.

Strategy implementations are loaded lazily so importing the package does not
load concrete LLM providers before generation needs them.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from nooa.strategies.base import GenerationStrategy, RuntimeServices

# NOTE: CodeActLiteStrategy and ReflexionStrategy are experimental. The
# FutureWarning gate lives on the top-level package (nooa.__getattr__),
# so importing them from here - `from nooa.strategies import
# CodeActLiteStrategy` - is an intentional un-gated (warning-free) escape hatch.

# =============================================================================
# Default Strategy Override
# =============================================================================
# Context variable for overriding the default strategy globally.
# When None (default), get_default_strategy() returns a fresh CodeActStrategy().
# Use set_default_strategy() to override for all agents in the current context.

_default_strategy_var: ContextVar[GenerationStrategy | None] = ContextVar(
    "default_strategy", default=None
)

_STRATEGY_EXPORTS = {
    "CurrentCall": "nooa.strategies.current_call",
    "CompositeStrategy": "nooa.strategies.composite",
    "TemplateStrategy": "nooa.strategies.template",
    "CodeActStrategy": "nooa.strategies.codeact",
    "CodeActLiteStrategy": "nooa.strategies.codeact_lite",
    "ReflexionStrategy": "nooa.strategies.reflexion",
    "PredictStrategy": "nooa.strategies.predict",
    "Prefill": "nooa.strategies.prefill",
    "InspectInputsPrefill": "nooa.strategies.prefill",
}


def get_default_strategy() -> GenerationStrategy:
    """Get the default strategy for agents without an explicit strategy.

    Returns the strategy set via set_default_strategy(), or creates a fresh
    CodeActStrategy() instance if not set.

    Returns:
        GenerationStrategy instance to use as default

    Example:
        # In actor.py / decorators.py:
        strategy = call_strategy or decorator_strategy or get_default_strategy()
    """
    strategy = _default_strategy_var.get()
    if strategy is None:
        from nooa.config import CodeActConfig
        from nooa.strategies.codeact import CodeActStrategy

        return CodeActStrategy(config=CodeActConfig())
    return strategy


def set_default_strategy(strategy: GenerationStrategy | None) -> None:
    """Set the default strategy for all agents in the current async context.

    This allows overriding the default strategy (CodeActStrategy) without
    modifying agent classes. Useful for:
    - Evaluation pipelines that want to test different strategies
    - Testing with a specific strategy across all agents
    - Temporarily switching strategies for a block of code

    Args:
        strategy: GenerationStrategy instance to use as default, or None to
                  reset to CodeActStrategy (the library default returned by
                  get_default_strategy())

    Example:
        from nooa import set_default_strategy, CodeActStrategy
        from nooa.config import CodeActConfig

        # Override default for all agents
        set_default_strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))

        # Run evaluation - all agents use CodeActStrategy
        results = await evaluator.run()

        # Reset to library default
        set_default_strategy(None)
    """
    _default_strategy_var.set(strategy)


def __getattr__(name: str) -> Any:
    module_name = _STRATEGY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "GenerationStrategy",
    "RuntimeServices",
    "CurrentCall",
    "CompositeStrategy",
    "TemplateStrategy",
    "CodeActStrategy",
    "CodeActLiteStrategy",
    "ReflexionStrategy",
    "PredictStrategy",
    # Prefill plugins
    "Prefill",
    "InspectInputsPrefill",
    # Default strategy functions
    "get_default_strategy",
    "set_default_strategy",
]
