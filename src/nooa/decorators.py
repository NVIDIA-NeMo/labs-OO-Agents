# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent decorators and configuration.

The @strategy decorator is used to override generation strategies on specific methods.
Agent configuration is done via class-level parameters: class MyAgent(Agent, llm=llm)
"""

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from nooa.context_blocks import ScopedContext
from nooa.ellipsis_detection import has_ellipsis_body, has_ellipsis_marker

if TYPE_CHECKING:
    from nooa.config.truncation_config import TruncationConfig
    from nooa.strategies import GenerationStrategy as GenerationStrategyABC
    from nooa.unifiedllm import UnifiedLLM

P = ParamSpec("P")
R = TypeVar("R")


def strategy(
    strategy_instance: "GenerationStrategyABC | None" = None,
    context: "ScopedContext | dict[str, Any] | None" = None,
    *,
    llm: "UnifiedLLM | None" = None,
    truncation: "TruncationConfig | None" = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Strategy decorator for agent methods.

    This decorator serves two purposes:
    1. At class definition time: Attaches metadata for AgentMeta metaclass to process
    2. At runtime (for dynamically-defined methods): Creates runtime wrapper for execution

    Args:
        strategy_instance: Strategy to use (e.g., ReflexionStrategy())
        context: Context block overrides. Accepts either:
            - A plain dict ``{key: str | Context | DynamicContext | None}`` for context-only overrides.
            - A ``ScopedContext`` instance when you also need event filtering.
            Applied in _prepare_context() between strategy overrides and scoped blocks.
        llm: Optional LLM override for this method
        truncation: Optional TruncationConfig override for this method. Fields set
            here take precedence over the agent-level truncation config. Unset fields
            inherit from the agent-level config.

    Examples:
        from nooa import Context
        from nooa.context_blocks import ScopedContext

        # Simple context dict (recommended):
        @strategy(CodeActStrategy(), context={"focus": "security", "self": None})
        async def analyze(self): ...

        # With event filtering (use ScopedContext):
        @strategy(ScopedContext(context={"focus": "security"}, events=EventQuery.current_call()))
        async def solve_with_reflection(self, problem: str): ...

    Returns:
        Decorator function

    Raises:
        ValueError: If multiple @strategy decorators are stacked
        TypeError: If context is not a dict or ScopedContext instance
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Check for duplicate @strategy decorators
        if hasattr(func, "_strategy_override"):
            raise ValueError(f"Cannot stack multiple @strategy decorators on {func.__name__}")

        # Extract context and events from ScopedContext or plain dict
        final_context = None
        final_events = None
        if context is not None:
            if isinstance(context, dict):
                final_context = context
            elif isinstance(context, ScopedContext):
                final_context = context.context
                final_events = context.events
            else:
                raise TypeError(
                    f"@strategy context parameter must be a dict or ScopedContext, got {type(context).__name__}. "
                    f"Use context={{...}} or ScopedContext(context={{...}}, events={{...}})"
                )

        # Attach metadata for metaclass to read (when used at class definition time)
        setattr(func, "_strategy_override", strategy_instance)  # noqa: B010
        setattr(func, "_strategy_llm", llm)  # noqa: B010
        setattr(func, "_strategy_context", final_context)  # noqa: B010
        setattr(func, "_strategy_events", final_events)  # noqa: B010
        setattr(func, "_strategy_truncation", truncation)  # noqa: B010

        # Strategies currently have a single-result contract. Deterministic
        # generators do not need @strategy and are handled by AgentMeta; generated
        # streams need a separate strategy protocol rather than this decorator.
        if inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
            if has_ellipsis_marker(func):
                raise TypeError(
                    f"@strategy method '{func.__name__}' is an LLM-generated generator. "
                    f"Generation methods must produce one final result; streaming "
                    f"generation is not supported."
                )
            raise TypeError(
                f"@strategy cannot be applied to deterministic generator method "
                f"'{func.__name__}'. Remove @strategy; deterministic generators are "
                f"supported directly as agent methods."
            )

        # Also create runtime wrapper for dynamically-defined methods
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@strategy method '{func.__name__}' must be async")

        needs_gen = has_ellipsis_body(func)

        strat = None
        if needs_gen:
            if strategy_instance is not None:
                strat = strategy_instance
            else:
                from nooa.strategies import get_default_strategy

                strat = get_default_strategy()

        # Detect standalone functions (no 'self' as first parameter).
        # Each call creates a fresh agent stub — no state, no history.
        _params = list(inspect.signature(func).parameters)
        if not _params or _params[0] != "self":
            if needs_gen:
                from nooa.standalone import create_standalone_wrapper

                return create_standalone_wrapper(func, strat, llm)
            return func  # type: ignore[return-value]  # non-generation standalone: nothing to wrap

        from nooa.runtime.method_wrapper import create_agent_method_wrapper

        wrapper = create_agent_method_wrapper(
            func,
            needs_generation=needs_gen,
            needs_tracing=not getattr(func, "_no_trace", False),
            strategy=strat,
            cached_source_code=None,
        )

        # Attach additional metadata specific to @strategy decorator
        setattr(wrapper, "_plan_llm", llm)  # noqa: B010
        setattr(wrapper, "_strategy_context", final_context)  # noqa: B010
        setattr(wrapper, "_strategy_events", final_events)  # noqa: B010
        setattr(wrapper, "_strategy_truncation", truncation)  # noqa: B010

        # Also attach to original function (needed for _execute_task)
        setattr(func, "_agent_decorator", "auto")  # noqa: B010
        setattr(func, "_needs_generation", needs_gen)  # noqa: B010
        setattr(func, "_plan_llm", llm)  # noqa: B010
        if strat:
            setattr(func, "_plan_strategy", strat)  # noqa: B010

        return wrapper

    return decorator
