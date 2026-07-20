# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy-level method conditions.

Preconditions and postconditions are small invariant hooks that run around
generated strategy methods. They are intended for checks that Pydantic
return-type validation cannot express, such as live agent state or per-call
process invariants.
"""

from collections.abc import Callable, Iterable
from typing import Any

MethodPrecondition = Callable[..., Any]
MethodPostcondition = Callable[..., Any]


class InvariantError(ValueError):
    """A model-correctable method invariant failure.

    Strategy runtimes catch this exception from postconditions and route it
    through their existing validation-retry feedback. Preconditions run before
    generation and fail fast. Other condition exceptions propagate as normal
    infrastructure/programming errors.
    """


def normalize_conditions(
    name: str, conditions: Iterable[Any] | None
) -> tuple[Callable[..., Any], ...]:
    """Normalize config conditions into a tuple of callables."""

    if conditions is None:
        return ()
    if isinstance(conditions, str | bytes) or not isinstance(conditions, Iterable):
        raise ValueError(f"{name} must be an iterable of callables")

    normalized: list[Callable[..., Any]] = []
    for condition in conditions:
        if not callable(condition):
            raise ValueError(f"{name} must be callables, got {type(condition).__name__}")
        normalized.append(condition)
    return tuple(normalized)


def _run_condition(kind: str, func: Callable[..., Any], *args: Any) -> None:
    try:
        func(*args)
    except InvariantError:
        raise
    except Exception as e:
        raise RuntimeError(f"method {kind} {func.__name__!r} failed") from e


def run_preconditions(agent: Any, call: Any, preconditions: Iterable[Any]) -> None:
    """Run pre-call conditions for ``call``."""

    for precondition in normalize_conditions("preconditions", preconditions):
        _run_condition("precondition", precondition, agent, call)


def run_postconditions(agent: Any, result: Any, call: Any, postconditions: Iterable[Any]) -> None:
    """Run post-result conditions for ``result``."""

    for postcondition in normalize_conditions("postconditions", postconditions):
        _run_condition("postcondition", postcondition, agent, result, call)
