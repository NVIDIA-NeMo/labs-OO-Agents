# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject dynamic method addition on Agent classes and instances.

The agent class is fixed; methods are defined in the class body. Helpers
written by generated code live in REPL scope, not on the agent.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from nooa.errors import DynamicMethodAdditionError

_REJECT_HELP = (
    "Define helpers as plain functions at the top of the cell:\n"
    "    def helper(x): ...\n"
    "    helper(value)\n"
    "For per-item LLM work over a list, decorate a standalone async function:\n"
    "    @strategy(PredictStrategy())\n"
    "    async def classify(text: str) -> str: ...\n"
    "    results = await asyncio.gather(*(classify(t) for t in items))"
)


def _is_method_like(value: Any) -> bool:
    """Return True if ``value`` is a function/method/static-or-classmethod or partial wrapping one."""
    if inspect.isfunction(value) or inspect.ismethod(value):
        return True
    if isinstance(value, (staticmethod, classmethod)):
        return True
    if isinstance(value, functools.partial):
        return _is_method_like(value.func)
    return False


def guard_dynamic_method(target: Any, name: str, value: Any) -> None:
    """Raise ``DynamicMethodAdditionError`` for method-like writes to public attributes.

    Private names (starting with ``_``) are exempt — they hold framework
    internals like unsubscribe handlers and cached bound methods.

    Class construction uses ``type.__setattr__`` directly to bypass this guard.
    """
    if name.startswith("_"):
        return
    if not _is_method_like(value):
        return

    kind = "class" if isinstance(target, type) else "instance"
    target_name = target.__name__ if isinstance(target, type) else type(target).__name__
    raise DynamicMethodAdditionError(
        f"Cannot attach callable to agent {kind} attribute '{target_name}.{name}'.\n" + _REJECT_HELP
    )
