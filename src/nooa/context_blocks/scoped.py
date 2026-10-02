# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary context-block overrides, inherited within an agent's calls."""

import contextvars
from typing import Any

from nooa.context_blocks.models import DynamicContext

# The default view reads the snapshot supplied through CurrentCall.
_scoped_blocks_var: contextvars.ContextVar[dict[str, str | DynamicContext | None] | None] = (
    contextvars.ContextVar("scoped_blocks", default=None)
)


class ScopedContext:
    """Override blocks inside a scope or through ``@strategy(context=...)``.

    Nested scopes inherit parent blocks; inner values replace matching keys.
    Use a context view to select events.

    Example:
        with ScopedContext(context={"focus": "Analyze security only"}):
            result = await agent.analyze(data)
    """

    def __init__(self, context: dict[str, str | DynamicContext | None] | None = None):
        self.context = context
        self._ctx_token: contextvars.Token[Any] | None = None

    def __enter__(self) -> "ScopedContext":
        parent = _scoped_blocks_var.get()
        self._ctx_token = _scoped_blocks_var.set({**(parent or {}), **(self.context or {})})
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._ctx_token is not None:
            _scoped_blocks_var.reset(self._ctx_token)
