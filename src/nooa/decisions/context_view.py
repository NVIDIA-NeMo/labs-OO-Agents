# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default supporting context for decision methods."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from nooa.context_blocks import Context, DynamicContext
from nooa.context_view import Block, ContextItem, evaluate_context_expression

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.strategies.current_call import CurrentCall


class DecisionContextView:
    """Include explicit decorator/scoped context and no implicit history."""

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        selected = {**(call.decorator_context or {}), **(call.scoped_context or {})}
        disabled = owner.context_manager.disabled()
        for key, value in selected.items():
            if key in disabled or value is None:
                continue
            if isinstance(value, Context):
                value = value.to_dynamic_context() if value.is_dynamic else value.value
            if isinstance(value, DynamicContext):
                expression = value.expr
                try:
                    value = await evaluate_context_expression(expression, owner=owner, call=call)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to resolve decision context block {key!r} "
                        f"from expression {expression!r}: {type(exc).__name__}: {exc}"
                    ) from exc
            yield Block(key=key, content=value)


__all__ = ["DecisionContextView"]
