# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one rule that picks the ``TruncationConfig`` in force: :func:`effective_truncation`."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.config.truncation_config import TruncationConfig

__all__ = ["effective_truncation"]

# The resolved config of the generated-method call running now: the agent's
# config merged with the method's @strategy(truncation=...). The runtime sets
# it around each generation call (nooa.runtime.actor._execute_with_generation)
# and resets it after; None outside one.
_current_truncation_config_var: contextvars.ContextVar[TruncationConfig | None] = (
    contextvars.ContextVar("current_truncation_config", default=None)
)


def effective_truncation(agent: Any = None) -> TruncationConfig:
    """The ``TruncationConfig`` in force here: read every configurable limit through it.

    First match wins:

    1. The method level: inside a generated method's call (and the code,
       tools and nested calls it runs), the agent's config merged with that
       method's ``@strategy(truncation=...)``.
    2. The agent level: ``agent._truncation`` of ``agent``, or, when
       ``agent`` is ``None``, of the agent whose method is running
       (``MyAgent(truncation=...)`` merged over ``class MyAgent(Agent,
       truncation=...)`` merged over the defaults; ``Agent._resolve_truncation``).
    3. ``DEFAULT_TRUNCATION_CONFIG`` (outside any agent).

    Read it when the limit is used, not when an object is built, so an
    override set later (a method call) applies. The lookup uses context
    variables: asyncio tasks inherit them, plain threads do not. Code that
    starts a thread runs its target in ``contextvars.copy_context().run(...)``
    to keep the limits of the caller; ``asyncio.to_thread`` already does.
    """
    override = _current_truncation_config_var.get()
    if override is not None:
        return override
    if agent is None:
        from nooa.runtime.context_vars import _parent_agent_var
        from nooa.util._context import _current_agent_var

        # The agent whose method is running: any agent method call sets
        # _parent_agent_var (nooa.runtime.method_wrapper); the implemented-method
        # path of the runtime also sets _current_agent_var.
        agent = _current_agent_var.get() or _parent_agent_var.get()
    config = getattr(agent, "_truncation", None) if agent is not None else None
    if config is not None:
        return config
    from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

    return DEFAULT_TRUNCATION_CONFIG
