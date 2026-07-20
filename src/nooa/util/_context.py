# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal context management for utility modules."""

from contextvars import ContextVar
from typing import Any

# ContextVar to store current agent during execution
_current_agent_var: ContextVar[Any] = ContextVar("current_agent", default=None)

# ContextVar to store current runtime during execution (for child agent inheritance)
_current_runtime_var: ContextVar[Any] = ContextVar("current_runtime", default=None)


def _current_agent() -> Any:
    """
    Get the current agent instance.

    Reads ``_current_agent_var``, which the runtime sets only around the
    implemented-method execution path (``runtime/actor.py``); it is not
    set for LLM-generated code. Callers therefore only get an agent when
    invoked from an implemented agent method.

    Returns:
        Current agent instance

    Raises:
        RuntimeError: If called outside of agent execution context
    """
    agent = _current_agent_var.get()
    if agent is None:
        raise RuntimeError(
            "No agent in context. This utility is only available from implemented agent methods."
        )
    return agent
