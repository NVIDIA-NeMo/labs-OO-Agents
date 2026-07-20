# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context variables for agent runtime.

This module exists to break circular imports between agent.py and runtime modules.
"""

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nooa.agent import Agent

# Context variable to detect when we're inside a generation session.
# Defined here (not in actor.py) to avoid circular imports with method_wrapper.py.
# method_wrapper.py needs this at module level, but actor.py has bidirectional
# deps with agent.py/metaclass.py that would cause circular imports.
_in_generation_session: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "in_generation_session", default=False
)

# Context variable tracking which EventManager._middleware_id values are currently
# inside an execute_python middleware chain.  Prevents infinite recursion when
# middleware-injected code triggers another execute_code() on the same agent.
# Defined here alongside _in_generation_session for the same circular-import reason.
_in_exec_middleware: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "in_exec_middleware", default=frozenset()
)

# Context variable for LLM inheritance from parent agent
# When a subagent is instantiated within generated code, it can inherit the parent's LLM
# if the subagent class was defined without an LLM (class MyAgent(Agent): ...)
_parent_agent_var: contextvars.ContextVar["Agent | None"] = contextvars.ContextVar(
    "parent_agent", default=None
)

# Task-local stack for agent call tracking.
# Uses immutable tuples to prevent accidental mutation and ensure
# correct behavior with asyncio.gather() and other parallel execution patterns.
# Each push/pop creates a new tuple, so parallel tasks have isolated stacks.
_agent_call_stack_var: contextvars.ContextVar[tuple[str | None, ...]] = contextvars.ContextVar(
    "agent_call_stack", default=()
)

# Task-local event render format for events produced by the active generation
# method. EventManager persists this into event.metadata so historical events
# render the same way after session resume and do not depend on call_id maps.
_current_event_format_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "current_event_format", default=None
)


def _get_agent_call_stack() -> tuple[str | None, ...]:
    """Get the agent call stack for the current async context.

    Returns an immutable tuple - use _push_agent_call_id() and _pop_agent_call_id()
    for stack operations.
    """
    return _agent_call_stack_var.get()


def _push_agent_call_id(call_id: str | None) -> None:
    """Push a call_id to the agent call stack.

    Creates a new immutable tuple with the added value. This ensures parallel
    tasks (via asyncio.gather) each have their own isolated stack - pushing
    in one task doesn't affect sibling tasks.

    ``call_id`` may be ``None`` when a ``@no_trace`` method at the top of the
    call graph (no traced ancestor) propagates its own ``parent_call_id``.
    """
    current = _agent_call_stack_var.get()
    _agent_call_stack_var.set((*current, call_id))


def _pop_agent_call_id() -> str | None:
    """Pop from the agent call stack.

    Creates a new immutable tuple without the last element.
    Returns the popped value or None if stack is empty.
    """
    current = _agent_call_stack_var.get()
    if not current:
        return None
    popped = current[-1]
    _agent_call_stack_var.set(current[:-1])
    return popped
