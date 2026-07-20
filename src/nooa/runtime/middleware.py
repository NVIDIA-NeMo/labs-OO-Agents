# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Middleware engine for intercepting LLM calls and code execution.

Register middleware via ``event_manager.intercept()``::

    agent.event_manager.intercept("llm_call", my_guardrail)
    agent.event_manager.intercept("execute_python", my_sandbox)

Three hooks are available:

- ``agent_call``: wraps the entire agent method (all turns, all code)
- ``llm_call``: wraps ``runtime.generate()`` (the LLM round-trip)
- ``execute_python``: wraps ``runtime.execute_code()`` (sandbox execution)

Each middleware is ``async def(ctx, nxt) -> result`` where *ctx* is a typed
context object and *nxt* calls the rest of the chain.

**intercept() vs on()** — both live on ``EventManager``:

- ``intercept("llm_call", fn)`` wraps a live operation (can transform / block)
- ``on("LLMOutput", fn)`` observes a recorded event (fire-and-forget, after
  the operation completes and the result is recorded)
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from nooa.agent import Agent
from nooa.events import ExecutionResult
from nooa.runtime.actor import ActorRuntime
from nooa.unifiedllm import LLMResponse

# Sentinel indicating that ``AgentCallContext.result`` has not been set yet.
# Distinguishes "middleware never ran the inner handler" from "method returned None".
_AGENT_RESULT_NOT_SET = object()

__all__ = [
    "MIDDLEWARE_AGENT_CALL",
    "MIDDLEWARE_LLM_CALL",
    "MIDDLEWARE_EXECUTE_PYTHON",
    "AgentCallContext",
    "AgentCallMiddleware",
    "AgentCallNext",
    "LLMCallContext",
    "LLMCallMiddleware",
    "LLMCallNext",
    "ExecutePythonContext",
    "ExecutePythonMiddleware",
    "ExecutePythonNext",
]

# ---------------------------------------------------------------------------
# Middleware kind constants
# ---------------------------------------------------------------------------

MIDDLEWARE_AGENT_CALL = "agent_call"
MIDDLEWARE_LLM_CALL = "llm_call"
MIDDLEWARE_EXECUTE_PYTHON = "execute_python"

# ---------------------------------------------------------------------------
# Context objects (passed to middleware)
# ---------------------------------------------------------------------------


class AgentCallContext(BaseModel):
    """Context for ``agent_call`` middleware.

    Wraps the entire agent method execution — all LLM turns, all code
    executions, the final return.

    Attributes:
        agent: The agent instance.
        method_name: Name of the method being called.
        args: Positional arguments (excluding ``self``).
        kwargs: Keyword arguments.
        result: ``None`` on the way *in*; set to the method's return value
                by the innermost handler on the way *out*.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Agent | None = None
    method_name: str = ""
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}
    result: Any = _AGENT_RESULT_NOT_SET


class LLMCallContext(BaseModel):
    """Context for ``llm_call`` middleware.

    Attributes:
        messages: The prompt messages list (mutable — middleware may edit).
        params: Extra keyword arguments forwarded to ``acall()``
                (tools, output_model, etc.).  Middleware may add / remove keys.
        agent: The agent instance that owns the runtime.
        runtime: The ``ActorRuntime`` instance.
        response: ``None`` on the way *in*; set to the ``LLMResponse`` by the
                  innermost handler on the way *out*.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, Any]]
    params: dict[str, Any] = {}
    agent: Agent | None = None
    runtime: ActorRuntime | None = None
    response: LLMResponse | None = None


class ExecutePythonContext(BaseModel):
    """Context for ``execute_python`` middleware.

    Attributes:
        code: The Python source about to be executed (mutable).
        params: Keyword arguments forwarded to the underlying execution
                (builtins, validate, timeout, etc.).
        agent: The agent instance that owns the runtime.
        runtime: The ``ActorRuntime`` instance.
        result: ``None`` on the way *in*; set to ``ExecutionResult`` by the
                innermost handler on the way *out*.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    params: dict[str, Any] = {}
    agent: Agent | None = None
    runtime: ActorRuntime | None = None
    result: ExecutionResult | None = None


# ---------------------------------------------------------------------------
# Typed signatures
# ---------------------------------------------------------------------------

AgentCallNext = Callable[[AgentCallContext], Awaitable[AgentCallContext]]
LLMCallNext = Callable[[LLMCallContext], Awaitable[LLMCallContext]]
ExecutePythonNext = Callable[[ExecutePythonContext], Awaitable[ExecutePythonContext]]

AgentCallMiddleware = Callable[[AgentCallContext, AgentCallNext], Awaitable[AgentCallContext]]
LLMCallMiddleware = Callable[[LLMCallContext, LLMCallNext], Awaitable[LLMCallContext]]
ExecutePythonMiddleware = Callable[
    [ExecutePythonContext, ExecutePythonNext], Awaitable[ExecutePythonContext]
]
