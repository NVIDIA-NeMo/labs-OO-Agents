# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Middleware engine for intercepting LLM calls and code execution.

Register middleware via ``event_manager.intercept()``::

    agent.event_manager.intercept("llm_call", my_guardrail)
    agent.event_manager.intercept("execute_python", my_sandbox)

Four hooks are available:

- ``agent_call``: wraps an instrumented async agent method (all turns, all code).
  Does **not** apply to sync methods — use ``agent_call_sync`` for those.
  Also does not apply to ``@no_trace`` methods (unless generated or decorated
  with ``@strategy``), ``staticmethod`` / ``classmethod``, or methods inherited
  from non-Agent bases — see :class:`AgentCallContext`.
- ``agent_call_sync``: wraps instrumented synchronous (``def``) agent methods.
  Uses a **synchronous** ``(ctx, call_next) -> ctx`` signature so no event loop
  is needed and the sync calling convention is preserved — see
  :class:`AgentCallContext`.
- ``llm_call``: wraps ``runtime.generate()`` (the LLM round-trip)
- ``execute_python``: wraps ``runtime.execute_code()`` (sandbox execution)

``agent_call`` and ``agent_call_sync`` share :class:`AgentCallContext` so a
policy author can reuse the same context shape for both.

``agent_call`` middleware is ``async def(ctx, nxt) -> ctx`` where *nxt* is async.
``agent_call_sync`` middleware is ``def(ctx, nxt) -> ctx`` where *nxt* is sync.
All other middleware is ``async def(ctx, nxt) -> result``.

**intercept() vs on()** — both live on ``EventManager``:

- ``intercept("llm_call", fn)`` wraps a live operation (can transform / block)
- ``on("LLMResponse", fn)`` observes a recorded event (fire-and-forget, after
  the operation completes and the result is recorded)
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nooa.agent import Agent
from nooa.events import ExecutionResult
from nooa.runtime.actor import ActorRuntime
from nooa.unifiedllm import CacheBoundary, LLMResponse, UnifiedLLM

# Sentinel indicating that ``AgentCallContext.result`` has not been set yet.
# Distinguishes "middleware never ran the inner handler" from "method returned None".
_AGENT_RESULT_NOT_SET = object()

__all__ = [
    "MIDDLEWARE_AGENT_CALL",
    "MIDDLEWARE_AGENT_CALL_SYNC",
    "MIDDLEWARE_LLM_CALL",
    "MIDDLEWARE_EXECUTE_PYTHON",
    "AgentCallContext",
    "AgentCallMiddleware",
    "AgentCallNext",
    "SyncAgentCallMiddleware",
    "SyncAgentCallNext",
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
MIDDLEWARE_AGENT_CALL_SYNC = "agent_call_sync"
MIDDLEWARE_LLM_CALL = "llm_call"
MIDDLEWARE_EXECUTE_PYTHON = "execute_python"

# ---------------------------------------------------------------------------
# Context objects (passed to middleware)
# ---------------------------------------------------------------------------


class AgentCallContext(BaseModel):
    """Context for ``agent_call`` and ``agent_call_sync`` middleware.

    Wraps the entire execution of an instrumented agent method.  For async
    methods this covers all LLM turns, all code executions, and the final
    return.  For synchronous (``def``) methods it covers the single call frame
    — use ``agent_call_sync`` to intercept them (same context shape, sync
    calling convention).

    .. warning::
       Coverage is narrower than "every agent method".

       ``agent_call`` (async): runs only in the wrapper the metaclass builds
       for traced async methods.  ``agent_call_sync`` (sync): runs in the
       wrapper for traced synchronous (``def``) methods.  A method executes
       outside *both* chains when it is any of:

       - marked ``@no_trace`` and left unwrapped by the metaclass — a
         ``@no_trace`` method that is generated or carries ``@strategy`` keeps
         its async wrapper, and the middleware chain with it;
       - a ``staticmethod`` or ``classmethod`` — skipped as a non-plain function;
       - inherited from a base that is not itself an ``Agent``.

       This holds however the method is reached, including from generated CodeAct
       Python.  ``@no_trace``, ``staticmethod`` / ``classmethod``, and
       non-Agent-inherited methods still emit no middleware events regardless of
       which hook is registered.

       When ``agent_call`` middleware is registered but a sync method has no
       corresponding ``agent_call_sync`` guard, a ``RuntimeWarning`` is emitted
       to surface the gap.

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
        messages: Public dictionaries and read-only responses. Replace a response
                  with a dictionary to edit it and discard its native state.
        params: Extra keyword arguments forwarded to ``acall()``
                (tools, output_model, etc.).  Middleware may add / remove keys.
        agent: The agent instance that owns the runtime.
        runtime: The ``ActorRuntime`` instance.
        client: Effective client for this call, including method-level overrides.
                Read-only: route overrides belong in params, not a replacement client.
        filtered_history: An event query restricted the rendered history. Consumers
                          must not treat this request as the complete event archive.
        response: ``None`` on the way *in*; set to the ``LLMResponse`` by the
                  innermost handler on the way *out*.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, Any] | LLMResponse | CacheBoundary]
    params: dict[str, Any] = {}
    agent: Agent | None = None
    runtime: ActorRuntime | None = None
    client: UnifiedLLM | None = Field(default=None, frozen=True)
    filtered_history: bool = False
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

# Synchronous variants for agent_call_sync middleware.
SyncAgentCallNext = Callable[[AgentCallContext], AgentCallContext]
SyncAgentCallMiddleware = Callable[[AgentCallContext, SyncAgentCallNext], AgentCallContext]
