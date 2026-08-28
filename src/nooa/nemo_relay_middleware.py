# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Relay integration via the ``intercept()`` middleware API.

When ``nemo_relay`` is installed, this module provides three middleware functions
that route LLM calls, code execution, and agent method calls through the NeMo Relay
pipeline (guardrails, intercepts, event subscribers, ATIF export).

Usage::

    from nooa.nemo_relay_middleware import install_nemo_relay

    # Inside an async context where nemo_relay scope is active:
    uninstall = install_nemo_relay(agent.event_manager)
    try:
        result = await agent.my_method()
    finally:
        uninstall()

Or use ``nemo_relay_scope()`` which handles install/uninstall automatically::

    from nooa.nemo_relay_middleware import nemo_relay_scope

    async with nemo_relay_scope(agent, "my-agent"):
        result = await agent.my_method()

Requirements:
    ``nemo_relay`` must be installed (``uv sync --extra nemo-relay``).
    If not installed, ``install_nemo_relay()`` and ``nemo_relay_scope()`` raise
    ``ImportError`` with install instructions.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nooa.runtime.middleware import (
    MIDDLEWARE_AGENT_CALL,
    MIDDLEWARE_EXECUTE_PYTHON,
    MIDDLEWARE_LLM_CALL,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nooa.runtime.event_manager import EventManager
    from nooa.runtime.middleware import (
        AgentCallContext,
        AgentCallNext,
        ExecutePythonContext,
        ExecutePythonNext,
        LLMCallContext,
        LLMCallNext,
    )

try:
    import nemo_relay  # type: ignore[import]
    from nemo_relay import LLMRequest  # type: ignore[import]

    _HAS_NEMO_RELAY = True
except ImportError:
    _HAS_NEMO_RELAY = False
    nemo_relay = None  # type: ignore[assignment]
    LLMRequest = None  # type: ignore[assignment,misc]

# The ContextVar holding NeMo Relay's mutable ScopeStack. It is private
# upstream (not in ``nemo_relay.__all__``), so resolve it defensively: if a
# future release renames it we degrade to the old shared-stack behaviour —
# correct for sequential nesting, broken only for overlapping calls, which is
# exactly where we are today. ``tests/test_nemo_relay_middleware.py`` asserts
# the attribute still exists so an upstream rename fails in CI rather than
# silently reverting. See nemo_relay_agent_call_middleware for why it matters.
_SCOPE_STACK_VAR = getattr(nemo_relay, "_scope_stack_var", None) if _HAS_NEMO_RELAY else None

# Current NeMo Relay scope as ``(handle, owning asyncio task)``. Unlike the
# upstream ScopeStack this holds an *immutable* handle, so a child task that
# sets it cannot disturb its parent or its siblings.
_current_relay_scope: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "_nooa_relay_scope", default=None
)

_INSTALL_MSG = (
    "nemo_relay is required for NeMo Relay integration. Install it with "
    "`uv sync --extra nemo-relay` (or `uv add nemo-relay`)."
)

# Keys stripped from params before exposing to NeMo Relay guardrails/events.
_SENSITIVE_KEYS: frozenset[str] = frozenset({"api_key", "api_base", "base_url"})

# Keys that hold non-JSON-serializable objects (Tool instances, Pydantic models).
# These are converted or removed before constructing LLMRequest.
_NON_SERIALIZABLE_KEYS: frozenset[str] = frozenset({"tools", "output_model"})

# Keys from LLMRequest.content that map directly to ctx.params.
# If a NeMo Relay request intercept modifies these, we propagate them.
# NOTE: "tools" is intentionally excluded — they are non-serializable Tool
# instances and must not be overwritten with JSON from the Rust boundary.
_PROPAGATABLE_LLM_PARAMS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    }
)

# Tool execution args that NeMo Relay intercepts may modify.
_PROPAGATABLE_TOOL_PARAMS: frozenset[str] = frozenset({"tool_call_id", "timeout"})


def _extract_model_name(agent: Any) -> str:
    """Return the model name from an agent's LLM client, if available."""
    if agent is None:
        return ""
    llm = getattr(agent, "_llm", None)
    if llm is None:
        return ""
    return getattr(llm, "model", "")


# ---------------------------------------------------------------------------
# Middleware functions
# ---------------------------------------------------------------------------


async def nemo_relay_llm_middleware(
    ctx: LLMCallContext,
    nxt: LLMCallNext,
) -> LLMCallContext:
    """Route LLM calls through the NeMo Relay LLM pipeline.

    Strips sensitive keys, wraps the call through ``nemo_relay.llm.execute()``,
    and returns the original ``LLMResponse`` to the caller.  The JSON-serialized
    response is what flows through NeMo Relay guardrails and ATIF export.
    """
    assert nemo_relay is not None

    # Get model name from the agent's LLM client (not in params).
    model_name = _extract_model_name(ctx.agent)
    safe_params = {
        k: v
        for k, v in ctx.params.items()
        if k not in _SENSITIVE_KEYS and k not in _NON_SERIALIZABLE_KEYS
    }
    safe_params["messages"] = ctx.messages
    # Tools are excluded via _NON_SERIALIZABLE_KEYS.  Do NOT re-add them:
    # including a "tools" key in request.content triggers an AttributeError
    # ('dict' object has no attribute 'name') inside NeMo Relay's native pipeline.
    # The old hooks-based integration avoided this because it intercepted at
    # unifiedllm's layer, where api_params never contained a tools key.
    request = LLMRequest({}, safe_params)  # type: ignore[misc]

    captured_ctx: LLMCallContext | None = None

    async def _wrapper(req: Any) -> Any:
        nonlocal captured_ctx
        # Apply request intercept modifications back to ctx.
        # NeMo Relay request intercepts can transform the LLMRequest (e.g. inject
        # system messages, modify headers).  The modified request is passed
        # here as `req`; we must propagate those changes to ctx so the rest
        # of the nooa middleware chain (and the actual LLM call) sees them.
        if hasattr(req, "content") and isinstance(req.content, dict):
            intercepted = req.content
            intercepted_msgs = intercepted.get("messages")
            if intercepted_msgs is not None:
                ctx.messages = intercepted_msgs
            # Propagate any supported param changes from the intercept.
            for key in _PROPAGATABLE_LLM_PARAMS:
                if key in intercepted:
                    ctx.params[key] = intercepted[key]
        # Call the rest of the middleware chain → eventually hits acall()
        captured_ctx = await nxt(ctx)
        # Return JSON to NeMo Relay (guardrails, events, ATIF).
        resp = captured_ctx.response
        if resp is None:
            return {}
        # UnifiedLLM owns the serialization boundary; provider SDK response
        # objects never cross into Relay or other framework consumers.
        to_wire = getattr(resp, "to_wire", None)
        if callable(to_wire):
            return to_wire()
        if hasattr(resp, "model_dump"):
            return resp.model_dump(mode="json")  # type: ignore[union-attr]
        if hasattr(resp, "assistant_message"):
            result: dict[str, Any] = {"message": resp.assistant_message}
            if getattr(resp, "usage", None):
                result["usage"] = resp.usage
            if getattr(resp, "finish_reason", None):
                result["finish_reason"] = resp.finish_reason
            return result
        return {}

    # Note: nemo_relay.llm.execute() returns the pre-guardrail response.
    # Sanitize-response guardrails transform data for NeMo Relay internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original response.  Conditional-execution guardrails that
    # reject raise GuardrailRejected, which propagates naturally.
    await nemo_relay.llm.execute(model_name, request, _wrapper, model_name=model_name)  # type: ignore[union-attr]

    if captured_ctx is not None:
        return captured_ctx
    # NeMo Relay guardrails blocked the LLM call (never invoked _wrapper).
    # Raise an explicit error instead of returning ctx without a response,
    # which would trigger a confusing generic RuntimeError downstream.
    raise RuntimeError(
        "NeMo Relay guardrail blocked the LLM call — the request was rejected "
        "before reaching the LLM. Check your NeMo Relay guardrail configuration."
    )


async def nemo_relay_tool_middleware(
    ctx: ExecutePythonContext,
    nxt: ExecutePythonNext,
) -> ExecutePythonContext:
    """Route code execution through the NeMo Relay tool pipeline.

    Extracts the meaningful return value from ``ExecutionResult``, serializes
    it via ``BestEffortAnyCodec`` for NeMo Relay inspection, and returns the
    original ``ExecutionResult`` to the caller.
    """
    assert nemo_relay is not None

    args = {
        "code": ctx.code,
        **{k: v for k, v in ctx.params.items() if k in _PROPAGATABLE_TOOL_PARAMS},
    }
    codec = nemo_relay.typed.BestEffortAnyCodec()  # type: ignore[union-attr]

    captured_ctx: ExecutePythonContext | None = None

    async def _wrapper(inner_args: Any) -> Any:
        nonlocal captured_ctx
        # Apply request intercept modifications back to ctx.
        # NeMo Relay tool request intercepts can transform args (e.g. rewrite code,
        # modify timeout).  The modified args are passed here as `inner_args`;
        # we must propagate changes to ctx so the actual execution sees them.
        if isinstance(inner_args, dict):
            if "code" in inner_args:
                ctx.code = inner_args["code"]
            for key in _PROPAGATABLE_TOOL_PARAMS:
                if key in inner_args:
                    ctx.params[key] = inner_args[key]
        # Call the rest of the middleware chain → eventually executes code
        captured_ctx = await nxt(ctx)
        result = captured_ctx.result
        if result is None:
            return codec.to_json(None)

        # Extract meaningful return value (returned_value > signal > stdout)
        from nooa.events import _NO_RETURN

        rv = getattr(result, "returned_value", _NO_RETURN)
        if rv is _NO_RETURN:
            sig = getattr(result, "signal", None)
            if sig is not None:
                sig_data = getattr(sig, "result", None)
                if isinstance(sig_data, dict) and "result" in sig_data:
                    rv = sig_data["result"]
                else:
                    rv = None
            else:
                rv = getattr(result, "stdout", None) or None

        return codec.to_json(rv)

    # Note: nemo_relay.tools.execute() returns the pre-guardrail result.
    # Sanitize-response guardrails transform data for NeMo Relay internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original result.  Conditional-execution guardrails that reject
    # raise GuardrailRejected, which propagates naturally.
    await nemo_relay.tools.execute("execute_python", args, _wrapper)  # type: ignore[union-attr]

    if captured_ctx is not None:
        return captured_ctx
    # NeMo Relay guardrails blocked code execution (never invoked _wrapper).
    raise RuntimeError(
        "NeMo Relay guardrail blocked code execution — the request was rejected "
        "before running. Check your NeMo Relay guardrail configuration."
    )


def _ambient_handle() -> Any:
    """Return the current top-of-stack scope handle, or ``None`` if unavailable."""
    try:
        return nemo_relay.scope.get_handle()  # type: ignore[union-attr]
    except Exception:
        _logger.debug("nemo_relay: scope.get_handle() failed", exc_info=True)
        return None


async def nemo_relay_agent_call_middleware(
    ctx: AgentCallContext,
    nxt: AgentCallNext,
) -> AgentCallContext:
    """Wrap each agent method call in a NeMo Relay Function scope.

    Pushes a ``ScopeType.Function`` scope named ``"AgentClass.method_name"``
    before the method executes and pops it after, giving ATIF per-method
    granularity.

    Concurrency
    -----------
    NeMo Relay keeps a single mutable LIFO scope stack in a ContextVar, and
    child asyncio tasks inherit it *by reference*.  Two agent calls that
    overlap in time (e.g. ``asyncio.gather(self.a(), self.b())`` in generated
    code) therefore interleave their pushes on one stack, and whichever
    finishes first cannot pop — ``pop()`` rejects any handle that is not on
    top.  The stack is then permanently desynchronised: scopes never emit end
    events, later scopes reparent under the leaked one, and the root pop
    raises out of :func:`nemo_relay_scope`.

    The fix is to give a concurrently-dispatched call its own scope stack, so
    its pushes and pops can never interleave with a sibling's.  Parent linkage
    survives isolation because ``scope.push(handle=...)`` accepts an explicit
    parent handle from another stack.  A call that runs in the *same* task as
    its parent keeps the shared stack — sequential nesting is already correct,
    and staying on the shared stack preserves ``scope_local`` registrations
    made on ancestor scopes.

    Known limitation: a concurrently-dispatched call runs on its own stack, so
    ``nemo_relay.scope_local`` registrations made on an *ancestor* scope do not
    apply to it.  Global guardrails/intercepts/subscribers are unaffected, as
    are scope-locals registered on the call's own scope.  Overlapping calls
    previously corrupted the stack outright, so this is strictly an
    improvement, but it is a real behavioural difference worth knowing.
    """
    assert nemo_relay is not None

    prev = _current_relay_scope.get()
    if prev is None:
        # install_nemo_relay() used without nemo_relay_scope(): adopt whatever
        # scope is currently active as the parent.
        parent, owner_task = _ambient_handle(), None
    else:
        parent, owner_task = prev

    task = asyncio.current_task()
    stack_token = None
    if owner_task is not task and _SCOPE_STACK_VAR is not None:
        # Concurrent sibling — isolate its stack so pops stay in order.
        stack_token = _SCOPE_STACK_VAR.set(nemo_relay.create_scope_stack())  # type: ignore[union-attr]

    scope_name = f"{type(ctx.agent).__name__}.{ctx.method_name}"
    handle = nemo_relay.scope.push(  # type: ignore[union-attr]
        scope_name, nemo_relay.ScopeType.Function, handle=parent
    )
    token = _current_relay_scope.set((handle, task))
    try:
        return await nxt(ctx)
    finally:
        try:
            _current_relay_scope.reset(token)
        except ValueError:
            # Token created in a different Context (manual __aenter__/__aexit__
            # across a task boundary). Fall back to restoring the value.
            _current_relay_scope.set(prev)
        try:
            nemo_relay.scope.pop(handle)  # type: ignore[union-attr]
        except Exception:
            # With per-task isolation this is genuinely exceptional — it means
            # the stack is out of sync, so log loudly rather than hiding it.
            _logger.warning(
                "nemo_relay_agent_call_middleware: scope.pop() failed for %r; "
                "the NeMo Relay scope stack may be out of sync",
                scope_name,
                exc_info=True,
            )
        # Must happen AFTER the pop: restoring the stack var first makes the
        # handle unreachable and the pop fails with "scope handle not found".
        if stack_token is not None and _SCOPE_STACK_VAR is not None:
            _SCOPE_STACK_VAR.reset(stack_token)


# ---------------------------------------------------------------------------
# Install / uninstall helpers
# ---------------------------------------------------------------------------


def install_nemo_relay(event_manager: EventManager) -> Callable[[], None]:
    """Register NeMo Relay middleware on an event manager.

    Returns an uninstall function that removes all three middleware.

    Raises:
        ImportError: If ``nemo_relay`` is not installed.
    """
    if not _HAS_NEMO_RELAY:
        raise ImportError(_INSTALL_MSG)

    unsub_agent = event_manager.intercept(MIDDLEWARE_AGENT_CALL, nemo_relay_agent_call_middleware)
    unsub_llm = event_manager.intercept(MIDDLEWARE_LLM_CALL, nemo_relay_llm_middleware)
    unsub_exec = event_manager.intercept(MIDDLEWARE_EXECUTE_PYTHON, nemo_relay_tool_middleware)

    def uninstall() -> None:
        unsub_agent()
        unsub_llm()
        unsub_exec()

    return uninstall


@asynccontextmanager
async def nemo_relay_scope(
    agent: Any,
    scope_name: str,
) -> AsyncIterator[Any]:
    """Async context manager that activates NeMo Relay for an agent.

    Pushes a NeMo Relay scope, installs middleware on the agent's event manager,
    and cleans up on exit::

        async with nemo_relay_scope(agent, "research-agent") as handle:
            result = await agent.research("quantum computing")
            # handle.uuid available for ATIF export

    Args:
        agent: The Agent instance whose event_manager will get middleware.
        scope_name: Human-readable name for the NeMo Relay scope.

    Yields:
        The NeMo Relay scope handle (has ``.uuid`` for ATIF correlation).

    Raises:
        ImportError: If ``nemo_relay`` is not installed.
    """
    if not _HAS_NEMO_RELAY:
        raise ImportError(_INSTALL_MSG)

    uninstall = install_nemo_relay(agent.event_manager)
    try:
        with nemo_relay.scope.scope(  # type: ignore[union-attr]
            scope_name,
            nemo_relay.ScopeType.Agent,  # type: ignore[union-attr]
        ) as handle:
            # Seed the root so agent calls parent off it explicitly rather than
            # off whatever happens to be on top of the shared stack.  The pop
            # of this scope is deliberately NOT guarded: with per-task stack
            # isolation it can only fail if the stack is genuinely corrupt, and
            # that must surface rather than leave every later trace misrooted.
            token = _current_relay_scope.set((handle, asyncio.current_task()))
            try:
                yield handle
            finally:
                try:
                    _current_relay_scope.reset(token)
                except ValueError:
                    _current_relay_scope.set(None)
    finally:
        uninstall()
