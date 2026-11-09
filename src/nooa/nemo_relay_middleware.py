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
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.runtime.middleware import (
    MIDDLEWARE_AGENT_CALL,
    MIDDLEWARE_EXECUTE_PYTHON,
    MIDDLEWARE_LLM_CALL,
)

_logger = logging.getLogger(__name__)


def _reconcile_messages(originals, public):
    """Only unchanged JSON entries at the same position recover their turn.

    Insertions/deletions conservatively demote shifted turns to portable dicts.
    No native fields cross the relay's JSON boundary.
    """
    return [
        originals[index]
        if index < len(originals)
        and isinstance(originals[index], (LLMResponse, CacheBoundary))
        and message == originals[index].public_message()
        else message
        for index, message in enumerate(public)
    ]


def _relay_response(response: LLMResponse) -> dict[str, Any]:
    """Project the canonical response only when NeMo Relay needs wire JSON."""
    result: dict[str, Any] = {"finish_reason": response.finish_reason}
    if response.content or response.tool_calls or response.reasoning:
        # Relay's generic ATIF extractor recognizes assistant_message, not
        # message. Keep this a public projection, never the provider response.
        result["assistant_message"] = response.public_message()
    else:
        result["content"] = ""
    if response.reasoning:
        result["reasoning"] = response.reasoning
    if response.usage is not None:
        result["usage"] = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "cached_tokens": response.usage.cached_input_tokens,
            "cache_creation_input_tokens": response.usage.cache_write_input_tokens,
            "reasoning_tokens": response.usage.reasoning_tokens,
            "cost_usd": response.usage.cost_usd,
        }
    return result


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

    # Method-level overrides live on the effective client, not agent._llm.
    model_name = ctx.client.model if ctx.client is not None else _extract_model_name(ctx.agent)
    safe_params = {
        k: v
        for k, v in ctx.params.items()
        if k not in _SENSITIVE_KEYS and k not in _NON_SERIALIZABLE_KEYS
    }
    original_messages = list(ctx.messages)
    safe_params["messages"] = [
        dict(message) if isinstance(message, (LLMResponse, CacheBoundary)) else message
        for message in original_messages
    ]
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
                ctx.messages = _reconcile_messages(original_messages, intercepted_msgs)
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
        return _relay_response(resp)

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
        assert nemo_relay is not None
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

        encoded = codec.to_json(rv)
        error = getattr(result, "error", None)
        cancelled = getattr(result, "cancelled", False)
        if error is not None:
            content = f"{type(error).__name__}: {error}"
        elif cancelled:
            content = "Execution cancelled"
        else:
            content = (
                encoded if isinstance(encoded, str) else json.dumps(encoded, ensure_ascii=False)
            )
        # ATIF content must be text or content parts. Retain typed data in
        # result while giving numeric/structured returns a readable observation.
        payload = {
            "schema": "nemo.relay.tool_result@1",
            "content": content,
            "is_error": error is not None or cancelled,
            "result": encoded,
            "cache": {"replayable": False},  # Python cells mutate live REPL state.
        }
        return nemo_relay.ToolExecutionResult(payload)

    # Note: nemo_relay.tools.execute() returns the pre-guardrail result.
    # Sanitize-response guardrails transform data for NeMo Relay internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original result.  Conditional-execution guardrails that reject
    # raise GuardrailRejected, which propagates naturally.
    await nemo_relay.tools.execute(  # type: ignore[union-attr]
        "execute_python", args, _wrapper, tool_call_id=ctx.params.get("tool_call_id")
    )

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
    survives isolation through Relay's public propagation and stack-binding
    APIs. A call that runs in the *same* task as
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
    stack_context = nullcontext()
    if owner_task is not task:
        # Preserve causal parent/root while isolating concurrent mutations.
        stack = nemo_relay.create_scope_stack_from_propagation(  # type: ignore[union-attr]
            nemo_relay.capture_propagation_context()
        )
        stack_context = nemo_relay.use_scope_stack(stack)  # type: ignore[union-attr,assignment]

    scope_name = f"{type(ctx.agent).__name__}.{ctx.method_name}"
    with (
        stack_context,
        nemo_relay.scope.scope(  # type: ignore[union-attr]
            scope_name, nemo_relay.ScopeType.Function, handle=parent
        ) as handle,
    ):
        token = _current_relay_scope.set((handle, task))
        try:
            return await nxt(ctx)
        finally:
            _current_relay_scope.reset(token)


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
