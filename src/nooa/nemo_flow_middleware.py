# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Flow integration via the ``intercept()`` middleware API.

The upstream ``nemo_flow`` package was renamed to ``nemo_relay`` (every
``nemo_flow`` release on PyPI is yanked with that reason). This module imports
``nemo_relay`` under the local alias ``nemo_flow`` — the runtime API is
unchanged, so the nooa-facing names (``nemo_flow_scope`` etc.) are kept for
backward compatibility.

When ``nemo_relay`` is installed, this module provides three middleware functions
that route LLM calls, code execution, and agent method calls through the NeMo Flow
pipeline (guardrails, intercepts, event subscribers, ATIF export).

Usage::

    from nooa.nemo_flow_middleware import install_nemo_flow

    # Inside an async context where nemo_flow scope is active:
    uninstall = install_nemo_flow(agent.event_manager)
    try:
        result = await agent.my_method()
    finally:
        uninstall()

Or use ``nemo_flow_scope()`` which handles install/uninstall automatically::

    from nooa.nemo_flow_middleware import nemo_flow_scope

    async with nemo_flow_scope(agent, "my-agent"):
        result = await agent.my_method()

Requirements:
    ``nemo_relay`` must be installed (``uv sync --extra nemo-relay``).
    If not installed, ``install_nemo_flow()`` and ``nemo_flow_scope()`` raise
    ``ImportError`` with install instructions.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
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
    # nemo_flow was renamed to nemo_relay upstream; alias it so the rest of
    # this module's nemo_flow.* references keep working unchanged.
    import nemo_relay as nemo_flow  # type: ignore[import]
    from nemo_relay import LLMRequest  # type: ignore[import]

    _HAS_NEMO_FLOW = True
except ImportError:
    _HAS_NEMO_FLOW = False
    nemo_flow = None  # type: ignore[assignment]
    LLMRequest = None  # type: ignore[assignment,misc]

_INSTALL_MSG = (
    "nemo_flow is required for NeMo Flow integration. The package was renamed "
    "to nemo_relay; install it with `uv sync --extra nemo-relay` "
    "(or `uv add nemo-relay`)."
)

# Keys stripped from params before exposing to NeMo Flow guardrails/events.
_SENSITIVE_KEYS: frozenset[str] = frozenset({"api_key", "api_base", "base_url"})

# Keys that hold non-JSON-serializable objects (Tool instances, Pydantic models).
# These are converted or removed before constructing LLMRequest.
_NON_SERIALIZABLE_KEYS: frozenset[str] = frozenset({"tools", "output_model"})

# Keys from LLMRequest.content that map directly to ctx.params.
# If a NeMo Flow request intercept modifies these, we propagate them.
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

# Tool execution args that NeMo Flow intercepts may modify.
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


async def nemo_flow_llm_middleware(
    ctx: LLMCallContext,
    nxt: LLMCallNext,
) -> LLMCallContext:
    """Route LLM calls through the NeMo Flow LLM pipeline.

    Strips sensitive keys, wraps the call through ``nemo_flow.llm.execute()``,
    and returns the original ``LLMResponse`` to the caller.  The JSON-serialized
    response is what flows through NeMo Flow guardrails and ATIF export.
    """
    assert nemo_flow is not None

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
    # ('dict' object has no attribute 'name') inside NeMo Flow's native pipeline.
    # The old hooks-based integration avoided this because it intercepted at
    # unifiedllm's layer, where api_params never contained a tools key.
    request = LLMRequest({}, safe_params)  # type: ignore[misc]

    captured_ctx: LLMCallContext | None = None

    async def _wrapper(req: Any) -> Any:
        nonlocal captured_ctx
        # Apply request intercept modifications back to ctx.
        # NeMo Flow request intercepts can transform the LLMRequest (e.g. inject
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
        # Return JSON to NeMo Flow (guardrails, events, ATIF).
        resp = captured_ctx.response
        if resp is None:
            return {}
        # Prefer the raw litellm ModelResponse (Pydantic) — gives NeMo Flow the
        # full OpenAI-style structure matching what the old hooks-based
        # integration returned via captured_response.model_dump(mode="json").
        raw = getattr(resp, "raw_response", None)
        if raw is not None and hasattr(raw, "model_dump"):
            return raw.model_dump(mode="json")
        # Pydantic response (e.g. passed directly)
        if hasattr(resp, "model_dump"):
            return resp.model_dump(mode="json")  # type: ignore[union-attr]
        # Fallback: manual serialization from unifiedllm.LLMResponse dataclass.
        if hasattr(resp, "assistant_message"):
            result: dict[str, Any] = {"message": resp.assistant_message}
            if resp.usage:
                result["usage"] = resp.usage
            if resp.finish_reason:
                result["finish_reason"] = resp.finish_reason
            return result
        return {}

    # Note: nemo_flow.llm.execute() returns the pre-guardrail response.
    # Sanitize-response guardrails transform data for NeMo Flow internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original response.  Conditional-execution guardrails that
    # reject raise GuardrailRejected, which propagates naturally.
    await nemo_flow.llm.execute(model_name, request, _wrapper, model_name=model_name)  # type: ignore[union-attr]

    if captured_ctx is not None:
        return captured_ctx
    # NeMo Flow guardrails blocked the LLM call (never invoked _wrapper).
    # Raise an explicit error instead of returning ctx without a response,
    # which would trigger a confusing generic RuntimeError downstream.
    raise RuntimeError(
        "NeMo Flow guardrail blocked the LLM call — the request was rejected "
        "before reaching the LLM. Check your NeMo Flow guardrail configuration."
    )


async def nemo_flow_tool_middleware(
    ctx: ExecutePythonContext,
    nxt: ExecutePythonNext,
) -> ExecutePythonContext:
    """Route code execution through the NeMo Flow tool pipeline.

    Extracts the meaningful return value from ``ExecutionResult``, serializes
    it via ``BestEffortAnyCodec`` for NeMo Flow inspection, and returns the
    original ``ExecutionResult`` to the caller.
    """
    assert nemo_flow is not None

    args = {
        "code": ctx.code,
        **{k: v for k, v in ctx.params.items() if k in _PROPAGATABLE_TOOL_PARAMS},
    }
    codec = nemo_flow.typed.BestEffortAnyCodec()  # type: ignore[union-attr]

    captured_ctx: ExecutePythonContext | None = None

    async def _wrapper(inner_args: Any) -> Any:
        nonlocal captured_ctx
        # Apply request intercept modifications back to ctx.
        # NeMo Flow tool request intercepts can transform args (e.g. rewrite code,
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

        # Extract meaningful return value (same priority as original _nemo_flow.py)
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

    # Note: nemo_flow.tools.execute() returns the pre-guardrail result.
    # Sanitize-response guardrails transform data for NeMo Flow internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original result.  Conditional-execution guardrails that reject
    # raise GuardrailRejected, which propagates naturally.
    await nemo_flow.tools.execute("execute_python", args, _wrapper)  # type: ignore[union-attr]

    if captured_ctx is not None:
        return captured_ctx
    # NeMo Flow guardrails blocked code execution (never invoked _wrapper).
    raise RuntimeError(
        "NeMo Flow guardrail blocked code execution — the request was rejected "
        "before running. Check your NeMo Flow guardrail configuration."
    )


async def nemo_flow_agent_call_middleware(
    ctx: AgentCallContext,
    nxt: AgentCallNext,
) -> AgentCallContext:
    """Wrap each agent method call in a NeMo Flow Function scope.

    Pushes a ``ScopeType.Function`` scope named ``"AgentClass.method_name"``
    before the method executes and pops it after, giving ATIF per-method
    granularity.
    """
    assert nemo_flow is not None

    scope_name = f"{type(ctx.agent).__name__}.{ctx.method_name}"
    handle = nemo_flow.scope.push(scope_name, nemo_flow.ScopeType.Function)  # type: ignore[union-attr]
    try:
        return await nxt(ctx)
    finally:
        try:
            nemo_flow.scope.pop(handle)  # type: ignore[union-attr]
        except Exception:
            _logger.debug("nemo_flow_agent_call_middleware: scope.pop() failed", exc_info=True)


# ---------------------------------------------------------------------------
# Install / uninstall helpers
# ---------------------------------------------------------------------------


def install_nemo_flow(event_manager: EventManager) -> Callable[[], None]:
    """Register NeMo Flow middleware on an event manager.

    Returns an uninstall function that removes all three middleware.

    Raises:
        ImportError: If ``nemo_flow`` is not installed.
    """
    if not _HAS_NEMO_FLOW:
        raise ImportError(_INSTALL_MSG)

    unsub_agent = event_manager.intercept(MIDDLEWARE_AGENT_CALL, nemo_flow_agent_call_middleware)
    unsub_llm = event_manager.intercept(MIDDLEWARE_LLM_CALL, nemo_flow_llm_middleware)
    unsub_exec = event_manager.intercept(MIDDLEWARE_EXECUTE_PYTHON, nemo_flow_tool_middleware)

    def uninstall() -> None:
        unsub_agent()
        unsub_llm()
        unsub_exec()

    return uninstall


@asynccontextmanager
async def nemo_flow_scope(
    agent: Any,
    scope_name: str,
) -> AsyncIterator[Any]:
    """Async context manager that activates NeMo Flow for an agent.

    Pushes a NeMo Flow scope, installs middleware on the agent's event manager,
    and cleans up on exit::

        async with nemo_flow_scope(agent, "research-agent") as handle:
            result = await agent.research("quantum computing")
            # handle.uuid available for ATIF export

    Args:
        agent: The Agent instance whose event_manager will get middleware.
        scope_name: Human-readable name for the NeMo Flow scope.

    Yields:
        The NeMo Flow scope handle (has ``.uuid`` for ATIF correlation).

    Raises:
        ImportError: If ``nemo_flow`` is not installed.
    """
    if not _HAS_NEMO_FLOW:
        raise ImportError(_INSTALL_MSG)

    uninstall = install_nemo_flow(agent.event_manager)
    try:
        with nemo_flow.scope.scope(  # type: ignore[union-attr]
            scope_name,
            nemo_flow.ScopeType.Agent,  # type: ignore[union-attr]
        ) as handle:
            yield handle
    finally:
        uninstall()
