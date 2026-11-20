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
import re
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from nooa._relay_atif import export_nemo_relay_atif as export_nemo_relay_atif
from nooa._relay_telemetry import inspection_snapshot
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
    if response.content or response.tool_calls or response.reasoning:
        # Preserve the payload shape used by existing response sanitizers.
        # Export formatting consumes only the final sanitized output.
        message = response.public_message()
        result["message"] = {
            **{key: message[key] for key in ("role", "tool_calls") if key in message},
            **message,
        }
    else:
        result["content"] = ""
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

# Immutable parent identity is copied into child asyncio contexts. Every
# NOOA scope has an independent native stack, so no task-owner registry is needed.
_current_relay_scope: ContextVar[Any] = ContextVar("_nooa_relay_scope", default=None)


def _reset_scope(token, previous) -> None:
    try:
        _current_relay_scope.reset(token)
    except ValueError:
        _current_relay_scope.set(previous)
        _logger.warning(
            "NeMo Relay context reset failed for scope %s; telemetry may be incomplete",
            getattr(previous, "uuid", None),
        )


@contextmanager
def _relay_stack() -> Iterator[None]:
    assert nemo_relay is not None
    stack = nemo_relay.create_scope_stack_from_propagation(nemo_relay.capture_propagation_context())
    binding = nemo_relay.use_scope_stack(stack)
    binding.__enter__()
    try:
        yield
    finally:
        try:
            binding.__exit__(None, None, None)
        except Exception:
            _logger.warning(
                "NeMo Relay stack binding exit failed for stack %s; telemetry may be incomplete",
                id(stack),
            )


@contextmanager
def _method_scope(name: str, scope_type: Any, parent: Any = None) -> Iterator[Any]:
    """Record status without copying exception text or masking method outcomes."""
    assert nemo_relay is not None
    handle = nemo_relay.scope.push(name, scope_type, handle=parent)
    status = "OK"
    try:
        yield handle
    except asyncio.CancelledError:
        status = "UNSET"
        raise
    except BaseException:
        status = "ERROR"
        raise
    finally:
        try:
            nemo_relay.scope.pop(handle, metadata={"otel.status_code": status})
        except Exception:
            _logger.warning(
                "NeMo Relay pop failed for %s (%s); telemetry may be incomplete", name, handle.uuid
            )


_INSTALL_MSG = (
    "nemo_relay is required for NeMo Relay integration. Install it with "
    "`uv sync --extra nemo-relay` (or `uv add nemo-relay`)."
)


def _require_relay() -> None:
    if not _HAS_NEMO_RELAY:
        raise ImportError(_INSTALL_MSG)
    _check_relay_version()


@cache
def _check_relay_version() -> None:
    """Validate once per process; installing packages requires a restart."""
    try:
        installed = version("nemo-relay")
    except PackageNotFoundError:
        raise ImportError(_INSTALL_MSG) from None
    match = re.fullmatch(r"0\.11\.([0-9]+)(?:\.post[0-9]+)?(?:\+[^ ]+)?", installed)
    if match is None or int(match[1]) < 1:
        raise ImportError(
            f"NOOA requires nemo-relay >=0.11.1,<0.12; found {installed}. " + _INSTALL_MSG
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
    _require_relay()
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
        assert nemo_relay is not None
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
        data, attributes = inspection_snapshot(_relay_response(resp))
        if isinstance(data, dict) and attributes["nooa.output.preview.incomplete"]:
            data.update(attributes)
        return data

    # Note: nemo_relay.llm.execute() returns the pre-guardrail response.
    # Sanitize-response guardrails transform data for NeMo Relay internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original response.  Conditional-execution guardrails that
    # reject raise GuardrailRejected, which propagates naturally.
    from nooa._relay_callback import execute_callback

    execute = nemo_relay.llm.execute
    await execute_callback(
        lambda callback: execute(model_name, request, callback, model_name=model_name),
        _wrapper,
    )

    if captured_ctx is not None:
        return captured_ctx
    # An execution intercept/cache bypassed NOOA (never invoked _wrapper).
    # Raise an explicit error instead of returning ctx without a response,
    # which would trigger a confusing generic RuntimeError downstream.
    raise RuntimeError(
        "NeMo Relay short-circuited the LLM call. NOOA does not support cached or "
        "replacement responses; disable response caching/execution short circuits."
    )


async def nemo_relay_tool_middleware(
    ctx: ExecutePythonContext,
    nxt: ExecutePythonNext,
) -> ExecutePythonContext:
    """Route code execution through the NeMo Relay tool pipeline.

    Capture a bounded inspection snapshot of the meaningful return value.
    The original ``ExecutionResult`` still reaches the caller unchanged.
    """
    _require_relay()
    assert nemo_relay is not None

    args = {
        "code": ctx.code,
        **{k: v for k, v in ctx.params.items() if k in _PROPAGATABLE_TOOL_PARAMS},
    }

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

        error = getattr(result, "error", None)
        cancelled = getattr(result, "cancelled", False)
        if error is not None:
            rv = error
        elif cancelled:
            rv = "Execution cancelled"
        # Keep the legacy value shape for bounded inspection JSON. Display
        # formatting happens only after response/event sanitizers, at export.
        data, attributes = inspection_snapshot(rv)
        return nemo_relay.ToolExecutionResult(
            data, annotation={"is_error": error is not None or bool(cancelled), **attributes}
        )

    # Note: nemo_relay.tools.execute() returns the pre-guardrail result.
    # Sanitize-response guardrails transform data for NeMo Relay internals
    # (ATIF export, event subscribers) but the caller always receives
    # the original result.  Conditional-execution guardrails that reject
    # raise GuardrailRejected, which propagates naturally.
    from nooa._relay_callback import execute_callback

    execute = nemo_relay.tools.execute
    await execute_callback(
        lambda callback: execute(
            "execute_python", args, callback, tool_call_id=ctx.params.get("tool_call_id") or None
        ),
        _wrapper,
    )

    if captured_ctx is not None:
        return captured_ctx
    # An execution intercept/cache bypassed NOOA (never invoked _wrapper).
    raise RuntimeError(
        "NeMo Relay short-circuited Python execution. NOOA does not support cached or "
        "replacement execution results; disable response caching/execution short circuits."
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

    Every call has an independent native stack, including nested and concurrent
    calls. Trace parentage survives; scope-local registrations are not inherited.
    Register guardrails/redactors globally when they must cover the call tree.
    """
    _require_relay()
    assert nemo_relay is not None

    prev = _current_relay_scope.get()
    if prev is None:
        # install_nemo_relay() used without nemo_relay_scope(): adopt whatever
        # scope is currently active as the parent.
        parent = _ambient_handle()
    else:
        parent = prev

    scope_name = f"{type(ctx.agent).__name__}.{ctx.method_name}"
    with (
        _relay_stack(),
        _method_scope(  # type: ignore[union-attr]
            scope_name, nemo_relay.ScopeType.Function, parent=parent
        ) as handle,
    ):
        token = _current_relay_scope.set(handle)
        try:
            return await nxt(ctx)
        finally:
            _reset_scope(token, prev)


# ---------------------------------------------------------------------------
# Install / uninstall helpers
# ---------------------------------------------------------------------------


@dataclass
class _RelayInstallation:
    users: int
    unsubscribe: tuple[Callable[[], None], ...]


def install_nemo_relay(event_manager: EventManager) -> Callable[[], None]:
    """Register NeMo Relay middleware on an event manager.

    Overlapping installations share one middleware chain. The last uninstall
    removes it; each returned uninstall function is idempotent.

    Raises:
        ImportError: If ``nemo_relay`` is not installed.
    """
    _require_relay()

    installation = getattr(event_manager, "_nooa_relay_installation", None)
    if not isinstance(installation, _RelayInstallation):
        # If any registration fails, undo the ones already installed.
        with ExitStack() as rollback:
            unsubscribe = []
            for kind, middleware in (
                (MIDDLEWARE_AGENT_CALL, nemo_relay_agent_call_middleware),
                (MIDDLEWARE_LLM_CALL, nemo_relay_llm_middleware),
                (MIDDLEWARE_EXECUTE_PYTHON, nemo_relay_tool_middleware),
            ):
                remove = event_manager.intercept(kind, middleware)
                unsubscribe.append(remove)
                rollback.callback(remove)
            installation = _RelayInstallation(0, tuple(unsubscribe))
            setattr(event_manager, "_nooa_relay_installation", installation)  # noqa: B010
            rollback.pop_all()
    installation.users += 1
    active = True

    def uninstall() -> None:
        nonlocal active
        if not active:
            return
        active = False
        installation.users -= 1
        if installation.users == 0:
            for unsubscribe in installation.unsubscribe:
                unsubscribe()
            delattr(event_manager, "_nooa_relay_installation")

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

    Runs and agent methods have independent stacks automatically. Enclosing
    scope-local guardrails, redactors and subscribers do not transfer to isolated
    children; register shared policies globally. Trace parentage is preserved.

    Yields:
        The NeMo Relay scope handle (has ``.uuid`` for ATIF correlation).

    Raises:
        ImportError: If ``nemo_relay`` is not installed.
    """
    _require_relay()

    uninstall = install_nemo_relay(agent.event_manager)
    previous = _current_relay_scope.get()
    try:
        with (
            _relay_stack(),
            _method_scope(  # type: ignore[union-attr]
                scope_name,
                nemo_relay.ScopeType.Agent,  # type: ignore[union-attr]
            ) as handle,
        ):
            token = _current_relay_scope.set(handle)
            try:
                yield handle
            finally:
                _reset_scope(token, previous)
    finally:
        uninstall()
