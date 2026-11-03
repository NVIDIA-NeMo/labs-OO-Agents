# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared method wrapper logic for agent methods.

This module provides the unified wrapper logic used by both:
- AgentMeta metaclass (for methods defined at class creation time)
- @strategy decorator (for dynamically-defined methods via exec)

Having this in one place eliminates duplication and ensures consistent behavior
for context variable management, tracing hooks, and execution routing.
"""

import asyncio
import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any
from uuid import uuid4

# These imports are safe at module level (no circular dependencies)
from nooa.context_blocks.scoped import _scoped_blocks_var, _scoped_events_var
from nooa.events import AfterAgentCall, BeforeAgentCall
from nooa.runtime.context_vars import (
    _get_agent_call_stack,
    _in_generation_session,
    _parent_agent_var,
    _pop_agent_call_id,
    _push_agent_call_id,
)
from nooa.runtime.hooks import call_after_hook, call_before_hook

if TYPE_CHECKING:
    from nooa.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)


async def _flush_litellm_journal() -> None:
    # Three yields drain litellm's GLOBAL_LOGGING_WORKER chain before joining the
    # journal POST thread; running the join in a worker avoids blocking the loop.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    from nooa.runtime.async_safety import _in_agent_context
    from nooa.tracing._litellm_journal import flush_pending

    token = _in_agent_context.set(False)
    try:
        await asyncio.to_thread(flush_pending)
    finally:
        _in_agent_context.reset(token)


def create_agent_method_wrapper(
    original_func: Callable[..., Any],
    *,
    needs_generation: bool,
    needs_tracing: bool,
    strategy: "GenerationStrategy | None",
    cached_source_code: str | None = None,
) -> Callable[..., Any]:
    """Create a wrapper for an agent method with unified behavior.

    This wrapper handles:
    - Instrumentation hooks (before/after agent call)
    - Call stack management (push/pop call_id)
    - Parent agent context for LLM inheritance
    - Scoped blocks isolation between agents
    - Routing to runtime for execution

    Args:
        original_func: The original async function to wrap
        needs_generation: Whether method body is ellipsis (needs LLM generation)
        needs_tracing: Whether method should be traced
        strategy: Strategy instance to use (or None for auto-resolution)
        cached_source_code: Pre-extracted source code for tracing (optional)

    Returns:
        Wrapped async function with all instrumentation and routing
    """

    # Mutable single-element list so that @no_trace applied *after* @strategy
    # (i.e. as the outer decorator) can flip the flag retroactively.
    # The wrapper checks _tracing_enabled[0] at call time rather than the
    # original `needs_tracing` bool so both decorator orderings work:
    #   @strategy @no_trace  (no_trace inner — @strategy sees _no_trace at creation)
    #   @no_trace @strategy  (no_trace outer — updates _tracing_enabled after creation)
    _tracing_enabled = [needs_tracing]

    @wraps(original_func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Unified wrapper for agent methods."""
        # Generation-specific runtime checks and setup
        resolved_strategy = strategy
        if needs_generation:
            # Note: We no longer raise on nested calls during generation.
            # Instead, we route them through _execute_task to avoid deadlock.

            # Lazy import: only needed for generation path, avoids loading
            # generated_code.py machinery for non-generation methods
            from nooa.strategies.base import RuntimeServices
            from nooa.strategies.generated_code import ArgumentValidator

            # Resolve truncation config for error message formatting:
            # - Agent methods: self._truncation (the agent's config)
            # - Strategy helper methods: args[0] is the runtime (by convention)
            # - Anything else (malformed call, non-Agent class): fall back to defaults
            if hasattr(self, "_truncation"):
                _tc = self._truncation
            elif args and isinstance(args[0], RuntimeServices):
                _tc = args[0].truncation_config
            else:
                from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

                _tc = DEFAULT_TRUNCATION_CONFIG

            # Strip framework kwargs before validation — they're consumed by
            # _execute_with_generation, not the user's method signature. Must
            # cover every name that _execute_with_generation pops, or the call
            # is rejected here before it can ever get there.
            #
            # Only on the agent path: the strategy-helper path below routes to
            # execute_nested(), which pops nothing, so stripping there would
            # drop these names silently into CurrentCall's prompt arguments
            # instead of rejecting them as the unexpected kwargs they are.
            _fw_kwargs: dict[str, Any] = {}
            if hasattr(self, "runtime"):
                _fw_kwargs = {
                    _name: kwargs.pop(_name)
                    for _name in ("_session_locals", "_strategy", "llm")
                    if _name in kwargs
                }
            try:
                ArgumentValidator().validate(original_func, args, kwargs, _tc)
            finally:
                # Restore so _execute_with_generation can pop them
                kwargs.update(_fw_kwargs)

            # Strategy resolution if not provided
            if resolved_strategy is None:
                # Lazy import: avoids circular dependency with strategies/__init__.py
                from nooa.strategies import get_default_strategy

                resolved_strategy = get_default_strategy()

        # Route based on available attributes (duck-typing)
        if hasattr(self, "runtime"):
            # === Agent method path ===
            runtime = self.runtime
            call_id = str(uuid4())
            parent_call_id = runtime._agent_call_id

            # Build trace attributes
            trace_attrs = _build_trace_attributes(
                needs_generation,
                resolved_strategy,
                cached_source_code,
            )

            # hook_context is set inside the middleware/fast-path blocks below
            hook_context = None

            # Enforce max nesting depth before pushing the new call ID
            execution_config = getattr(self, "_execution_config", None)
            if execution_config is not None:
                current_depth = len(_get_agent_call_stack())
                if current_depth >= execution_config.max_nesting_depth:
                    raise RuntimeError(
                        f"Exceeded maximum nesting depth of {execution_config.max_nesting_depth} "
                        f"for {type(self).__name__}.{original_func.__name__}. "
                        f"Current depth: {current_depth}. "
                        f"Increase ExecutionConfig(max_nesting_depth=...) to allow deeper nesting."
                    )

            # For @no_trace methods, push the parent's call_id rather than our own
            # so that any child methods find the nearest *traced* ancestor in the
            # stack and become its children in the span tree.
            _push_agent_call_id(call_id if _tracing_enabled[0] else parent_call_id)

            # Check if we're entering a DIFFERENT agent's method (subagent call)
            # Scoped blocks should propagate within the same agent but NOT across
            # agent boundaries. This fixes the bug where CodeActStrategy's
            # execution_context block would leak to a subagent using a different strategy.
            current_parent = _parent_agent_var.get()
            is_subagent_call = current_parent is not None and current_parent is not self

            # Clear scoped blocks and events only when entering a different agent
            scoped_blocks_token = None
            scoped_events_token = None
            if is_subagent_call:
                scoped_blocks_token = _scoped_blocks_var.set(None)
                scoped_events_token = _scoped_events_var.set(None)

            # Set parent agent context for LLM inheritance by subagents
            parent_token = _parent_agent_var.set(self)

            # Emit a generic agent-call lifecycle event for subscribers (e.g.
            # ATIF arms its standalone cascade on the top-level call). is_top_level
            # = no agent was active before this call. Defensive: never break the call.
            is_top_level = current_parent is None
            try:
                self.event_manager.add(
                    BeforeAgentCall(
                        method_name=original_func.__name__,
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        is_top_level=is_top_level,
                        needs_generation=needs_generation,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("agent-call: BeforeAgentCall emission failed", exc_info=True)

            result = None
            exception_caught = None

            try:
                # --- agent_call middleware wraps entire method execution ---
                em = self.event_manager
                has_agent_mw = bool(em._middleware.get("agent_call"))

                # Shared dispatch logic used by both middleware and fast path.
                async def _dispatch(a: tuple[Any, ...], kw: dict[str, Any]) -> Any:
                    if needs_generation:
                        if _in_generation_session.get():
                            return await runtime._execute_task(wrapper, a, kw)
                        else:
                            return await runtime._call_plan(wrapper, a, kw)
                    else:
                        return await original_func(self, *a, **kw)

                if has_agent_mw:
                    from nooa.runtime.middleware import (
                        _AGENT_RESULT_NOT_SET,
                        AgentCallContext,
                    )

                    ac_ctx = AgentCallContext(
                        agent=self,
                        method_name=original_func.__name__,
                        args=args,
                        kwargs=kwargs,
                    )

                    async def _core_agent(ctx: AgentCallContext) -> AgentCallContext:
                        # Tracing hooks fire INSIDE middleware so they
                        # see the post-middleware args/kwargs.
                        nonlocal hook_context
                        if _tracing_enabled[0]:
                            hook_context = call_before_hook(
                                "before_agent_call",
                                agent=self,
                                method_name=original_func.__name__,
                                args=ctx.args,
                                kwargs=ctx.kwargs,
                                call_id=call_id,
                                parent_call_id=parent_call_id,
                                **trace_attrs,
                            )
                        ctx.result = await _dispatch(ctx.args, ctx.kwargs)
                        await _flush_litellm_journal()
                        return ctx

                    ac_ctx = await em.run_middleware("agent_call", ac_ctx, _core_agent)
                    if ac_ctx.result is _AGENT_RESULT_NOT_SET:
                        raise RuntimeError(
                            "agent_call middleware returned without setting ctx.result. "
                            "Short-circuiting middleware must set ctx.result before returning."
                        )
                    result = ac_ctx.result
                else:
                    # Fast path — no agent_call middleware
                    if _tracing_enabled[0]:
                        hook_context = call_before_hook(
                            "before_agent_call",
                            agent=self,
                            method_name=original_func.__name__,
                            args=args,
                            kwargs=kwargs,
                            call_id=call_id,
                            parent_call_id=parent_call_id,
                            **trace_attrs,
                        )
                    result = await _dispatch(args, kwargs)
                    await _flush_litellm_journal()

                return result
            except Exception as e:
                exception_caught = e
                raise
            finally:
                # Completion event (fires on success and exception). Defensive.
                try:
                    self.event_manager.add(
                        AfterAgentCall(
                            method_name=original_func.__name__,
                            call_id=call_id,
                            parent_call_id=parent_call_id,
                            is_top_level=is_top_level,
                            needs_generation=needs_generation,
                            success=exception_caught is None,
                            exception_type=(
                                type(exception_caught).__name__
                                if exception_caught is not None
                                else None
                            ),
                        )
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("agent-call: AfterAgentCall emission failed", exc_info=True)
                # Reset scoped blocks/events context if we cleared it
                if scoped_blocks_token is not None:
                    _scoped_blocks_var.reset(scoped_blocks_token)
                if scoped_events_token is not None:
                    _scoped_events_var.reset(scoped_events_token)
                # Reset parent agent context
                _parent_agent_var.reset(parent_token)
                _pop_agent_call_id()
                # Only fire after_agent_call if before_agent_call ran
                # (skipped when middleware short-circuits).
                if hook_context is not None:
                    call_after_hook(
                        "after_agent_call",
                        hook_context,
                        agent=self,
                        method_name=original_func.__name__,
                        result=result,
                        exception=exception_caught,
                    )

        elif needs_generation and args and isinstance(args[0], RuntimeServices):  # pyright: ignore[reportPossiblyUnboundVariable]
            # === Strategy method path ===
            # First argument implements RuntimeServices Protocol
            runtime = args[0]
            call_args = args[1:]  # Skip runtime parameter
            call_kwargs = kwargs

            if not resolved_strategy:
                raise ValueError(
                    f"@strategy method {original_func.__name__} on strategy requires strategy parameter. "
                    f"Usage: @strategy(SomeStrategy())"
                )

            # Lazy import: only needed for strategy method path
            from nooa.strategies.current_call import CurrentCall

            # Build CurrentCall from method signature
            call = CurrentCall.from_method(original_func, call_args, call_kwargs)

            # Execute nested strategy
            return await runtime.execute_nested(resolved_strategy, call)

        elif not needs_generation:
            # === Direct execution path (non-generation methods without runtime) ===
            # This handles regular methods called before runtime is set
            return await original_func(self, *args, **kwargs)

        else:
            # Check if this is an Agent instance missing initialization
            # Import here to avoid circular dependency
            from nooa.agent import Agent

            if isinstance(self, Agent) and not hasattr(self, "runtime"):
                raise RuntimeError(
                    f"Agent {type(self).__name__} is not properly initialized.\n"
                    f"\n"
                    f"The agent's __init__() method must call super().__init__() to set up "
                    f"critical infrastructure (runtime, event manager, context, blocks).\n"
                    f"\n"
                    f"Expected pattern:\n"
                    f"  def __init__(self, ...):\n"
                    f"      super().__init__()\n"
                    f"      # Your initialization code here\n"
                    f"\n"
                    f"Without super().__init__(), generation methods cannot execute."
                )

            # Original error for non-Agent cases
            raise ValueError(
                f"@strategy method {original_func.__name__} on {type(self).__name__} requires "
                f"RuntimeServices as first argument after self. "
                f"Expected: async def {original_func.__name__}(self, runtime: RuntimeServices, ...)"
            )

    # Attach metadata for introspection
    setattr(wrapper, "_agent_decorator", "auto")  # noqa: B010
    setattr(wrapper, "_needs_generation", needs_generation)  # noqa: B010
    setattr(wrapper, "_plan_strategy", strategy)  # noqa: B010
    # Expose the mutable flag so @no_trace applied after @strategy can flip it
    setattr(wrapper, "_tracing_enabled", _tracing_enabled)  # noqa: B010

    return wrapper


def create_sync_agent_method_wrapper(
    original_func: Callable[..., Any],
    *,
    needs_tracing: bool,
    cached_source_code: str | None = None,
) -> Callable[..., Any]:
    """Create a sync wrapper for a sync agent method (tracing only).

    Sync methods on Agent subclasses cannot use the async wrapper directly —
    making them awaitable would change the calling convention. This wrapper:

    - Fires before/after_agent_call hooks so sync helpers produce AGENT spans.
    - Pushes/pops the agent call stack so nested calls have correct parent linkage.
    - Skips agent_call middleware (middleware is async and would need an event loop).
    - Skips the generation path entirely (sync methods can't await an LLM).

    When `self.runtime` is not yet set (i.e. inside `Agent.__init__` while
    `_resolve_llm`/`_resolve_truncation`/etc. are running) the wrapper short-circuits
    to the original function with no hook firing. This mirrors the async wrapper's
    fall-through at the bottom of `create_agent_method_wrapper`.

    Args:
        original_func: The original sync function to wrap
        needs_tracing: Whether method should be traced
        cached_source_code: Pre-extracted source code for tracing (optional)

    Returns:
        Wrapped sync function with tracing instrumentation
    """
    # Mirrors the async wrapper: a mutable list lets `@no_trace` applied AFTER
    # this wrapper (outer decorator) flip the flag retroactively.
    _tracing_enabled = [needs_tracing]

    @wraps(original_func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Fast path: no runtime yet (e.g. `Agent.__init__` calling helpers
        # before `self.runtime` is assigned). Skip all instrumentation.
        if not hasattr(self, "runtime"):
            return original_func(self, *args, **kwargs)

        runtime = self.runtime
        call_id = str(uuid4())
        parent_call_id = runtime._agent_call_id

        trace_attrs = _build_trace_attributes(
            needs_generation=False,
            strategy=None,
            cached_source_code=cached_source_code,
        )

        # @no_trace methods propagate the parent's id so children find the
        # nearest traced ancestor — same semantics as the async wrapper.
        _push_agent_call_id(call_id if _tracing_enabled[0] else parent_call_id)

        # Same agent-call event as the async wrapper. The sync wrapper doesn't
        # set _parent_agent_var, so read it directly for is_top_level.
        is_top_level = _parent_agent_var.get() is None
        try:
            self.event_manager.add(
                BeforeAgentCall(
                    method_name=original_func.__name__,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                    is_top_level=is_top_level,
                    needs_generation=False,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("agent-call: BeforeAgentCall emission failed (sync)", exc_info=True)

        hook_context = None
        result = None
        exception_caught: Exception | None = None
        try:
            if _tracing_enabled[0]:
                hook_context = call_before_hook(
                    "before_agent_call",
                    agent=self,
                    method_name=original_func.__name__,
                    args=args,
                    kwargs=kwargs,
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                    **trace_attrs,
                )
            result = original_func(self, *args, **kwargs)
            return result
        except Exception as e:
            exception_caught = e
            raise
        finally:
            try:
                self.event_manager.add(
                    AfterAgentCall(
                        method_name=original_func.__name__,
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        is_top_level=is_top_level,
                        needs_generation=False,
                        success=exception_caught is None,
                        exception_type=(
                            type(exception_caught).__name__
                            if exception_caught is not None
                            else None
                        ),
                    )
                )
            except Exception:  # noqa: BLE001
                logger.debug("agent-call: AfterAgentCall emission failed (sync)", exc_info=True)
            _pop_agent_call_id()
            if hook_context is not None:
                call_after_hook(
                    "after_agent_call",
                    hook_context,
                    agent=self,
                    method_name=original_func.__name__,
                    result=result,
                    exception=exception_caught,
                )

    setattr(wrapper, "_agent_decorator", "auto")  # noqa: B010
    setattr(wrapper, "_needs_generation", False)  # noqa: B010
    setattr(wrapper, "_plan_strategy", None)  # noqa: B010
    setattr(wrapper, "_tracing_enabled", _tracing_enabled)  # noqa: B010
    setattr(wrapper, "_original", original_func)  # noqa: B010

    return wrapper


def _build_trace_attributes(
    needs_generation: bool,
    strategy: Any | None,
    cached_source_code: str | None,
) -> dict[str, Any]:
    """Build trace attributes for instrumentation hooks.

    Args:
        needs_generation: Whether method needs LLM generation
        strategy: Strategy instance (if any)
        cached_source_code: Pre-extracted source code (if any)

    Returns:
        Dict of trace attributes to pass to hooks
    """
    trace_attrs: dict[str, Any] = {}

    if needs_generation and strategy:
        trace_attrs["strategy.name"] = (
            strategy.name if hasattr(strategy, "name") else type(strategy).__name__
        )
        # Strategy-specific config
        if hasattr(strategy, "max_iterations"):
            trace_attrs["strategy.max_iterations"] = strategy.max_iterations
        if hasattr(strategy, "max_retries"):
            trace_attrs["strategy.max_retries"] = strategy.max_retries
    elif cached_source_code:
        # Non-generation method with source code for tracing
        trace_attrs["source_code"] = cached_source_code

    return trace_attrs
