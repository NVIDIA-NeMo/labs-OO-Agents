# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hook-based instrumentation for nooa.

Provides optional callbacks at key execution points, enabling external
instrumentation (OpenTelemetry, logging, etc.) without coupling to core runtime.

Design principles:
- Protocol-based (type-safe, no runtime dependencies)
- Context objects passed between before/after pairs
- Simple types only (no OpenTelemetry imports in core)
- Fast context variable lookup (<1% overhead when not installed)

Usage:
    from nooa.runtime.hooks import set_hooks, get_hooks, InstrumentationHooks

    class MyHooks:
        def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
            return {"start_time": time.time()}

        def after_agent_call(self, agent, method_name, result, exception, context):
            elapsed = time.time() - context["start_time"]
            print(f"{method_name} took {elapsed:.2f}s")
        # ... implement other hooks

    set_hooks(MyHooks())
"""

import logging
import time
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class InstrumentationHooks(Protocol):
    """Optional instrumentation hooks for nooa operations.

    All hook methods are called defensively - exceptions in hooks
    never affect agent execution.

    The context object returned by before_* methods is passed to the
    corresponding after_* method, enabling stateful instrumentation
    (span tracking, timing, etc.) without globals.
    """

    def before_agent_call(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        call_id: str,
        parent_call_id: str | None,
        **extra_kwargs: Any,
    ) -> Any:
        """Called before @strategy method execution.

        Args:
            agent: Agent instance
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments
            call_id: Unique ID for this call
            parent_call_id: Parent call ID (if nested), or None
            **extra_kwargs: Additional context for trace correlation

        Returns:
            Context object passed to after_agent_call
        """
        ...

    def after_agent_call(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        **kwargs: Any,
    ) -> None:
        """Called after @strategy method completes.

        Args:
            agent: Agent instance
            method_name: Name of the method that was called
            result: Return value (if successful)
            exception: Exception (if failed)
            context: Context object from before_agent_call, or ``None``
                     when middleware short-circuits before ``before_agent_call``
                     has a chance to run.
            **kwargs: Additional context for trace correlation
        """
        ...

    def before_generation(
        self,
        agent: Any,
        method_name: str,
        strategy: str,
        generation_id: str,
        parent_generation_id: str | None,
        **kwargs: Any,
    ) -> Any:
        """Called before LLM generation session.

        Args:
            agent: Agent instance
            method_name: Name of the method being generated
            strategy: Strategy name (e.g., "PurePythonStrategy")
            generation_id: Unique ID for this generation session
            parent_generation_id: Parent generation ID (if nested), or None
            **kwargs: Additional context for trace correlation

        Returns:
            Context object passed to after_generation
        """
        ...

    def after_generation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after LLM generation completes.

        Args:
            agent: Agent instance
            method_name: Name of the method that was generated
            result: Return value (if successful)
            exception: Exception (if failed)
            context: Context object from before_generation
            generation_id: Generation session ID
            **kwargs: Additional context for trace correlation
        """
        ...

    def before_code_execution(
        self,
        agent: Any,
        code: str,
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called before executing generated code.

        Args:
            agent: Agent instance
            code: Python code to execute
            execution_id: Unique ID for this execution
            generation_id: ID of the generation session this execution belongs to
            **kwargs: Additional context (e.g., tool_call_id for trace correlation)

        Returns:
            Context object passed to after_code_execution
        """
        ...

    def after_code_execution(
        self,
        agent: Any,
        code: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after code execution completes.

        Args:
            agent: Agent instance
            code: Python code that was executed
            result: Return value (if successful)
            exception: Exception (if failed)
            context: Context object from before_code_execution
            execution_id: Execution ID
            **kwargs: Additional context (e.g., tool_call_id for trace correlation)
        """
        ...

    def before_method_invocation(
        self,
        agent: Any,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        invocation_id: str,
        **extra_kwargs: Any,
    ) -> Any:
        """Called before invoking a generated method.

        This is called when the generated method is actually invoked with arguments,
        after it has been defined via code_execution.

        Args:
            agent: Agent instance
            method_name: Name of the method being invoked
            args: Positional arguments to the method
            kwargs: Keyword arguments to the method
            invocation_id: Unique ID for this invocation
            **extra_kwargs: Additional context for trace correlation

        Returns:
            Context object passed to after_method_invocation
        """
        ...

    def after_method_invocation(
        self,
        agent: Any,
        method_name: str,
        result: Any,
        exception: Exception | None,
        context: Any,
        invocation_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after method invocation completes.

        Args:
            agent: Agent instance
            method_name: Name of the method that was invoked
            result: Return value (if successful)
            exception: Exception (if failed)
            context: Context object from before_method_invocation
            invocation_id: Invocation ID
            **kwargs: Additional context for trace correlation
        """
        ...

    def before_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        execution_id: str,
        generation_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Called before executing a tool (other than code execution).

        This is used for tools like return_result, or any custom tools
        that are not Python code execution.

        Args:
            agent: Agent instance
            tool_name: Name of the tool being executed (e.g., "return_result")
            arguments: Tool arguments as a dictionary
            execution_id: Unique ID for this execution
            generation_id: ID of the generation session this execution belongs to
            **kwargs: Additional context (e.g., tool_call_id for trace correlation)

        Returns:
            Context object passed to after_tool_execution
        """
        ...

    def after_tool_execution(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        exception: Exception | None,
        context: Any,
        execution_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after tool execution completes.

        Args:
            agent: Agent instance
            tool_name: Name of the tool that was executed
            arguments: Tool arguments that were passed
            result: Return value (if successful)
            exception: Exception (if failed)
            context: Context object from before_tool_execution
            execution_id: Execution ID
            **kwargs: Additional context (e.g., tool_call_id for trace correlation)
        """
        ...

    def on_messages_built(
        self,
        agent: Any,
        method_name: str,
        messages: list[dict[str, Any]],
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        """Called after messages are built, before LLM call.

        Captures the rendered message list for tracing (e.g., system message
        snapshots). This is a point-in-time observation, not a before/after pair.

        Args:
            agent: Agent instance
            method_name: Name of the method being generated
            messages: The rendered messages list (OpenAI format)
            generation_id: ID of the current generation session
            **kwargs: Additional context
        """
        ...


# Global hooks storage using context variable
# This allows different hooks per async context (e.g., testing)
_instrumentation_hooks_var: ContextVar[InstrumentationHooks | None] = ContextVar(
    "instrumentation_hooks", default=None
)


def set_hooks(hooks: InstrumentationHooks | None) -> None:
    """Install or remove instrumentation hooks.

    Args:
        hooks: InstrumentationHooks implementation, or None to remove

    Example:
        from nooa.runtime.hooks import set_hooks

        # Install hooks
        set_hooks(MyOpenTelemetryHooks())

        # Remove hooks
        set_hooks(None)
    """
    _instrumentation_hooks_var.set(hooks)


def get_hooks() -> InstrumentationHooks | None:
    """Get current instrumentation hooks.

    Returns:
        Current InstrumentationHooks implementation, or None if not installed

    Note:
        This is a fast context variable lookup with <1% overhead.
    """
    return _instrumentation_hooks_var.get()


def _record_tracing_overhead(elapsed_s: float) -> None:
    """Record time spent inside tracing hook callbacks, if metrics are active."""
    try:
        from nooa.runtime.harness_metrics import get_harness_metrics

        get_harness_metrics().tracing_overhead(elapsed_s)
    except Exception:
        logger.debug("Failed to record tracing overhead metric", exc_info=True)


def call_before_hook(hook_name: str, **kwargs: Any) -> Any:
    """Defensively call a before_* hook method. Returns context or None.

    Args:
        hook_name: Name of the hook method (e.g., "before_generation")
        **kwargs: Arguments to pass to the hook

    Returns:
        Context object from the hook, or None if no hooks or hook fails

    Example:
        hook_context = call_before_hook("before_generation",
            agent=agent, method_name="process", strategy="PurePython",
            generation_id=gen_id, parent_generation_id=None)
    """
    hooks = get_hooks()
    if not hooks:
        return None
    t0 = time.perf_counter()
    try:
        return getattr(hooks, hook_name)(**kwargs)
    except Exception:
        logger.warning("Hook %s failed", hook_name, exc_info=True)
        return None
    finally:
        _record_tracing_overhead(time.perf_counter() - t0)


def call_after_hook(hook_name: str, context: Any, **kwargs: Any) -> None:
    """Defensively call an after_* hook method.

    Args:
        hook_name: Name of the hook method (e.g., "after_generation")
        context: Context object returned by the corresponding before_* hook
        **kwargs: Arguments to pass to the hook (should include result and exception)

    Example:
        call_after_hook("after_generation", hook_context,
            agent=agent, method_name="process", result=result,
            exception=None, generation_id=gen_id)
    """
    hooks = get_hooks()
    if not hooks:
        return
    t0 = time.perf_counter()
    try:
        getattr(hooks, hook_name)(context=context, **kwargs)
    except Exception:
        logger.warning("Hook %s failed", hook_name, exc_info=True)
    finally:
        _record_tracing_overhead(time.perf_counter() - t0)


__all__ = [
    "InstrumentationHooks",
    "set_hooks",
    "get_hooks",
    "call_before_hook",
    "call_after_hook",
]
