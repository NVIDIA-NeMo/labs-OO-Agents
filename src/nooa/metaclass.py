# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic metaclass for automatic method wrapping with ellipsis bodies.

This module provides the AgentMeta metaclass that automatically wraps async methods
with ellipsis bodies for LLM code generation. Tracing can be enabled per-class by
setting _enable_tracing = True as a class attribute.

When a class enables tracing, sync `def` methods are wrapped for tracing too —
they get before/after_agent_call hooks, source-code capture, and proper parent
linkage in the call stack. Sync methods can't generate (no async LLM) and don't
run agent_call middleware (which is async-only).

The metaclass works on any class, not just Agent or GenerationStrategy.
"""

import inspect
from abc import ABCMeta
from collections.abc import Callable
from typing import Any

from nooa.ellipsis_detection import has_ellipsis_body, has_ellipsis_marker


class AgentMeta(ABCMeta):
    """Generic metaclass for auto-wrapping ellipsis methods and tracing helpers.

    Inherits from ABCMeta to support abstract base classes.
    Automatically wraps qualifying methods at class creation time.

    Auto-wrapping criteria:
    - Generatable: async + ellipsis body (all: public, private, dunder)
    - Traceable (async): all async methods (if class sets _enable_tracing = True)
    - Traceable (sync): all sync `def` methods except dunder names
      (if class sets _enable_tracing = True)
    - Traceable (generators): methods containing `yield` get a generator-specific
      wrapper (see `create_async_gen_agent_method_wrapper`). `yield` plus an
      ellipsis body is rejected — see `_reject_ellipsis_generator`.

    Sync methods can't generate or run async middleware; they get tracing only.
    Properties, classmethods, and staticmethods are skipped (they aren't plain
    functions, so `inspect.isfunction` returns False for them).

    The wrapped methods use duck-typing to route calls:
    - If self has 'runtime' attribute → calls self.runtime._call_plan()
    - If first arg has 'agent' and 'events' → calls runtime.execute_nested()
    - Otherwise → calls original function directly

    This makes the metaclass work on any class, not just Agent/Strategy.
    """

    def __new__(
        mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> type:
        """Create new class with auto-wrapped methods.

        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace (methods and attributes)
            **kwargs: Additional kwargs (llm, blocks, etc.) for __init_subclass__

        Returns:
            New class with wrapped methods
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Check if this class enables tracing (convention-based)
        # (guard for dynamic method addition is installed via __setattr__ below)
        # Classes set _enable_tracing = True to opt into tracing
        should_trace_class = getattr(cls, "_enable_tracing", False)

        # Process each method in this class's namespace (not inherited)
        for attr_name in list(namespace.keys()):
            attr_value: Any = namespace.get(attr_name)

            # Skip if already wrapped (e.g. by @strategy decorator)
            if hasattr(attr_value, "_agent_decorator"):
                continue

            # "Is this an ellipsis-bodied generator?" is a question about the
            # SOURCE, so it follows `__wrapped__` and looks inside
            # staticmethod/classmethod descriptors. Kept separate from — and
            # ahead of — the dispatch below, which is a question about the
            # CALLING CONVENTION and must not unwrap. See _reject_ellipsis_generator.
            mcs._reject_ellipsis_generator(name, attr_name, attr_value)

            # Dispatch on `attr_value`'s OWN shape. Unwrapping here would be
            # wrong: a decorator can change what calling the attribute returns.
            # `@contextlib.contextmanager` wraps a generator function but returns
            # a context manager, so dispatching on the unwrapped shape would
            # install the generator wrapper and break `with agent.session()`.
            # A decorator that preserves generator-ness is indistinguishable from
            # one that consumes the generator, so it is left on the plain path.
            #
            # Async generators (`async def` + `yield`) are NOT coroutine functions,
            # so this must come before the iscoroutinefunction check — otherwise
            # they fall through to the sync branch (`inspect.isfunction` is True
            # for them) and get a wrapper whose span closes before the body runs.
            if inspect.isasyncgenfunction(attr_value):
                # === Async generator path (tracing only) ===
                should_trace = mcs._should_trace(attr_name, attr_value, should_trace_class)
                if should_trace:
                    wrapped = mcs._create_async_gen_wrapper(attr_value)
                    type.__setattr__(cls, attr_name, wrapped)

            elif inspect.iscoroutinefunction(attr_value):
                # === Async method path (generation + tracing) ===
                should_generate = mcs._should_generate(attr_name, attr_value)
                if should_generate:
                    # Generation methods are traced by default; @no_trace suppresses the span.
                    should_trace = not getattr(attr_value, "_no_trace", False)
                else:
                    should_trace = mcs._should_trace(attr_name, attr_value, should_trace_class)

                if should_generate or should_trace:
                    strategy = mcs._resolve_strategy(attr_value)
                    wrapped = mcs._create_wrapper(
                        attr_value, should_generate, should_trace, strategy
                    )
                    # Use type.__setattr__ to bypass our own __setattr__ guard
                    # during class construction.
                    type.__setattr__(cls, attr_name, wrapped)

            elif inspect.isfunction(attr_value):
                # === Sync method path (tracing only) ===
                # `inspect.isfunction` is False for property/classmethod/staticmethod
                # descriptors, so those are naturally skipped.
                is_generator = inspect.isgeneratorfunction(attr_value)

                # Skip dunders to avoid wrapping __init__/__init_subclass__/__setattr__/
                # __getattribute__ etc. — risk of infinite recursion or running before
                # the runtime exists. Custom dunders have to be async to be traced.
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue

                if mcs._should_trace(attr_name, attr_value, should_trace_class):
                    make = (
                        mcs._create_sync_gen_wrapper if is_generator else mcs._create_sync_wrapper
                    )
                    type.__setattr__(cls, attr_name, make(attr_value))

        return cls

    def __setattr__(cls, name: str, value: Any) -> None:
        from nooa.runtime.method_guard import guard_dynamic_method

        guard_dynamic_method(cls, name, value)
        super().__setattr__(name, value)

    @staticmethod
    def _should_generate(method_name: str, method_obj: Callable[..., Any]) -> bool:
        """Check if method needs LLM generation.

        Args:
            method_name: Name of the method
            method_obj: Method object

        Returns:
            True if method has ellipsis body (needs generation)
        """
        return has_ellipsis_body(method_obj)

    @staticmethod
    def _should_trace(
        method_name: str, method_obj: Callable[..., Any], should_trace_class: bool
    ) -> bool:
        """Check if method should be traced.

        When should_trace_class is True, all async methods are traced
        (public, private, dunder) unless explicitly opted-out with @no_trace.

        Args:
            method_name: Name of the method
            method_obj: Method object
            should_trace_class: True if class has _enable_tracing = True

        Returns:
            True if method should have tracing hooks
        """
        if not should_trace_class:
            return False  # Class doesn't enable tracing

        if getattr(method_obj, "_no_trace", False):
            return False  # Explicit opt-out

        return True

    @staticmethod
    def _resolve_strategy(method_obj: Callable[..., Any]) -> Any:
        """Get strategy from @strategy decorator or default.

        Args:
            method_obj: Method object

        Returns:
            Strategy instance (from @strategy decorator or None if default)
        """
        # Check for @strategy decorator metadata
        if hasattr(method_obj, "_strategy_override"):
            return getattr(method_obj, "_strategy_override")  # noqa: B009

        # Return None for default - will be resolved at runtime
        # This avoids circular imports during class creation
        return None

    @staticmethod
    def _extract_source_code(func: Callable[..., Any]) -> str | None:
        """Extract source code from function, returning None if unavailable."""
        try:
            return inspect.getsource(func)
        except (OSError, TypeError):
            return None

    @staticmethod
    def _reject_ellipsis_generator(class_name: str, method_name: str, attr_value: Any) -> None:
        """Reject a generator method whose body is an ellipsis stub.

        An ellipsis body means "the LLM writes this". Generation only happens on
        the coroutine path, and a generator function is never a coroutine
        function — so an ellipsis-bodied generator would silently skip
        generation and run as an ordinary generator yielding whatever its body
        literally contains. The two markers contradict each other, so this is
        always an authoring mistake rather than a supported combination.

        This is a question about the source, not about the calling convention,
        so unlike the dispatch it deliberately looks through `__wrapped__` and
        into staticmethod/classmethod descriptors — neither of which reaches the
        dispatch branches, and both of which can hide the contradiction.
        Ellipsis detection unwraps too, so the two agree on what they inspect.
        """
        # staticmethod/classmethod are not functions; inspect the wrapped one.
        method_obj = (
            attr_value.__func__
            if isinstance(attr_value, (staticmethod, classmethod))
            else attr_value
        )
        # Gate on the object's own type, not merely `callable`. This runs on
        # every namespace entry, and `has_ellipsis_body` -> `inspect.getsource`
        # rejects anything that is not a real function: `functools.partial` in
        # particular passes `isgeneratorfunction` (which unwraps partials) but
        # blows up in `getsource` (which does not), crashing class creation.
        if not (inspect.isfunction(method_obj) or inspect.ismethod(method_obj)):
            return

        unwrapped = inspect.unwrap(method_obj)
        if not (inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)):
            return

        if not has_ellipsis_marker(method_obj):
            return

        is_async = inspect.isasyncgenfunction(unwrapped)
        shape = (
            "an async generator (async def with yield)"
            if is_async
            else "a generator (def with yield)"
        )
        raise TypeError(
            f"{class_name}.{method_name} is {shape} with the `...` generation marker, "
            f"but generator methods cannot be LLM-generated: generation methods "
            f"produce one final result, so generated streams are not supported.\n"
            f"Either remove `...` and write the deterministic generator body in full, "
            f"or remove `yield` and make it a plain `async def` generation method."
        )

    @staticmethod
    def _create_async_gen_wrapper(original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create a tracing wrapper for an async generator method.

        Generators can't generate and can't run async agent_call middleware, so
        this is tracing-only. The wrapper scopes the call id to each resumption
        of the body — see `create_async_gen_agent_method_wrapper`.
        """
        from nooa.runtime.method_wrapper import create_async_gen_agent_method_wrapper

        cached_source_code = AgentMeta._extract_source_code(original_func)
        return create_async_gen_agent_method_wrapper(
            original_func,
            needs_tracing=True,
            cached_source_code=cached_source_code,
        )

    @staticmethod
    def _create_sync_gen_wrapper(original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create a tracing wrapper for a sync generator method.

        Same span shape as the async generator wrapper — see
        `create_sync_gen_agent_method_wrapper`.
        """
        from nooa.runtime.method_wrapper import create_sync_gen_agent_method_wrapper

        cached_source_code = AgentMeta._extract_source_code(original_func)
        return create_sync_gen_agent_method_wrapper(
            original_func,
            needs_tracing=True,
            cached_source_code=cached_source_code,
        )

    @staticmethod
    def _create_sync_wrapper(original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create a tracing wrapper for a sync method.

        Sync methods can't generate (no async LLM call) and can't run async
        agent_call middleware, so this is tracing-only. Source code is captured
        at class creation time for the span.
        """
        from nooa.runtime.method_wrapper import create_sync_agent_method_wrapper

        cached_source_code = AgentMeta._extract_source_code(original_func)
        return create_sync_agent_method_wrapper(
            original_func,
            needs_tracing=True,
            cached_source_code=cached_source_code,
        )

    @staticmethod
    def _create_wrapper(
        original_func: Callable[..., Any],
        needs_generation: bool,
        needs_tracing: bool,
        strategy: Any,
    ) -> Callable[..., Any]:
        """Create wrapper for async methods (generation or tracing-only).

        Delegates to the shared wrapper logic in method_wrapper.py.

        Args:
            original_func: Original async function
            needs_generation: Whether method needs LLM generation
            needs_tracing: Whether method should be traced
            strategy: Strategy instance to use

        Returns:
            Wrapped async function
        """
        # Import shared wrapper logic
        from nooa.runtime.method_wrapper import create_agent_method_wrapper

        # Extract source code once at class creation time (cached via closure)
        # Only for tracing-only methods (generation methods don't need source code in traces)
        cached_source_code = None
        if not needs_generation and needs_tracing:
            cached_source_code = AgentMeta._extract_source_code(original_func)

        # Use shared wrapper logic
        wrapper = create_agent_method_wrapper(
            original_func,
            needs_generation=needs_generation,
            needs_tracing=needs_tracing,
            strategy=strategy,
            cached_source_code=cached_source_code,
        )

        # Attach additional metadata for introspection (shared wrapper sets some already)
        setattr(wrapper, "_original", original_func)  # noqa: B010

        return wrapper


def no_trace(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to opt-out of tracing for public methods.

    Use this decorator on public async methods that should NOT be traced,
    even though they would normally be traced by the metaclass.

    Note: This only disables tracing, NOT generation. If the method has an
    ellipsis body, it will still be generated by the LLM.

    Args:
        func: Function to mark as non-traceable

    Returns:
        Same function with _no_trace marker

    Example:
        class MyAgent(Agent, llm=llm):
            @no_trace
            async def utility_method(self):
                ...  # Generated but NOT traced
    """
    setattr(func, "_no_trace", True)  # noqa: B010
    # If @no_trace is applied *after* @strategy (i.e. as the outer decorator),
    # the wrapper already exists with _tracing_enabled baked in. Flip it now
    # so both decorator orderings suppress hooks correctly:
    #   @strategy @no_trace  (no_trace inner — @strategy sees _no_trace at creation time)
    #   @no_trace @strategy  (no_trace outer — this branch updates the existing wrapper)
    tracing_enabled = getattr(func, "_tracing_enabled", None)
    if tracing_enabled is not None:
        tracing_enabled[0] = False
    return func
