# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Late-bound per-method LLM overrides for ``@strategy(llm=...)``.

``@strategy(llm=my_client)`` pins a method to a concrete client at import
time, so every instance of the class shares it forever. Passing a
**callable** instead defers the choice to generation time, where the agent
instance is in hand::

    class Researcher(Agent, llm=fast):
        @strategy(llm=lambda self: self.big_model)
        async def analyze(self, doc: str) -> str: ...

        @strategy(llm=lambda self: self.big if self.retries > 2 else self.fast)
        async def solve(self, problem: str) -> str: ...

The callable receives the agent instance and must return a
:class:`~nooa.unifiedllm.UnifiedLLM`. It is invoked on **every** generation
call for that method — so return an existing client, don't construct one.
Re-resolving per call is what makes the second example above work.

Standalone ``@strategy`` functions (no ``self`` parameter) have no instance
to bind against, so they reject callables at decoration time — see
:func:`validate_method_llm_spec`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


def _is_llm_client(spec: Any) -> bool:
    """True if *spec* is an LLM client rather than a resolver callable.

    Subclassing :class:`~nooa.unifiedllm.UnifiedLLM` is the normal case, but
    the framework has never *required* it — ``Agent(llm=...)`` accepts any
    object with the client interface, and several fakes in the test-suite
    duck-type it. Narrowing to a strict ``isinstance`` here would break them,
    so fall back to probing for ``acall`` (one of the two abstract methods on
    ``UnifiedLLM``; ``call`` is the sync twin, but generation is async).

    The ``isinstance`` check runs first so that a ``UnifiedLLM`` subclass
    which also defines ``__call__`` is still classified as a client, never as
    a resolver. Resolvers are plain functions and have no ``acall``, so the
    two categories don't overlap in practice.

    Classes are excluded before the probe: ``hasattr(MyLLMClass, "acall")`` is
    True for the *unbound* function on the class, so passing the class itself
    (a plausible "factory" spelling) would sail through validation and then
    fail on an unbound-method call deep inside generation — exactly the late
    failure this module exists to prevent. A class is not a client; if it's
    meant as a resolver it will be treated as one and called, which is the
    behaviour a factory spelling implies anyway.
    """
    import inspect

    from nooa.unifiedllm import UnifiedLLM

    if isinstance(spec, UnifiedLLM):
        return True
    if inspect.isclass(spec):
        return False
    return hasattr(spec, "acall")


def validate_method_llm_spec(spec: Any, func_name: str, *, standalone: bool = False) -> None:
    """Validate an ``@strategy(llm=...)`` value at decoration time.

    Catches the two mistakes that would otherwise surface much later, in
    the middle of a generation call:

    - a value that is neither a ``UnifiedLLM`` nor callable (e.g. a model
      name string — there is no alias registry; pass a client or a callable)
    - a callable on a standalone function, which has no instance to bind to

    Args:
        spec: The value passed as ``@strategy(llm=...)``.
        func_name: Decorated function name, for the error message.
        standalone: True if the decorated function has no ``self`` parameter.

    Raises:
        TypeError: If *spec* is unusable. Raised at decoration time, so the
            failure points at the offending ``@strategy`` line.
    """
    if _is_llm_client(spec):
        return

    if callable(spec):
        if standalone:
            raise TypeError(
                f"@strategy(llm=...) on standalone function '{func_name}' must be a "
                f"UnifiedLLM instance, not a callable. Callables resolve against an "
                f"agent instance, and standalone functions have none. Pass a client "
                f"directly, or make '{func_name}' a method on an Agent subclass."
            )
        return

    raise TypeError(
        f"@strategy(llm=...) on '{func_name}' must be a UnifiedLLM instance or a "
        f"callable taking the agent and returning one, got {type(spec).__name__}. "
        f"For a per-instance model, use llm=lambda self: self.my_client."
    )


def resolve_method_llm(spec: Any, agent: Any, method_name: str) -> UnifiedLLM:
    """Resolve a ``@strategy(llm=...)`` value against an agent instance.

    Args:
        spec: A ``UnifiedLLM``, or a callable taking the agent instance.
        agent: The agent the method is executing on.
        method_name: Method name, for error messages.

    Returns:
        The resolved ``UnifiedLLM``.

    Raises:
        TypeError: If *spec* is not a client or callable, or if a callable
            returns something that is not a ``UnifiedLLM``.
        RuntimeError: If the callable itself raises. The original exception
            is chained, but the message names the method and makes clear the
            failure came from the ``llm=`` callable rather than from
            generation — a bare AttributeError from a typo'd attribute is
            otherwise very hard to place.
    """
    if _is_llm_client(spec):
        return cast("UnifiedLLM", spec)

    if not callable(spec):
        raise TypeError(
            f"@strategy(llm=...) for '{method_name}' must be a UnifiedLLM instance or "
            f"a callable returning one, got {type(spec).__name__}."
        )

    try:
        resolved = spec(agent)
    except Exception as exc:
        raise RuntimeError(
            f"The @strategy(llm=...) callable for '{method_name}' raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not _is_llm_client(resolved):
        raise TypeError(
            f"The @strategy(llm=...) callable for '{method_name}' must return a "
            f"UnifiedLLM instance, got {type(resolved).__name__}."
        )

    return cast("UnifiedLLM", resolved)


__all__ = ["resolve_method_llm", "validate_method_llm_spec"]
