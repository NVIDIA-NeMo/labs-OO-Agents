# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Late-bound per-method LLM overrides for ``@strategy(llm=...)``.

``@strategy(llm=my_client)`` pins a method to a concrete client at import
time, so every instance of the class shares it forever. Two later-bound
spellings avoid that:

* a **registry alias / litellm model string**, resolved lazily on the first
  generation call for each agent instance through
  :func:`nooa.unifiedllm.get_llm_client` and cached on the instance::

    class SupportAgent(Agent, llm="gpt-5"):
        @strategy(llm="gpt-5-mini")
        async def summarize(self, text: str) -> str: ...

  Strings need no client object at import time — a module that only
  *declares* models doesn't construct any LLM I/O machinery until the
  method is actually called, matching the framework's lazy-I/O pattern.
  The resolved client is cached per agent instance, so repeated calls
  share one client rather than constructing a new one per call.

* a **callable** taking the agent instance, invoked on every generation
  call for that method — so return an existing client, don't construct
  one::

    class Researcher(Agent, llm=fast):
        @strategy(llm=lambda self: self.big_model)
        async def analyze(self, doc: str) -> str: ...

        @strategy(llm=lambda self: self.big if self.retries > 2 else self.fast)
        async def solve(self, problem: str) -> str: ...

Standalone ``@strategy`` functions (no ``self`` parameter) have no instance
to bind against, so they reject callables at decoration time — but strings
are fine, since a string resolves without an instance. See
:func:`validate_method_llm_spec`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

# Attribute on an agent instance holding its resolved string-alias clients:
# {alias: UnifiedLLM}. Keyed on the instance (not the method) so a class
# with several methods sharing an alias constructs one client.
_INSTANCE_LLM_CACHE_ATTR = "_strategy_llm_alias_cache"


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


def resolve_alias(
    alias: str,
    cache: dict[str, Any] | None,
    method_name: str,
    *,
    origin: str = "@strategy(llm=...)",
) -> UnifiedLLM:
    """Resolve a registry alias / litellm model string via ``get_llm_client``.

    Shared by every call site that accepts an alias string — the
    ``@strategy(llm=...)`` decorator (caching on the agent instance, see
    :func:`_resolve_alias`) and standalone ``@strategy`` functions (caching
    in the wrapper closure, since a fresh agent stub is built per call).
    One resolution path means one error message, one cache policy, and one
    place to change if client construction ever grows validation.

    Args:
        alias: Registry key or litellm-supported model string.
        cache: Caller-owned ``{alias: client}`` dict, or ``None`` to resolve
            without caching (for owners that cannot host a dict).
        method_name: Method (or standalone function) name, for error messages.
        origin: Human-readable source of the alias, for error messages.

    Returns:
        The resolved :class:`~nooa.unifiedllm.UnifiedLLM`.

    Raises:
        RuntimeError: If ``get_llm_client`` raises. The original exception
            is chained, but the message names the method and the alias so a
            typo'd model name points at the line that named it.
        TypeError: If resolution returns something that is not a client.
    """
    if cache is not None:
        cached = cache.get(alias)
        if cached is not None:
            return cast("UnifiedLLM", cached)

    from nooa.unifiedllm import get_llm_client

    try:
        client = get_llm_client(alias)
    except Exception as exc:
        raise RuntimeError(
            f"The {origin} alias {alias!r} for '{method_name}' could not be "
            f"resolved by get_llm_client: {type(exc).__name__}: {exc}"
        ) from exc

    if not _is_llm_client(client):
        raise TypeError(
            f"The {origin} alias {alias!r} for '{method_name}' resolved to "
            f"{type(client).__name__}, not a UnifiedLLM instance."
        )

    if cache is not None:
        cache[alias] = client
    return client


def _resolve_alias(alias: str, agent: Any, method_name: str) -> UnifiedLLM:
    """Resolve a registry alias / litellm model string against *agent*.

    The client is constructed once per (agent instance, alias) pair and
    cached on the instance, so a second call reuses the client (and its
    token budgeting bookkeeping) rather than rebuilding it.

    Args:
        alias: Registry key or litellm-supported model string.
        agent: The agent instance to cache the client on.
        method_name: Method name, for error messages.

    Returns:
        The resolved :class:`~nooa.unifiedllm.UnifiedLLM`.
    """
    cache = getattr(agent, _INSTANCE_LLM_CACHE_ATTR, None)
    if isinstance(cache, dict):
        return resolve_alias(alias, cache, method_name)

    # Fresh instance, or a duck-typed stub that refuses new attributes
    # (e.g. __slots__). Resolve without caching rather than failing the call.
    try:
        cache = {}
        setattr(agent, _INSTANCE_LLM_CACHE_ATTR, cache)
    except (AttributeError, TypeError):
        return resolve_alias(alias, None, method_name)
    return resolve_alias(alias, cache, method_name)


def validate_method_llm_spec(spec: Any, func_name: str, *, standalone: bool = False) -> None:
    """Validate an ``@strategy(llm=...)`` value at decoration time.

    Catches the mistakes that would otherwise surface much later, in the
    middle of a generation call:

    - a value that is not a client, alias string, or callable (e.g. an int
      or a list)
    - an empty or non-string passed where an alias was meant
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

    if isinstance(spec, str):
        if not spec:
            raise TypeError(
                f"@strategy(llm=...) on '{func_name}' got an empty string. Pass a "
                f"registry alias or litellm model string, or a client / callable."
            )
        return

    if callable(spec):
        if standalone:
            raise TypeError(
                f"@strategy(llm=...) on standalone function '{func_name}' must be a "
                f"UnifiedLLM instance or an alias string, not a callable. Callables "
                f"resolve against an agent instance, and standalone functions have "
                f"none. Pass a client or a string alias, or make '{func_name}' a "
                f"method on an Agent subclass."
            )
        return

    raise TypeError(
        f"@strategy(llm=...) on '{func_name}' must be a UnifiedLLM instance, a "
        f"registry alias / model string, or a callable taking the agent and "
        f"returning one, got {type(spec).__name__}. For a per-instance model, use "
        f"llm=lambda self: self.my_client."
    )


def resolve_method_llm(spec: Any, agent: Any, method_name: str) -> UnifiedLLM:
    """Resolve a ``@strategy(llm=...)`` value against an agent instance.

    Args:
        spec: A ``UnifiedLLM``, a registry alias / litellm model string
            (resolved lazily and cached per agent instance), or a callable
            taking the agent instance.
        agent: The agent the method is executing on.
        method_name: Method name, for error messages.

    Returns:
        The resolved ``UnifiedLLM``.

    Raises:
        TypeError: If *spec* is not a client, string, or callable, or if a
            callable returns something that is not a ``UnifiedLLM``.
        RuntimeError: If a string alias cannot be resolved by
            ``get_llm_client``, or if the callable itself raises. The original
            exception is chained, but the message names the method and makes
            clear the failure came from the ``llm=`` value rather than from
            generation — a bare AttributeError from a typo'd attribute or a
            404 from a typo'd model name is otherwise very hard to place.
    """
    if _is_llm_client(spec):
        return cast("UnifiedLLM", spec)

    if isinstance(spec, str):
        return _resolve_alias(spec, agent, method_name)

    if not callable(spec):
        raise TypeError(
            f"@strategy(llm=...) for '{method_name}' must be a UnifiedLLM instance, a "
            f"registry alias / model string, or a callable returning one, got "
            f"{type(spec).__name__}."
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


__all__ = ["resolve_alias", "resolve_method_llm", "validate_method_llm_spec"]
