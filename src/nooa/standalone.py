# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone generation functions — LLM-powered functions with no agent state.

A standalone generation function is an async function decorated with ``@strategy``
that has no ``self`` parameter.  Each call gets a fresh agent stub (no shared state,
history disappears after the call).  Context blocks can be supplied via the decorator:

    @strategy(CodeActStrategy(), ScopedContext(context={"role": "expert"}), llm=llm)
    async def summarise(text: str) -> str:
        \"\"\"Summarise {text} in one sentence.\"\"\"
        ...

    result = await summarise("long article...")

Design constraints:
- No doc(self) / system_prompt — framework_blocks is empty.
- No persistent state — EventManager and ContextManager are fresh each invocation.
- History disappears — each call uses its own agent instance.
- Full strategy support — same CodeActStrategy / PredictStrategy machinery as agents.
- exec_globals correctness — the stub agent class is rooted in the function's own
  module so ``filter_module_globals`` exposes the caller's types to generated code.
"""

import contextvars
import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import Any
from uuid import uuid4

# Set by ``install_atif`` to allow the standalone wrapper to cascade an active
# ATIF exporter onto the fresh child agent's EventManager. Without this hook,
# events fired inside a ``@strategy(...)`` function called from an instrumented
# agent would silently bypass the parent's trajectory.
#
# Type-annotated as ``Any`` to avoid an import cycle: ``nooa.atif``
# would import this module via ``standalone`` users, and adding a reverse
# import here would create a cycle.
_atif_exporter_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_atif_exporter_var", default=None
)

# Cache of per-module _StandaloneAgent subclasses, keyed by module __name__.
# Each subclass has __module__ set to the function's module so that
# ActorRuntime.execute_code's filter_module_globals picks up the right symbols.
_agent_cls_cache: dict[str, type] = {}

# Per-Python-process token, generated once at module import.
#
# Blended into the standalone ``_agent_id`` (alongside the decorated function's
# identity) so the resulting ``prompt_cache_key`` shards behave correctly:
#
# - Within one process, all calls to the same standalone function share a
#   shard — fan-out like ``asyncio.gather(*(summarise(t) for t in texts))``
#   keeps shared-prefix cache hits.
# - Two separate Python processes running the same code get different shards
#   — 50 parallel benchmark trials (cybergym-style) spread across 50 shards
#   instead of compounding load on one (the ~15 req/min per-shard ceiling).
#
# Caveat: ``multiprocessing.fork`` workers inherit the parent's token; they
# would collide on shards. Spawn-style multiprocessing and separate
# ``python script.py`` invocations are unaffected.
_PROCESS_AGENT_ID = uuid4().hex[:8]


def _get_agent_cls(module_name: str) -> type:
    """Return a _StandaloneAgent subclass whose ``__module__`` is *module_name*.

    The class is cached so we never create more than one per module.
    """
    if module_name in _agent_cls_cache:
        return _agent_cls_cache[module_name]

    from nooa.config import TruncationConfig
    from nooa.context_blocks.render_config import RenderConfig
    from nooa.runtime.actor import ActorRuntime
    from nooa.runtime.context_manager import ContextManager
    from nooa.runtime.event_manager import EventManager

    class _StandaloneAgent:
        """Minimal agent stub: no framework blocks, fresh state per call."""

        event_query = None
        _execution_config = None

        def __init__(self, llm: Any, agent_id: str) -> None:
            self._llm = llm
            self._truncation = TruncationConfig()
            self.render_config = RenderConfig()
            self.event_manager = EventManager()
            self.context_manager = ContextManager()
            self.runtime = ActorRuntime(self)
            # Used by ActorRuntime to default prompt_cache_key per agent.
            # The wrapper passes a deterministic id derived from the
            # decorated function's identity so repeated calls share a shard.
            self._agent_id = agent_id

    cls = type("_StandaloneAgent", (_StandaloneAgent,), {})
    # Setting __module__ makes inspect.getmodule(cls) return the function's module,
    # so execute_code's filter_module_globals exposes the caller's types.
    cls.__module__ = module_name
    _agent_cls_cache[module_name] = cls
    return cls


def _make_adapter(func: Callable[..., Any], strategy: Any = None) -> Callable[..., Any]:
    """Create a method-like adapter (with *self*) from a standalone function.

    ActorRuntime._execute_with_generation expects a method whose first parameter
    is ``self`` (it skips it when building CurrentCall).  This adapter:

    - prepends ``self`` to the signature so the runtime parses args correctly
    - has an ``...`` body so ``has_ellipsis_body`` returns True
    - shares the original function's ``__globals__`` so ``get_type_hints`` can
      resolve forward references (PEP 563 / ``from __future__ import annotations``)
    - carries all ``_plan_*`` / ``_strategy_*`` metadata the runtime reads
    """
    orig_sig = inspect.signature(func)
    new_params = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *orig_sig.parameters.values(),
    ]
    new_sig = orig_sig.replace(parameters=new_params)

    # Template coroutine with an ellipsis body — has_ellipsis_body detects this.
    # Reuse func's own __globals__ so get_type_hints resolves annotations in
    # the same scope the function was defined in. Works for module-level
    # functions and for REPL-defined functions (where func.__module__ is None).
    async def _tmpl(self: Any, *args: Any, **kwargs: Any) -> Any: ...

    adapted = types.FunctionType(_tmpl.__code__, func.__globals__, func.__name__)

    adapted.__doc__ = func.__doc__
    adapted.__name__ = func.__name__
    adapted.__qualname__ = func.__qualname__
    adapted.__annotations__ = dict(func.__annotations__)
    adapted.__signature__ = new_sig  # type: ignore[attr-defined]
    adapted.__module__ = func.__module__

    # Metadata that _execute_with_generation reads off the method object
    adapted._plan_llm = getattr(func, "_plan_llm", None)  # type: ignore[attr-defined]
    adapted._plan_strategy = strategy or getattr(func, "_plan_strategy", None)  # type: ignore[attr-defined]
    adapted._strategy_context = getattr(func, "_strategy_context", None)  # type: ignore[attr-defined]
    adapted._strategy_events = getattr(func, "_strategy_events", None)  # type: ignore[attr-defined]
    adapted._needs_generation = True  # type: ignore[attr-defined]
    adapted._agent_decorator = "auto"  # type: ignore[attr-defined]

    return adapted


def create_standalone_wrapper(
    func: Callable[..., Any],
    strategy: Any,
    llm: Any,
) -> Callable[..., Any]:
    """Wrap a standalone generation function so it can be called directly.

    Each invocation creates a fresh agent stub (no shared state, history resets).
    LLM resolution order: ``@strategy(llm=…)`` → parent-agent cascade → RuntimeError.

    Args:
        func: Original async function with an ellipsis body.
        strategy: GenerationStrategy instance resolved from the decorator.
        llm: Optional explicit LLM client from ``@strategy(..., llm=...)``.
            Always a client, never a resolver callable: standalone functions
            have no agent instance to bind one against, so ``@strategy``
            rejects callables here at decoration time.

    Returns:
        Async callable with the same signature as *func*.
    """

    # Build the adapter once — func and strategy are decoration-time constants.
    _adapted = _make_adapter(func, strategy=strategy)

    # Deterministic agent_id per (process, decorated function). See
    # ``_PROCESS_AGENT_ID`` above for the design rationale; in short:
    #
    #   - same function, same process → same shard (fan-out cache hits)
    #   - different functions, same process → different shards (prefixes
    #     already diverge, no benefit to sharing)
    #   - same function, different processes → different shards (50
    #     parallel ``python script.py`` invocations spread their load
    #     instead of stacking on one shard)
    _standalone_agent_id = f"standalone:{_PROCESS_AGENT_ID}:{func.__module__}:{func.__qualname__}"

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        resolved_llm = llm
        if resolved_llm is None:
            # Cascade: inherit LLM from a calling agent if we're inside one
            from nooa.runtime.context_vars import _parent_agent_var

            parent = _parent_agent_var.get()
            if parent is not None:
                resolved_llm = getattr(parent, "_llm", None)
        if resolved_llm is None:
            raise RuntimeError(
                f"No LLM client for standalone function '{func.__name__}'. "
                f"Pass llm=<client> to @strategy(..., llm=<client>)."
            )

        # Fresh agent per call — no shared state, history resets automatically
        agent_cls = _get_agent_cls(func.__module__)
        agent = agent_cls(llm=resolved_llm, agent_id=_standalone_agent_id)

        # ATIF exporter cascade: if an exporter is active in the surrounding
        # context, attach it to this child agent's
        # EventManager so the child's events flow into the parent's
        # trajectory as a subagent_trajectories[] entry. Detach on exit.
        _attached_exporter: Any = None
        _candidate = _atif_exporter_var.get()
        if _candidate is not None and hasattr(_candidate, "_attach_child"):
            try:
                _candidate._attach_child(agent.event_manager, child_agent_name=func.__name__)
                _attached_exporter = _candidate
            except Exception:  # noqa: BLE001
                # Cascade is best-effort; never break the wrapped call.
                _attached_exporter = None
        try:
            return await agent.runtime._execute_with_generation(
                _adapted, args, kwargs, func.__name__
            )
        finally:
            if _attached_exporter is not None:
                try:
                    _attached_exporter._detach_child(agent.event_manager)
                except Exception:  # noqa: BLE001
                    pass

    wrapper._standalone = True  # type: ignore[attr-defined]
    wrapper._needs_generation = True  # type: ignore[attr-defined]
    wrapper._plan_strategy = strategy  # type: ignore[attr-defined]
    wrapper._plan_llm = llm  # type: ignore[attr-defined]
    return wrapper
