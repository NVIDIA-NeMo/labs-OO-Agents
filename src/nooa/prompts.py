# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt inspection utilities for nooa.

Inspect the exact prompts that would be sent to the LLM — with real agent
state and real arguments — without triggering an actual LLM call:

    agent = MyAgent(llm=client)
    # ... lots of things happen
    await nooa.print_prompt(agent.analyze, my_data)
"""

import inspect
import sys
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# PromptData
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptData:
    """All prompt sections for a single agent method call."""

    system_prompt: str | None
    """Formatted system message."""

    task_prompt: str
    """The task user-message from the strategy's _build_task_message template."""

    inspect_prefill: str | None
    """InspectInputsPrefill code, or user-configured prefill for PurePython."""

    pre_ellipsis: str | None
    """Setup code from the method body that appears before the ``...`` marker."""

    strategy_name: str
    """e.g. 'CodeActStrategy'"""

    method_path: str
    """e.g. 'MyAgent.analyze'"""


# ---------------------------------------------------------------------------
# Task prompt
# ---------------------------------------------------------------------------
async def _get_task_prompt(runtime: Any, strategy: Any, call: Any) -> str:
    """Build the task prompt text for *strategy* and *call*.

    Reads ``_build_task_message.__doc__`` from the *original* (unwrapped)
    function on the strategy class and expands it via the runtime's own
    ``expand_variables``.  Falls back to the raw method docstring when the
    strategy has no such method, or when the rendered template is empty.
    """
    build_fn = getattr(type(strategy), "_build_task_message", None)
    if build_fn is not None:
        original = getattr(build_fn, "_original", build_fn)
        template = inspect.getdoc(original) or ""
        if template:
            result = await runtime.expand_variables(template, {"original_call": call})
            if result:
                return str(result)
    return call.docstring or "(no docstring)"


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def _get_prefill(strategy: Any, call: Any, agent: Any) -> tuple[str | None, str | None]:
    """Return ``(inspect_code, pre_ellipsis_code)`` for *strategy* and *call*.

    Mirrors the runtime prefill path (``CodeActStrategy._run_prefill``): the
    configured ``CodeActConfig.prefill`` plugin drives the inspect prefill,
    resolved against the agent's truncation config. ``prefill`` may be ``None``
    (inspect prefill disabled) or a custom ``Prefill`` instance.

    For strategies that don't have explicit prefill support, ``pre_ellipsis``
    is still returned so that code the user wrote is never silently discarded.
    """
    from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
    from nooa.strategies.codeact import CodeActStrategy
    from nooa.strategies.codeact_lite import CodeActLiteStrategy
    from nooa.strategies.pure_python import PurePythonStrategy

    pre_ellipsis = call.pre_ellipsis_code

    if isinstance(strategy, (CodeActStrategy, CodeActLiteStrategy)):
        prefill = strategy.config.prefill
        if prefill is None:
            return None, pre_ellipsis
        truncation_config = getattr(agent, "_truncation", DEFAULT_TRUNCATION_CONFIG)
        return prefill.get_code(call, config=truncation_config), pre_ellipsis

    if isinstance(strategy, PurePythonStrategy):
        # PurePythonStrategy always has a prefill (InspectInputsPrefill by default)
        return strategy.prefill.get_code(call), pre_ellipsis

    return None, pre_ellipsis


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
async def _build_prompt_data_from_agent(
    agent: Any,
    wrapper: Any,
    original_func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> PromptData:
    """Build :class:`PromptData` from a live agent instance.

    .. warning::
        Calls ``agent.runtime._build_messages()``, which updates the agent's
        ``DynamicContext`` resolved-value cache as a side effect.  Avoid
        calling this mid-session with speculative arguments if your agent
        relies on cached DynamicContext values between turns.
    """
    from nooa.runtime.actor import _current_strategy_var
    from nooa.strategies import get_default_strategy
    from nooa.strategies.current_call import CurrentCall

    strategy = getattr(wrapper, "_plan_strategy", None) or get_default_strategy()
    call = CurrentCall.from_method(original_func, args=args, kwargs=kwargs)

    # Pre-expand docstring with call arguments (mirrors actor.py behavior).
    # Without this, {param} placeholders in the user's docstring remain literal
    # when embedded via {original_call.docstring} in the strategy template.
    if call.docstring and call.kwargs:
        import dataclasses

        expanded_doc = await agent.runtime.expand_variables(
            call.docstring, extra_context=call.kwargs, error_mode="silent"
        )
        call = dataclasses.replace(call, docstring=expanded_doc)

    # _build_messages falls back to _current_strategy_var when the method
    # has no explicit _plan_strategy (i.e. uses the default strategy).
    token = _current_strategy_var.set(strategy)
    try:
        messages = await agent.runtime._build_messages(wrapper, args, kwargs)
    finally:
        _current_strategy_var.reset(token)

    system_prompt = next(
        (m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )

    task_prompt = await _get_task_prompt(agent.runtime, strategy, call)
    inspect_code, pre_ellipsis = _get_prefill(strategy, call, agent)

    return PromptData(
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        inspect_prefill=inspect_code,
        pre_ellipsis=pre_ellipsis,
        strategy_name=type(strategy).__name__,
        method_path=f"{type(agent).__name__}.{call.method_name}",
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_prompt_data(data: PromptData) -> str:
    """Format *data* as a human-readable plain-text string."""
    parts: list[str] = []

    if data.system_prompt is not None:
        class_name = data.method_path.split(".")[0]
        parts.append(f"=== SYSTEM PROMPT  [{class_name}] ===")
        parts.append("")
        parts.append(data.system_prompt)
        parts.append("")

    parts.append(f"=== TASK PROMPT  [{data.method_path}] ===")
    parts.append("")
    parts.append(data.task_prompt)
    parts.append("")

    prefill_parts: list[str] = []
    if data.inspect_prefill is not None:
        prefill_parts.append(data.inspect_prefill)
    if data.pre_ellipsis is not None:
        if prefill_parts:
            prefill_parts.append("# --- pre-ellipsis ---")
        prefill_parts.append(data.pre_ellipsis)

    if prefill_parts:
        parts.append(f"=== PREFILL  [{data.strategy_name}] ===")
        parts.append("")
        parts.append("\n".join(prefill_parts))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def build_prompt_data(method: Any, /, *args: Any, **kwargs: Any) -> PromptData:
    """Build prompt data for *method* without printing.

    Args:
        method: Bound generation method on an :class:`~nooa.Agent` instance.
        *args, **kwargs: Arguments the method would be called with.

    Returns:
        :class:`PromptData` with all sections populated.

    Note:
        Calling this on a method that has no ``...`` body (not a generation
        method) will still return a ``PromptData``, but the output may not be
        meaningful since non-generation methods are not dispatched to the LLM.

    Example::

        data = await nooa.build_prompt_data(agent.analyze, my_data)
        print(data.task_prompt)
    """
    if not (hasattr(method, "__self__") and hasattr(method, "__func__")):
        raise TypeError(
            f"build_prompt_data() requires a bound agent method, got {type(method).__name__!r}. "
            "Pass a method like agent.analyze, not a bare function."
        )
    from nooa.agent import Agent

    agent = method.__self__
    if not isinstance(agent, Agent):
        raise TypeError(
            f"build_prompt_data() requires a method on an Agent instance, "
            f"got {type(agent).__name__!r}."
        )
    wrapper = method.__func__
    original_func = getattr(wrapper, "_original", wrapper)
    return await _build_prompt_data_from_agent(agent, wrapper, original_func, args, kwargs)


async def print_prompt(method: Any, /, *args: Any, **kwargs: Any) -> None:
    """Print the prompts that would be sent to the LLM for *method*.

    Uses real agent state and real arguments — decorator-scoped context blocks,
    dynamic blocks, and actual parameter values are all reflected accurately.

    Output is plain text written to stdout.

    Args:
        method: Bound generation method on an :class:`~nooa.Agent` instance.
        *args: Positional arguments the method would be called with.
        **kwargs: Keyword arguments the method would be called with.

    Example::

        agent = MyAgent(llm=client)
        # ... lots of things happen
        await nooa.print_prompt(agent.analyze, my_data)
    """
    data = await build_prompt_data(method, *args, **kwargs)
    sys.stdout.write(render_prompt_data(data))
    sys.stdout.flush()
