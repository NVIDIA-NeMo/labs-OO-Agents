# SPDX-License-Identifier: Apache-2.0
"""NOOA-owned lifecycle for logical LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from nooa.runtime.hooks import call_after_hook, call_before_hook


@dataclass
class LLMCall:
    call_id: str
    context: Any
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


def begin_llm_call(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> LLMCall:
    """Begin one logical (possibly retried) provider call."""
    call_id = uuid4().hex
    ctx = call_before_hook(
        "before_llm_call",
        call_id=call_id,
        model=model,
        messages=messages,
        tools=tools,
        invocation_parameters=kwargs,
    )
    return LLMCall(call_id, ctx, model, messages, tools)


def end_llm_call(
    call: LLMCall, response: Any = None, exception: BaseException | None = None
) -> None:
    """Finish a logical provider call. Instrumentation failures are isolated."""
    call_after_hook(
        "after_llm_call",
        call.context,
        call_id=call.call_id,
        model=call.model,
        messages=call.messages,
        tools=call.tools,
        response=response,
        exception=exception,
    )
