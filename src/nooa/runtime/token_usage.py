# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-level token usage accumulator.

Tracks cumulative input and output token counts across all LLM calls made
during a single Harbor task evaluation.  Uses a ContextVar so the accumulator
is isolated per asyncio task and does not leak between concurrent runs.

Usage (from runner.py)::

    from nooa.runtime.token_usage import start_task_tokens, get_task_tokens

    start_task_tokens()
    result = await agent._run_evaluation(...)
    tokens = get_task_tokens()
    # tokens == {"n_input_tokens": 1234, "n_output_tokens": 567}

The accumulator is fed by ``actor.py``'s LLM metrics bridge, which calls
``accumulate_tokens()`` whenever unifiedllm fires a ``"token_usage"`` event.
"""

from __future__ import annotations

from contextvars import ContextVar

# Each task gets its own mutable dict so concurrent runs don't interfere.
# Value is None when no task is active (outside runner context).
_task_tokens_var: ContextVar[dict[str, int] | None] = ContextVar("task_tokens", default=None)


def start_task_tokens() -> None:
    """Initialize the accumulator for the current task.  Call once before _run_evaluation."""
    _task_tokens_var.set({"n_input_tokens": 0, "n_output_tokens": 0})


def accumulate_tokens(input_tokens: int, output_tokens: int) -> None:
    """Add token counts from one LLM call to the running task total.

    Safe to call even when no accumulator is active (e.g. in tests).
    """
    counter = _task_tokens_var.get()
    if counter is not None:
        counter["n_input_tokens"] += input_tokens
        counter["n_output_tokens"] += output_tokens


def get_task_tokens() -> dict[str, int]:
    """Return a snapshot of accumulated token counts for the current task."""
    counter = _task_tokens_var.get()
    return dict(counter) if counter else {"n_input_tokens": 0, "n_output_tokens": 0}
