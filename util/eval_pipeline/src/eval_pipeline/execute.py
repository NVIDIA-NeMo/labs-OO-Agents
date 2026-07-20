# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task execution stage of the pipeline."""

import asyncio
import time
from pathlib import Path
from typing import Any, Protocol

from .models import ExecutionResult, Task

# Type alias for agent input: (args, kwargs)
AgentInput = tuple[tuple[Any, ...], dict[str, Any]]


class Agent(Protocol):
    """Protocol for agents that can execute tasks."""

    async def run(self, input: AgentInput) -> Any:
        """Execute a task and return result.

        Args:
            input: Tuple of (args, kwargs) for the method call
        """
        ...


async def execute_task(
    agent: Agent,
    task: Task,
    trace_file: Path | None = None,
    timeout_seconds: float | None = None,
) -> ExecutionResult:
    """Execute a single task with an agent.

    Args:
        agent: Agent to execute the task
        task: Task to execute
        trace_file: Path where trace will be written, or None when using OTLP (caller supplies path later).
        timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.

    Returns:
        ExecutionResult with actual output and timing
    """
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)

    error = None
    actual = None

    start = time.perf_counter()
    try:
        if timeout_seconds is not None:
            async with asyncio.timeout(timeout_seconds):
                actual = await agent.run(task.input)
        else:
            actual = await agent.run(task.input)
    except TimeoutError:
        elapsed = time.perf_counter() - start
        error = f"Timeout after {elapsed:.1f}s (limit: {timeout_seconds}s)"
    except Exception as e:
        error = str(e)
    latency_ms = (time.perf_counter() - start) * 1000

    return ExecutionResult(
        task_id=task.id,
        input=task.input,
        expected=task.expected,
        actual=actual,
        trace_file=trace_file,
        latency_ms=latency_ms,
        error=error,
    )
