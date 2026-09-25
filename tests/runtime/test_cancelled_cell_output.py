# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cancelled cell keeps the output it produced before the cancel.

``execute_code`` re-raises ``asyncio.CancelledError`` and attaches a partial
``ExecutionResult`` (``cancelled=True``) built from the stdout/stderr captured
up to the cancel point, so the strategy can show the model what ran.
"""

import asyncio

import pytest

from nooa import Agent
from nooa.events import ExecutionResult
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()

_CELL = """\
import sys
print("before cancel")
print("warning before cancel", file=sys.stderr)
started.set()
await blocker.wait()
print("after cancel")
"""


@pytest.fixture
def test_agent():
    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    return TestAgent()


@pytest.mark.asyncio
async def test_cancelled_cell_attaches_partial_result(test_agent):
    started = asyncio.Event()
    blocker = asyncio.Event()
    task = asyncio.create_task(
        test_agent.runtime.execute_code(
            _CELL,
            wrap_in_function=True,
            validate=False,
            builtins={"started": started, "blocker": blocker},
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task

    partial = getattr(excinfo.value, "execution_result", None)
    assert isinstance(partial, ExecutionResult)
    assert partial.cancelled is True
    assert partial.success is False
    assert partial.error is None
    assert "before cancel" in partial.stdout
    assert "after cancel" not in partial.stdout
    assert "warning before cancel" in partial.stderr


@pytest.mark.asyncio
async def test_completed_cell_is_not_marked_cancelled(test_agent):
    result = await test_agent.runtime.execute_code("print('hi')", validate=False)
    assert result.cancelled is False
    assert result.success is True


_OUTER_CELL = """\
print("outer before inner")
await runtime.execute_code(
    inner_code,
    wrap_in_function=True,
    validate=False,
    builtins={"started": started, "blocker": blocker},
)
"""


@pytest.mark.asyncio
async def test_nested_cancel_keeps_the_innermost_partial_result(test_agent):
    """One CancelledError passes through both frames; the inner cell's output wins."""
    started = asyncio.Event()
    blocker = asyncio.Event()
    task = asyncio.create_task(
        test_agent.runtime.execute_code(
            _OUTER_CELL,
            wrap_in_function=True,
            validate=False,
            builtins={
                "runtime": test_agent.runtime,
                "inner_code": _CELL,
                "started": started,
                "blocker": blocker,
            },
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await task

    partial = excinfo.value.execution_result
    assert "before cancel" in partial.stdout
    assert "warning before cancel" in partial.stderr
    assert "outer before inner" not in partial.stdout
