# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run_with_deadline``: only the cell deadline's cancel is recognisable as such."""

import asyncio

import pytest

from nooa.runtime.cell_deadline import deadline_cancel, run_with_deadline


async def _waiter(seen):
    try:
        await asyncio.sleep(30)
    except asyncio.CancelledError:
        state = deadline_cancel()
        seen.append(state)
        if state is not None:
            state.note("job x is still running.")
        raise


async def test_the_deadline_cancel_is_recognised_and_its_notes_end_the_error():
    seen = []
    with pytest.raises(TimeoutError) as caught:
        await run_with_deadline(_waiter(seen), 0.05)
    assert seen[0] is not None
    assert str(caught.value) == (
        "Code execution timed out after 0.05 seconds. "
        "Check for infinite loops or blocking operations. job x is still running."
    )
    assert deadline_cancel() is None  # outside the cell


async def test_a_gathered_task_sees_the_deadline_too():
    seen = []

    async def cell():  # the cell starts the tasks: they copy its context
        await asyncio.gather(_waiter(seen), _waiter(seen))

    with pytest.raises(TimeoutError):
        await run_with_deadline(cell(), 0.05)
    assert len(seen) == 2 and all(state is not None for state in seen)


async def test_another_cancel_is_not_the_deadline():
    seen = []
    task = asyncio.ensure_future(run_with_deadline(_waiter(seen), 30))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [None]


async def test_the_cells_own_timeout_error_is_left_alone():
    async def cell():
        raise TimeoutError("my own")

    with pytest.raises(TimeoutError, match="^my own$"):
        await run_with_deadline(cell(), 30)
