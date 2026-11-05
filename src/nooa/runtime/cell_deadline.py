# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell deadline: tell its cancellation apart from every other one.

``run_with_deadline(awaitable, timeout)`` runs a cell under ``asyncio.timeout``
and publishes a ``CellDeadline`` in a ContextVar while it runs. Code awaited
by the cell (also in tasks it gathers) asks ``deadline_cancel()`` in its
``except CancelledError``: a ``CellDeadline`` means the deadline cancelled it
(and nothing else did), so it may hand live work over instead of killing it,
and ``note()`` what it did; the notes end the cell's timeout error text.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable

__all__ = ["CellDeadline", "deadline_cancel", "run_with_deadline", "timeout_message"]


class CellDeadline:
    """The running cell's deadline: whether it fired, and what was handed over."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.notes: list[str] = []
        self.finished = False
        self._cm: asyncio.Timeout | None = None

    def expired(self) -> bool:
        """The deadline fired (it is cancelling, or has cancelled, the cell)."""
        return self._cm is not None and self._cm.expired()

    def note(self, text: str) -> None:
        """A sentence for the cell's timeout error (say what still runs and how to reach it)."""
        if text and text not in self.notes:
            self.notes.append(text)


_current: contextvars.ContextVar[CellDeadline | None] = contextvars.ContextVar(
    "nooa_cell_deadline", default=None
)


def deadline_cancel() -> CellDeadline | None:
    """The cell's ``CellDeadline`` when its deadline is what cancels the caller, else None.

    Call it in ``except asyncio.CancelledError``. A user, turn or session
    cancel, or a cancel after the cell ended, is None.
    """
    state = _current.get()
    if state is None or state.finished or not state.expired():
        return None
    return state


def timeout_message(timeout: float, notes: list[str] | None = None) -> str:
    """The cell's timeout error text, with the deadline's notes after it."""
    text = (
        f"Code execution timed out after {timeout} seconds. "
        "Check for infinite loops or blocking operations."
    )
    return " ".join([text, *(notes or [])])


async def run_with_deadline[T](awaitable: Awaitable[T], timeout: float) -> T:
    """Await ``awaitable`` in this task under ``timeout``; ``TimeoutError`` names the notes."""
    state = CellDeadline(timeout)
    token = _current.set(state)
    try:
        async with asyncio.timeout(timeout) as cm:
            state._cm = cm
            return await awaitable
    except TimeoutError:
        if not state.expired():
            raise  # the cell's own TimeoutError, not the deadline
        raise TimeoutError(timeout_message(timeout, state.notes)) from None
    finally:
        state.finished = True
        _current.reset(token)
