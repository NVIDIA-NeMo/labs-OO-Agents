# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep a native Relay call and its Python callback's lifetime aligned."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def execute_callback(
    execute: Callable[[Callable[..., Awaitable[Any]]], Awaitable[Any]],
    callback: Callable[..., Awaitable[Any]],
) -> Any:
    """Execute one callback without exporting its exception text to Relay.

    Relay can cancel its native future before the Python callback has finished
    unwinding. Wait for that callback before allowing its enclosing scopes to
    close. This waits only for this call, including after repeated cancellation.
    """
    completed = asyncio.Event()
    started = False
    cancelled = False
    failure: BaseException | None = None
    caller = asyncio.current_task()
    cancellations_before = caller.cancelling() if caller is not None else 0

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal started, failure
        if cancelled:
            # The native call may have queued its callback just before cancellation.
            raise asyncio.CancelledError
        started = True
        try:
            return await callback(*args, **kwargs)
        except asyncio.CancelledError as exc:
            failure = exc
            raise asyncio.CancelledError from None
        except BaseException as exc:
            failure = exc
            # Native scope metadata includes exception text. Keep the original
            # exception in Python and expose only this fixed message to Relay.
            raise RuntimeError("NOOA callback failed") from None
        finally:
            completed.set()

    try:
        result = await execute(wrapped)
    except asyncio.CancelledError:
        cancelled = True
        if started and not completed.is_set():
            waiter = asyncio.create_task(completed.wait())
            while not completed.is_set():
                try:
                    await asyncio.shield(waiter)
                except asyncio.CancelledError:
                    # A second cancellation must not let enclosing scopes close
                    # while the native callback is still cleaning up its children.
                    continue
        if failure is not None and (caller is None or caller.cancelling() == cancellations_before):
            raise failure from None
        raise
    except BaseException:
        if failure is not None:
            raise failure from None
        raise
    if failure is not None:
        # A native execution hook may have converted the generic callback error.
        raise failure from None
    return result
