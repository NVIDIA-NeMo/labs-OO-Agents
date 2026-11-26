# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep a native Relay call and its Python callback's lifetime aligned."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

_CANCELLATION_GRACE_SECONDS = 1.0
_logger = logging.getLogger(__name__)


async def execute_callback(
    execute: Callable[[Callable[..., Awaitable[Any]]], Awaitable[Any]],
    callback: Callable[..., Awaitable[Any]],
) -> Any:
    """Execute one callback without exporting its exception text to Relay.

    Relay can cancel its native future before the Python callback has finished
    unwinding. Wait for that callback before allowing its enclosing scopes to
    close. Cleanup has one bounded grace period, including after repeated
    cancellation; user code that suppresses cancellation cannot hold up callers.
    """
    loop = asyncio.get_running_loop()
    completed = loop.create_future()
    callback_task: asyncio.Task[Any] | None = None
    started = False
    cancelled = False
    failure: BaseException | None = None
    caller = asyncio.current_task()
    cancellations_before = caller.cancelling() if caller is not None else 0

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal started, failure, callback_task
        if cancelled:
            # The native call may have queued its callback just before cancellation.
            raise asyncio.CancelledError
        started = True
        callback_task = asyncio.current_task()
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
            if not completed.done():
                completed.set_result(None)

    try:
        result = await execute(wrapped)
    except asyncio.CancelledError:
        cancelled = True
        if started and not completed.done():
            deadline = loop.time() + _CANCELLATION_GRACE_SECONDS
            if (
                callback_task is not None
                and callback_task is not caller
                and not callback_task.done()
                and not callback_task.cancelling()
            ):
                callback_task.cancel()
            while not completed.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    if callback_task is not None and callback_task is not caller:
                        callback_task.cancel()
                    _logger.warning(
                        "NeMo Relay callback task %s exceeded cancellation cleanup grace (%s s); "
                        "callback cancellation requested, but user work may still be running",
                        id(callback_task),
                        _CANCELLATION_GRACE_SECONDS,
                    )
                    break
                try:
                    # asyncio.wait does not cancel the shared completion future
                    # and creates no detached waiter task when we stop waiting.
                    await asyncio.wait((completed,), timeout=remaining)
                except asyncio.CancelledError:
                    # Repeated cancellation cannot extend the cleanup deadline.
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
