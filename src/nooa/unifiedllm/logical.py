# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral logical calls, outside the shared transport retry owner.

Preparation and media recovery are invocation-local policies. The send callback
runs execution.execute_sync/async (or an old-style subclass's one-call hook),
not another logical call. Recovery is bounded independently of transport retry;
failed logical calls roll back staged changes even on cancellation/BaseException.
"""

from collections.abc import Awaitable, Callable

from nooa.llm_types import LLMResponse


def logical_sync[T](
    prepare: Callable[[], T],
    send: Callable[[T], LLMResponse],
    *,
    recovery_attempts: int = 0,
    recover: Callable[[Exception], bool] | None = None,
    rollback: Callable[[], None] | None = None,
) -> LLMResponse:
    """Prepare/send, recover media if permitted, commit only on success."""
    try:
        for _ in range(recovery_attempts):
            try:
                return send(prepare())
            except Exception as exc:
                if recover is None or not recover(exc):
                    raise
        return send(prepare())
    except BaseException:
        if rollback is not None:
            rollback()
        raise


async def logical_async[T](
    prepare: Callable[[], T],
    send: Callable[[T], Awaitable[LLMResponse]],
    *,
    recovery_attempts: int = 0,
    recover: Callable[[Exception], bool] | None = None,
    rollback: Callable[[], None] | None = None,
) -> LLMResponse:
    """Async logical_sync; cancellation never enters media recovery."""
    try:
        for _ in range(recovery_attempts):
            try:
                return await send(prepare())
            except Exception as exc:
                if recover is None or not recover(exc):
                    raise
        return await send(prepare())
    except BaseException:
        if rollback is not None:
            rollback()
        raise
