# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The overall per-attempt deadline on a provider call.

``read_timeout`` only bounds the gap between bytes, so a stalled stream that
still trickles keepalive bytes never trips it and the attempt awaits forever.
``request_timeout`` bounds the whole attempt and cancels it on expiry.
"""

import asyncio

import pytest

from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.unifiedllm import (
    _await_provider_task,
    _run_async_provider_call,
)


def test_http_config_request_timeout_default_and_opt_out():
    assert HttpConfig().request_timeout == 600.0
    assert HttpConfig(request_timeout=None).request_timeout is None


async def test_deadline_times_out_and_cancels_the_unit():
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(asyncio.TimeoutError):
        await _run_async_provider_call(hang, request_timeout=0.05)

    await asyncio.sleep(0.01)
    assert cancelled.is_set()


async def test_fast_call_under_deadline_returns():
    async def quick():
        return "ok"

    assert await _run_async_provider_call(quick, request_timeout=5.0) == "ok"


async def test_none_disables_the_deadline():
    async def quick():
        return "ok"

    assert await _run_async_provider_call(quick, request_timeout=None) == "ok"


async def test_await_provider_task_times_out_and_cancels():
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(hang())
    with pytest.raises(asyncio.TimeoutError):
        await _await_provider_task(task, request_timeout=0.05)

    await asyncio.sleep(0.01)
    assert cancelled.is_set()


async def test_deadline_survives_caller_cancel():
    """A caller cancel does not disable the deadline; a stalled task is still cancelled."""
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(hang())
    outer = asyncio.create_task(_await_provider_task(task, request_timeout=0.05))
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    # The caller is gone, but the deadline still fires and cancels the stalled task.
    assert not cancelled.is_set()
    await asyncio.sleep(0.15)
    assert cancelled.is_set()
    assert task.cancelled()


async def test_caller_cancel_wins_over_deadline():
    """A caller cancel that lands with the deadline raises CancelledError, not TimeoutError."""
    outer: asyncio.Task[None] | None = None

    async def hang():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # The deadline is tearing this task down; cancel the caller in the same window.
            assert outer is not None
            outer.cancel()
            raise

    task = asyncio.create_task(hang())
    outer = asyncio.create_task(_await_provider_task(task, request_timeout=0.05))
    with pytest.raises(asyncio.CancelledError):
        await outer


async def test_await_provider_task_shields_a_caller_cancel():
    finished = asyncio.Event()

    async def work():
        await asyncio.sleep(0.05)
        finished.set()
        return "done"

    task = asyncio.create_task(work())
    outer = asyncio.create_task(_await_provider_task(task, request_timeout=None))
    await asyncio.sleep(0.01)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer

    # The shield keeps the provider task running despite the caller's cancel.
    await asyncio.sleep(0.1)
    assert finished.is_set()
