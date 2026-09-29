# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owners must not release shared resources before component cleanup finishes."""

import asyncio

from nooa.runtime.event_manager import EventManager


async def test_repeated_cancellation_and_concurrent_close_wait_for_all_callbacks():
    manager = EventManager()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def first():
        calls.append("first")

    async def second():
        entered.set()
        await release.wait()
        calls.append("second")
        await manager.aclose()  # Recursive owner close must not deadlock.

    manager.on_close(first)
    manager.on_close(second)
    closer = asyncio.create_task(manager.aclose())
    await entered.wait()
    concurrent = asyncio.create_task(manager.aclose())
    try:
        for _ in range(2):
            closer.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not closer.done()
            assert not concurrent.done()
            assert calls == []
    finally:
        release.set()
        results = await asyncio.gather(closer, concurrent, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    assert calls == ["second", "first"]
    await manager.aclose()
    assert calls == ["second", "first"]


async def test_close_callback_can_reenter_via_a_child_task():
    manager = EventManager()
    children = []
    completed_inside_callback = []

    async def callback():
        child = asyncio.create_task(manager.aclose())
        children.append(child)
        done, _ = await asyncio.wait([child], timeout=0.2)
        completed_inside_callback.append(child in done)

    manager.on_close(callback)
    await manager.aclose()
    await asyncio.gather(*children)
    assert completed_inside_callback == [True]


async def test_cancelled_callback_does_not_drop_remaining_cleanup():
    manager = EventManager()
    calls = []

    async def first():
        calls.append("first")

    async def second():
        calls.append("second")
        raise asyncio.CancelledError

    manager.on_close(first)
    manager.on_close(second)
    results = await asyncio.gather(manager.aclose(), manager.aclose(), return_exceptions=True)
    assert results == [None, None]
    assert calls == ["second", "first"]
    await manager.aclose()
    assert calls == ["second", "first"]
