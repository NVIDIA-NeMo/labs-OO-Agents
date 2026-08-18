# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locking and cancellation contracts for concurrent shell consumers."""

from __future__ import annotations

import asyncio
import shlex

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
async def session(tmp_path):
    shell = BashSession(cwd=tmp_path)
    await shell.start()
    try:
        yield shell
    finally:
        await shell.close()


async def _collect(stream) -> list[tuple[str, str]]:
    return [item async for item in stream]


class _ObservedLock:
    """Expose when a queued caller has actually attempted lock acquisition."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock
        self.attempted = asyncio.Event()

    async def __aenter__(self):
        self.attempted.set()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *_exc) -> None:
        self._lock.release()


async def test_paused_stream_blocks_queued_run_until_closed(session, tmp_path):
    release = tmp_path / "release-first-stream"
    stream = session.run_stream(
        f"printf first; while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done"
    )
    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "first")
    observed_lock = _ObservedLock(session._lock)
    session._lock = observed_lock
    queued = asyncio.create_task(session.run("printf second"))

    await asyncio.wait_for(observed_lock.attempted.wait(), timeout=3)
    assert not queued.done()
    await asyncio.wait_for(stream.aclose(), timeout=3)

    assert await asyncio.wait_for(queued, timeout=2) == ("second", "", 0)


async def test_queued_second_stream_receives_only_its_own_output(session, tmp_path):
    release = tmp_path / "release-queued-stream"
    first = session.run_stream(
        "printf first; "
        f"while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done; "
        "printf forbidden"
    )
    assert await asyncio.wait_for(anext(first), timeout=3) == ("stdout", "first")
    observed_lock = _ObservedLock(session._lock)
    session._lock = observed_lock
    second_task = asyncio.create_task(_collect(session.run_stream("printf second")))

    await asyncio.wait_for(observed_lock.attempted.wait(), timeout=3)
    assert not second_task.done()
    await asyncio.wait_for(first.aclose(), timeout=3)
    second_events = await asyncio.wait_for(second_task, timeout=2)

    assert second_events == [("stdout", "second"), ("__done__", "0,0")]


async def test_cancelling_run_waiting_for_lock_is_harmless(session, tmp_path):
    release = tmp_path / "release-lock-holder"
    holder = session.run_stream(
        f"printf held; while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done"
    )
    assert await asyncio.wait_for(anext(holder), timeout=3) == ("stdout", "held")
    observed_lock = _ObservedLock(session._lock)
    session._lock = observed_lock
    waiting = asyncio.create_task(session.run("printf forbidden"))
    await asyncio.wait_for(observed_lock.attempted.wait(), timeout=3)
    assert not waiting.done()

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.wait_for(holder.aclose(), timeout=3)

    assert await asyncio.wait_for(session.run("printf recovered"), timeout=2) == (
        "recovered",
        "",
        0,
    )


async def test_cancelling_stream_waiting_for_lock_is_harmless(session, tmp_path):
    release = tmp_path / "release-stream-holder"
    holder = session.run_stream(
        f"printf held; while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done"
    )
    assert await asyncio.wait_for(anext(holder), timeout=3) == ("stdout", "held")
    observed_lock = _ObservedLock(session._lock)
    session._lock = observed_lock

    async def consume_waiter() -> list[tuple[str, str]]:
        return await _collect(session.run_stream("printf forbidden"))

    waiting = asyncio.create_task(consume_waiter())
    await asyncio.wait_for(observed_lock.attempted.wait(), timeout=3)
    assert not waiting.done()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.wait_for(holder.aclose(), timeout=3)

    assert await asyncio.wait_for(session.run("printf recovered"), timeout=2) == (
        "recovered",
        "",
        0,
    )


async def test_repeated_close_is_idempotent_and_releases_lock_once(session):
    stream = session.run_stream("printf started; sleep 30")
    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")

    await asyncio.wait_for(stream.aclose(), timeout=3)
    await asyncio.wait_for(stream.aclose(), timeout=1)

    assert await asyncio.wait_for(session.run("printf recovered"), timeout=2) == (
        "recovered",
        "",
        0,
    )


def test_cancelled_stream_remains_recoverable_after_event_loop_change(tmp_path):
    session = BashSession(cwd=tmp_path)

    async def cancel_on_first_loop():
        await session.start()
        stream = session.run_stream("printf started; sleep 30")
        assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")
        await asyncio.wait_for(stream.aclose(), timeout=3)

    async def use_on_second_loop():
        return await asyncio.wait_for(session.run("printf recovered"), timeout=3)

    try:
        asyncio.run(cancel_on_first_loop())
        assert asyncio.run(use_on_second_loop()) == ("recovered", "", 0)
    finally:
        asyncio.run(session.close())
