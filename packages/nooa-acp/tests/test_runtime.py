# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ACP adapter's private live-session ownership shim."""

from __future__ import annotations

import asyncio

import pytest
from nooa_acp._runtime import (
    SessionBusyError,
    SessionRuntime,
    SessionRuntimeClosedError,
    SessionRuntimePool,
)

# Bounds a hang, not the expected duration.
_HANG_TIMEOUT = 30


class _RuntimeValue:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


async def test_same_session_rejects_a_second_foreground_turn():
    runtime = SessionRuntime("one", object())

    async def _claim_again() -> None:
        async with runtime.turn():
            pass

    async with runtime.turn():
        assert runtime.busy is True
        # Bounded: if turns start queueing instead of failing fast — the exact
        # regression — this wedges on the inner lock and hangs the suite.
        with pytest.raises(SessionBusyError):
            await asyncio.wait_for(_claim_again(), timeout=_HANG_TIMEOUT)


async def test_simultaneous_turn_claims_do_not_queue():
    """Exactly one simultaneous caller wins; the other fails rather than queues."""
    runtime = SessionRuntime("one", object())
    start = asyncio.Event()
    release = asyncio.Event()
    outcomes: list[str] = []

    async def claim() -> None:
        await start.wait()
        try:
            async with runtime.turn():
                outcomes.append("entered")
                await release.wait()
        except SessionBusyError:
            outcomes.append("busy")

    tasks = [asyncio.create_task(claim()) for _ in range(2)]
    start.set()
    while not outcomes:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*tasks)

    assert sorted(outcomes) == ["busy", "entered"]


async def test_different_sessions_run_foreground_turns_concurrently():
    pool: SessionRuntimePool[str] = SessionRuntimePool()
    first = await pool.add("first", "A")
    second = await pool.add("second", "B")
    both_entered = asyncio.Event()
    entered: set[str] = set()

    async def run(runtime: SessionRuntime[str]) -> None:
        async with runtime.turn() as value:
            entered.add(value)
            if len(entered) == 2:
                both_entered.set()
            await both_entered.wait()

    # Deadlock detector: if the two sessions serialized, neither would reach
    # both_entered and the gather would hang. Generous so a loaded runner
    # cannot flake it; a real serialization bug still fails, just later.
    await asyncio.wait_for(asyncio.gather(run(first), run(second)), timeout=30)
    assert entered == {"A", "B"}
    await pool.close()


async def test_close_waits_for_active_turn_and_is_idempotent():
    value = _RuntimeValue()
    runtime = SessionRuntime("one", value)
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def active_turn() -> None:
        async with runtime.turn():
            turn_started.set()
            await release_turn.wait()

    turn_task = asyncio.create_task(active_turn())
    await turn_started.wait()
    close_task = asyncio.create_task(runtime.close())
    # A single yield is satisfied by scheduling latency — _close_once has not
    # even started — so it passes with the turn lock removed entirely.
    for _ in range(20):
        await asyncio.sleep(0)
    assert close_task.done() is False
    assert value.close_calls == 0
    release_turn.set()
    await asyncio.gather(turn_task, close_task)
    await runtime.close()

    assert value.close_calls == 1
    assert runtime.is_closed is True
    with pytest.raises(SessionRuntimeClosedError):
        async with runtime.turn():
            pass


async def test_close_cleanup_survives_caller_cancellation():
    value = _RuntimeValue()
    runtime = SessionRuntime("one", value)
    turn_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def active_turn() -> None:
        async with runtime.turn():
            turn_started.set()
            await release_turn.wait()

    turn_task = asyncio.create_task(active_turn())
    await turn_started.wait()
    cancelled_close = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    cancelled_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_close

    release_turn.set()
    await turn_task
    await runtime.close()
    assert runtime.is_closed is True
    assert value.close_calls == 1


async def test_pool_remove_and_close_release_each_runtime_once():
    pool: SessionRuntimePool[_RuntimeValue] = SessionRuntimePool()
    first_value = _RuntimeValue()
    second_value = _RuntimeValue()
    await pool.add("first", first_value)
    await pool.add("second", second_value)

    assert await pool.remove("first") is first_value
    assert await pool.ids() == ("second",)
    await pool.close()
    await pool.close()

    assert first_value.close_calls == 1
    assert second_value.close_calls == 1
    with pytest.raises(SessionRuntimeClosedError):
        await pool.add("third", _RuntimeValue())


async def test_remove_unregisters_even_when_teardown_fails():
    """A failing close must not strand the session id in the pool.

    The runtime is torn down regardless; leaving the entry registered makes the
    id permanently unusable — later loads report "already loaded" and later
    prompts reach a closed runtime.
    """

    class _Failing:
        async def close(self) -> None:
            raise RuntimeError("teardown blew up")

    pool: SessionRuntimePool[_Failing] = SessionRuntimePool()
    await pool.add("one", _Failing())

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await pool.remove("one")

    assert await pool.ids() == ()
    with pytest.raises(KeyError):
        await pool.get("one")


async def test_cancelled_remove_keeps_id_reserved_until_teardown_finishes():
    """Cancellation must not expose the id while its old runtime is still live."""
    started = asyncio.Event()
    release = asyncio.Event()

    class _Slow:
        async def close(self) -> None:
            started.set()
            await release.wait()

    pool: SessionRuntimePool[_Slow] = SessionRuntimePool()
    await pool.add("one", _Slow())

    remover = asyncio.create_task(pool.remove("one"))
    await asyncio.wait_for(started.wait(), timeout=_HANG_TIMEOUT)
    remover.cancel()
    with pytest.raises(asyncio.CancelledError):
        await remover

    assert await pool.ids() == ("one",)
    with pytest.raises(ValueError, match="already registered"):
        await pool.add("one", _Slow())

    release.set()
    for _ in range(20):
        if await pool.ids() == ():
            break
        await asyncio.sleep(0)
    assert await pool.ids() == ()
