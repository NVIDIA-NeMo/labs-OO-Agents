# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed owner-thread cancel dispatch must not permanently wedge cancellation.

``_request_cancel_on_owner`` is reached from ``swap_agent``/``cancel_for_transition``
via ``_cancel_turn_in_lifecycle``. Unlike ``request_cancel``, it did not roll back
``_cancel_requested`` when dispatching the cancel failed (a done ``ConcurrentFuture``
returning ``False`` from ``.cancel()``, or a closing loop raising ``RuntimeError``
from ``call_soon_threadsafe``). That left ``_cancel_requested`` stuck ``True``
forever, so the early "already requested" guard turned every later cancel request
into a silent no-op that never actually cancelled anything.
"""

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

import pytest
from nooa_coder.interactive.local_agent import LocalAgentRunner


class _FakeQueueManager:
    def set_notify_callback(self, callback: Any) -> None:
        pass

    def channels(self) -> list[Any]:
        return []

    def handles(self) -> list[Any]:
        return []


class _FakeUserMessages:
    def qsize(self) -> int:
        return 0

    def snapshot(self) -> list[Any]:
        return []

    def set_on_get(self, callback: Any) -> None:
        pass


class _FakeAgent:
    def __init__(self) -> None:
        self.queue_manager = _FakeQueueManager()
        self._user_messages_in = _FakeUserMessages()
        self.cwd = None


def _make_runner() -> LocalAgentRunner:
    return LocalAgentRunner(_FakeAgent(), emit_text=lambda text: None, agent_id="test-agent")


@pytest.mark.asyncio
async def test_failed_cancel_dispatch_rolls_back_cancel_requested():
    runner = _make_runner()
    runner._lifecycle_state = "active"
    runner._in_handle = True
    runner._task = asyncio.get_running_loop().create_future()
    runner._source_task = None
    already_done_future: ConcurrentFuture = ConcurrentFuture()
    already_done_future.set_result(None)
    runner._source_future = already_done_future

    result = runner._request_cancel_on_owner(force=False, notify=True)

    assert result is False
    assert runner._cancel_requested is False
    assert runner._notify_cancelled is False


@pytest.mark.asyncio
async def test_failed_cancel_dispatch_does_not_wedge_later_cancel_requests():
    """Without the rollback, a second call would find ``_cancel_requested`` still
    True and hit the early "already requested" guard, returning True without ever
    dispatching a real cancel -- a permanent, silent no-op."""
    runner = _make_runner()
    runner._lifecycle_state = "active"
    runner._in_handle = True
    runner._task = asyncio.get_running_loop().create_future()
    runner._source_task = None
    already_done_future: ConcurrentFuture = ConcurrentFuture()
    already_done_future.set_result(None)
    runner._source_future = already_done_future

    first = runner._request_cancel_on_owner(force=False, notify=True)
    assert first is False

    # A real cancel becomes dispatchable again: swap in a pending future whose
    # .cancel() succeeds, and confirm the second call actually dispatches it
    # instead of returning a stale True from the "already requested" guard.
    live_future: ConcurrentFuture = ConcurrentFuture()
    runner._source_future = live_future

    second = runner._request_cancel_on_owner(force=False, notify=True)

    assert second is True
    assert runner._cancel_requested is True
    assert live_future.cancelled()
