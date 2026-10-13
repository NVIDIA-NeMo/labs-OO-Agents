# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission-owned provider attempt lifecycle, independent of transport policy."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from nooa.unifiedllm.admission import _run_admitted_attempt


@pytest.mark.asyncio
async def test_admitted_success_returns_original_result_and_releases_once():
    """A release-only controller needs no provider-feedback extension."""
    events = []
    result = object()

    class Permit:
        def release(self):
            """Record the sole cleanup after the provider returns."""
            events.append("release")

    async def provider():
        """Return a result without adaptation or an extra provider attempt."""
        events.append("call")
        return result

    assert await _run_admitted_attempt(provider, Permit()) is result
    assert events == ["call", "release"]


@pytest.mark.asyncio
async def test_synchronous_provider_factory_failure_still_releases():
    """Even a failure before constructing the provider coroutine owns cleanup."""
    events = []
    error = RuntimeError("factory failed")

    class Permit:
        def release(self):
            """Return the slot when the provider factory fails synchronously."""
            events.append("release")

    def provider() -> Coroutine[Any, Any, None]:
        """Fail before returning a coroutine to await."""
        raise error

    with pytest.raises(RuntimeError) as caught:
        await _run_admitted_attempt(provider, Permit())
    assert caught.value is error
    assert events == ["release"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_non_overload_and_cancellation_release_without_feedback(cancelled: bool):
    """Cancellation is cleanup, not an overload signal or a retry opportunity."""
    events = []
    error = asyncio.CancelledError() if cancelled else ValueError("invalid request")

    class Permit:
        async def cooldown(self, delay_s):
            """Record unexpected feedback for failures that cannot request it."""
            events.append(("cooldown", delay_s))

        def release(self):
            """Release capacity on every exit, including BaseException exits."""
            events.append("release")

    async def provider():
        """Preserve the provider's exact failure object through the boundary."""
        raise error

    with pytest.raises(type(error)) as caught:
        await _run_admitted_attempt(provider, Permit())
    assert caught.value is error
    assert events == ["release"]


@pytest.mark.asyncio
async def test_shared_feedback_completes_before_permit_is_released():
    """Pending feedback keeps the lease until the admission policy confirms it."""
    events = []
    feedback_started = asyncio.Event()
    acknowledge_feedback = asyncio.Event()

    class Overloaded(Exception):
        status_code = 429
        headers = {"Retry-After": "1"}

    error = Overloaded("provider failed")

    class Permit:
        async def cooldown(self, delay_s):
            """Represent the asynchronous acknowledgement of shared feedback."""
            events.append(("cooldown", delay_s))
            feedback_started.set()
            await acknowledge_feedback.wait()
            events.append("acknowledged")

        def release(self):
            """Record release only after feedback no longer owns the lease."""
            events.append("release")

    async def provider():
        """Supply overload metadata directly to the admission-owned wrapper."""
        raise error

    task = asyncio.create_task(_run_admitted_attempt(provider, Permit()))
    try:
        await asyncio.wait_for(feedback_started.wait(), timeout=1)
        assert events == [("cooldown", 1.0)]
        assert not task.done()
        acknowledge_feedback.set()
        with pytest.raises(Overloaded) as caught:
            await task
        assert caught.value is error
        assert events == [("cooldown", 1.0), "acknowledged", "release"]
    finally:
        acknowledge_feedback.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_feedback_still_returns_capacity():
    """A provider deadline can cancel feedback without stranding its permit."""
    events = []
    feedback_started = asyncio.Event()

    class Overloaded(Exception):
        status_code = 503
        headers = {"Retry-After": "1"}

    class Permit:
        async def cooldown(self, delay_s):
            """Wait until the provider task's deadline cancels this feedback."""
            feedback_started.set()
            await asyncio.Event().wait()

        def release(self):
            """Record cleanup after cancellation of the feedback coroutine."""
            events.append("release")

    async def provider():
        """Trigger shared cooldown before cancellation reaches the task."""
        raise Overloaded()

    task = asyncio.create_task(_run_admitted_attempt(provider, Permit()))
    try:
        await asyncio.wait_for(feedback_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["release"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
