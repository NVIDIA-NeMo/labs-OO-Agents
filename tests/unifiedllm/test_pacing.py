# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local joint concurrency, request pacing, and cooldown contract tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from nooa.unifiedllm import AdmissionControlConfig, AdmissionTimeoutError
from nooa.unifiedllm.admission import (
    AdmissionPolicy,
    CooldownAdmissionPermit,
    _admission_controller_scope,
    _get_or_create_group,
    _reset_admission_groups_for_tests,
)
from nooa.unifiedllm.unifiedllm import _run_async_provider_call


@pytest.fixture(autouse=True)
def isolated_groups():
    """Keep named group state independent between contract tests."""
    _reset_admission_groups_for_tests()
    yield
    _reset_admission_groups_for_tests()


def policy(
    *,
    rate: float | None = None,
    cooldown: float | None = None,
    limit: int = 4,
    timeout: float | None = None,
    group: str = "pacing-test",
) -> AdmissionPolicy:
    """Build a local controller with one shared named scheduler."""
    return AdmissionPolicy(
        max_in_flight=limit,
        concurrency_group=group,
        api_base=None,
        queue_timeout=timeout,
        requests_per_second=rate,
        max_cooldown=cooldown,
    )


@pytest.mark.parametrize("name", ["requests_per_second", "max_cooldown"])
@pytest.mark.parametrize("value", [True, "2", object()])
def test_policy_options_reject_non_numeric_values(name: str, value: Any):
    """Reject ambiguous configuration before starting provider work."""
    with pytest.raises(TypeError, match=name):
        AdmissionControlConfig(max_in_flight=1, **{name: value})


@pytest.mark.parametrize("name", ["requests_per_second", "max_cooldown"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 10**500])
def test_policy_options_reject_non_finite_or_nonpositive_values(name: str, value: Any):
    """Reject impossible durations and rates, including numeric overflow."""
    with pytest.raises(ValueError, match=name):
        AdmissionControlConfig(max_in_flight=1, **{name: value})


def test_rate_reciprocal_must_be_finite():
    """Even a finite positive subnormal rate can have an infinite interval."""
    with pytest.raises(ValueError, match="finite reciprocal"):
        AdmissionControlConfig(max_in_flight=1, requests_per_second=5e-324)
    with pytest.raises(ValueError, match="finite reciprocal"):
        policy(rate=5e-324)


@pytest.mark.parametrize("field", ["requests_per_second", "max_cooldown"])
def test_external_controller_cannot_silently_ignore_local_options(field: str):
    """Policy is owned by an injected controller, not the wrapper config."""

    class Controller:
        async def acquire(self, observer):
            """Bypass admission for this config validation probe."""
            return None

    with pytest.raises(ValueError, match="cannot be combined"):
        AdmissionControlConfig(controller=Controller(), **{field: 1.0})


@pytest.mark.parametrize("field", ["rate", "cooldown"])
def test_named_group_requires_consistent_policy(field: str):
    """Wrappers cannot weaken or redefine an existing group's policy."""
    policy(rate=10, cooldown=1)
    kwargs = {"rate": 10, "cooldown": 1}
    kwargs[field] = 2
    with pytest.raises(ValueError, match="Conflicting"):
        policy(**kwargs)
    with pytest.raises(ValueError, match="Conflicting"):
        policy()


@pytest.mark.asyncio
async def test_shared_pacing_is_fifo_and_does_not_hold_concurrency():
    """Separate controller instances share one non-bursty admission clock."""
    owner, alias = policy(rate=25), policy(rate=25)
    initial = await owner.acquire(lambda _: None)
    initial.release()
    starts: list[tuple[int, float]] = []
    observations: list[dict[str, Any]] = []

    async def worker(index: int):
        """Acquire, record grant time, and return concurrency immediately."""
        permit = await alias.acquire(observations.append)
        starts.append((index, time.monotonic()))
        permit.release()

    tasks = []
    for index in range(6):
        tasks.append(asyncio.create_task(worker(index)))
        await asyncio.sleep(0)
    group = _get_or_create_group(owner.identity, owner.display_name, None)
    assert group is not None
    assert group.active == 0
    assert group.queued == 6
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert [index for index, _ in starts] == list(range(6))
    assert all(b[1] - a[1] >= 0.035 for a, b in zip(starts, starts[1:], strict=False))
    assert all(item["outcome"] == "admitted_after_wait" for item in observations)
    assert group.active == group.queued == 0


@pytest.mark.asyncio
async def test_concurrency_remains_independent_of_pacing():
    """Time eligibility cannot dispatch work when both slots are occupied."""
    controller = policy(rate=100, limit=2)
    first = await controller.acquire(lambda _: None)
    second = await controller.acquire(lambda _: None)
    queued = asyncio.create_task(controller.acquire(lambda _: None))
    await asyncio.sleep(0.04)
    assert not queued.done()
    first.release()
    third = await asyncio.wait_for(queued, 1)
    second.release()
    third.release()


@pytest.mark.asyncio
async def test_idle_time_does_not_accumulate_burst_credits():
    """After idling, only one immediate grant is available at a time."""
    controller = policy(rate=25)
    first = await controller.acquire(lambda _: None)
    first.release()
    await asyncio.sleep(0.08)
    second = await controller.acquire(lambda _: None)
    started = time.monotonic()
    second.release()
    third = await controller.acquire(lambda _: None)
    assert time.monotonic() - started >= 0.035
    third.release()


@pytest.mark.asyncio
async def test_cooldown_extends_never_shortens_and_is_capped():
    """All peers honor the longest outstanding report within its bound."""
    controller = policy(cooldown=0.12)
    held = await controller.acquire(lambda _: None)
    assert isinstance(held, CooldownAdmissionPermit)
    await held.cooldown(100)
    group = _get_or_create_group(controller.identity, controller.display_name, None)
    assert group is not None
    deadline = group._cooldown_until
    assert 0.10 < deadline - time.monotonic() <= 0.12
    await held.cooldown(0.001)
    assert group._cooldown_until == deadline
    held.release()
    second = await asyncio.wait_for(controller.acquire(lambda _: None), 1)
    assert time.monotonic() >= deadline
    second.release()


@pytest.mark.asyncio
async def test_active_call_feedback_reschedules_head_without_revoking_other_calls():
    """A later overload pushes back a paced head, not existing active work."""
    controller = policy(rate=25, cooldown=0.15)
    held = await controller.acquire(lambda _: None)
    queued = asyncio.create_task(controller.acquire(lambda _: None))
    await asyncio.sleep(0.01)
    await held.cooldown(0.10)
    until = time.monotonic() + 0.085
    await asyncio.sleep(0.05)
    assert not queued.done()
    next_permit = await asyncio.wait_for(queued, 1)
    assert time.monotonic() >= until
    held.release()
    next_permit.release()


@pytest.mark.asyncio
async def test_disabled_cooldown_and_released_permit_feedback_are_noops():
    """The default policy is unchanged and expired leases cannot delay peers."""
    controller = policy()
    permit = await controller.acquire(lambda _: None)
    await permit.cooldown(10)
    permit.release()
    await permit.cooldown(10)
    observations = []
    probe = await controller.acquire(observations.append)
    assert observations[0]["outcome"] == "immediate"
    probe.release()


@pytest.mark.asyncio
async def test_queue_timeout_covers_rate_waiting_without_a_slot():
    """A rate-blocked acquisition terminates without debiting concurrency."""
    controller = policy(rate=1, timeout=0.01)
    permit = await controller.acquire(lambda _: None)
    permit.release()
    observations = []
    with pytest.raises(AdmissionTimeoutError):
        await controller.acquire(observations.append)
    group = _get_or_create_group(controller.identity, controller.display_name, None)
    assert group is not None
    assert group.active == group.queued == 0
    assert observations[0]["outcome"] == "timeout"


@pytest.mark.asyncio
async def test_cancelled_timer_head_wakes_successor_and_preserves_pace():
    """Cancelling the task that owns the timer cannot strand the FIFO queue."""
    controller = policy(rate=25)
    first = await controller.acquire(lambda _: None)
    first.release()
    started = time.monotonic()
    head = asyncio.create_task(controller.acquire(lambda _: None))
    tail = asyncio.create_task(controller.acquire(lambda _: None))
    await asyncio.sleep(0)
    head.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head
    survivor = await asyncio.wait_for(tail, 1)
    assert time.monotonic() - started >= 0.035
    survivor.release()


@pytest.mark.asyncio
async def test_failed_observer_does_not_refund_pacing():
    """A grant consumes spacing even when subsequent observation fails."""
    controller = policy(rate=25)

    def fail(_):
        """Reject an already granted admission to probe cleanup."""
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        await controller.acquire(fail)
    started = time.monotonic()
    permit = await controller.acquire(lambda _: None)
    assert time.monotonic() - started >= 0.035
    permit.release()


@pytest.mark.asyncio
async def test_pacing_and_feedback_work_across_threads_and_event_loops():
    """A shared group coordinates timers on independent running loops."""
    controller = policy(rate=50, cooldown=0.08)
    held = await controller.acquire(lambda _: None)
    await held.cooldown(0.06)
    deadline = time.monotonic() + 0.05
    held.release()

    def run_worker():
        """Create an independent loop and return its grant timestamp."""

        async def work():
            """Release local concurrency before closing the worker's loop."""
            permit = await controller.acquire(lambda _: None)
            timestamp = time.monotonic()
            permit.release()
            return timestamp

        return asyncio.run(work())

    stamps = sorted(await asyncio.gather(*(asyncio.to_thread(run_worker) for _ in range(4))))
    assert stamps[0] >= deadline
    assert all(b - a >= 0.016 for a, b in zip(stamps, stamps[1:], strict=False))


@pytest.mark.asyncio
async def test_mixed_cancellation_soak_leaves_no_tasks_or_waiters():
    """Repeated timer ownership changes do not retain queue entries/tasks."""
    controller = policy(rate=2_000, cooldown=0.01)

    async def worker():
        """Consume and release each successful grant during the soak."""
        permit = await controller.acquire(lambda _: None)
        permit.release()

    for _ in range(5):
        initial = await controller.acquire(lambda _: None)
        await initial.cooldown(0.004)
        initial.release()
        tasks = [asyncio.create_task(worker()) for _ in range(40)]
        await asyncio.sleep(0)
        for task in tasks[::2]:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(
            result is None or isinstance(result, asyncio.CancelledError) for result in results
        )
    group = _get_or_create_group(controller.identity, controller.display_name, None)
    assert group is not None
    assert group.active == group.queued == 0
    assert not group._waiters


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["parser", "feedback", "none"])
async def test_feedback_preserves_original_error_and_always_releases(failure: str, caplog):
    """Optional feedback must neither mask a provider error nor leak a lease."""
    events = []

    class BrokenHeaders(dict):
        def items(self):
            """Simulate an adapter failure while exposing header metadata."""
            raise RuntimeError("secret-feedback-detail")

    class Overloaded(Exception):
        status_code = 429
        headers = BrokenHeaders() if failure == "parser" else {"Retry-After": "1"}

    error = Overloaded("secret-provider-detail")

    class Permit:
        async def cooldown(self, delay_s):
            """Report feedback, optionally simulating an unavailable service."""
            events.append(("cooldown", delay_s))
            if failure == "feedback":
                raise RuntimeError("secret-feedback-detail")

        def release(self):
            """Record cleanup after feedback, including its failure path."""
            events.append("release")

    class Controller:
        async def acquire(self, observer):
            """Return a custom cooldown-capable permit for a provider attempt."""
            return Permit()

    async def provider():
        """Expose the exact original error object to the caller."""
        raise error

    with _admission_controller_scope(Controller()):
        with pytest.raises(Overloaded) as caught:
            await _run_async_provider_call(provider)
    assert caught.value is error
    assert events == (["release"] if failure == "parser" else [("cooldown", 1.0), "release"])
    assert "secret-feedback-detail" not in caplog.text
    assert "secret-provider-detail" not in caplog.text


@pytest.mark.asyncio
async def test_existing_release_only_permit_keeps_its_original_contract():
    """External controllers are not required to adopt cooldown feedback."""
    events = []

    class Permit:
        def release(self):
            """Return the existing controller's slot without any new hooks."""
            events.append("release")

    class Controller:
        async def acquire(self, observer):
            """Keep the existing acquire(observer) signature."""
            return Permit()

    class Overloaded(Exception):
        status_code = 429
        headers = {"Retry-After": "1"}

    async def provider():
        """Fail with metadata an old controller may safely ignore."""
        raise Overloaded()

    with _admission_controller_scope(Controller()):
        with pytest.raises(Overloaded):
            await _run_async_provider_call(provider)
    assert events == ["release"]
