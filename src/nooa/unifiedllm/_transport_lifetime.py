# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral ownership of actual admitted wire tasks and deadlines.

The legacy facade forwards these identities, including Gym's ContextVar. Native
SDK adapters use this module without importing legacy provider implementations.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any

from .admission import _current_admission_controller, _run_admitted_attempt

logger = logging.getLogger(__name__)

# Optional harness metrics callback — set by the agent framework via ContextVar.
# No reverse import needed: the callback is injected by actor.py at session start.
_llm_metrics_callback: ContextVar[Callable[[str, Any], None] | None] = ContextVar(
    "llm_metrics_callback", default=None
)


def _record_llm_metric(event: str, detail: Any = None) -> None:
    """Fire-and-forget metric recording. No-op if no callback is set.

    Swallows any exception from the callback — instrumentation must never
    break the LLM call flow.
    """
    cb = _llm_metrics_callback.get()
    if cb is not None:
        try:
            cb(event, detail)
        except Exception as e:  # noqa: BLE001
            logger.debug("Metric callback failed for event %r: %s", event, e)


def _record_admission_observation(detail: dict[str, Any]) -> None:
    """Record one admission outcome on the generation span and harness metrics."""
    _record_llm_metric("llm_queue", detail)
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if span and span.is_recording():
            span.add_event("llm.queue", attributes=detail)
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not record LLM admission event: %s", e)


# Private consumer lease hook, captured by the existing shielded transport owner.
# It neither admits nor executes provider work; Gym retains its HTTP reservations
# through actual exit rather than caller cancellation.
_provider_lifetime: ContextVar[Any] = ContextVar("_provider_lifetime", default=None)


async def _run_async_provider_call[T](
    call: Callable[[], Coroutine[Any, Any, T]],
    *,
    request_timeout: float | None,
    _wire_owner=None,
    _attempt_owner=None,
    _hidden_retries=False,
) -> T:
    """Dispatch after admission; the actual shielded task owns wire resources.

    Queued cancellation creates neither a task nor an extra pool lease. Once
    dispatched, caller cancellation remains prompt while the provider retains
    its admission permit and transport lease through streaming/actual exit.
    """
    lifetime = _provider_lifetime.get()
    admission_controller = _current_admission_controller()
    if _attempt_owner is not None:
        _attempt_owner.writer.admission()
    permit = (
        await admission_controller.acquire(_record_admission_observation)
        if admission_controller is not None
        else None
    )
    acquired = False
    try:
        if _wire_owner is not None:
            _wire_owner.acquire()
            acquired = True

        started = False

        async def owned_call():
            nonlocal started
            started = True
            if _attempt_owner is not None:
                _attempt_owner.writer.entering_provider(hidden_retries=_hidden_retries)
            with lifetime() if lifetime is not None else nullcontext():
                return (
                    await _run_admitted_attempt(call, permit)
                    if permit is not None
                    else await call()
                )

        def release_task_resources(completed):
            # A done callback also covers cancellation before coroutine entry.
            # Admission owns its normal finally; do not release it twice.
            if not started and permit is not None:
                permit.release()
            if _wire_owner is not None:
                _wire_owner.release()

        task = asyncio.create_task(owned_call())
        task.add_done_callback(release_task_resources)
    except BaseException:
        if permit is not None:
            permit.release()
        if acquired:
            _wire_owner.release()
        raise
    return await _await_provider_task(task, request_timeout)


async def _await_provider_task[T](task: asyncio.Task[T], request_timeout: float | None) -> T:
    """Await a shielded provider task, bounded by ``request_timeout`` if set.

    The deadline is armed on the task, not on this await, so it still fires
    after the caller cancels. A caller cancel raises CancelledError, even when
    it lands with the deadline.
    """
    task.add_done_callback(_consume_async_provider_result)
    if request_timeout is None:
        return await asyncio.shield(task)

    timed_out = False

    def _on_deadline() -> None:
        nonlocal timed_out
        if not task.done():
            timed_out = True
            task.cancel()

    handle = asyncio.get_running_loop().call_later(request_timeout, _on_deadline)
    task.add_done_callback(lambda _done: handle.cancel())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if timed_out and (current is None or not current.cancelling()):
            raise TimeoutError from None
        raise


def _consume_async_provider_result(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except BaseException:
        pass
