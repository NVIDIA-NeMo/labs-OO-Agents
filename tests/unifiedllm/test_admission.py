# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for process-local UnifiedLLM admission control."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.unifiedllm import (
    AdmissionController,
    AdmissionTimeoutError,
    CompletionClient,
    ResponsesClient,
    RetryConfig,
)
from nooa.unifiedllm.admission import (
    AdmissionPolicy,
    _get_or_create_group,
    _reset_admission_groups_after_fork,
    _reset_admission_groups_for_tests,
)
from nooa.unifiedllm.unifiedllm import (
    _litellm_acompletion,
    _record_admission_observation,
    _run_async_provider_call,
)


@pytest.fixture(autouse=True)
def _isolated_admission_groups():
    _reset_admission_groups_for_tests()
    yield
    _reset_admission_groups_for_tests()


def _policy(
    limit: int | None,
    *,
    group: str = "test-group",
    timeout: float | None = None,
) -> AdmissionPolicy:
    return AdmissionPolicy(
        max_in_flight=limit,
        concurrency_group=group,
        queue_timeout=timeout,
        api_base=None,
    )


def _chat_response(content: str = "ok") -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="test-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    )


def _responses_response(content: str = "ok") -> MagicMock:
    response = MagicMock()
    response.output = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
    ]
    response.output_text = content
    response.usage = None
    return response


def test_injected_controller_cannot_be_combined_with_local_policy():
    class Controller:
        async def acquire(self, observer: Callable[[dict[str, Any]], None]):
            del observer
            return None

    with pytest.raises(ValueError, match="cannot be combined"):
        CompletionClient(
            "test-model",
            admission_controller=Controller(),
            max_in_flight=1,
            concurrency_group="local",
        )


@pytest.mark.asyncio
async def test_completion_client_uses_injected_admission_controller():
    events: list[str] = []

    class Permit:
        def release(self) -> None:
            events.append("release")

    class Controller:
        async def acquire(self, observer: Callable[[dict[str, Any]], None]) -> Permit:
            events.append("acquire")
            observer(
                {
                    "group": "injected",
                    "outcome": "immediate",
                    "queued": False,
                    "wait_s": 0.0,
                    "queue_depth": 0,
                    "max_in_flight": 1,
                }
            )
            return Permit()

    controller: AdmissionController = Controller()
    client = CompletionClient(
        "test-model",
        admission_controller=controller,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    try:
        with patch("litellm.acompletion", AsyncMock(return_value=_chat_response())):
            response = await client.acall([{"role": "user", "content": "hello"}])
    finally:
        await client.aclose()

    assert response.content == "ok"
    assert events == ["acquire", "release"]
    assert client.admission_controller is controller
    assert "admission_controller" not in client.config


@pytest.mark.parametrize("value", [0, -1])
def test_max_in_flight_must_be_positive(value: int):
    with pytest.raises(ValueError, match="greater than zero"):
        _policy(value)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_max_in_flight_must_be_an_integer(value: Any):
    with pytest.raises(TypeError, match="positive integer"):
        _policy(value)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_queue_timeout_must_be_positive_and_finite(value: float):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        _policy(1, timeout=value)


def test_limit_requires_group_or_absolute_api_base():
    with pytest.raises(ValueError, match="requires concurrency_group or api_base"):
        AdmissionPolicy(
            max_in_flight=1,
            concurrency_group=None,
            queue_timeout=None,
            api_base=None,
        )


def test_unbounded_client_preserves_non_http_provider_base_url():
    policy = AdmissionPolicy(
        max_in_flight=None,
        concurrency_group=None,
        queue_timeout=None,
        api_base="provider-specific://socket",
    )

    assert policy.identity is None


def test_named_group_rejects_conflicting_limits():
    _policy(2)

    with pytest.raises(ValueError, match="existing=2, requested=3"):
        _policy(3)


def test_endpoint_identity_is_opaque_and_normalized():
    first = AdmissionPolicy(
        max_in_flight=1,
        concurrency_group=None,
        queue_timeout=None,
        api_base="HTTPS://Gateway.Example:443/v1/?token=secret#fragment",
    )
    second = AdmissionPolicy(
        max_in_flight=None,
        concurrency_group=None,
        queue_timeout=None,
        api_base="https://gateway.example/v1",
    )

    assert first.identity == second.identity
    assert first.display_name == second.display_name
    assert first.display_name is not None
    assert first.display_name.startswith("endpoint:")
    assert "gateway" not in first.display_name
    assert "secret" not in first.display_name


@pytest.mark.asyncio
async def test_waiters_are_admitted_fifo():
    policy = _policy(1)
    first = await policy.acquire(lambda _detail: None)
    assert first is not None
    order: list[int] = []

    async def worker(index: int) -> None:
        permit = await policy.acquire(lambda _detail: None)
        assert permit is not None
        order.append(index)
        permit.release()

    tasks = []
    for index in range(4):
        tasks.append(asyncio.create_task(worker(index)))
        await asyncio.sleep(0)

    first.release()
    await asyncio.gather(*tasks)

    assert order == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_queued_cancellation_does_not_dispatch_or_leak_capacity():
    policy = _policy(1)
    first = await policy.acquire(lambda _detail: None)
    assert first is not None
    observations: list[dict[str, Any]] = []

    cancelled = asyncio.create_task(policy.acquire(observations.append))
    await asyncio.sleep(0)
    survivor = asyncio.create_task(policy.acquire(observations.append))
    await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    first.release()
    permit = await asyncio.wait_for(survivor, timeout=1)
    assert permit is not None
    permit.release()

    assert [item["outcome"] for item in observations] == [
        "cancelled",
        "admitted_after_wait",
    ]


@pytest.mark.asyncio
async def test_cancellation_storm_removes_waiters_immediately():
    policy = _policy(1, group="cancellation-storm")
    first = await policy.acquire(lambda _detail: None)
    assert first is not None
    tasks = [asyncio.create_task(policy.acquire(lambda _detail: None)) for _ in range(1_000)]
    await asyncio.sleep(0)

    assert policy.identity is not None
    group = _get_or_create_group(policy.identity, policy.display_name or "", None)
    assert group is not None
    assert group.queued == 1_000

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert group.queued == 0
    assert not group._waiters
    first.release()


@pytest.mark.asyncio
async def test_unbounded_call_preserves_existing_provider_handoff():
    policy = AdmissionPolicy(
        max_in_flight=None,
        concurrency_group=None,
        queue_timeout=None,
        api_base=None,
    )
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    provider_exited = asyncio.Event()

    async def admitted_call() -> Any:
        raise AssertionError("unbounded calls must use the compatibility path")

    async def provider(**_kwargs: Any) -> litellm.ModelResponse:
        provider_started.set()
        try:
            await release_provider.wait()
            return _chat_response()
        finally:
            provider_exited.set()

    async def existing_handoff() -> Any:
        return await _litellm_acompletion({})

    with patch("litellm.acompletion", AsyncMock(side_effect=provider)):
        task = asyncio.create_task(
            _run_async_provider_call(
                admitted_call,
                policy,
                unadmitted_call=existing_handoff,
            )
        )
        await asyncio.wait_for(provider_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not provider_exited.is_set()
        release_provider.set()
        await asyncio.wait_for(provider_exited.wait(), timeout=1)


@pytest.mark.asyncio
async def test_queue_timeout_is_terminal_for_retry_wrapper():
    policy = _policy(1, timeout=0.01)
    first = await policy.acquire(lambda _detail: None)
    assert first is not None
    attempts = 0

    async def queued_attempt() -> None:
        nonlocal attempts
        attempts += 1
        await policy.acquire(lambda _detail: None)

    from nooa.unifiedllm.retry import with_retry

    try:
        with pytest.raises(AdmissionTimeoutError):
            await with_retry(
                queued_attempt,
                config=RetryConfig(
                    max_retries=3,
                    rate_limit_extra_retries=0,
                    base_delay=0,
                    jitter_factor=0,
                ),
            )
    finally:
        first.release()

    assert attempts == 1


@pytest.mark.asyncio
async def test_dispatched_cancellation_holds_slot_until_provider_exits():
    policy = _policy(1)
    first_started = asyncio.Event()
    finish_first = asyncio.Event()
    second_started = asyncio.Event()

    async def first_provider() -> str:
        first_started.set()
        await finish_first.wait()
        return "first"

    async def second_provider() -> str:
        second_started.set()
        return "second"

    first_task = asyncio.create_task(_run_async_provider_call(first_provider, policy))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    second_task = asyncio.create_task(_run_async_provider_call(second_provider, policy))
    await asyncio.sleep(0.02)
    assert not second_started.is_set()

    finish_first.set()
    assert await asyncio.wait_for(second_task, timeout=1) == "second"


@pytest.mark.asyncio
async def test_forked_child_starts_with_clean_process_local_accounting():
    policy = _policy(1, group="forked-group")
    inherited = await policy.acquire(lambda _detail: None)
    assert inherited is not None

    # Exercise the registered after-fork callback directly. The inherited
    # permit belongs to the parent's orphaned group; this policy must create a
    # clean group in the child registry instead of inheriting active=1.
    _reset_admission_groups_after_fork()
    child = await asyncio.wait_for(policy.acquire(lambda _detail: None), timeout=1)
    assert child is not None

    child.release()
    inherited.release()


@pytest.mark.asyncio
async def test_completion_and_responses_share_one_named_group():
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def track(response_factory: Callable[[], Any], **_kwargs: Any) -> Any:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return response_factory()

    async def chat_provider(**kwargs: Any) -> Any:
        return await track(_chat_response, **kwargs)

    async def responses_provider(**kwargs: Any) -> Any:
        return await track(_responses_response, **kwargs)

    completion = CompletionClient(
        "test-chat",
        concurrency_group="shared-gateway",
        max_in_flight=2,
    )
    responses = ResponsesClient(
        "test-responses",
        concurrency_group="shared-gateway",
        max_in_flight=2,
    )
    try:
        with (
            patch("litellm.acompletion", AsyncMock(side_effect=chat_provider)),
            patch("litellm.aresponses", AsyncMock(side_effect=responses_provider)),
        ):
            calls = [
                completion.acall([{"role": "user", "content": f"chat-{index}"}])
                for index in range(3)
            ]
            calls.extend(
                responses.acall([{"role": "user", "content": f"response-{index}"}])
                for index in range(3)
            )
            results = await asyncio.gather(*calls)
    finally:
        await completion.aclose()
        await responses.aclose()

    assert len(results) == 6
    assert peak == 2


@pytest.mark.asyncio
async def test_endpoint_alias_without_limit_joins_existing_budget():
    active = 0
    peak = 0

    async def provider(**_kwargs: Any) -> litellm.ModelResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _chat_response()

    owner = CompletionClient(
        "owner",
        api_base="https://gateway.example/v1/",
        max_in_flight=1,
    )
    alias = CompletionClient(
        "alias",
        api_base="https://GATEWAY.example:443/v1?ignored=true",
    )
    try:
        with patch("litellm.acompletion", AsyncMock(side_effect=provider)):
            await asyncio.gather(
                owner.acall([{"role": "user", "content": "one"}]),
                alias.acall([{"role": "user", "content": "two"}]),
            )
    finally:
        await owner.aclose()
        await alias.aclose()

    assert peak == 1


@pytest.mark.asyncio
async def test_retry_reacquires_and_backoff_does_not_hold_slot():
    calls_by_model: dict[str, int] = {}
    order: list[str] = []
    first_attempt_finished = asyncio.Event()

    class Retryable502(Exception):
        status_code = 502

    async def provider(**kwargs: Any) -> litellm.ModelResponse:
        model = kwargs["model"]
        calls_by_model[model] = calls_by_model.get(model, 0) + 1
        attempt = calls_by_model[model]
        order.append(f"{model}-{attempt}")
        if model == "retry-model" and attempt == 1:
            first_attempt_finished.set()
            raise Retryable502("502 Bad Gateway")
        return _chat_response(model)

    retrying = CompletionClient(
        "retry-model",
        concurrency_group="retry-group",
        max_in_flight=1,
        retry_config=RetryConfig(
            max_retries=1,
            rate_limit_extra_retries=0,
            base_delay=0.05,
            jitter_factor=0,
        ),
    )
    other = CompletionClient(
        "other-model",
        concurrency_group="retry-group",
        max_in_flight=1,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    try:
        with patch("litellm.acompletion", AsyncMock(side_effect=provider)):
            retry_task = asyncio.create_task(retrying.acall([{"role": "user", "content": "retry"}]))
            await asyncio.wait_for(first_attempt_finished.wait(), timeout=1)
            other_result = await other.acall([{"role": "user", "content": "other"}])
            retry_result = await retry_task
    finally:
        await retrying.aclose()
        await other.aclose()

    assert other_result.content == "other-model"
    assert retry_result.content == "retry-model"
    assert order == ["retry-model-1", "other-model-1", "retry-model-2"]


def test_group_coordinates_multiple_event_loops():
    active = 0
    peak = 0
    completed = 0
    state_lock = threading.Lock()
    start = threading.Barrier(3)

    async def run_batch() -> None:
        nonlocal active, peak, completed
        policy = _policy(2, group="cross-loop")

        async def worker() -> None:
            nonlocal active, peak, completed
            permit = await policy.acquire(lambda _detail: None)
            assert permit is not None
            with state_lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.002)
            with state_lock:
                active -= 1
                completed += 1
            permit.release()

        await asyncio.gather(*(worker() for _ in range(10)))

    def thread_target() -> None:
        start.wait()
        asyncio.run(run_batch())

    threads = [threading.Thread(target=thread_target) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert completed == 20
    assert peak == 2


def test_admission_observation_records_otel_event_and_harness_metrics():
    span = MagicMock()
    span.is_recording.return_value = True
    harness = HarnessMetrics()
    detail = {
        "group": "shared-gateway",
        "outcome": "admitted_after_wait",
        "queued": True,
        "wait_s": 0.25,
        "queue_depth": 3,
        "max_in_flight": 2,
    }

    from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

    token = _llm_metrics_callback.set(
        lambda event, value: harness.record_llm_queue(value) if event == "llm_queue" else None
    )
    try:
        with patch("opentelemetry.trace.get_current_span", return_value=span):
            _record_admission_observation(detail)
    finally:
        _llm_metrics_callback.reset(token)

    span.add_event.assert_called_once_with("llm.queue", attributes=detail)
    attrs = harness.to_span_attributes()
    assert attrs["harness.llm_queue.admissions"] == 1
    assert attrs["harness.llm_queue.queued"] == 1
    assert attrs["harness.llm_queue.max_depth"] == 3
    assert attrs["harness.llm_queue.wait.total_s"] == 0.25


def test_call_cap_rejection_records_harness_metric():
    harness = HarnessMetrics()

    harness.record_llm_queue(
        {
            "group": "application-run",
            "outcome": "call_cap",
            "queued": False,
            "wait_s": 0.0,
            "queue_depth": 0,
            "max_in_flight": 4,
            "max_calls": 100,
            "admitted_calls": 100,
        }
    )

    attrs = harness.to_span_attributes()
    assert attrs["harness.llm_queue.call_cap_rejections"] == 1
    assert attrs.get("harness.llm_queue.admissions", 0) == 0
