# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise admission pacing through real loopback HTTP provider transports."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import litellm
import pytest

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionControl,
    AdmissionControlConfig,
    AdmissionTimeoutError,
    BrokerAdmissionConfig,
    CompletionClient,
    ResponsesClient,
    RetryConfig,
    UnifiedLLM,
)
from nooa.unifiedllm import admission as admission_module

_NO_RETRY = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
_FAST_RETRY = RetryConfig(
    max_retries=1,
    rate_limit_extra_retries=0,
    base_delay=0.01,
    rate_limit_base_delay=0.01,
    max_delay=0.01,
    jitter_factor=0,
)
_MESSAGES = [{"role": "user", "content": "Synthetic local test"}]


@pytest.fixture(autouse=True)
def _isolated_groups() -> Any:
    """Prevent one test's rate or cooldown deadline from affecting another."""
    admission_module._reset_admission_groups_for_tests()
    yield
    admission_module._reset_admission_groups_for_tests()


@dataclass
class _Request:
    """Record actual gateway arrival and response time without request secrets."""

    path: str
    arrived_at: float
    status: int | None = None
    replied_at: float | None = None


class _Gateway:
    """Minimal OpenAI-compatible HTTP fixture with explicitly controlled outcomes."""

    def __init__(self, respond: Callable[[_Request], Awaitable[tuple[int, dict[str, str]]]]):
        """Store the response policy and own the fixture's observable state."""
        self.respond = respond
        self.requests: list[_Request] = []
        self.errors: list[Exception] = []
        self.handlers: set[asyncio.Task[Any]] = set()
        self.server: asyncio.Server | None = None

    @property
    def api_base(self) -> str:
        """Expose a fresh loopback-only endpoint for this fixture."""
        assert self.server is not None and self.server.sockets
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Read one real HTTP request and return a valid Chat or Responses payload."""
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode("ascii").split("\r\n")
            method, path, _version = lines[0].split(" ", 2)
            assert method == "POST"
            headers = {
                key.lower(): value.strip()
                for line in lines[1:]
                if line
                for key, value in [line.split(":", 1)]
            }
            await reader.readexactly(int(headers.get("content-length", "0")))
            request = _Request(path, time.monotonic())
            self.requests.append(request)
            status, response_headers = await self.respond(request)
            request.status = status
            payload = (
                _success_payload(path)
                if status == 200
                else {
                    "error": {
                        "message": "Synthetic gateway overload",
                        "type": "rate_limit_error" if status == 429 else "server_error",
                        "code": "rate_limit_exceeded" if status == 429 else "server_overloaded",
                    }
                }
            )
            body = json.dumps(payload).encode("utf-8")
            response_headers = {
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Connection": "close",
                **response_headers,
            }
            raw_headers = "".join(f"{key}: {value}\r\n" for key, value in response_headers.items())
            request.replied_at = time.monotonic()
            writer.write(f"HTTP/1.1 {status} Test\r\n{raw_headers}\r\n".encode("ascii") + body)
            await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.errors.append(error)
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.handlers.discard(task)


def _success_payload(path: str) -> dict[str, Any]:
    """Produce minimal valid responses for the actual LiteLLM/SDK parsers."""
    if path == "/v1/chat/completions":
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    assert path == "/v1/responses"
    return {
        "id": "resp-local",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg-local",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


@asynccontextmanager
async def _gateway(
    respond: Callable[[_Request], Awaitable[tuple[int, dict[str, str]]]],
) -> AsyncIterator[_Gateway]:
    """Own every socket and handler so test failures leave no background work."""
    gateway = _Gateway(respond)
    gateway.server = await asyncio.start_server(gateway.handle, "127.0.0.1", 0)
    try:
        yield gateway
    finally:
        gateway.server.close()
        await gateway.server.wait_closed()
        handlers = list(gateway.handlers)
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        assert not gateway.errors, gateway.errors


def _base_client(api: str, endpoint: str, retry: RetryConfig = _NO_RETRY) -> UnifiedLLM:
    """Disable hidden SDK retries so wire requests represent NOOA provider attempts."""
    client_type = CompletionClient if api == "chat" else ResponsesClient
    return client_type(
        "openai/gpt-4o-mini",
        api_base=endpoint,
        api_key="local-test-key",  # noqa: S106 -- inert loopback-only test key
        stream=False,
        num_retries=0,
        max_retries=0,
        cache_breakpoint=None,
        retry_config=retry,
    )


async def _wait_until(predicate: Callable[[], bool], timeout: float = 10) -> None:
    """Wait for a causal condition, with a generous deadline instead of fixed sleeps."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _local_counts(client: AdmissionControl) -> tuple[int, int]:
    """Inspect slot accounting only to synchronize deterministic queue tests."""
    group = admission_module._groups[client.controller.identity]
    return group.active, group.queued


@asynccontextmanager
async def _controlled(
    api: str,
    endpoint: str,
    backend: str,
    *,
    timeout: float | None = None,
    retry: RetryConfig = _NO_RETRY,
    rate: float | None = None,
    max_cooldown: float | None = None,
) -> AsyncIterator[tuple[AdmissionControl, AdmissionBroker | None]]:
    """Exercise the same transport contract for local and broker-owned policies."""
    base = _base_client(api, endpoint, retry)
    broker = None
    controller = None
    try:
        if backend == "broker":
            broker = AdmissionBroker(
                max_in_flight=1,
                group="wire-test",
                requests_per_second=rate,
                max_cooldown=max_cooldown,
            ).start()
            controller = broker.controller(queue_timeout=timeout)
            config = AdmissionControlConfig(controller=controller)
        else:
            config = AdmissionControlConfig(
                max_in_flight=1,
                concurrency_group="wire-test",
                queue_timeout=timeout,
                requests_per_second=rate,
                max_cooldown=max_cooldown,
            )
        yield AdmissionControl(base, config), broker
    finally:
        await base.aclose()
        if controller is not None:
            controller.close()
        if broker is not None:
            broker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_real_gateway_transport_response_contract(api: str) -> None:
    """Both unmocked HTTP provider paths parse the fixture's successful payload."""

    async def success(_request: _Request) -> tuple[int, dict[str, str]]:
        """Accept one request through the real provider transport."""
        return 200, {}

    async with _gateway(success) as gateway:
        client = _base_client(api, gateway.api_base)
        try:
            result = await asyncio.wait_for(client.acall(_MESSAGES), timeout=10)
        finally:
            await client.aclose()
    assert result.content == "ok"
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_pacing_prevents_synthetic_rate_quota_rejections(api: str) -> None:
    """Concurrency-only bursts retry/fail; paced calls all pass a local HTTP quota."""
    reports: dict[str, dict[str, Any]] = {}
    for label, rate in (("concurrency_only", None), ("paced", 4.0)):
        last_success = float("-inf")

        async def quota(request: _Request) -> tuple[int, dict[str, str]]:
            """Enforce a synthetic arrival quota independently of concurrent calls."""
            nonlocal last_success
            # 120ms minimum spacing leaves 130ms transport headroom at 4rps.
            if request.arrived_at - last_success < 0.12:
                return 429, {}
            last_success = request.arrived_at
            return 200, {}

        async with _gateway(quota) as gateway:
            base = _base_client(api, gateway.api_base, _FAST_RETRY)
            client = AdmissionControl(
                base,
                AdmissionControlConfig(
                    max_in_flight=6,
                    concurrency_group=label,
                    requests_per_second=rate,
                ),
            )
            started = time.monotonic()
            try:
                outcomes = await asyncio.wait_for(
                    asyncio.gather(
                        *(client.acall(_MESSAGES) for _ in range(6)), return_exceptions=True
                    ),
                    timeout=15,
                )
            finally:
                await base.aclose()
            reports[label] = {
                "successes": sum(not isinstance(value, BaseException) for value in outcomes),
                "overloads": sum(request.status == 429 for request in gateway.requests),
                "provider_attempts": len(gateway.requests),
                "runtime_s": round(time.monotonic() - started, 3),
            }
    print(json.dumps({"synthetic_rate_quota": api, **reports}, sort_keys=True))
    assert reports["concurrency_only"]["overloads"] > 0
    assert reports["concurrency_only"]["provider_attempts"] > 6
    assert reports["paced"]["successes"] == 6
    assert reports["paced"]["overloads"] == 0
    assert reports["paced"]["provider_attempts"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("backend", ["local", "broker"])
@pytest.mark.parametrize("status", [429, 503, 529])
async def test_final_overload_delays_already_queued_http_callers(
    api: str, backend: str, status: int
) -> None:
    """Even a terminal provider attempt shares Retry-After before its slot releases."""
    finish_first = asyncio.Event()

    async def respond(request: _Request) -> tuple[int, dict[str, str]]:
        """Gate the overload response until the later callers are already queued."""
        if len(gateway.requests) == 1:
            await finish_first.wait()
            return status, {"Retry-After": "1"}
        return 200, {}

    async with _gateway(respond) as gateway:
        async with _controlled(api, gateway.api_base, backend, max_cooldown=0.5) as (
            client,
            broker,
        ):
            tasks = [asyncio.create_task(client.acall(_MESSAGES))]
            try:
                await _wait_until(lambda: len(gateway.requests) == 1)
                tasks.extend(asyncio.create_task(client.acall(_MESSAGES)) for _ in range(2))
                await _wait_until(
                    lambda: (
                        broker.snapshot().queued == 2 if broker else _local_counts(client)[1] == 2
                    )
                )
                finish_first.set()
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=10
                )
                assert isinstance(outcomes[0], Exception)
                assert getattr(outcomes[0], "status_code", None) == status
                assert [outcome.content for outcome in outcomes[1:]] == ["ok", "ok"]
                assert len(gateway.requests) == 3
                first, second, _third = gateway.requests
                assert first.replied_at is not None
                assert second.arrived_at - first.replied_at >= 0.45
                if broker:
                    await _wait_until(lambda: broker.snapshot().active == 0)
                    assert broker.snapshot().admitted_calls == 3
                    assert broker.snapshot().queued == 0
                else:
                    assert _local_counts(client) == (0, 0)
            finally:
                finish_first.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_retry_reacquires_after_shared_cooldown_through_http(api: str) -> None:
    """One NOOA retry produces exactly two paced wire attempts and no permit leak."""

    async def respond(_request: _Request) -> tuple[int, dict[str, str]]:
        """Reject exactly the first wire attempt, then permit its NOOA retry."""
        return (429, {"Retry-After": "1"}) if len(gateway.requests) == 1 else (200, {})

    async with _gateway(respond) as gateway:
        async with _controlled(
            api, gateway.api_base, "broker", retry=_FAST_RETRY, max_cooldown=0.4
        ) as (client, broker):
            result = await asyncio.wait_for(client.acall(_MESSAGES), timeout=10)
            assert result.content == "ok"
            assert len(gateway.requests) == 2
            first, second = gateway.requests
            assert first.replied_at is not None
            assert second.arrived_at - first.replied_at >= 0.35
            assert broker is not None
            await _wait_until(lambda: broker.snapshot().active == 0)
            assert broker.snapshot().admitted_calls == 2
            assert broker.snapshot().queued == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_pacing_timeout_and_queued_cancellation_make_no_http_attempt(api: str) -> None:
    """A rate wait is subject to queue deadlines and cancellation without dispatch."""

    async def success(_request: _Request) -> tuple[int, dict[str, str]]:
        """Accept dispatched calls, allowing absent dispatches to be counted."""
        return 200, {}

    async with _gateway(success) as gateway:
        async with _controlled(api, gateway.api_base, "local", rate=2) as (client, _broker):
            assert (await client.acall(_MESSAGES)).content == "ok"
            timeout_client = AdmissionControl(
                client.base_llm,
                AdmissionControlConfig(
                    max_in_flight=1,
                    concurrency_group="wire-test",
                    queue_timeout=0.02,
                    requests_per_second=2,
                ),
            )
            with pytest.raises(AdmissionTimeoutError):
                await timeout_client.acall(_MESSAGES)
            queued = asyncio.create_task(client.acall(_MESSAGES))
            await _wait_until(lambda: _local_counts(client)[1] == 1)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert len(gateway.requests) == 1
            assert _local_counts(client) == (0, 0)
            assert (await asyncio.wait_for(client.acall(_MESSAGES), timeout=10)).content == "ok"
            assert len(gateway.requests) == 2
            assert gateway.requests[1].arrived_at - gateway.requests[0].arrived_at >= 0.4


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_dispatched_cancelled_caller_still_shares_overload_feedback(api: str) -> None:
    """A shielded HTTP attempt retains its slot and reports overload after caller cancellation."""
    finish_first = asyncio.Event()

    async def respond(_request: _Request) -> tuple[int, dict[str, str]]:
        """Keep the first HTTP attempt alive until its caller has been cancelled."""
        if len(gateway.requests) == 1:
            await finish_first.wait()
            return 429, {"Retry-After": "1"}
        return 200, {}

    async with _gateway(respond) as gateway:
        async with _controlled(api, gateway.api_base, "local", max_cooldown=0.5) as (
            client,
            _broker,
        ):
            first = asyncio.create_task(client.acall(_MESSAGES))
            second = None
            try:
                await _wait_until(lambda: len(gateway.requests) == 1)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert _local_counts(client) == (1, 0)
                second = asyncio.create_task(client.acall(_MESSAGES))
                await _wait_until(lambda: _local_counts(client)[1] == 1)
                finish_first.set()
                assert (await asyncio.wait_for(second, timeout=10)).content == "ok"
                assert len(gateway.requests) == 2
                assert gateway.requests[0].replied_at is not None
                assert gateway.requests[1].arrived_at - gateway.requests[0].replied_at >= 0.45
                assert _local_counts(client) == (0, 0)
            finally:
                finish_first.set()
                if second is not None and not second.done():
                    second.cancel()
                await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
                await _wait_until(lambda: _local_counts(client) == (0, 0))


def _broker_http_child(
    api: str,
    endpoint: str,
    config: BrokerAdmissionConfig,
    start: Any,
    reports: Any,
) -> None:
    """Use the real client transport from a spawned interpreter against one shared broker."""

    async def run() -> str:
        """Make one independently owned HTTP attempt under the parent's policy."""
        controller = config.controller()
        base = _base_client(api, endpoint)
        client = AdmissionControl(base, AdmissionControlConfig(controller=controller))
        try:
            result = await client.acall(_MESSAGES)
            return result.content or ""
        except litellm.RateLimitError:
            return "429"
        finally:
            await base.aclose()
            controller.close()

    reports.put(("ready", api))
    if not start.wait(timeout=40):
        raise RuntimeError("Parent did not release the prepared child")
    reports.put(("outcome", api, asyncio.run(run())))


@pytest.mark.asyncio
async def test_broker_cooldown_is_shared_across_spawned_http_clients() -> None:
    """A Chat overload delays a queued Responses call in another process on the host."""
    finish_first = asyncio.Event()

    async def respond(_request: _Request) -> tuple[int, dict[str, str]]:
        """Wait for the second process to queue before rejecting the first call."""
        if len(gateway.requests) == 1:
            await finish_first.wait()
            return 429, {"Retry-After": "1"}
        return 200, {}

    context = multiprocessing.get_context("spawn")
    reports = context.Queue()
    starts = [context.Event(), context.Event()]
    processes = []
    async with _gateway(respond) as gateway:
        with AdmissionBroker(
            max_in_flight=1, max_cooldown=0.5, group="children-wire-test"
        ) as broker:
            try:
                for api, start in zip(("chat", "responses"), starts, strict=True):
                    process = context.Process(
                        target=_broker_http_child,
                        args=(api, gateway.api_base, broker.controller_config(), start, reports),
                    )
                    process.start()
                    processes.append(process)
                ready = [await asyncio.to_thread(reports.get, timeout=40) for _ in range(2)]
                assert {tuple(item) for item in ready} == {
                    ("ready", "chat"),
                    ("ready", "responses"),
                }
                starts[0].set()
                await _wait_until(lambda: len(gateway.requests) == 1)
                starts[1].set()
                await _wait_until(lambda: broker.snapshot().queued == 1)
                finish_first.set()
                outcomes = [await asyncio.to_thread(reports.get, timeout=40) for _ in range(2)]
                assert {tuple(item) for item in outcomes} == {
                    ("outcome", "chat", "429"),
                    ("outcome", "responses", "ok"),
                }
                for process in processes:
                    await asyncio.to_thread(process.join, 10)
                    assert not process.is_alive()
                    assert process.exitcode == 0
                assert len(gateway.requests) == 2
                first, second = gateway.requests
                assert first.path == "/v1/chat/completions"
                assert second.path == "/v1/responses"
                assert first.replied_at is not None
                assert second.arrived_at - first.replied_at >= 0.45
                await _wait_until(lambda: broker.snapshot().active == 0)
                assert broker.snapshot().admitted_calls == 2
                assert broker.snapshot().queued == 0
            finally:
                finish_first.set()
                for start in starts:
                    start.set()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    await asyncio.to_thread(process.join, 5)
                reports.close()
                reports.join_thread()
