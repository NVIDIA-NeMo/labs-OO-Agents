# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback validation for mixed-client UnifiedLLM admission control."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import litellm

from nooa.unifiedllm import (
    AdmissionControl,
    AdmissionControlConfig,
    CompletionClient,
    ResponsesClient,
    RetryConfig,
)


@dataclass
class Result:
    name: str
    successful: int
    total: int
    overloads: int
    gateway_peak: int
    elapsed_s: float


class GatewayState:
    def __init__(
        self,
        capacity: int,
        latency_s: float,
        *,
        expected_burst: int | None = None,
    ):
        self.capacity = capacity
        self.latency_s = latency_s
        self.expected_burst = expected_burst
        self.active = 0
        self.peak = 0
        self.overloads = 0
        self.arrivals = 0
        self.lock = threading.Lock()
        self.burst_arrived = threading.Event()

    def enter(self) -> bool:
        with self.lock:
            self.arrivals += 1
            if self.expected_burst is not None and self.arrivals >= self.expected_burst:
                self.burst_arrived.set()
            if self.active >= self.capacity:
                self.overloads += 1
                return False
            self.active += 1
            self.peak = max(self.peak, self.active)
            return True

    def exit(self) -> None:
        with self.lock:
            self.active -= 1

    def wait_for_burst(self) -> None:
        if self.expected_burst is not None:
            self.burst_arrived.wait(timeout=2)


class Gateway(ThreadingHTTPServer):
    state: GatewayState


class Handler(BaseHTTPRequestHandler):
    server: Gateway

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)

        if not self.server.state.enter():
            self._write_json(
                503,
                {
                    "error": {
                        "message": "simulated gateway capacity exceeded",
                        "type": "server_error",
                        "code": "overloaded",
                    }
                },
            )
            return

        try:
            self.server.state.wait_for_burst()
            time.sleep(self.server.state.latency_s)
            if self.path.rstrip("/").endswith("responses"):
                self._write_json(200, _responses_payload())
            else:
                self._write_json(200, _chat_payload())
        finally:
            self.server.state.exit()

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _chat_payload() -> dict[str, Any]:
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _responses_payload() -> dict[str, Any]:
    return {
        "id": "resp-local",
        "object": "response",
        "created_at": 0,
        "model": "local-model",
        "status": "completed",
        "output": [
            {
                "id": "msg-local",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


NO_RETRY = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
MESSAGES = [{"role": "user", "content": "validate admission"}]


def _clients(
    api_base: str,
    *,
    group: str | None = None,
    limit: int | None = None,
) -> tuple[Any, Any, Any]:
    common: dict[str, Any] = {
        "api_base": api_base,
        "api_key": "local-test-key",
        "retry_config": NO_RETRY,
        "num_retries": 0,
    }
    clients: tuple[Any, Any, Any] = (
        CompletionClient("openai/chat-a", **common),
        CompletionClient("openai/chat-b", **common),
        ResponsesClient("openai/responses", **common),
    )
    if limit is None:
        return clients
    config = AdmissionControlConfig(
        max_in_flight=limit,
        concurrency_group=group,
    )
    return (
        AdmissionControl(clients[0], config),
        AdmissionControl(clients[1], config),
        AdmissionControl(clients[2], config),
    )


async def _close_clients(
    clients: tuple[Any, Any, Any],
) -> None:
    await asyncio.gather(*(client.aclose() for client in clients))


async def _burst(
    state: GatewayState,
    api_base: str,
    *,
    group: str | None,
    limit: int | None,
    total: int,
) -> Result:
    clients = _clients(api_base, group=group, limit=limit)
    started = time.perf_counter()
    try:

        async def invoke(index: int) -> bool:
            client = clients[index % len(clients)]
            try:
                response = await client.acall(MESSAGES)
            except Exception:
                return False
            return response.content == "ok"

        outcomes = await asyncio.gather(*(invoke(index) for index in range(total)))
    finally:
        await _close_clients(clients)

    return Result(
        name="protected" if limit is not None else "unbounded",
        successful=sum(outcomes),
        total=total,
        overloads=state.overloads,
        gateway_peak=state.peak,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


async def _shadow_workflows(
    state: GatewayState,
    api_base: str,
    *,
    workflows: int,
    limit: int,
) -> Result:
    clients = _clients(api_base, group="staged-workload", limit=limit)
    started = time.perf_counter()
    try:

        async def workflow(index: int) -> bool:
            try:
                for client in clients:
                    response = await client.acall([{"role": "user", "content": f"target-{index}"}])
                    if response.content != "ok":
                        return False
            except Exception:
                return False
            return True

        outcomes = await asyncio.gather(*(workflow(index) for index in range(workflows)))
    finally:
        await _close_clients(clients)

    return Result(
        name="staged_workload",
        successful=sum(outcomes),
        total=workflows,
        overloads=state.overloads,
        gateway_peak=state.peak,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


async def _process_batch(api_base: str, process_id: int) -> int:
    clients = _clients(
        api_base,
        group="process-local-topology",
        limit=2,
    )
    try:

        async def invoke(index: int) -> bool:
            client = clients[index % 2]
            try:
                response = await client.acall(
                    [{"role": "user", "content": f"process-{process_id}-call-{index}"}]
                )
            except Exception:
                return False
            return response.content == "ok"

        outcomes = await asyncio.gather(*(invoke(index) for index in range(6)))
        return sum(outcomes)
    finally:
        await _close_clients(clients)


def _process_worker(
    api_base: str,
    process_id: int,
    start_event: Any,
    ready_event: Any,
    result_queue: Any,
) -> None:
    ready_event.set()
    if not start_event.wait(timeout=10):
        raise TimeoutError("Parent did not release the child-process start gate")
    result_queue.put(_process_batch_result(api_base, process_id))


def _process_batch_result(api_base: str, process_id: int) -> int:
    return asyncio.run(_process_batch(api_base, process_id))


def _multi_process_topology(state: GatewayState, api_base: str) -> Result:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    ready_events = [context.Event() for _ in range(2)]
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_process_worker,
            args=(api_base, process_id, start_event, ready_events[process_id], result_queue),
        )
        for process_id in range(2)
    ]
    started = time.perf_counter()
    for process in processes:
        process.start()
    if not all(ready.wait(timeout=10) for ready in ready_events):
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
        result_queue.close()
        result_queue.join_thread()
        raise RuntimeError("Child processes did not reach the start gate")
    start_event.set()
    for process in processes:
        process.join(timeout=20)
    if any(process.is_alive() or process.exitcode != 0 for process in processes):
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError("Child-process topology validation failed")

    successful = sum(result_queue.get(timeout=2) for _ in processes)
    result_queue.close()
    result_queue.join_thread()
    return Result(
        name="two_processes_limit_2",
        successful=successful,
        total=12,
        overloads=state.overloads,
        gateway_peak=state.peak,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


async def main() -> None:
    litellm.suppress_debug_info = True
    state = GatewayState(capacity=3, latency_s=0.03, expected_burst=20)
    server = Gateway(("127.0.0.1", 0), Handler)
    server.state = state
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    api_base = f"http://127.0.0.1:{server.server_port}/v1"

    results: list[Result] = []
    try:
        results.append(await _burst(state, api_base, group=None, limit=None, total=20))

        state = GatewayState(capacity=3, latency_s=0.03)
        server.state = state
        results.append(
            await _burst(
                state,
                api_base,
                group="loopback-protected",
                limit=3,
                total=20,
            )
        )

        state = GatewayState(capacity=5, latency_s=0.02)
        server.state = state
        results.append(
            await _shadow_workflows(
                state,
                api_base,
                workflows=15,
                limit=5,
            )
        )

        state = GatewayState(capacity=10, latency_s=0.1)
        server.state = state
        results.append(_multi_process_topology(state, api_base))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    print(json.dumps([asdict(result) for result in results], indent=2))

    unbounded, protected, shadow, multiprocess = results
    if not (
        unbounded.overloads > 0
        and unbounded.successful < unbounded.total
        and protected.successful == protected.total
        and protected.overloads == 0
        and protected.gateway_peak == 3
        and shadow.successful == shadow.total
        and shadow.overloads == 0
        and shadow.gateway_peak == 5
        and multiprocess.successful == multiprocess.total
        and multiprocess.overloads == 0
        and multiprocess.gateway_peak == 4
    ):
        raise SystemExit("Admission-control validation did not meet its invariants")


if __name__ == "__main__":
    asyncio.run(main())
