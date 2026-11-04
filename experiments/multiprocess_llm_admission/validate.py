# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate shared multiprocess admission against a real local HTTP gateway."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    BrokerAdmissionConfig,
)

PROCESSES = 8
CALLS_PER_PROCESS = 25
GATEWAY_CAPACITY = 4
CALL_CAP = 19
PROVIDER_LATENCY_S = 0.03


class _GatewayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0
        self.overloads = 0

    def enter(self) -> bool:
        with self.lock:
            if self.active >= GATEWAY_CAPACITY:
                self.overloads += 1
                return False
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            return True

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _Gateway(ThreadingHTTPServer):
    daemon_threads = True
    state: _GatewayState


class _GatewayHandler(BaseHTTPRequestHandler):
    server: _Gateway

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        if not self.server.state.enter():
            self.send_response(503)
            self.end_headers()
            return
        try:
            time.sleep(PROVIDER_LATENCY_S)
            self.send_response(200)
            self.end_headers()
        finally:
            self.server.state.leave()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        del format, args


@dataclass(slots=True)
class _Result:
    name: str
    requested: int
    successful: int
    overloaded: int
    capped: int
    errors: int
    gateway_peak: int
    gateway_calls: int
    gateway_overloads: int
    elapsed_s: float


def _worker(
    config: BrokerAdmissionConfig | None,
    target_url: str,
    calls: int,
    start: Any,
    results: Any,
) -> None:
    async def run() -> tuple[int, int, int, int]:
        controller = config.controller() if config is not None else None
        limits = httpx.Limits(max_connections=calls, max_keepalive_connections=0)
        async with httpx.AsyncClient(timeout=10, limits=limits, trust_env=False) as client:

            async def invoke(index: int) -> str:
                permit = None
                if controller is not None:
                    try:
                        permit = await controller.acquire(lambda _detail: None)
                    except AdmissionCallCapError:
                        return "capped"
                    except Exception:
                        return "error"
                try:
                    response = await client.post(target_url, json={"call": index})
                except Exception:
                    return "error"
                finally:
                    if permit is not None:
                        permit.release()
                return "ok" if response.status_code == 200 else "overloaded"

            outcomes = await asyncio.gather(*(invoke(index) for index in range(calls)))
            return (
                outcomes.count("ok"),
                outcomes.count("overloaded"),
                outcomes.count("capped"),
                outcomes.count("error"),
            )

    start.wait()
    results.put(asyncio.run(run()))


def _run(
    name: str,
    gateway: _Gateway,
    target_url: str,
    config: BrokerAdmissionConfig | None,
    *,
    processes: int,
    calls_per_process: int,
) -> _Result:
    gateway.state = _GatewayState()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(config, target_url, calls_per_process, start, result_queue),
        )
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    started = time.perf_counter()
    start.set()
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)

    records: list[tuple[int, int, int, int]] = []
    for _ in workers:
        try:
            records.append(result_queue.get(timeout=2))
        except queue.Empty:
            records.append((0, 0, 0, calls_per_process))
    return _Result(
        name=name,
        requested=processes * calls_per_process,
        successful=sum(record[0] for record in records),
        overloaded=sum(record[1] for record in records),
        capped=sum(record[2] for record in records),
        errors=sum(record[3] for record in records),
        gateway_peak=gateway.state.peak,
        gateway_calls=gateway.state.calls,
        gateway_overloads=gateway.state.overloads,
        elapsed_s=round(time.perf_counter() - started, 3),
    )


def main() -> int:
    gateway = _Gateway(("127.0.0.1", 0), _GatewayHandler)
    gateway.state = _GatewayState()
    thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    thread.start()
    host = str(gateway.server_address[0])
    port = int(gateway.server_address[1])
    target_url = f"http://{host}:{port}/v1/chat/completions"
    try:
        baseline = _run(
            "unprotected",
            gateway,
            target_url,
            None,
            processes=PROCESSES,
            calls_per_process=CALLS_PER_PROCESS,
        )
        with AdmissionBroker(
            group="multiprocess-stress",
            max_in_flight=GATEWAY_CAPACITY,
        ) as broker:
            protected = _run(
                "parent_broker",
                gateway,
                target_url,
                broker.controller_config(queue_timeout=10),
                processes=PROCESSES,
                calls_per_process=CALLS_PER_PROCESS,
            )
            protected_snapshot = asdict(broker.snapshot())

        cap_processes = 4
        cap_calls_per_process = 8
        with AdmissionBroker(
            group="multiprocess-call-cap",
            max_in_flight=GATEWAY_CAPACITY,
            max_calls=CALL_CAP,
        ) as broker:
            capped = _run(
                "parent_broker_call_cap",
                gateway,
                target_url,
                broker.controller_config(queue_timeout=10),
                processes=cap_processes,
                calls_per_process=cap_calls_per_process,
            )
            cap_snapshot = asdict(broker.snapshot())
    finally:
        gateway.shutdown()
        gateway.server_close()
        thread.join(timeout=2)

    errors = []
    if baseline.gateway_overloads == 0:
        errors.append("unprotected baseline did not reproduce gateway overload")
    if (
        protected.successful != PROCESSES * CALLS_PER_PROCESS
        or protected.gateway_overloads
        or protected.gateway_peak > GATEWAY_CAPACITY
    ):
        errors.append("parent broker failed the multiprocess concurrency criterion")
    expected_capped = cap_processes * cap_calls_per_process - CALL_CAP
    if capped.gateway_calls != CALL_CAP or capped.capped != expected_capped:
        errors.append("parent broker failed the exact call-cap criterion")

    print(
        json.dumps(
            {
                "configuration": {
                    "processes": PROCESSES,
                    "calls_per_process": CALLS_PER_PROCESS,
                    "gateway_capacity": GATEWAY_CAPACITY,
                    "call_cap": CALL_CAP,
                },
                "baseline": asdict(baseline),
                "protected": asdict(protected),
                "protected_broker": protected_snapshot,
                "call_cap": asdict(capped),
                "call_cap_broker": cap_snapshot,
                "validation_errors": errors,
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
