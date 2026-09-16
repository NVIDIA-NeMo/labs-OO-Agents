# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one LLM concurrency ceiling and call cap across spawned processes.

Set ``NOOA_LLM_MODEL`` to a configured model alias before running this example.
The parent owns the broker; children receive only serializable connection
settings and construct their own UnifiedLLM clients.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os

from nooa.unifiedllm import AdmissionBroker, BrokerAdmissionConfig, get_llm_client


def child_process(
    worker_id: int,
    model: str,
    admission: BrokerAdmissionConfig,
) -> None:
    async def run() -> None:
        llm = get_llm_client(
            model,
            admission_controller=admission.controller(),
        )
        try:
            response = await llm.acall(
                [{"role": "user", "content": f"Reply with worker {worker_id}."}]
            )
            print(f"worker={worker_id} response={response.content!r}")
        finally:
            await llm.aclose()

    asyncio.run(run())


def main() -> None:
    model = os.environ.get("NOOA_LLM_MODEL")
    if not model:
        raise SystemExit("Set NOOA_LLM_MODEL to a configured model alias")

    context = multiprocessing.get_context("spawn")
    with AdmissionBroker(
        group="example-application-run",
        max_in_flight=2,
        max_calls=4,
    ) as broker:
        config = broker.controller_config(queue_timeout=30)
        workers = [
            context.Process(target=child_process, args=(index, model, config)) for index in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"Example worker {worker.pid} exited with {worker.exitcode}")

        print(broker.snapshot())


if __name__ == "__main__":
    main()
