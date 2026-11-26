# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real task completion during resource cancellation, without field mutation."""

import asyncio

import pytest
from nooa_atom.agent.work_manager import WorkManager

from nooa.runtime.channels import QueueManager


class EntryBarrier:
    def __init__(self, fail):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail
        self.error = OSError("controlled entry failure")
        self.exits = []
        self.sends = []

    async def __aenter__(self):
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise self.error
        return self

    async def __aexit__(self, *exc):
        self.exits.append(exc)

    async def send(self, item):
        self.sends.append(item)
        await asyncio.Event().wait()


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("fail,with_request", [(True, True), (True, False), (False, True)])
async def test_cancel_keeps_real_owner_task_across_request_cleanup(fail, with_request, concurrent):
    # Repeated barrier scheduling, with the two negative controls from the
    # original race. No sleeps of real duration or manufactured _task states.
    baseline = asyncio.all_tasks()
    for iteration in range(10):
        channel = QueueManager().queue("jobs")
        manager = WorkManager(channel)
        resource = EntryBarrier(fail)
        job = manager.launch(resource, name=f"resource_{iteration}")
        owner_task = job._task
        request_task = None
        try:
            await resource.started.wait()
            request = job.send("work") if with_request else None
            if request is not None:
                request_task = request._task
            await asyncio.sleep(0)  # real request is now waiting on entry
            cancellation = asyncio.create_task(job.cancel())
            cancellations = [cancellation]
            if concurrent:
                cancellations.append(asyncio.create_task(job.cancel()))
            # FIFO: cancel starts its request gather before entry can finish.
            resource.release.set()
            await asyncio.gather(*cancellations)
            await asyncio.sleep(0)  # completion callbacks have released contexts
            assert owner_task.done() and job._task is None
            assert all(t.done() and t.exception() is None for t in cancellations)
            if with_request:
                assert request_task.done() and request._task is None
                assert request.status == "cancelled"
            if fail and with_request:
                assert job.status == "failed"
                assert "controlled entry failure" in job.error
                assert job._error is resource.error  # observation only; resources are not awaitable
            else:
                assert job.status == "cancelled"
            assert resource.exits == ([] if fail else [(None, None, None)])
            assert resource.sends == []
            await asyncio.gather(job.cancel(), job.cancel())  # finished is idempotent
            assert not manager.running()
            assert channel.snapshot() == []
        finally:
            await manager.aclose()
            await asyncio.gather(
                owner_task, *([request_task] if request_task else []), return_exceptions=True
            )
    assert not [t for t in asyncio.all_tasks() - baseline if not t.done()]


@pytest.mark.parametrize("fail", [False, True])
async def test_cancel_already_finished_task_preserves_outcome(fail):
    channel = QueueManager().queue("jobs")
    manager = WorkManager(channel)
    error = ValueError("original error")

    async def work():
        if fail:
            raise error
        return "result"

    job = manager.launch(work())
    owner_task = job._task
    try:
        await owner_task
        await asyncio.sleep(0)
        assert job._task is None
        before = (job.status, job.result, job.error, channel.snapshot())
        await asyncio.gather(job.cancel(), job.cancel())
        assert (job.status, job.result, job.error, channel.snapshot()) == before
        if fail:
            with pytest.raises(ValueError) as caught:
                await job
            assert caught.value is error
        else:
            assert await job == "result"
    finally:
        await manager.aclose()
