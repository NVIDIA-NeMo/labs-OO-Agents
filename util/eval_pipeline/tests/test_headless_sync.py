# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic persistence barriers without a server or SQLite connection."""

import asyncio
import threading
from contextlib import asynccontextmanager

import httpx
import pytest

from eval_pipeline.headless_backend import _make_headless_app

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _client(monkeypatch, writer):
    monkeypatch.setattr("nooa.viewer.otlp_store.init_db", lambda: None)
    monkeypatch.setattr("nooa.viewer.otlp_store.ingest_batch_write_bytes", writer)
    app = _make_headless_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


def _observe_barrier(monkeypatch):
    queued = asyncio.Event()

    class ObservedQueue(asyncio.Queue):
        def put_nowait(self, item):
            super().put_nowait(item)
            if not isinstance(item, bytes):
                queued.set()

    monkeypatch.setattr("eval_pipeline.headless_backend.asyncio.Queue", ObservedQueue)
    return queued


@pytest.mark.parametrize("batch_limit", [1, 32])
async def test_sync_waits_for_preceding_batch_but_not_later_arrivals(monkeypatch, batch_limit):
    monkeypatch.setattr("eval_pipeline.headless_backend._INGEST_MAX_BATCH", batch_limit)
    started = {name: threading.Event() for name in (b"first", b"second", b"later")}
    release = {name: threading.Event() for name in started}
    persisted = []
    barrier_queued = _observe_barrier(monkeypatch)

    def writer(batch):
        started[batch[0]].set()
        assert release[batch[0]].wait(10), "test did not release writer"
        persisted.extend(batch)
        return [{} for _ in batch]

    async with _client(monkeypatch, writer) as client:
        task = None
        try:
            await client.post("/v1/traces", content=b"first")
            assert await asyncio.to_thread(started[b"first"].wait, 5)
            await client.post("/v1/traces", content=b"second")
            task = asyncio.create_task(client.post("/v1/sync"))
            await asyncio.wait_for(barrier_queued.wait(), 5)
            await client.post("/v1/traces", content=b"later")
            release[b"first"].set()
            assert await asyncio.to_thread(started[b"second"].wait, 5)
            assert not task.done(), "sync acknowledged an unwritten preceding batch"
            release[b"second"].set()
            response = await asyncio.wait_for(task, 5)
            assert response.status_code == 200
            assert response.json() == {"synced": True}
            assert persisted == [b"first", b"second"]
        finally:
            for event in release.values():
                event.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failure", ["exception", "skipped_payload"])
async def test_sync_reports_write_failure(monkeypatch, failure):
    def writer(batch):
        if failure == "exception":
            raise OSError("disk full")
        return []  # The store omits results for payloads that fail during ingestion.

    async with _client(monkeypatch, writer) as client:
        await client.post("/v1/traces", content=b"payload")
        for _ in range(2):
            response = await client.post("/v1/sync")
            assert response.status_code == 500
            assert "error" in response.json()


async def test_cancelled_sync_does_not_stop_worker(monkeypatch):
    started, release = threading.Event(), threading.Event()
    queued = _observe_barrier(monkeypatch)
    persisted = []

    def writer(batch):
        started.set()
        assert release.wait(10), "test did not release writer"
        persisted.extend(batch)
        return [{} for _ in batch]

    async with _client(monkeypatch, writer) as client:
        task = None
        try:
            await client.post("/v1/traces", content=b"payload")
            assert await asyncio.to_thread(started.wait, 5)
            task = asyncio.create_task(client.post("/v1/sync"))
            await asyncio.wait_for(queued.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            response = await asyncio.wait_for(client.post("/v1/sync"), 5)
            assert response.status_code == 200
            assert persisted == [b"payload"]
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
