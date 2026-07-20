# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the OTLP ingest endpoint and write-queue worker.

httpx's ASGITransport only handles HTTP — it does not trigger the FastAPI
lifespan, so the background worker is never started via the normal app
startup path.  We therefore test the two concerns in isolation:

  1. The endpoint handler: PUT raw bytes onto the queue, return {"queued": True}.
     No JSON parsing happens on the event loop — the handler is pure I/O.
  2. The worker function: drain the queue into batches, call
     otlp_store.ingest_batch_write_bytes(batch) for each batch, and survive
     exceptions without dying.
"""

import asyncio
import json as _json
import time as _time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store():
    store = MagicMock()
    store.init_db.return_value = 0
    store.get_stats.return_value = {"sessions": 0, "experiments": 0}
    store.ingest_batch_write_bytes.return_value = [
        {"session_id": "s", "experiment": "e", "span_count": 1}
    ]
    store.DB_PATH = "/tmp/test_traces.db"
    return store


@pytest.fixture()
async def http_client(mock_store, tmp_path):
    """Async HTTP client wired to the FastAPI app with a mocked store.

    The lifespan is NOT triggered (httpx limitation), so no worker task runs.
    Suitable for testing the endpoint handler in isolation.
    """
    with patch("nooa.viewer.main.otlp_store", mock_store):
        from nooa.viewer.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------------
# Endpoint tests — verify the handler queues the body and returns immediately
# ---------------------------------------------------------------------------


class TestOtlpIngestEndpoint:
    async def test_returns_200_queued_true(self, http_client):
        resp = await http_client.post("/v1/traces", json={"resourceSpans": []})
        assert resp.status_code == 200
        assert resp.json() == {"queued": True}

    async def test_does_not_call_ingest_synchronously(self, http_client, mock_store):
        """The endpoint must return before ingest_batch_write_bytes() is called (non-blocking)."""
        await http_client.post("/v1/traces", json={"resourceSpans": []})
        # Without a running worker, ingest_batch_write_bytes should NOT have been called yet.
        mock_store.ingest_batch_write_bytes.assert_not_called()


# ---------------------------------------------------------------------------
# Worker tests — verify the background task drains the queue correctly
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="class")
def _fresh_write_executor():
    """Ensure _write_executor is alive — a prior TestClient shutdown may have killed it."""
    import nooa.viewer.main as main_mod

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer-test")
    original = main_mod._write_executor
    main_mod._write_executor = executor
    yield
    executor.shutdown(wait=False)
    main_mod._write_executor = original


class TestIngestWorker:
    async def test_worker_calls_ingest_batch_write_bytes_with_queued_body(self, mock_store):
        from nooa.viewer.main import _ingest_worker

        queue: asyncio.Queue = asyncio.Queue()
        raw = b'{"resourceSpans": [{"spans": [{"spanId": "abc"}]}]}'
        await queue.put(raw)

        with patch("nooa.viewer.main._ingest_queue", queue):
            with patch("nooa.viewer.main.otlp_store", mock_store):
                task = asyncio.create_task(_ingest_worker())
                await asyncio.sleep(0.1)
                task.cancel()

        # Worker batches all pending items; one payload → called with [raw]
        mock_store.ingest_batch_write_bytes.assert_called_once_with([raw])

    async def test_worker_batches_multiple_bodies(self, mock_store):
        """All payloads already in the queue are batched into one transaction."""
        from nooa.viewer.main import _ingest_worker

        queue: asyncio.Queue = asyncio.Queue()
        payloads = [_json.dumps({"resourceSpans": [], "seq": i}).encode() for i in range(4)]
        for p in payloads:
            await queue.put(p)

        with patch("nooa.viewer.main._ingest_queue", queue):
            with patch("nooa.viewer.main.otlp_store", mock_store):
                task = asyncio.create_task(_ingest_worker())
                await asyncio.sleep(0.1)
                task.cancel()

        # All 4 payloads were in the queue — one batch call.
        mock_store.ingest_batch_write_bytes.assert_called_once_with(payloads)

    async def test_worker_survives_ingest_exception(self, mock_store):
        """An exception from ingest_batch_write_bytes() must not kill the worker."""
        from nooa.viewer.main import _ingest_worker

        queue: asyncio.Queue = asyncio.Queue()

        good_result = [{"session_id": "s", "experiment": "e", "span_count": 1}]
        mock_store.ingest_batch_write_bytes.side_effect = [RuntimeError("db locked"), good_result]

        bad_payload = b'{"resourceSpans": "bad"}'
        good_payload = b'{"resourceSpans": []}'

        await queue.put(bad_payload)

        async def _wait_for_call_count(n: int, timeout: float = 5.0) -> None:
            deadline = _time.monotonic() + timeout
            while _time.monotonic() < deadline:
                if mock_store.ingest_batch_write_bytes.call_count >= n:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError(
                f"timed out waiting for ingest_batch_write_bytes call_count >= {n}; "
                f"got {mock_store.ingest_batch_write_bytes.call_count}"
            )

        with patch("nooa.viewer.main._ingest_queue", queue):
            with patch("nooa.viewer.main.otlp_store", mock_store):
                task = asyncio.create_task(_ingest_worker())
                # Wait for the bad payload to be drained and raise — without
                # this barrier, a slow CI runner can leave the queue holding
                # both payloads at the next ``put`` so the worker batches them
                # into a single call and the second-call assertion fails.
                await _wait_for_call_count(1)
                await queue.put(good_payload)
                await _wait_for_call_count(2)
                task.cancel()

        assert mock_store.ingest_batch_write_bytes.call_count == 2
        assert mock_store.ingest_batch_write_bytes.call_args_list[1].args[0] == [good_payload]


# ---------------------------------------------------------------------------
# Event-loop isolation test — verify GET handlers don't block POST ingest
# ---------------------------------------------------------------------------


class TestEventLoopIsolation:
    """Verify that slow GET handlers (DB reads) don't delay POST /v1/traces.

    Root cause of the original DROP bug: route handlers were ``async def``
    and called otlp_store functions synchronously on the event loop.  Under
    parallel eval runs, a user clicking through the UI triggered heavy SQLite
    reads that blocked the event loop, preventing the POST handler from reading
    BSP export bodies in time.

    The fix: route handlers are now ``def`` (sync), which FastAPI/anyio runs
    in a thread pool.  The event loop stays free for the POST ingest path
    regardless of how long the GET takes.

    This test simulates a 2 s blocking DB read in a GET handler concurrent
    with a POST /v1/traces, and asserts the POST returns in under 1 s.
    """

    @pytest.mark.flaky(reruns=2, reason="CI timing: 1s threshold sensitive to runner load")
    async def test_post_not_blocked_by_slow_get(self, mock_store):
        """POST /v1/traces must return quickly while a slow GET is in-flight."""
        from nooa.viewer.main import app

        # Simulate a heavy SQLite read (2 s) in the evaluations-tab endpoint.
        def slow_list_sessions(*args, **kwargs):
            _time.sleep(2.0)
            return []

        mock_store.list_sessions.side_effect = slow_list_sessions

        with patch("nooa.viewer.main.otlp_store", mock_store):
            with patch("nooa.viewer.eval_routes.otlp_store", mock_store):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    t0 = asyncio.get_event_loop().time()

                    # Fire GET (slow) and POST (should be fast) concurrently.
                    get_task = asyncio.create_task(client.get("/api/eval/experiments"))
                    post_task = asyncio.create_task(
                        client.post(
                            "/v1/traces",
                            content=b'{"resourceSpans":[]}',
                            headers={"Content-Type": "application/json"},
                        )
                    )
                    get_resp, post_resp = await asyncio.gather(get_task, post_task)

                    post_elapsed = asyncio.get_event_loop().time() - t0

        assert post_resp.status_code == 200, f"POST failed: {post_resp.text}"
        assert post_elapsed < 1.0, (
            f"POST /v1/traces took {post_elapsed:.2f}s while GET was in-flight — "
            "event loop is being blocked by a sync DB read in the GET handler"
        )
