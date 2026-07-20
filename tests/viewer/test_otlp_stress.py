# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stress test for the OTLP ingest endpoint under parallel load.

Simulates 10 concurrent "agents" (like Opus with max context) each posting
multiple large trace payloads simultaneously.  Verifies:

  - All HTTP requests return 200 {"queued": True}.
  - Every queued payload is eventually written to the store (no silent drops).
  - The write queue drains fully (queue is empty after join).

Design notes:
  - httpx ASGITransport does NOT trigger FastAPI lifespan, so we start
    _ingest_worker manually via asyncio.create_task().
  - asyncio.Queue.join() waits until every task_done() has been called,
    giving us a clean "all flushed" signal without polling.
  - Payload size is tuned to approximate a realistic Opus max-context trace:
    a span attribute string of ~4 KB repeated across many spans per batch.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATTR_VALUE_4KB = "x" * 4_096  # simulate a large LLM prompt/response attribute


def _make_payload(n_spans: int = 10) -> dict:
    """Build a realistic-ish OTLP JSON payload with *n_spans* spans."""
    spans = [
        {
            "traceId": f"trace{i:032x}",
            "spanId": f"span{i:016x}",
            "name": f"agent.turn.{i}",
            "kind": 1,
            "startTimeUnixNano": str(1_700_000_000_000_000_000 + i * 1_000_000),
            "endTimeUnixNano": str(1_700_000_000_000_000_000 + i * 1_000_000 + 500_000),
            "attributes": [
                {"key": "llm.input_messages", "value": {"stringValue": _ATTR_VALUE_4KB}},
                {"key": "llm.output_messages", "value": {"stringValue": _ATTR_VALUE_4KB}},
                {"key": "session.id", "value": {"stringValue": "stress-test-session"}},
            ],
        }
        for i in range(n_spans)
    ]
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "service.name", "value": {"stringValue": "nooa-stress"}}]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store():
    store = MagicMock()
    store.init_db.return_value = 0
    store.get_stats.return_value = {"sessions": 0, "experiments": 0}
    store.ingest.return_value = {"session_id": "s", "experiment": "e", "span_count": 1}
    store.ingest_batch_write_bytes.return_value = [
        {"session_id": "s", "experiment": "e", "span_count": 1}
    ]
    store.DB_PATH = "/tmp/stress_test.db"
    return store


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


class TestIngestStress:
    @pytest.mark.stress
    async def test_10_parallel_agents_all_delivered(self, mock_store):
        """10 concurrent agents each post 5 large batches — all must be ingested."""
        N_AGENTS = 10
        BATCHES_PER_AGENT = 5
        SPANS_PER_BATCH = 10  # 10 spans × ~8 KB attrs ≈ 80 KB per payload

        total_batches = N_AGENTS * BATCHES_PER_AGENT

        # Use a fresh queue bound to this test's event loop to avoid
        # cross-loop contamination between test runs.
        fresh_queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("nooa.viewer.main.otlp_store", mock_store):
            with patch("nooa.viewer.main._ingest_queue", fresh_queue):
                from nooa.viewer.main import _ingest_worker, app

                async def agent_session(agent_id: int, client: AsyncClient) -> list[int]:
                    """Post BATCHES_PER_AGENT payloads and return all HTTP status codes."""
                    statuses = []
                    for batch in range(BATCHES_PER_AGENT):
                        payload = _make_payload(n_spans=SPANS_PER_BATCH)
                        # Tag each payload so we can track it if needed
                        payload["_agent"] = agent_id
                        payload["_batch"] = batch
                        resp = await client.post("/v1/traces", json=payload)
                        statuses.append(resp.status_code)
                    return statuses

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # Start the worker manually (lifespan not triggered by ASGITransport)
                    worker = asyncio.create_task(_ingest_worker())

                    t0 = time.monotonic()
                    all_statuses = await asyncio.gather(
                        *[agent_session(i, client) for i in range(N_AGENTS)]
                    )
                    post_elapsed = time.monotonic() - t0

                    # Wait for every queued payload to be processed
                    await asyncio.wait_for(fresh_queue.join(), timeout=30.0)
                    drain_elapsed = time.monotonic() - t0

                    worker.cancel()

        # All HTTP responses must be 200
        flat_statuses = [s for agent in all_statuses for s in agent]
        assert len(flat_statuses) == total_batches, (
            f"Expected {total_batches} responses, got {len(flat_statuses)}"
        )
        non_200 = [s for s in flat_statuses if s != 200]
        assert not non_200, f"Non-200 responses: {non_200}"

        # Every payload must have been ingested — the worker batches items, so
        # count total payloads across all ingest_batch_write_bytes() calls.
        total_ingested = sum(
            len(call.args[0]) for call in mock_store.ingest_batch_write_bytes.call_args_list
        )
        assert total_ingested == total_batches, (
            f"Expected {total_batches} payloads ingested, got {total_ingested}"
        )

        # Queue must be fully drained
        assert fresh_queue.empty(), "Queue should be empty after join()"

        # Timing: HTTP phase should be fast (all 200s before drain)
        # Just assert it completed in reasonable time (no hard limit — CI may be slow)
        assert drain_elapsed < 30.0, f"Drain took too long: {drain_elapsed:.1f}s"
        _ = post_elapsed  # available for debugging if needed

    @pytest.mark.stress
    async def test_queue_survives_ingest_errors_under_load(self, mock_store):
        """Worker must not die when ingest() raises — remaining items still processed."""
        N_AGENTS = 5
        BATCHES_PER_AGENT = 4
        total_batches = N_AGENTS * BATCHES_PER_AGENT

        # Every other call raises — worker must survive and process the rest
        call_count = 0

        def flaky_ingest(payloads):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("simulated db contention")
            return [{"session_id": "s", "experiment": "e", "span_count": 1}] * len(payloads)

        mock_store.ingest_batch_write_bytes.side_effect = flaky_ingest

        fresh_queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("nooa.viewer.main.otlp_store", mock_store):
            with patch("nooa.viewer.main._ingest_queue", fresh_queue):
                from nooa.viewer.main import _ingest_worker, app

                async def agent_session(agent_id: int, client: AsyncClient) -> list[int]:
                    statuses = []
                    for _ in range(BATCHES_PER_AGENT):
                        resp = await client.post("/v1/traces", json=_make_payload())
                        statuses.append(resp.status_code)
                    return statuses

                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    worker = asyncio.create_task(_ingest_worker())

                    all_statuses = await asyncio.gather(
                        *[agent_session(i, client) for i in range(N_AGENTS)]
                    )

                    await asyncio.wait_for(fresh_queue.join(), timeout=30.0)
                    worker.cancel()

        # All HTTP responses must still be 200 — the queue never rejects
        flat_statuses = [s for agent in all_statuses for s in agent]
        assert all(s == 200 for s in flat_statuses), (
            f"Non-200: {[s for s in flat_statuses if s != 200]}"
        )

        # Every payload was attempted (worker didn't die after errors).
        # The worker batches payloads, so count total payloads across all calls.
        total_attempted = sum(
            len(call.args[0]) for call in mock_store.ingest_batch_write_bytes.call_args_list
        )
        assert total_attempted == total_batches, (
            f"Expected {total_batches} payloads attempted, got {total_attempted}"
        )
