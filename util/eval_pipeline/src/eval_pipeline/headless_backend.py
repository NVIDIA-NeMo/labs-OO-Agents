# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless OTLP backend for eval pipeline scoring.

Spins up a minimal FastAPI/uvicorn server in a background thread that:
- Accepts OTLP span payloads at POST /v1/traces
- Stores them in a per-run temporary SQLite database
- Serves them back at GET /api/trace?session_id=... (same API as the full viewer)

Subprocess workers POST spans over HTTP — crossing the process boundary — and
scorers fetch them back via ``TraceExplorer.from_viewer`` pointing at the local
endpoint. No InMemorySpanExporter needed; all span data travels over HTTP so
asyncio and subprocess engines behave identically.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Request as _FastAPIRequest
from fastapi.responses import JSONResponse as _JSONResponse
from starlette.requests import ClientDisconnect as _ClientDisconnect

log = logging.getLogger(__name__)

_INGEST_MAX_BATCH = 32


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _make_headless_app():
    """Build a minimal FastAPI app: OTLP ingest + trace read routes, no UI."""
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware

    from nooa.viewer import otlp_store
    from nooa.viewer.trace_routes import router as trace_router

    _ingest_queue: asyncio.Queue[bytes | asyncio.Event] = asyncio.Queue()
    _write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="headless-writer")

    async def _ingest_worker() -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await _ingest_queue.get()
            # Sentinel event from /v1/sync — everything before it is written.
            if isinstance(item, asyncio.Event):
                item.set()
                _ingest_queue.task_done()
                continue
            batch: list[bytes] = [item]
            while len(batch) < _INGEST_MAX_BATCH:
                try:
                    next_item = _ingest_queue.get_nowait()
                    if isinstance(next_item, asyncio.Event):
                        next_item.set()
                        _ingest_queue.task_done()
                        continue
                    batch.append(next_item)
                except asyncio.QueueEmpty:
                    break
            try:
                await loop.run_in_executor(
                    _write_executor, otlp_store.ingest_batch_write_bytes, batch
                )
            except Exception:
                log.exception("headless ingest_worker: failed to write batch of %d", len(batch))
            finally:
                for _ in batch:
                    _ingest_queue.task_done()

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        otlp_store.init_db()
        worker = asyncio.create_task(_ingest_worker())
        try:
            yield
        finally:
            if not _ingest_queue.empty():
                await _ingest_queue.join()
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            _write_executor.shutdown(wait=True)

    app = fastapi.FastAPI(title="Eval OTLP Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Standard trace read routes (GET /api/trace, GET /api/traces, etc.)
    app.include_router(trace_router)

    @app.post("/v1/traces")
    async def ingest(request: _FastAPIRequest):
        try:
            body = await request.body()
        except _ClientDisconnect:
            return _JSONResponse(status_code=499, content={"error": "client disconnected"})
        await _ingest_queue.put(body)
        return _JSONResponse(content={"queued": True})

    @app.post("/v1/journal/messages")
    async def journal_messages(request: _FastAPIRequest):
        body = await request.json()
        if not isinstance(body, list):
            return _JSONResponse(status_code=400, content={"error": "Body must be a list"})
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _write_executor, otlp_store.ingest_journal_messages, body
        )
        return _JSONResponse(content=result)

    @app.post("/v1/journal/calls")
    async def journal_calls(request: _FastAPIRequest):
        body = await request.json()
        if not isinstance(body, dict) or not body.get("call_id") or not body.get("session_id"):
            return _JSONResponse(
                status_code=400, content={"error": "call_id and session_id are required"}
            )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _write_executor, otlp_store.ingest_journal_call, body
            )
        except ValueError as exc:
            return _JSONResponse(status_code=400, content={"error": str(exc)})
        return _JSONResponse(content=result)

    @app.post("/v1/journal/blocks")
    async def journal_blocks(request: _FastAPIRequest):
        body = await request.json()
        if not isinstance(body, list):
            return _JSONResponse(
                status_code=400, content={"error": "Body must be a list of block objects"}
            )
        session_id = request.headers.get("X-Session-Id") or ""
        if not session_id:
            items_with_sid = [i for i in body if isinstance(i, dict) and i.get("session_id")]
            if items_with_sid:
                session_id = items_with_sid[0]["session_id"]
        if not session_id:
            return _JSONResponse(
                status_code=400,
                content={
                    "error": ("session id required (X-Session-Id header or session_id on items)")
                },
            )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _write_executor, otlp_store.ingest_journal_blocks, session_id, body
        )
        return _JSONResponse(content=result)

    @app.post("/v1/sync")
    async def sync():
        """Wait until all spans queued before this call are written.

        Inserts a sentinel Event into the queue.  The ingest worker
        processes items in order; when it reaches the sentinel it sets
        the event, guaranteeing everything enqueued before it has been
        written to SQLite.  Unlike Queue.join(), this is not affected
        by new items arriving from other concurrent tasks.
        """
        event = asyncio.Event()
        await _ingest_queue.put(event)
        await asyncio.wait_for(event.wait(), timeout=30)
        return _JSONResponse(content={"synced": True})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class HeadlessOtlpBackend:
    """Lifecycle manager for a local headless OTLP backend.

    Starts a minimal FastAPI/uvicorn server in a background daemon thread.
    Each instance gets its own temporary SQLite database, so concurrent eval
    runs in separate processes don't interfere.

    Usage::

        backend = HeadlessOtlpBackend()
        base_url = backend.start()   # "http://127.0.0.1:<port>"
        try:
            enable_tracing(
                exporters=[exporters.journal(endpoint=f"{base_url}/v1/traces")], ...
            )
            # ... run eval ...
        finally:
            backend.stop()
    """

    def __init__(self):
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._port: int | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        """Start the backend. Returns base URL, e.g. 'http://127.0.0.1:54321'."""
        from nooa.viewer import otlp_store

        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "eval_traces.db"
        self._port = _find_free_port()

        # Redirect otlp_store to the temp DB before uvicorn opens any connections.
        # Thread-local connections in otlp_store are lazily opened using DB_PATH at
        # first access, so patching here (before the uvicorn thread starts) is safe.
        otlp_store.DB_PATH = db_path

        import uvicorn

        app = _make_headless_app()
        config = uvicorn.Config(app, host="127.0.0.1", port=self._port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._wait_ready()
        return f"http://127.0.0.1:{self._port}"

    def _wait_ready(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=0.5)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(
            f"Headless OTLP backend on port {self._port} did not become ready within {timeout}s"
        )

    def stop(self) -> None:
        """Shut down the server and delete the temporary database."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log.warning("Headless backend thread did not terminate within timeout")
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
