# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined Trace + Evaluation Viewer backend.

Single FastAPI app serving both trace viewer and evaluation viewer APIs.
Serves a React SPA from FRONTEND_DIR (Vite build output) with a catch-all
for client-side routing.
"""

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.requests import ClientDisconnect
from starlette.staticfiles import StaticFiles

load_dotenv()

from . import FRONTEND_DIR, memory_routes, otlp_store  # noqa: E402
from .annotation_routes import router as annotation_router  # noqa: E402
from .eval_routes import router as eval_router  # noqa: E402
from .explorer_routes import router as explorer_router  # noqa: E402
from .memory_routes import router as memory_router  # noqa: E402
from .trace_routes import router as trace_router  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write queue — decouple HTTP ingest latency from SQLite write latency.
#
# Under parallel eval runs, many subprocesses POST spans simultaneously.
# Calling otlp_store.ingest() directly in the async handler blocks the event
# loop on SQLite I/O, causing export timeouts and dropped spans.
#
# Instead: accept the POST into an asyncio.Queue and return 200 immediately.
# A single background task drains the queue serially — SQLite gets one writer
# at a time, the event loop is never blocked, and HTTP latency is near-zero.
# ---------------------------------------------------------------------------

_INGEST_QUEUE_MAXSIZE = int(os.environ.get("NOOA_VIEWER_INGEST_QUEUE_MAXSIZE", "16"))
_MAX_OTLP_BODY_BYTES = int(os.environ.get("NOOA_VIEWER_MAX_OTLP_BODY_BYTES", str(8 * 1024 * 1024)))
_MAX_JOURNAL_BODY_BYTES = int(
    os.environ.get("NOOA_VIEWER_MAX_JOURNAL_BODY_BYTES", str(2 * 1024 * 1024))
)
_MAX_JOURNAL_ITEMS = int(os.environ.get("NOOA_VIEWER_MAX_JOURNAL_ITEMS", "1000"))

_ingest_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_INGEST_QUEUE_MAXSIZE)
_QUEUE_WARN_THRESHOLD = max(1, _INGEST_QUEUE_MAXSIZE * 3 // 4)

# Single-writer thread pool: exactly one thread owns the write connection.
# Using max_workers=1 ensures serial SQLite writes with no concurrent access.
# The write thread uses otlp_store._get_write_db() (a thread-local connection)
# so it never shares a connection object with the event-loop read path.
_write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer")


_INGEST_MAX_BATCH = 32  # max payloads to commit in one SQLite transaction

_AUTH_TOKEN_ENV = "NOOA_VIEWER_AUTH_TOKEN"
_AUTH_COOKIE_NAME = "nooa_viewer_auth"
_PUBLIC_PATHS = frozenset({"/api/eval/health"})


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is limited to the local machine."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ensure_viewer_bind_is_safe(host: str) -> None:
    """Reject remote binds unless the operator configured bearer-token auth."""
    if not is_loopback_host(host) and not os.environ.get(_AUTH_TOKEN_ENV):
        raise ValueError(
            f"Refusing to bind viewer to {host!r} without {_AUTH_TOKEN_ENV}. "
            "Set a strong token or bind to 127.0.0.1."
        )


def _configured_auth_token() -> str | None:
    token = os.environ.get(_AUTH_TOKEN_ENV, "").strip()
    return token or None


def _request_auth_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value
    return request.cookies.get(_AUTH_COOKIE_NAME)


async def _read_limited_body(request: Request, max_bytes: int) -> bytes:
    """Read a request body without allowing an unbounded allocation."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from None

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="Request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_limited_json(request: Request, max_bytes: int) -> object:
    body = await _read_limited_body(request, max_bytes)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc


def _validate_journal_items(body: object) -> list[dict]:
    if not isinstance(body, list):
        raise HTTPException(status_code=400, detail="Body must be a list of objects")
    if len(body) > _MAX_JOURNAL_ITEMS:
        raise HTTPException(status_code=413, detail="Too many journal items")
    return body


async def _ingest_worker() -> None:
    """Drain _ingest_queue, writing batches to SQLite in a dedicated writer thread.

    Design:
    - Awaits the first queued item (yields to event loop between batches).
    - Greedily drains up to _INGEST_MAX_BATCH additional items already in the
      queue — all committed in a single SQLite transaction, amortising WAL sync
      (db.commit()) across the batch.
    - Runs ingest_batch_write() in a single-thread executor so SQLite I/O
      never blocks the event loop.  Parallel BSP exporters can always deliver
      spans without HTTP timeouts.
    - The writer thread uses _get_write_db() (thread-local connection), separate
      from the event-loop read connection, preventing the concurrent-access
      corruption that a default thread-pool executor caused.
    """
    loop = asyncio.get_running_loop()
    while True:
        # Wait for first item — yields control to event loop
        batch = [await _ingest_queue.get()]
        # Drain additional items already waiting (non-blocking)
        while len(batch) < _INGEST_MAX_BATCH:
            try:
                batch.append(_ingest_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        remaining = _ingest_queue.qsize()
        t0 = time.monotonic()
        try:
            await loop.run_in_executor(_write_executor, otlp_store.ingest_batch_write_bytes, batch)
        except Exception:
            log.exception("[ingest_worker] Failed to write batch of %d to SQLite", len(batch))
        finally:
            elapsed_ms = (time.monotonic() - t0) * 1000
            if remaining > 0 or elapsed_ms > 500:
                log.info(
                    "[ingest_worker] batch=%d  queued=%d  write=%.0fms",
                    len(batch),
                    remaining,
                    elapsed_ms,
                )
            for _ in batch:
                _ingest_queue.task_done()


# Suppress all successful (2xx/3xx) access logs — they're noise during eval runs.
# Errors (4xx/5xx) still appear.  Our own diagnostic log.info/warning messages
# go through the "nooa.viewer" logger, not "uvicorn.access", so they're unaffected.


class _QuietAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Keep 4xx/5xx responses visible
        return '" 4' in msg or '" 5' in msg


logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Frontend: %s", FRONTEND_DIR)
    log.info("Initializing SQLite trace store...")
    try:
        count = otlp_store.init_db()
    except otlp_store.DatabaseBusyAtStartup as exc:
        # Fail loudly with a fixable diagnostic — the process would
        # otherwise come up "healthy" but every journal/span POST would
        # hang on the lock.
        log.error("SQLite trace store is not writable:\n%s", exc)
        raise SystemExit(1) from exc
    log.info("Database ready: %d sessions in %s", count, otlp_store.DB_PATH)

    worker = asyncio.create_task(_ingest_worker())
    try:
        yield
    finally:
        # Drain the queue before shutdown so in-flight spans aren't lost.
        if not _ingest_queue.empty():
            log.info("Flushing %d pending ingest(s)…", _ingest_queue.qsize())
            await _ingest_queue.join()
        worker.cancel()
        memory_routes.close_stores()
        _write_executor.shutdown(wait=True)
        log.info("Shutdown complete")


app = FastAPI(title="NVIDIA OO Agents Viewer", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def viewer_auth_middleware(request: Request, call_next):
    """Require a configured bearer token for API, ingest, and SPA requests.

    Local loopback usage remains frictionless when no token is configured. A
    remote bind is rejected at startup unless a token exists, so enabling this
    middleware is mandatory for the only remotely reachable mode.
    """
    expected = _configured_auth_token()
    if expected is None or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    supplied = _request_auth_token(request)
    if supplied is not None and hmac.compare_digest(supplied, expected):
        return await call_next(request)

    query_token = request.query_params.get("token")
    is_spa_get = request.method == "GET" and not request.url.path.startswith(("/api/", "/v1/"))
    if is_spa_get and query_token is not None and hmac.compare_digest(query_token, expected):
        query = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
        target = str(request.url.replace(query=urlencode(query)))
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            _AUTH_COOKIE_NAME,
            expected,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return response

    return JSONResponse(
        status_code=401,
        content={"error": "viewer authentication required"},
        headers={"WWW-Authenticate": "Bearer"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trace_router)
app.include_router(eval_router)
app.include_router(annotation_router)
app.include_router(explorer_router)
app.include_router(memory_router)


# ============================================================================
# OTLP ingest endpoint
# ============================================================================


@app.post("/v1/traces")
async def otlp_ingest(request: Request):
    """Accept OTLP JSON ExportTraceServiceRequest and queue for async SQLite write.

    Returns 200 immediately — the actual SQLite write happens in a dedicated
    writer thread.  This prevents parallel eval runs from blocking the event
    loop and causing BSP export timeouts or ClientDisconnect errors.

    JSON parsing is offloaded to a thread executor so large Opus traces
    (3-5 MB payloads) don't block the event loop while parsing.
    """
    try:
        body_bytes = await _read_limited_body(request, _MAX_OTLP_BODY_BYTES)
    except ClientDisconnect:
        # BSP exporter timed out and closed the connection before we read the
        # body — log at WARNING (not ERROR) since this is a transient backpressure
        # signal, not a bug.  The BSP will retry on the next export cycle.
        log.warning("[otlp_ingest] Client disconnected before body was read — BSP may retry")
        return JSONResponse(status_code=499, content={"error": "client disconnected"})
    qsize = _ingest_queue.qsize()
    if qsize >= _QUEUE_WARN_THRESHOLD:
        log.warning(
            "[otlp_ingest] Write queue backlog: %d pending — "
            "SQLite may not be keeping up with ingest rate.",
            qsize,
        )
    try:
        _ingest_queue.put_nowait(body_bytes)
    except asyncio.QueueFull:
        log.warning("[otlp_ingest] Write queue full: %d pending", _ingest_queue.qsize())
        return JSONResponse(
            status_code=503,
            content={"error": "ingest queue full", "retryable": True},
        )
    return JSONResponse(content={"queued": True})


# ============================================================================
# Sync endpoint — wait for ingest queue to drain
# ============================================================================


@app.post("/v1/sync")
async def sync_ingest():
    """Block until the ingest queue is fully drained and all spans are in SQLite.

    Called by subprocess_worker after flush_traces() to ensure spans are
    readable before the trace is fetched for scoring.
    """
    try:
        await asyncio.wait_for(_ingest_queue.join(), timeout=30.0)
    except TimeoutError:
        qsize = _ingest_queue.qsize()
        log.error(
            "[sync_ingest] Timeout waiting for queue drain — worker may be stuck. Queue size: %d",
            qsize,
        )
        return JSONResponse(
            status_code=503,
            content={"error": "timeout waiting for queue drain", "queue_size": qsize},
        )
    return JSONResponse(content={"synced": True})


# ============================================================================
# Message journal endpoints
# ============================================================================


@app.post("/v1/journal/messages")
async def journal_messages_ingest(request: Request):
    """Accept a batch of content-addressed message records.

    Body: list of ``{"h": "<hash>", "msg": {<message dict>}}`` objects.
    Already-stored hashes are silently skipped.

    Offloads the SQLite write to the single-writer executor so the event
    loop is never blocked — same pattern as /v1/traces ingest.
    """
    import sqlite3

    body = _validate_journal_items(await _read_limited_json(request, _MAX_JOURNAL_BODY_BYTES))
    n_items = len(body)
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        result = await loop.run_in_executor(
            _write_executor, otlp_store.ingest_journal_messages, body
        )
    except sqlite3.OperationalError as exc:
        log.warning("[journal/messages] SQLite write failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": f"sqlite busy: {exc}", "retryable": True},
        )
    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 500:
        log.warning(
            "[journal/messages] slow write: %.0fms  items=%d  (executor backlog likely)",
            elapsed_ms,
            n_items,
        )
    return JSONResponse(content=result)


@app.post("/v1/journal/calls")
async def journal_call_ingest(request: Request):
    """Accept a single LLM call record with input/output hash lists.

    Offloads the SQLite write to the single-writer executor.
    """
    import sqlite3

    body = await _read_limited_json(request, _MAX_JOURNAL_BODY_BYTES)
    if not isinstance(body, dict) or not body.get("call_id") or not body.get("session_id"):
        return JSONResponse(
            status_code=400,
            content={"error": "call_id and session_id are required"},
        )
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        result = await loop.run_in_executor(_write_executor, otlp_store.ingest_journal_call, body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except sqlite3.OperationalError as exc:
        log.warning(
            "[journal/calls] SQLite write failed: %s  call_id=%s",
            exc,
            body.get("call_id", "?"),
        )
        return JSONResponse(
            status_code=503,
            content={"error": f"sqlite busy: {exc}", "retryable": True},
        )
    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 500:
        log.warning(
            "[journal/calls] slow write: %.0fms  call_id=%s  (executor backlog likely)",
            elapsed_ms,
            body.get("call_id", "?"),
        )
    return JSONResponse(content=result)


@app.post("/v1/journal/blocks")
async def journal_blocks_ingest(request: Request):
    """Accept a batch of content-addressed message blocks for a session.

    Body: list of ``{"hash": "<sha256:...>", "content": "<utf-8 string>"}``.
    Per-session idempotent on hash.  The exporter posts the session id in
    the ``X-Session-Id`` header (the legacy journal callback also tags
    individual blobs with their session); we accept either source.

    Offloads the SQLite write to the single-writer executor so the event
    loop is never blocked.
    """
    import sqlite3

    body = _validate_journal_items(await _read_limited_json(request, _MAX_JOURNAL_BODY_BYTES))
    session_id = request.headers.get("X-Session-Id") or ""
    if not session_id:
        # Fallback: allow the session_id on each item (rarely used but
        # symmetric with /v1/journal/calls).
        items_with_sid = [i for i in body if isinstance(i, dict) and i.get("session_id")]
        if items_with_sid:
            session_id = items_with_sid[0]["session_id"]
    if not session_id:
        return JSONResponse(
            status_code=400,
            content={"error": "session id required (X-Session-Id header or session_id on items)"},
        )
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        result = await loop.run_in_executor(
            _write_executor, otlp_store.ingest_journal_blocks, session_id, body
        )
    except sqlite3.OperationalError as exc:
        log.warning("[journal/blocks] SQLite write failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": f"sqlite busy: {exc}", "retryable": True},
        )
    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > 500:
        log.warning(
            "[journal/blocks] slow write: %.0fms  items=%d  (executor backlog likely)",
            elapsed_ms,
            len(body),
        )
    return JSONResponse(content=result)


@app.get("/api/traces/{session_id:path}/calls")
def get_session_calls(session_id: str):
    """Return all LLM calls for a session with fully reconstructed messages."""
    if not otlp_store.session_exists(session_id):
        return JSONResponse(status_code=404, content={"error": f"Session not found: {session_id}"})
    return JSONResponse(content=otlp_store.get_session_calls(session_id))


# ============================================================================
# Unified refresh endpoint
# ============================================================================


@app.post("/api/refresh")
def refresh_all():
    """Return current store stats."""
    stats = otlp_store.get_stats()
    return {
        "status": "ok",
        "sessions_found": stats["sessions"],
        "experiments_found": stats["experiments"],
    }


# ============================================================================
# Frontend serving — React SPA
# NOTE: The catch-all GET route MUST be registered AFTER all API GET routes.
# ============================================================================

app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/{path:path}")
def spa_catchall(request: Request, path: str):
    """Serve static files if they exist, otherwise index.html for client-side routing."""
    file_path = (FRONTEND_DIR / path).resolve()
    if path and file_path.is_file() and file_path.is_relative_to(FRONTEND_DIR.resolve()):
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("NOOA_TRACE_VIEWER_HOST", "127.0.0.1")
    port = int(
        os.environ.get("NOOA_TRACE_VIEWER_PORT")
        or os.environ.get("NEMO_OO_TRACE_VIEWER_PORT", "5001")
    )
    ensure_viewer_bind_is_safe(host)
    log.info("Starting viewer on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
