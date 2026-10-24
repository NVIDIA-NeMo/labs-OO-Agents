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
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
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

_INGEST_MAX_BODY_BYTES = 16 * 1024 * 1024
_INGEST_QUEUE_MAX_ITEMS = 64


async def _read_limited_body(request: Request) -> bytes:
    """Read at most ``_INGEST_MAX_BODY_BYTES`` without buffering more."""
    raw_length = request.headers.get("content-length")
    if raw_length and raw_length.isdecimal() and int(raw_length) > _INGEST_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="ingest body too large")

    chunks: list[bytes] = []
    body_size = 0
    async for chunk in request.stream():
        body_size += len(chunk)
        if body_size > _INGEST_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="ingest body too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_limited_json(request: Request) -> object:
    return json.loads(await _read_limited_body(request))


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

_ingest_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_INGEST_QUEUE_MAX_ITEMS)
_QUEUE_WARN_THRESHOLD = 48

# Single-writer thread pool: exactly one thread owns the write connection.
# Using max_workers=1 ensures serial SQLite writes with no concurrent access.
# The write thread uses otlp_store._get_write_db() (a thread-local connection)
# so it never shares a connection object with the event-loop read path.
_write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer")


_INGEST_MAX_BATCH = 32  # max payloads to commit in one SQLite transaction


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
    # Stated at startup so a rejected Host is diagnosable from the log alone.
    extra_hosts = allowed_hosts()
    if "*" in extra_hosts:
        log.warning("Host check disabled (NOOA_VIEWER_ALLOWED_HOSTS=*) — DNS rebinding is possible")
    else:
        log.info(
            "Browser requests accepted for Host: localhost, IP literals, %s, %s.*%s",
            _OWN_HOSTNAME,
            _OWN_HOSTNAME,
            f", {', '.join(sorted(extra_hosts))}" if extra_hosts else "",
        )

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
        # After the writer thread is joined, so nothing is mid-write.
        otlp_store.close_db()
        log.info("Shutdown complete")


app = FastAPI(title="NVIDIA OO Agents Viewer", version="2.0.0", lifespan=lifespan)


SESSION_COOKIE = "nooa_viewer_session"


def viewer_auth_token() -> str | None:
    """Return the configured viewer token, or ``None`` when unset/blank."""
    return os.environ.get("NOOA_VIEWER_AUTH_TOKEN", "").strip() or None


def _is_loopback_client(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        # Starlette's in-process test transport uses this non-network host.
        return client_host == "testclient"


def _require_viewer_authorization(request: Request) -> None:
    """Authorize a protected viewer API route.

    Three independent credentials, checked as OR rather than else-if:

    1. Loopback origin — local agents ingest with no configuration, which is the
       default single-machine workflow.
    2. Session cookie — the browser, bootstrapped from ``?token=`` (see
       :func:`_maybe_issue_session_cookie`). The SPA cannot set an
       ``Authorization`` header on a navigation, so a header-only scheme locks
       the UI out entirely.
    3. ``Authorization: Bearer`` — remote programmatic clients (exporters).

    An earlier revision made the token *replace* the loopback allowance, which
    left no configuration in which both the UI and remote ingest worked.
    """
    if _is_loopback_client(request):
        return

    expected = viewer_auth_token()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="set NOOA_VIEWER_AUTH_TOKEN before exposing the viewer",
        )

    cookie = request.cookies.get(SESSION_COOKIE, "")
    if cookie and hmac.compare_digest(cookie, expected):
        return

    authorization = request.headers.get("Authorization", "")
    if hmac.compare_digest(authorization, f"Bearer {expected}"):
        return

    raise HTTPException(
        status_code=401,
        detail="viewer authorization required",
        headers={"WWW-Authenticate": "Bearer"},
    )


_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "NOOA_VIEWER_CORS_ORIGINS", "http://localhost:5001,http://127.0.0.1:5001"
    ).split(",")
    if origin.strip()
]
_protected = [Depends(_require_viewer_authorization)]


# Headers no non-browser client sends. Chrome 76+/Firefox 90+/Safari 16.4+ set
# Sec-Fetch-*; anything issuing a cross-origin fetch sets Origin. urllib and
# requests — which the OTLP exporter and journal client use — set neither.
_BROWSER_MARKER_HEADERS = ("sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest", "origin")

_OWN_HOSTNAME = socket.gethostname().lower()


def _request_is_browser_shaped(request: Request) -> bool:
    return any(h in request.headers for h in _BROWSER_MARKER_HEADERS)


def _host_without_port(raw: str) -> str:
    """``example.com:5001`` -> ``example.com``; ``[::1]:5001`` -> ``[::1]``."""
    raw = raw.strip().lower()
    if raw.startswith("["):
        end = raw.find("]")
        return raw[: end + 1] if end != -1 else raw
    return raw.rsplit(":", 1)[0] if ":" in raw else raw


def _host_is_our_own(host: str) -> bool:
    """Whether *host* is a name or address this machine legitimately answers to."""
    if host in {"localhost", "[::1]"}:
        return True
    try:
        # An IP literal cannot be rebound: the attack needs a name whose DNS the
        # attacker controls, and nobody can make a bare address their origin
        # without controlling that address.
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        pass
    # Prefix match, not equality: when the qualified name comes from a DNS
    # search domain, gethostname() returns only the leading label, so a client
    # addressing us as ``<host>.<domain>`` is legitimate. Comparing exactly
    # rejected real traffic and 400'd every span POST from such clients.
    return host == _OWN_HOSTNAME or host.startswith(_OWN_HOSTNAME + ".")


def allowed_hosts() -> set[str]:
    """Extra ``Host`` values accepted beyond this machine's own names."""
    configured = os.environ.get("NOOA_VIEWER_ALLOWED_HOSTS", "").strip()
    return {h.strip().lower() for h in configured.split(",") if h.strip()}


@app.middleware("http")
async def _enforce_host_header(request: Request, call_next):
    """Reject browser requests whose ``Host`` is not one of ours — DNS rebinding.

    A malicious page can publish a short-TTL record, have the browser fetch its
    own origin, then re-answer DNS with 127.0.0.1. The browser connects to this
    server still believing the origin is the attacker's, so same-origin policy
    lets their script *read* the response — and the request arrives from
    loopback, which :func:`_require_viewer_authorization` allows with no
    credential. Binding to localhost does not help; loopback is the target.

    Only browser-shaped requests are checked. Rebinding is inherently a browser
    attack, so exempting programmatic clients costs nothing — and it means this
    can never reject a span export, which is how an earlier, stricter version
    silently broke trace ingest.
    """
    extra = allowed_hosts()
    if "*" not in extra and _request_is_browser_shaped(request):
        host = _host_without_port(request.headers.get("host", ""))
        if host and not _host_is_our_own(host) and host not in extra:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        f"Host {host!r} is not this server. If you reach the viewer by "
                        "this name (a CNAME, or through a proxy), add it to "
                        "NOOA_VIEWER_ALLOWED_HOSTS; '*' disables the check."
                    )
                },
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Session-Id"],
)

app.include_router(trace_router, dependencies=_protected)
app.include_router(eval_router, dependencies=_protected)
app.include_router(annotation_router, dependencies=_protected)
app.include_router(explorer_router, dependencies=_protected)
app.include_router(memory_router, dependencies=_protected)


# ============================================================================
# OTLP ingest endpoint
# ============================================================================


@app.post("/v1/traces", dependencies=_protected)
async def otlp_ingest(request: Request):
    """Accept OTLP JSON ExportTraceServiceRequest and queue for async SQLite write.

    Returns 200 immediately — the actual SQLite write happens in a dedicated
    writer thread.  This prevents parallel eval runs from blocking the event
    loop and causing BSP export timeouts or ClientDisconnect errors.

    JSON parsing is offloaded to a thread executor so large Opus traces
    (3-5 MB payloads) don't block the event loop while parsing.
    """
    try:
        body_bytes = await _read_limited_body(request)
    except ClientDisconnect:
        log.warning("[otlp_ingest] Client disconnected before body was read — BSP may retry")
        return JSONResponse(status_code=499, content={"error": "client disconnected"})
    qsize = _ingest_queue.qsize()
    if qsize >= _QUEUE_WARN_THRESHOLD:
        log.warning("[otlp_ingest] Write queue backlog: %d pending", qsize)
    try:
        _ingest_queue.put_nowait(body_bytes)
    except asyncio.QueueFull:
        return JSONResponse(status_code=503, content={"error": "ingest queue is full"})
    return JSONResponse(content={"queued": True})


# ============================================================================
# Sync endpoint — wait for ingest queue to drain
# ============================================================================


@app.post("/v1/sync", dependencies=_protected)
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


@app.post("/v1/journal/messages", dependencies=_protected)
async def journal_messages_ingest(request: Request):
    """Accept a batch of content-addressed message records.

    Body: list of ``{"h": "<hash>", "msg": {<message dict>}}`` objects.
    Already-stored hashes are silently skipped.

    Offloads the SQLite write to the single-writer executor so the event
    loop is never blocked — same pattern as /v1/traces ingest.
    """
    import sqlite3

    body = await _read_limited_json(request)

    if not isinstance(body, list):
        return JSONResponse(
            status_code=400,
            content={"error": "Body must be a list of message objects"},
        )
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


@app.post("/v1/journal/calls", dependencies=_protected)
async def journal_call_ingest(request: Request):
    """Accept a single LLM call record with input/output hash lists.

    Offloads the SQLite write to the single-writer executor.
    """
    import sqlite3

    body = await _read_limited_json(request)

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


@app.post("/v1/journal/blocks", dependencies=_protected)
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

    body = await _read_limited_json(request)

    if not isinstance(body, list):
        return JSONResponse(
            status_code=400,
            content={"error": "Body must be a list of block objects"},
        )
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


@app.get("/api/traces/{session_id:path}/calls", dependencies=_protected)
def get_session_calls(session_id: str):
    """Return all LLM calls for a session with fully reconstructed messages."""
    if not otlp_store.session_exists(session_id):
        return JSONResponse(status_code=404, content={"error": f"Session not found: {session_id}"})
    return JSONResponse(content=otlp_store.get_session_calls(session_id))


# ============================================================================
# Unified refresh endpoint
# ============================================================================


@app.post("/api/refresh", dependencies=_protected)
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


_SESSION_MAX_AGE = 30 * 24 * 3600


def _maybe_issue_session_cookie(request: Request, path: str) -> RedirectResponse | None:
    """Trade a valid ``?token=`` for a session cookie, then drop it from the URL.

    The SPA cannot attach an ``Authorization`` header to a navigation, so this
    is how a browser authenticates. Stripping the token via redirect means only
    the one-time bootstrap link carries the secret — trace deep links shared
    afterwards are plain URLs, so the token stays out of chat logs, browser
    history and screenshots.

    ``SameSite=Lax`` is deliberate: it still sends the cookie on top-level GET
    navigations, so a shared link works on click, while withholding it from
    cross-site POSTs, which blocks CSRF against the write and playground routes.
    """
    expected = viewer_auth_token()
    supplied = request.query_params.get("token")
    if not expected or not supplied or not hmac.compare_digest(supplied, expected):
        return None

    remaining = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
    target = f"/{path}"
    if remaining:
        target += "?" + urlencode(remaining)

    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        expected,
        max_age=_SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
        # Only meaningful behind TLS; setting it over plain HTTP would make the
        # cookie unusable.
        secure=request.url.scheme == "https",
    )
    return response


@app.get("/{path:path}")
def spa_catchall(request: Request, path: str):
    """Serve static files if they exist, otherwise index.html for client-side routing."""
    bootstrap = _maybe_issue_session_cookie(request, path)
    if bootstrap is not None:
        return bootstrap
    file_path = (FRONTEND_DIR / path).resolve()
    if path and file_path.is_file() and file_path.is_relative_to(FRONTEND_DIR.resolve()):
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get("NOOA_TRACE_VIEWER_PORT")
        or os.environ.get("NEMO_OO_TRACE_VIEWER_PORT", "5001")
    )
    log.info("Starting viewer on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
