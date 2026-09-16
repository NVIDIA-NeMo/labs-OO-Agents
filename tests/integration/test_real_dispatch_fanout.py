# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two journal exporters receive the same UnifiedLLM call through SDK dispatch."""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import pytest


class _Recorder:
    """Local HTTP receiver recording the journal's actual POST bodies."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict | list]] = []
        self._lock = threading.Lock()
        self._server = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def start(self) -> str:
        """Start the recorder on an ephemeral port; return the base URL."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        recorder = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                try:
                    parsed: Any = json.loads(body) if body else None
                except json.JSONDecodeError:
                    parsed = body
                with recorder._lock:
                    recorder.posts.append((self.path, parsed))

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                body_out = b'{"ok":true}'
                self.send_header("Content-Length", str(len(body_out)))
                self.end_headers()
                self.wfile.write(body_out)

            def log_message(self, *args: Any, **kwargs: Any) -> None:
                pass  # silence default stderr access log

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def posts_to(self, path: str) -> list[Any]:
        with self._lock:
            return [body for p, body in self.posts if p == path]


@pytest.fixture
def two_recorders():
    a = _Recorder()
    b = _Recorder()
    a.start()
    b.start()
    try:
        yield a, b
    finally:
        a.stop()
        b.stop()


@pytest.mark.parametrize("transport", ["litellm", "direct"])
def test_real_dispatch_fans_out_to_both_recorders(two_recorders, mock_model_client, transport):
    """SDK dispatch fans out one journal record to both receivers."""

    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    base_a = f"http://127.0.0.1:{rec_a.port}"
    base_b = f"http://127.0.0.1:{rec_b.port}"

    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"{base_a}/v1/traces"),
            exporters.journal(endpoint=f"{base_b}/v1/traces"),
        ]
    )

    set_session("real-dispatch-fanout")

    with mock_model_client("fixed reply", transport) as client:
        response = client.call([{"role": "user", "content": "hello fanout"}])
    assert response.content == "fixed reply"

    # force_flush() joins in-flight POST daemon threads, but the daemon
    # only returns *after the recorder has accepted the body*; the
    # recorder side of the connection is then closed by the worker
    # thread.  The recorder's request handler appends to its list
    # before sending the response, so once the daemon returns we know
    # the recorder has the post.  Add a brief poll for safety against
    # any kernel-level scheduling jitter under heavy pytest output.
    from nooa.tracing import _provider

    assert _provider is not None
    _provider.force_flush()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        calls_a = rec_a.posts_to("/v1/journal/calls")
        calls_b = rec_b.posts_to("/v1/journal/calls")
        if calls_a and calls_b:
            break
        time.sleep(0.02)
    else:
        calls_a = rec_a.posts_to("/v1/journal/calls")
        calls_b = rec_b.posts_to("/v1/journal/calls")

    assert len(calls_a) == 1, (
        f"recorder A got {len(calls_a)} call POSTs; full posts: {rec_a.posts!r}"
    )
    assert len(calls_b) == 1, (
        f"recorder B got {len(calls_b)} call POSTs; this is the fan-out "
        f"bug: only one same-class callback received log_success_event. "
        f"full posts: {rec_b.posts!r}"
    )

    # Both destinations must receive the *same* logical record -- a bug
    # that fanned out *different* records per destination would slip
    # through a "non-empty on both" assertion.
    assert calls_a[0]["call_id"] == calls_b[0]["call_id"]
    assert calls_a[0]["session_id"] == calls_b[0]["session_id"]
    assert calls_a[0]["input_skeleton"] == calls_b[0]["input_skeleton"]
    assert calls_a[0]["output_messages"] == calls_b[0]["output_messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["litellm", "direct"])
async def test_real_dispatch_async_fans_out_to_both_recorders(
    two_recorders, mock_model_client, transport
):
    """The asynchronous client has the same fan-out contract."""

    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    base_a = f"http://127.0.0.1:{rec_a.port}"
    base_b = f"http://127.0.0.1:{rec_b.port}"

    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"{base_a}/v1/traces"),
            exporters.journal(endpoint=f"{base_b}/v1/traces"),
        ]
    )
    set_session("async-real-dispatch-fanout")

    async with mock_model_client("fixed reply", transport) as client:
        response = await client.acall([{"role": "user", "content": "hello async fanout"}])
    assert response.content == "fixed reply"

    import asyncio

    from nooa.tracing import _provider

    assert _provider is not None
    _provider.force_flush()

    # Background journal delivery can finish after the model call returns.
    sid = "async-real-dispatch-fanout"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        calls_a = [b for b in rec_a.posts_to("/v1/journal/calls") if b.get("session_id") == sid]
        calls_b = [b for b in rec_b.posts_to("/v1/journal/calls") if b.get("session_id") == sid]
        if calls_a and calls_b:
            break
        await asyncio.sleep(0.02)

    assert len(calls_a) == 1, (
        f"recorder A got {len(calls_a)} call POSTs for session {sid!r} on "
        f"async path; full posts: {rec_a.posts!r}"
    )
    assert len(calls_b) == 1, (
        f"recorder B got {len(calls_b)} call POSTs for session {sid!r} on "
        f"async path; full posts: {rec_b.posts!r}"
    )
    assert calls_a[0]["call_id"] == calls_b[0]["call_id"]
    assert calls_a[0]["input_skeleton"] == calls_b[0]["input_skeleton"]
    assert calls_a[0]["output_messages"] == calls_b[0]["output_messages"]
