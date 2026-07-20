# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-dispatch test: two journal exporters -> both receivers get every call.

T2 and T4 drive ``MessageJournalCallback`` by hand because litellm's
``mock_response`` shortcut bypasses the callback chain.  That sidestepped
the bug we're trying to guard against -- "litellm only delivers
``log_success_event`` to one of two same-class callbacks".  This test
plugs into litellm's ``custom_provider_map`` instead, which routes
``acompletion`` through the *full* callback chain without any network
call, then asserts both running HTTP recorders saw the journal POSTs.

The backends are minimal HTTP recorders rather than real
``HeadlessOtlpBackend``s because two of those in the same process
clobber each other's ``otlp_store`` module state.  All we need is to
verify that the journal exporter posts to *both* of them; the receiver-
side persistence and reconstruction is covered by T5.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import pytest


class _Recorder:
    """Tiny localhost HTTP server that records every POST body it sees,
    *and* answers OpenAI-shape ``/chat/completions`` so litellm can dispatch
    a real (network-roundtripping) call against it.

    The OpenAI shim is what makes this work as a fan-out test fixture:
    ``litellm.acompletion(model="openai/x", api_base=<recorder>)`` fires
    the full callback chain on the way in (``log_pre_api_call``) and out
    (``log_success_event``), unlike ``mock_response`` or
    ``custom_provider_map`` which short-circuit it.
    """

    _CHAT_RESPONSE = {
        "id": "resp-fanout",
        "object": "chat.completion",
        "created": 0,
        "model": "fanout-stub",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "fixed reply"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

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

                # OpenAI completions shim so litellm thinks it talked to
                # a real provider.
                if self.path.endswith("/chat/completions"):
                    body_out = json.dumps(_Recorder._CHAT_RESPONSE).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body_out)))
                    self.end_headers()
                    self.wfile.write(body_out)
                    return

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


@pytest.fixture
def llm_endpoint():
    """Spin up a third recorder that *also* serves OpenAI ``/chat/completions``,
    used as litellm's ``api_base``.  Distinct from the journal recorders
    so we don't conflate "the LLM call" with "the journal POSTs"."""
    rec = _Recorder()
    base = rec.start()
    try:
        yield rec, base
    finally:
        rec.stop()


def test_real_dispatch_fans_out_to_both_recorders(two_recorders, llm_endpoint):
    """Drive a real ``litellm.completion`` with two ``JournalExporter``s
    pointed at two recorders.  Both must receive a ``/v1/journal/calls``
    POST with the same call_id.  This is the test that guards against
    the original 'litellm only delivers log_success_event to one of two
    same-class callbacks' bug -- prior tests papered over it by driving
    the callbacks in a Python ``for`` loop, which can't possibly fail.

    Uses sync ``completion`` rather than ``acompletion`` because the
    installed litellm version reliably fires the callback chain on the
    sync path; the async path has timing/registration quirks that
    aren't worth working around for this integration test.  The
    fan-out fix lives entirely in :class:`MessageJournalCallback`'s
    destination list, which is shared between sync and async paths.
    """
    import litellm

    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    base_a = f"http://127.0.0.1:{rec_a.port}"
    base_b = f"http://127.0.0.1:{rec_b.port}"
    _llm_rec, llm_base = llm_endpoint

    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"{base_a}/v1/traces"),
            exporters.journal(endpoint=f"{base_b}/v1/traces"),
        ]
    )

    set_session("real-dispatch-fanout")

    response = litellm.completion(
        model="openai/gpt-fanout",
        messages=[{"role": "user", "content": "hello fanout"}],
        api_base=f"{llm_base}/v1",
        api_key="not-real",
    )
    assert response.choices[0].message.content == "fixed reply"

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
async def test_real_dispatch_async_fans_out_to_both_recorders(two_recorders, llm_endpoint):
    """Same fan-out invariant on the async path.  litellm's async success
    handler is a deferred task on the running loop, so the test must
    ``await`` after the call to give the loop time to run it -- a
    subtle gotcha that masked async dispatch as "broken" earlier."""
    import litellm

    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    base_a = f"http://127.0.0.1:{rec_a.port}"
    base_b = f"http://127.0.0.1:{rec_b.port}"
    _llm_rec, llm_base = llm_endpoint

    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"{base_a}/v1/traces"),
            exporters.journal(endpoint=f"{base_b}/v1/traces"),
        ]
    )
    set_session("async-real-dispatch-fanout")

    response = await litellm.acompletion(
        model="openai/gpt-fanout",
        messages=[{"role": "user", "content": "hello async fanout"}],
        api_base=f"{llm_base}/v1",
        api_key="not-real",
    )
    assert response.choices[0].message.content == "fixed reply"

    # Yield to the loop so litellm's async success task can run; then
    # force_flush joins the journal POST daemon threads as in the sync
    # case.  Without the await-sleep, ``asyncio.run`` would tear down
    # the loop before the deferred success task fires, and the test
    # would observe an empty recorder for entirely uninteresting
    # event-loop reasons.
    import asyncio

    await asyncio.sleep(0.1)

    from nooa.tracing import _provider

    assert _provider is not None
    _provider.force_flush()

    # Filter by session_id: prior tests in the same pytest run use
    # ``mock_response``, which schedules a deferred async-success log
    # task on the running loop.  ``asyncio.run`` tears down their loop
    # before the task fires, so it queues into *this* test's loop and
    # POSTs against our recorders with the prior test's session_id.
    # Counting only posts whose body matches our session is the only
    # robust way to avoid that test-pollution interaction.
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
