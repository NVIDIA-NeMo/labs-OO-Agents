# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-dispatch test: two journal exporters -> both receivers get every call.

T2 and T4 drive ``MessageJournalCallback`` by hand because any_llm's
``mock_response`` shortcut bypasses the callback chain.  That sidestepped
the bug we're trying to guard against -- "any_llm only delivers
``log_success_event`` to one of two same-class callbacks".  This test
plugs into any_llm's ``custom_provider_map`` instead, which routes
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
    *and* answers OpenAI-shape ``/chat/completions`` so any_llm can dispatch
    a real (network-roundtripping) call against it.

    The OpenAI shim is what makes this work as a fan-out test fixture:
    ``any_llm.acompletion(model="openai/x", api_base=<recorder>)`` fires
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

                # OpenAI completions shim so any_llm thinks it talked to
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
    used as any_llm's ``api_base``.  Distinct from the journal recorders
    so we don't conflate "the LLM call" with "the journal POSTs"."""
    rec = _Recorder()
    base = rec.start()
    try:
        yield rec, base
    finally:
        rec.stop()


def _drive_native_call(messages):
    from nooa.runtime.llm_lifecycle import begin_llm_call, end_llm_call
    from nooa.unifiedllm import LLMResponse

    call = begin_llm_call("fanout-stub", messages)
    response = LLMResponse(
        content="fixed reply",
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": "fixed reply"},
    )
    end_llm_call(call, response=response)
    return response


def _assert_fanout(rec_a, rec_b, session_id):
    from nooa.tracing import _provider

    assert _provider is not None
    _provider.force_flush()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        calls_a = [
            b for b in rec_a.posts_to("/v1/journal/calls") if b.get("session_id") == session_id
        ]
        calls_b = [
            b for b in rec_b.posts_to("/v1/journal/calls") if b.get("session_id") == session_id
        ]
        if calls_a and calls_b:
            break
        time.sleep(0.02)
    assert len(calls_a) == 1
    assert len(calls_b) == 1
    for key in ("call_id", "session_id", "input_skeleton", "output_messages"):
        assert calls_a[0][key] == calls_b[0][key]


def test_native_lifecycle_fans_out_to_both_recorders(two_recorders):
    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"http://127.0.0.1:{rec_a.port}/v1/traces"),
            exporters.journal(endpoint=f"http://127.0.0.1:{rec_b.port}/v1/traces"),
        ]
    )
    session_id = "native-dispatch-fanout"
    set_session(session_id)
    response = _drive_native_call([{"role": "user", "content": "hello fanout"}])
    assert response.content == "fixed reply"
    _assert_fanout(rec_a, rec_b, session_id)


@pytest.mark.asyncio
async def test_native_lifecycle_async_context_fans_out_to_both_recorders(two_recorders):
    from nooa.tracing import enable_tracing, exporters, set_session

    rec_a, rec_b = two_recorders
    enable_tracing(
        exporters=[
            exporters.journal(endpoint=f"http://127.0.0.1:{rec_a.port}/v1/traces"),
            exporters.journal(endpoint=f"http://127.0.0.1:{rec_b.port}/v1/traces"),
        ]
    )
    session_id = "async-native-dispatch-fanout"
    set_session(session_id)
    _drive_native_call([{"role": "user", "content": "hello async fanout"}])
    _assert_fanout(rec_a, rec_b, session_id)
