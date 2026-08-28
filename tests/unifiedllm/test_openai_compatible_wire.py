# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic wire contract for the real NOOA -> AnyLLM OpenAI-compatible path."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nooa.unifiedllm import LLMChunk, LLMResponse, get_llm_client, reload_registry


def test_openai_compatible_completion_and_stream_wire_contract(tmp_path, monkeypatch):
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass  # Never log request headers (which include the test credential).

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length", "0"))
            captured.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("authorization"),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            if captured[-1]["body"].get("stream"):
                events = [
                    {
                        "id": "x",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "wire-model",
                        "choices": [
                            {"index": 0, "delta": {"content": "wire"}, "finish_reason": None}
                        ],
                    },
                    {
                        "id": "x",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "wire-model",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ]
                payload = (
                    "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                    + "data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload.encode())))
                self.end_headers()
                self.wfile.write(payload.encode())
                return
            payload = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "wire-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    secret = "wire-secret-not-for-logs"
    monkeypatch.setenv("WIRE_TEST_API_KEY", secret)
    path = tmp_path / "models.yaml"
    path.write_text(
        f"""models:
  wire:
    model_name: wire-model
    provider: openai-compatible
    endpoint: http://127.0.0.1:{server.server_port}/v1
    api_key_env: WIRE_TEST_API_KEY
    request:
      max_output_tokens: 23
      temperature: 0.1
"""
    )
    reload_registry(path)
    client = get_llm_client("wire")
    try:
        response = client.call([{"role": "user", "content": "hi"}])
        chunks = list(client.stream([{"role": "user", "content": "stream"}]))
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert isinstance(response, LLMResponse)
    assert response.content == "hello"
    assert chunks and all(isinstance(chunk, LLMChunk) for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == "wire"
    assert len(captured) == 2
    for request in captured:
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] == f"Bearer {secret}"
        assert request["body"]["model"] == "wire-model"
        assert request["body"]["max_completion_tokens"] == 23
        assert "max_output_tokens" not in request["body"]
        assert "max_tokens" not in request["body"]
    assert captured[1]["body"]["stream"] is True
    # Public outputs cannot expose OpenAI/AnyLLM backend object types.
    assert "openai." not in repr(response).lower()
    assert "any_llm" not in repr(response).lower()
