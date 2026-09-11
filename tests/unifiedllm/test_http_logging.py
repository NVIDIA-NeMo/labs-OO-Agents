# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from nooa.unifiedllm.http_logging import enable_http_request_logging


def test_opaque_reasoning_state_is_redacted_from_http_debug_payloads(
    tmp_path, secret_header_server
) -> None:
    payload = {
        "input": [
            {"encrypted_content": "provider-secret"},
            {"type": "thinking", "thinking": "visible", "signature": "anthropic-secret"},
            {"type": "redacted_thinking", "data": "anthropic-redacted"},
            {"thought_signature": "gemini-secret"},
            {"_nooa_llm_state": {"payload": {"items": ["opaque"]}}},
        ]
    }

    disable = enable_http_request_logging(output_dir=tmp_path, verbose=False)
    try:
        httpx.post(f"http://127.0.0.1:{secret_header_server.server_port}/llm", json=payload)
    finally:
        disable()
    redacted = json.loads(next(tmp_path.glob("request_*.json")).read_text())

    assert redacted["input"] == [
        {"encrypted_content": "[REDACTED]"},
        {"type": "thinking", "thinking": "visible", "signature": "[REDACTED]"},
        {"type": "redacted_thinking", "data": "[REDACTED]"},
        {"thought_signature": "[REDACTED]"},
        {"_nooa_llm_state": "[REDACTED]"},
    ]


class _SecretHeaderHandler(BaseHTTPRequestHandler):
    status_code = 500

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("content-length", "0")))
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Set-Cookie", "sid=response-cookie-secret")
        self.send_header("X-Api-Key", "response-api-key-secret")
        self.send_header("X-Session-Token", "response-session-secret")
        self.send_header("X-Auth-Token", "response-auth-secret")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "access_token": "response-access-secret",
                    "nested": {"client_secret": "response-client-secret"},
                    "message": "safe",
                }
            ).encode()
        )

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def secret_header_server() -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SecretHeaderHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _post_to_server(server: ThreadingHTTPServer) -> httpx.Response:
    return httpx.post(
        f"http://127.0.0.1:{server.server_port}/llm",
        headers={"Authorization": "Bearer request-auth-secret"},
        json={
            "model": "header-redaction-model",
            "api_key": "request-api-key-secret",
        },
    )


def test_save_responses_redacts_sensitive_response_headers_and_bodies(
    tmp_path: Path, secret_header_server: ThreadingHTTPServer
) -> None:
    disable = enable_http_request_logging(
        output_dir=tmp_path, save_responses=True, errors_only=False, verbose=False
    )
    try:
        response = _post_to_server(secret_header_server)
    finally:
        disable()

    assert response.headers["x-session-token"] == "response-session-secret"

    response_log = json.loads(next(tmp_path.glob("response_*.json")).read_text())
    assert response_log["headers"]["set-cookie"] == "***REDACTED***"
    assert response_log["headers"]["x-api-key"] == "***REDACTED***"
    assert response_log["headers"]["x-session-token"] == "***REDACTED***"
    assert response_log["headers"]["x-auth-token"] == "***REDACTED***"
    assert response_log["body"]["access_token"] == "[REDACTED]"
    assert response_log["body"]["nested"]["client_secret"] == "[REDACTED]"

    persisted = json.dumps(response_log)
    assert "response-cookie-secret" not in persisted
    assert "response-api-key-secret" not in persisted
    assert "response-session-secret" not in persisted
    assert "response-auth-secret" not in persisted
    assert "response-access-secret" not in persisted
    assert "response-client-secret" not in persisted


def test_errors_only_jsonl_redacts_sensitive_headers_and_bodies(
    tmp_path: Path, secret_header_server: ThreadingHTTPServer
) -> None:
    disable = enable_http_request_logging(
        output_dir=tmp_path, save_responses=True, errors_only=True, verbose=False
    )
    try:
        response = _post_to_server(secret_header_server)
    finally:
        disable()

    assert response.status_code == 500

    entry = json.loads((tmp_path / "llm_errors.jsonl").read_text())
    assert entry["request"]["headers"]["authorization"] == "***REDACTED***"
    assert entry["request"]["body"]["api_key"] == "[REDACTED]"
    assert entry["response"]["headers"]["set-cookie"] == "***REDACTED***"
    assert entry["response"]["headers"]["x-api-key"] == "***REDACTED***"
    assert entry["response"]["headers"]["x-session-token"] == "***REDACTED***"
    assert entry["response"]["headers"]["x-auth-token"] == "***REDACTED***"
    assert entry["response"]["body"]["access_token"] == "[REDACTED]"
    assert entry["response"]["body"]["nested"]["client_secret"] == "[REDACTED]"

    persisted = json.dumps(entry)
    assert "request-auth-secret" not in persisted
    assert "request-api-key-secret" not in persisted
    assert "response-cookie-secret" not in persisted
    assert "response-api-key-secret" not in persisted
    assert "response-session-secret" not in persisted
    assert "response-auth-secret" not in persisted
    assert "response-access-secret" not in persisted
    assert "response-client-secret" not in persisted
